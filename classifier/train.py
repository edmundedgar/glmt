"""Fine-tune a single topic head on LLM-labeled data (spec Component 5,
minus the HTTP endpoint -- this is the offline/CLI version).

Usage:
    python -m classifier.train                                     # binary uspol data, 9743 rows
    python -m classifier.train --source freeform                   # freeform-consolidated data, "us-politics"
    python -m classifier.train --source freeform --topic gaming    # any other consolidated canonical label

Balances negatives to 2x the positive count per the spec's default sampling
ratio, freezes the encoder, and trains only the topic's nn.Linear(768, 1) head.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from classifier.model import MAX_TOKENS, TopicClassifier, load_pretrained

BINARY_LABELED_PATH = Path(__file__).parent.parent / "data" / "labeled_uspol.jsonl"
FREEFORM_LABELED_PATH = Path(__file__).parent.parent / "data" / "freeform_labels_consolidated.jsonl"
WEIGHTS_DIR = Path(__file__).parent / "weights"

NEGATIVE_RATIO = 2  # per spec: sample 2x positive count when no ratio is given
VAL_FRACTION = 0.2
LR = 2e-4
BATCH_SIZE = 32
MAX_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
F1_DEPLOY_THRESHOLD = 0.7
SEED = 0


def load_binary_labeled(path: Path) -> list[dict]:
    """(uri, text, label: 'yes'/'no', confidence) -- from llm_label.py."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_freeform_labeled(path: Path, topic: str) -> list[dict]:
    """(uri, text, labels: [...]) -- from consolidate_labels.py. Converts to
    the same binary yes/no shape by checking membership of `topic`."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            label = "yes" if topic in row["labels"] else "no"
            rows.append({"uri": row["uri"], "text": row["text"], "label": label})
    return rows


def balanced_split(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    positives = [r for r in rows if r["label"] == "yes"]
    negatives = [r for r in rows if r["label"] == "no"]
    rng.shuffle(negatives)
    negatives = negatives[: NEGATIVE_RATIO * len(positives)]
    print(f"using {len(positives)} positives + {len(negatives)} negatives (ratio {NEGATIVE_RATIO}:1)")

    def stratified_split(items: list[dict]) -> tuple[list[dict], list[dict]]:
        rng.shuffle(items)
        n_val = max(1, int(len(items) * VAL_FRACTION))
        return items[n_val:], items[:n_val]

    pos_train, pos_val = stratified_split(positives)
    neg_train, neg_val = stratified_split(negatives)
    train = pos_train + neg_train
    val = pos_val + neg_val
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.no_grad()
def embed_texts(encoder, tokenizer, texts: list[str], device: str, batch_size: int = 64) -> torch.Tensor:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
        ).to(device)
        out = encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        embeddings.append(out.last_hidden_state[:, 0, :].cpu())
    return torch.cat(embeddings, dim=0)


def evaluate(head: nn.Linear, embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
    with torch.no_grad():
        probs = torch.sigmoid(head(embeddings).squeeze(-1))
        preds = (probs > 0.5).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["binary", "freeform"], default="binary")
    parser.add_argument("--topic", default=None, help="defaults to 'uspol' for binary, 'us-politics' for freeform")
    parser.add_argument("--data-path", type=Path, default=None, help="override the default path for --source freeform (e.g. data/local_llm_bulk_labeled.jsonl)")
    args = parser.parse_args()

    if args.source == "binary":
        topic = args.topic or "uspol"
        rows = load_binary_labeled(args.data_path or BINARY_LABELED_PATH)
        print(f"source=binary path={args.data_path or BINARY_LABELED_PATH} topic={topic!r}")
    else:
        topic = args.topic or "us-politics"
        data_path = args.data_path or FREEFORM_LABELED_PATH
        rows = load_freeform_labeled(data_path, topic)
        print(f"source=freeform path={data_path} topic={topic!r}")

    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    train_rows, val_rows = balanced_split(rows, rng)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    encoder, tokenizer, _ = load_pretrained()
    encoder.to(device)

    train_embeddings = embed_texts(encoder, tokenizer, [r["text"] for r in train_rows], device).to(device)
    train_labels = torch.tensor([1.0 if r["label"] == "yes" else 0.0 for r in train_rows], device=device)
    val_embeddings = embed_texts(encoder, tokenizer, [r["text"] for r in val_rows], device).to(device)
    val_labels = torch.tensor([1.0 if r["label"] == "yes" else 0.0 for r in val_rows], device=device)

    classifier = TopicClassifier(encoder, [topic]).to(device)
    head = classifier.heads[topic]
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    n_train = train_embeddings.size(0)

    for epoch in range(MAX_EPOCHS):
        head.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            batch_emb = train_embeddings[idx]
            batch_labels = train_labels[idx]
            optimizer.zero_grad()
            logits = head(batch_emb).squeeze(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        head.eval()
        metrics = evaluate(head, val_embeddings, val_labels)
        print(
            f"epoch {epoch + 1}: train_loss={total_loss / n_train:.4f} "
            f"val_precision={metrics['precision']:.3f} val_recall={metrics['recall']:.3f} val_f1={metrics['f1']:.3f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {epoch + 1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    head.load_state_dict(best_state)
    final_metrics = evaluate(head, val_embeddings, val_labels)
    print(f"\nbest validation metrics: {final_metrics}")

    if final_metrics["f1"] > F1_DEPLOY_THRESHOLD:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = WEIGHTS_DIR / f"{topic}.pt"
        torch.save(best_state, weights_path)
        print(f"F1 {final_metrics['f1']:.3f} > {F1_DEPLOY_THRESHOLD} threshold -- saved weights to {weights_path}")
    else:
        print(
            f"F1 {final_metrics['f1']:.3f} <= {F1_DEPLOY_THRESHOLD} threshold -- "
            "not deploying (per spec, returning metrics with a warning instead)"
        )


if __name__ == "__main__":
    main()

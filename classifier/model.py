import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ENCODER_NAME = "cardiffnlp/tweet-topic-latest-multi"
MAX_TOKENS = 128


def load_pretrained():
    """Loads the full cardiffnlp checkpoint once and returns:
    - tokenizer
    - encoder: the frozen RoBERTa base (`.roberta`), shared across topic heads
    - pretrained_model: the full model, including cardiffnlp's own trained
      19-topic head -- useful as a baseline signal, separate from our
      per-topic heads which start untrained.

    Loading via AutoModelForSequenceClassification (rather than bare
    AutoModel) matters: this checkpoint has no trained pooler, and AutoModel
    would silently fabricate a randomly-initialized one to fill the gap.
    """
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(ENCODER_NAME)
    pretrained_model.eval()
    encoder = pretrained_model.roberta
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, tokenizer, pretrained_model


class TopicClassifier(nn.Module):
    """Frozen shared encoder with one independent binary head per topic.

    Only the per-topic nn.Linear(hidden_size, 1) heads are ever trained;
    the encoder stays frozen. A head that hasn't been fine-tuned yet is
    randomly initialized and carries no signal -- that's expected, not
    a bug, until a training pass runs.
    """

    def __init__(self, encoder, topic_names):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_size, 1) for name in topic_names})

    def embed(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # cardiffnlp/tweet-topic-latest-multi ships no trained pooler
            # (it's a RobertaForSequenceClassification checkpoint); the <s>
            # token's hidden state is the actual pretrained representation,
            # not out.pooler_output (see classifier/README notes).
            return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask, topic):
        embeddings = self.embed(input_ids, attention_mask)
        return self.heads[topic](embeddings).squeeze(-1)

    def predict(self, texts, topic, tokenizer, device="cpu"):
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
        ).to(device)
        logits = self.forward(inputs["input_ids"], inputs["attention_mask"], topic)
        return torch.sigmoid(logits)


class StackedTopicHeads(nn.Module):
    """Compiles many independently-trained per-topic nn.Linear(hidden, 1)
    heads into a single nn.Linear(hidden, n_topics).

    Mathematically identical to running each head separately -- each output
    neuron keeps its own independent weight row, so topics are still fully
    independent classifiers. The only change is representation: one matmul
    instead of a Python loop over N small matmuls, which matters once N gets
    into the hundreds/thousands (see classifier/benchmark.py).

    Training is unaffected -- keep training individual heads with
    classifier/train.py as usual, then compile the trained heads here for
    fast simultaneous inference.
    """

    def __init__(self, heads: dict[str, nn.Linear]):
        super().__init__()
        self.topic_names = list(heads.keys())
        hidden_size = next(iter(heads.values())).in_features
        self.linear = nn.Linear(hidden_size, len(heads))
        with torch.no_grad():
            for i, name in enumerate(self.topic_names):
                self.linear.weight[i] = heads[name].weight[0]
                self.linear.bias[i] = heads[name].bias[0]

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: (batch, hidden) -> logits: (batch, n_topics)"""
        return self.linear(embeddings)

    def topic_index(self, topic: str) -> int:
        return self.topic_names.index(topic)

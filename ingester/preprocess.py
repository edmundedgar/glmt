"""Cardiff NLP's recommended preprocessing for `cardiffnlp/tweet-topic-latest-multi`.

See https://huggingface.co/cardiffnlp/tweet-topic-latest-multi
"""


def preprocess(text: str) -> str:
    tokens = []
    for token in text.split(" "):
        if token.startswith("@") and len(token) > 1:
            token = "@user"
        elif token.startswith("http"):
            token = "http"
        tokens.append(token)
    return " ".join(tokens).strip()

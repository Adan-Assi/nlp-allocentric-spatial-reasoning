from sentence_transformers import SentenceTransformer
import torch

_embedding_model = None


def get_embedding_model():
    """
    Singleton loader for sentence embedding model.

    Prevents loading all-MiniLM-L6-v2 multiple times across modules.
    """
    global _embedding_model

    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading semantic embedding model on {device}...", flush=True)
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    return _embedding_model
import torch
from vahumana.memory.store import MemoryStore
from vahumana.emotion.circumplex import emotion_similarity


def retrieve_top_k(
    store: MemoryStore,
    query_emotion: torch.Tensor,
    k: int = 5,
) -> torch.Tensor | None:
    """
    Retrieve top-k memory gists by cosine similarity in emotion space.
    Returns a salience-weighted mean of gist vectors (emotional gist, not verbatim detail).
    """
    if len(store) == 0:
        return None

    stored_emotions = store.emotions()                         # (n, 2)
    q = query_emotion.view(1, 2).expand_as(stored_emotions)
    sims = emotion_similarity(q, stored_emotions)              # (n,)

    k_actual = min(k, len(store))
    top_indices = sims.topk(k_actual).indices

    top_sims = sims[top_indices].clamp(min=0.0)
    weights = top_sims / (top_sims.sum() + 1e-8)

    gists = store.gists()[top_indices]                         # (k, hidden_dim)
    return (weights.unsqueeze(-1) * gists).sum(dim=0)          # (hidden_dim,)

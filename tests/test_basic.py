import torch
import pytest
from vahumana.emotion.encoder import EmotionEncoder
from vahumana.emotion.circumplex import CircumplexPoint, emotion_distance, emotion_similarity
from vahumana.memory.diffusion import diffuse
from vahumana.memory.store import MemoryStore, MemoryEntry
from vahumana.memory.retrieval import retrieve_top_k


def test_emotion_encoder_output_shape():
    enc = EmotionEncoder(hidden_dim=64)
    h = torch.randn(2, 10, 64)
    assert enc(h).shape == (2, 2)


def test_emotion_encoder_output_range():
    enc = EmotionEncoder(hidden_dim=64)
    out = enc(torch.randn(4, 8, 64))
    assert out.abs().max().item() <= 1.0 + 1e-5


def test_emotion_encoder_salience_positive():
    enc = EmotionEncoder(hidden_dim=64)
    emotion = enc(torch.randn(3, 5, 64))
    assert (enc.salience(emotion) >= 0).all()


def test_circumplex_point_labels():
    assert CircumplexPoint(0.5, 0.5).label() == "happy/excited"
    assert CircumplexPoint(-0.5, 0.5).label() == "angry/afraid"
    assert CircumplexPoint(-0.5, -0.5).label() == "sad/depressed"
    assert CircumplexPoint(0.5, -0.5).label() == "calm/relaxed"


def test_emotion_distance_identical():
    a = torch.tensor([[1.0, 0.0]])
    assert emotion_distance(a, a).item() == pytest.approx(0.0, abs=1e-5)


def test_diffuse_clamps_output():
    out = diffuse(torch.tensor([[0.9, 0.9]]), torch.tensor([[0.01]]), base_noise_std=1.0)
    assert out.abs().max().item() <= 1.0


def test_diffuse_high_salience_low_noise():
    emotion = torch.tensor([[0.5, 0.5]])
    out = diffuse(emotion, torch.tensor([[1000.0]]), base_noise_std=0.1)
    assert torch.allclose(out, emotion, atol=1e-2)


def test_memory_store_respects_max_entries():
    store = MemoryStore(max_entries=3)
    for i in range(5):
        store.add(MemoryEntry(torch.tensor([i * 0.1, 0.0]), torch.zeros(16), salience=float(i)))
    assert len(store) == 3


def test_memory_store_evicts_lowest_salience():
    store = MemoryStore(max_entries=2)
    store.add(MemoryEntry(torch.tensor([0.1, 0.0]), torch.zeros(4), salience=1.0))
    store.add(MemoryEntry(torch.tensor([0.5, 0.0]), torch.zeros(4), salience=5.0))
    store.add(MemoryEntry(torch.tensor([0.9, 0.0]), torch.zeros(4), salience=3.0))
    assert 1.0 not in [e.salience for e in store._entries]


def test_retrieve_top_k_shape():
    store = MemoryStore()
    hidden_dim = 16
    for _ in range(10):
        store.add(MemoryEntry(torch.randn(2), torch.randn(hidden_dim), salience=1.0))
    result = retrieve_top_k(store, torch.tensor([1.0, 0.0]), k=3)
    assert result is not None and result.shape == (hidden_dim,)


def test_retrieve_returns_none_for_empty_store():
    assert retrieve_top_k(MemoryStore(), torch.tensor([1.0, 0.0]), k=5) is None

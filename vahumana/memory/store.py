from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class MemoryEntry:
    emotion: torch.Tensor        # (2,) — diffused valence/arousal
    hidden_gist: torch.Tensor    # (hidden_dim,) — mean-pooled hidden state
    salience: float
    text_snippet: Optional[str] = None


class MemoryStore:
    """Persistent in-memory store of consolidated emotional memories."""

    def __init__(self, max_entries: int = 1024):
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries.sort(key=lambda e: e.salience, reverse=True)
            self._entries = self._entries[: self.max_entries]

    def emotions(self) -> torch.Tensor:
        if not self._entries:
            return torch.empty(0, 2)
        return torch.stack([e.emotion for e in self._entries])

    def gists(self) -> torch.Tensor:
        if not self._entries:
            return torch.empty(0)
        return torch.stack([e.hidden_gist for e in self._entries])

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass
class CircumplexPoint:
    """A point in Russell's 2D Circumplex emotion space."""
    valence: float   # [-1, 1]  negative ↔ positive
    arousal: float   # [-1, 1]  calm ↔ excited

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.valence, self.arousal], dtype=torch.float32)

    def angle_degrees(self) -> float:
        return math.degrees(math.atan2(self.arousal, self.valence))

    def label(self) -> str:
        v, a = self.valence, self.arousal
        if v >= 0 and a >= 0:
            return "happy/excited"
        if v < 0 and a >= 0:
            return "angry/afraid"
        if v < 0 and a < 0:
            return "sad/depressed"
        return "calm/relaxed"


def emotion_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    return 1.0 - (a_norm * b_norm).sum(dim=-1)


def emotion_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    return (a_norm * b_norm).sum(dim=-1)

import torch
import torch.nn as nn


class EmotionEncoder(nn.Module):
    """Projects transformer hidden states into Russell's 2D emotion space (valence, arousal)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, 2),
            nn.Tanh(),  # constrain output to [-1, 1] for both valence and arousal
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        Returns:
            emotion: (batch, 2) — [valence, arousal] in [-1, 1]
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.mean(dim=1)
        return self.net(hidden_state)

    def salience(self, emotion: torch.Tensor) -> torch.Tensor:
        """Euclidean distance from the neutral origin — used to scale diffusion noise."""
        return emotion.norm(dim=-1, keepdim=True)

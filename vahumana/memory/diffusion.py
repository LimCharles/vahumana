import torch


def diffuse(
    emotion: torch.Tensor,
    salience: torch.Tensor,
    base_noise_std: float = 0.1,
    min_noise_std: float = 1e-4,
) -> torch.Tensor:
    """
    Apply Gaussian noise to an emotion vector during memory consolidation.

    Noise is scaled by 1/salience so that emotionally intense memories are
    preserved more faithfully — analogous to flashbulb memory effects.
    """
    salience_clamped = salience.clamp(min=min_noise_std)
    std = base_noise_std / salience_clamped
    noise = torch.randn_like(emotion) * std
    return (emotion + noise).clamp(-1.0, 1.0)

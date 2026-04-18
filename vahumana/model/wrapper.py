from __future__ import annotations
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from vahumana.emotion.encoder import EmotionEncoder
from vahumana.memory.store import MemoryStore, MemoryEntry
from vahumana.memory.diffusion import diffuse
from vahumana.memory.retrieval import retrieve_top_k
from vahumana.steering.hooks import HookManager

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_BLOCK_ATTR = "model.layers"   # Qwen3 / LLaMA family
DEFAULT_HIDDEN_DIM = 1024             # Qwen3-0.6B hidden size


class VahumanaModel(nn.Module):
    """
    Wraps a HuggingFace CausalLM with vahumana emotional memory augmentation.

    Defaults target Qwen3-0.6B but any CausalLM works — pass the correct
    hidden_dim and block_attr for the architecture:

        GPT-2:         hidden_dim=768,  block_attr="transformer.h"
        Qwen3-0.6B:    hidden_dim=1024, block_attr="model.layers"  (default)
        Qwen3-1.7B:    hidden_dim=2048, block_attr="model.layers"
        LLaMA-3-8B:    hidden_dim=4096, block_attr="model.layers"
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        block_attr: str = DEFAULT_BLOCK_ATTR,
        memory_size: int = 1024,
        top_k: int = 5,
        injection_scale: float = 0.05,
        base_noise_std: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.emotion_encoder = EmotionEncoder(hidden_dim)
        self.memory_store = MemoryStore(max_entries=memory_size)
        self.hook_manager = HookManager(base_model, block_attr=block_attr)
        self.top_k = top_k
        self.injection_scale = injection_scale
        self.base_noise_std = base_noise_std
        self._last_emotion: torch.Tensor | None = None

    def attach_hooks(self) -> None:
        self.hook_manager.attach(callback=self._on_block_output)

    def detach_hooks(self) -> None:
        self.hook_manager.detach()

    def _on_block_output(self, block_idx: int, hidden_state: torch.Tensor) -> None:
        emotion = self.emotion_encoder(hidden_state)          # (batch, 2)
        salience = self.emotion_encoder.salience(emotion)     # (batch, 1)
        self._last_emotion = emotion.mean(dim=0)              # (2,)

        diffused = diffuse(emotion, salience, self.base_noise_std)
        gist = hidden_state.mean(dim=1).mean(dim=0).detach()  # (hidden_dim,)

        self.memory_store.add(MemoryEntry(
            emotion=diffused.mean(dim=0).detach(),
            hidden_gist=gist,
            salience=salience.mean().item(),
        ))

    def forward(self, input_ids: torch.Tensor, **kwargs):
        self.hook_manager.clear_cache()
        outputs = self.base_model(input_ids=input_ids, **kwargs)
        if self._last_emotion is not None:
            retrieve_top_k(self.memory_store, self._last_emotion, k=self.top_k)
        return outputs

    def generate(self, **kwargs):
        self.hook_manager.clear_cache()
        return self.base_model.generate(**kwargs)

    @property
    def current_emotion(self) -> torch.Tensor | None:
        return self._last_emotion

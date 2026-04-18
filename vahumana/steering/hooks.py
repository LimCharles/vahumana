from __future__ import annotations
from typing import Callable, List
import torch
import torch.nn as nn


class HookManager:
    """
    Attaches forward hooks to each transformer block to intercept hidden states.

    Usage:
        manager = HookManager(model, block_attr="model.layers")  # Qwen3/LLaMA
        manager = HookManager(model, block_attr="transformer.h") # GPT-2
        manager.attach(callback_fn)
        # ... run inference ...
        manager.detach()
    """

    def __init__(self, model: nn.Module, block_attr: str = "model.layers"):
        self.model = model
        self.block_attr = block_attr
        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._hidden_states: List[torch.Tensor] = []

    def _get_blocks(self) -> nn.ModuleList:
        obj = self.model
        for attr in self.block_attr.split("."):
            obj = getattr(obj, attr)
        return obj

    def attach(self, callback: Callable[[int, torch.Tensor], None] | None = None) -> None:
        self._hidden_states.clear()
        blocks = self._get_blocks()

        for idx, block in enumerate(blocks):
            def make_hook(i: int):
                def hook(module, input, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    self._hidden_states.append(hs.detach())
                    if callback is not None:
                        callback(i, hs)
                return hook
            handle = block.register_forward_hook(make_hook(idx))
            self._handles.append(handle)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def last_hidden_states(self) -> List[torch.Tensor]:
        return list(self._hidden_states)

    def clear_cache(self) -> None:
        self._hidden_states.clear()

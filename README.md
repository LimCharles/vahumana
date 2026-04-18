# vahumana

> *vahumana* — the wandering spirit from Genshin Impact; those who lost their physical form yet persist as emotion and memory.

A research prototype exploring **human-like lossy reconstructive emotional memory** for large language models.

## Philosophy

Human memory is not a tape recorder. It is reconstructive, emotionally weighted, and lossy — we remember the *gist* of experiences, coloured by how they made us feel. Vivid emotional events are consolidated sharply; mundane ones blur into noise.

`vahumana` attempts to simulate this in transformer LLMs:

1. **Intercept** hidden states at each transformer block during inference.
2. **Project** them into Russell's 2D Circumplex emotion space (valence × arousal).
3. **Consolidate** them into a persistent memory store via Gaussian diffusion — important memories (high salience) receive less noise and are preserved more faithfully.
4. **Retrieve** emotionally-salient memories by cosine similarity and inject them back into the residual stream, biasing the model's next-token distribution.

The result is a model whose "mood" and "recall" evolve over a conversation in a psychologically plausible way, without any weight updates.

## Architecture

```
vahumana/
├── emotion/
│   ├── encoder.py       # MLP: hidden_dim → 128 → 2 (valence, arousal)
│   └── circumplex.py    # Russell's Circumplex geometry utilities
├── memory/
│   ├── diffusion.py     # Gaussian noise consolidation scaled by 1/salience
│   ├── store.py         # Persistent in-memory + optional disk store
│   └── retrieval.py     # Top-k cosine similarity retrieval in emotion space
├── steering/
│   └── hooks.py         # register_forward_hook on each transformer block
└── model/
    └── wrapper.py       # Wraps any HuggingFace CausalLM with vahumana augmentation
```

## Quickstart

```bash
pip install -e .
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from vahumana.model.wrapper import VahumanaModel

model_name = "Qwen/Qwen3-0.6B"  # or any HuggingFace CausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
base = AutoModelForCausalLM.from_pretrained(model_name)

# Qwen3 exposes blocks at model.layers; hidden_dim=1024 for Qwen3-0.6B
model = VahumanaModel(
    base_model=base,
    hidden_dim=1024,
    block_attr="model.layers",
)
model.attach_hooks()

inputs = tokenizer("I felt an overwhelming sense of", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Gaussian diffusion noise ∝ 1/salience | High-salience memories compress less, matching human flashbulb memory effects |
| 2D emotion space (not high-dim) | Interpretability — we can visualise emotional trajectories over a conversation |
| Residual stream injection | Minimally invasive; doesn't alter attention patterns directly |
| Cosine similarity retrieval | Emotion space is directional; angular distance captures valence/arousal proximity better than L2 |

## References

- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
- Anthropic Interpretability Team — residual stream as "world model" framing
- Diffusion-based memory consolidation inspired by sleep replay literature (Stickgold, 2005)

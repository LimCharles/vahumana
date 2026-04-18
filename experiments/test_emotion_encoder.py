"""Sanity check: run a forward pass and inspect emotion trajectories."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vahumana.model.wrapper import VahumanaModel, DEFAULT_MODEL, DEFAULT_BLOCK_ATTR, DEFAULT_HIDDEN_DIM
from vahumana.emotion.circumplex import CircumplexPoint


def main(
    model_name: str = DEFAULT_MODEL,
    block_attr: str = DEFAULT_BLOCK_ATTR,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(model_name)
    base.eval()

    model = VahumanaModel(base_model=base, hidden_dim=hidden_dim, block_attr=block_attr)
    model.attach_hooks()

    sentences = [
        "I felt an overwhelming surge of joy and love.",
        "The news filled me with dread and despair.",
        "It was a perfectly ordinary Tuesday.",
    ]

    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt")
            _ = model(**inputs)
            e = model.current_emotion
            if e is not None:
                pt = CircumplexPoint(valence=e[0].item(), arousal=e[1].item())
                print(f"  '{sentence[:45]}...'")
                print(f"    -> valence={pt.valence:.3f}, arousal={pt.arousal:.3f} [{pt.label()}]")
                print(f"    -> angle={pt.angle_degrees():.1f}deg, memories={len(model.memory_store)}")

    print(f"\nTotal memories stored: {len(model.memory_store)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--block-attr", default=DEFAULT_BLOCK_ATTR)
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    args = p.parse_args()
    main(args.model, args.block_attr, args.hidden_dim)

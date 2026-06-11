"""
infer.py
========
Run a trained mT5 on the span-corruption task and SHOW its predictions: take a
sentence, corrupt it the same way training does, feed the corrupted input, and
let the model generate the sentinel+span target.

USAGE:
    # untrained (random) model -- output will be gibberish, proves the loop runs:
    python infer.py --tokenizer tokenizer/spm.model

    # with a trained checkpoint (after we add checkpointing in the Colab loop):
    python infer.py --tokenizer tokenizer/spm.model --checkpoint checkpoints/step5000.pt

    # provide your own text:
    python infer.py --tokenizer tokenizer/spm.model --text "Ang kape ay sikat na inumin"
"""

import argparse
import torch

from model.config import MT5Config
from model.mt5 import build_model
from data.collator import SpanCorruptionCollator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/spm.model")
    ap.add_argument("--profile", default="tiny",
                    choices=["tiny", "small", "base"],
                    help="MUST match the profile the checkpoint was trained with")
    ap.add_argument("--checkpoint", default=None,
                    help="optional .pt file with model state_dict")
    ap.add_argument("--text", default="Ang kape ay sikat na inumin sa buong "
                    "mundo at maraming tao ang umiinom nito tuwing umaga")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)  # 0 = greedy
    ap.add_argument("--top_k", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = getattr(MT5Config, args.profile)()
    model = build_model(cfg).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        # Accept either a raw state_dict or a {'model': state_dict, ...} bundle.
        sd = state.get("model", state) if isinstance(state, dict) else state
        model.load_state_dict(sd)
        print(f"loaded checkpoint: {args.checkpoint}")
    else:
        print("no checkpoint -> using RANDOM weights (output will be gibberish)")

    collator = SpanCorruptionCollator(args.tokenizer, cfg, seed=0)
    sp = collator.sp

    # Corrupt the input exactly like training.
    ids = collator.encode_text(args.text)
    inp_ids, tgt_ids = collator.corrupt(ids)

    print(f"\noriginal : {sp.decode(ids)}")
    print(f"corrupted: {sp.decode(inp_ids)}")
    print(f"gold tgt : {sp.decode(tgt_ids)}")

    input_tensor = torch.tensor([inp_ids], dtype=torch.long, device=device)
    out = model.generate(
        input_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    pred_ids = out[0].tolist()
    # Trim trailing pad/eos for display.
    pred_ids = [t for t in pred_ids if t != cfg.pad_token_id]
    print(f"predicted: {sp.decode(pred_ids)}")


if __name__ == "__main__":
    main()
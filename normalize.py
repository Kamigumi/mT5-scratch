"""
normalize.py
============
Run the fine-tuned normalizer: noisy Taglish in -> clean text out. Unlike
infer.py (which does span-corruption reconstruction), this is the actual task
the model was fine-tuned for: plain seq2seq normalization.

USAGE
-----
  # normalize a single string
  python normalize.py --checkpoint P/finetune/gold_best.pt \
      --text "di ko inexpect yung prices grabe huhu"

  # normalize and SCORE against the held-out gold val set (shows input/gold/pred
  # side by side for a sample, so you can eyeball quality)
  python normalize.py --checkpoint P/finetune/gold_best.pt \
      --eval_gold P/data/manual_pairs.jsonl --n 15
"""

import argparse
import torch

from model.config import MT5Config
from model.mt5 import build_model
from data.supervised_collator import load_pairs, split_pairs
import sentencepiece as spm


def load_model(checkpoint, profile, device):
    cfg = getattr(MT5Config, profile)()
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    sd = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


def normalize_one(model, sp, cfg, text, device, max_new_tokens=128):
    """Encode noisy text, generate clean text greedily."""
    ids = sp.encode(text, out_type=int)[: cfg_max_in(cfg) - 1]
    ids.append(cfg.eos_token_id)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.0)
    pred = [t for t in out[0].tolist() if t not in (cfg.pad_token_id, cfg.eos_token_id)]
    return sp.decode(pred)


def cfg_max_in(cfg):
    return 256  # match the collator's max_input_length default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/spm.model")
    ap.add_argument("--profile", default="small", choices=["tiny", "small", "base"])
    ap.add_argument("--text", default=None, help="single noisy string to normalize")
    ap.add_argument("--eval_gold", default=None,
                    help="path to gold .jsonl; shows input/gold/pred on held-out val")
    ap.add_argument("--n", type=int, default=15, help="how many val examples to show")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    model, cfg = load_model(args.checkpoint, args.profile, device)
    print(f"loaded {args.checkpoint} on {device}\n")

    if args.text:
        pred = normalize_one(model, sp, cfg, args.text, device, args.max_new_tokens)
        print(f"INPUT : {args.text}")
        print(f"OUTPUT: {pred}")

    if args.eval_gold:
        # Recreate the SAME held-out split used in finetuning (seed=42, val=120).
        pairs = load_pairs(args.eval_gold)
        _, val = split_pairs(pairs, val_size=120, seed=42)
        show = val[: args.n]
        print(f"=== held-out gold sample ({len(show)} of {len(val)}) ===\n")
        for i, ex in enumerate(show, 1):
            pred = normalize_one(model, sp, cfg, ex["input"], device, args.max_new_tokens)
            print(f"[{i}] NOISY : {ex['input']}")
            print(f"    GOLD  : {ex['target']}")
            print(f"    PRED  : {pred}")
            print()


if __name__ == "__main__":
    main()
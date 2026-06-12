"""
finetune.py
===========
Supervised fine-tuning for noisy->clean Taglish normalization. One script, two
stages selected by --stage:

  STAGE 1 (synthetic): train on the 100k rule-generated pairs. Many examples,
      moderate LR, a few epochs. Teaches the GENERAL noisy->clean mapping.

  STAGE 2 (gold): train on the ~1k hand-annotated real-Taglish pairs. Few
      examples, LOW LR, few epochs, with a held-out val set. Adapts to the REAL
      distribution you care about.

The transfer chain:
    pretrained base  --(stage synthetic)-->  synthetic model
                     --(stage gold)------->  final normalizer
Each stage starts from the previous via --init.

KEY DIFFERENCES FROM PRETRAINING (train_colab.py):
  - supervised pairs, not span corruption (uses SupervisedCollator)
  - EPOCH-based (we have finite labeled data) rather than infinite streaming
  - gold stage holds out a val set and reports val loss for honest evaluation
  - --init loads weights ONLY (fresh optimizer), since we're starting a new task

USAGE:
  # Stage 1: from pretrained base (best val checkpoint = step92000)
  python finetune.py --project P --stage synthetic \
      --init P/checkpoints/step92000.pt --epochs 3 --batch_size 32 --lr 1e-4

  # Stage 2: from the stage-1 result
  python finetune.py --project P --stage gold \
      --init P/finetune/synthetic_final.pt --epochs 5 --batch_size 16 --lr 3e-5
"""

import argparse
import os
import math
import random
import torch
from torch.optim import AdamW

from model.config import MT5Config
from model.mt5 import build_model
from data.supervised_collator import SupervisedCollator, load_pairs, split_pairs


def batches(pairs, batch_size, shuffle=True):
    """Yield batches over pairs for one epoch."""
    order = list(range(len(pairs)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [pairs[j] for j in order[i:i + batch_size]]
        if chunk:
            yield chunk


@torch.no_grad()
def evaluate(model, val_pairs, collator, device, use_bf16, batch_size=32):
    model.eval()
    total, n = 0.0, 0
    for batch in batches(val_pairs, batch_size, shuffle=False):
        b = collator(batch)
        input_ids = b["input_ids"].to(device)
        labels = b["labels"].to(device)
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)
        else:
            _, loss = model(input_ids, labels=labels)
        total += loss.item() * len(batch)
        n += len(batch)
    model.train()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--stage", required=True, choices=["synthetic", "gold"])
    ap.add_argument("--init", required=True,
                    help="checkpoint to initialize weights from")
    ap.add_argument("--profile", default="small", choices=["tiny", "small", "base"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--max_input_length", type=int, default=256)
    ap.add_argument("--max_target_length", type=int, default=256)
    ap.add_argument("--val_size", type=int, default=120,
                    help="held-out pairs (gold stage only)")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device: {device} | bf16: {use_bf16} | stage: {args.stage}")

    tok_path = os.path.join(args.project, "tokenizer", "spm.model")
    out_dir = os.path.join(args.project, "finetune")
    os.makedirs(out_dir, exist_ok=True)

    cfg = getattr(MT5Config, args.profile)()
    model = build_model(cfg).to(device)

    # Load init weights ONLY (fresh optimizer for the new task).
    state = torch.load(args.init, map_location=device)
    sd = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(sd)
    print(f"initialized from {args.init}")

    collator = SupervisedCollator(tok_path, cfg,
                                  max_input_length=args.max_input_length,
                                  max_target_length=args.max_target_length)

    # --- data per stage ---
    if args.stage == "synthetic":
        data_path = os.path.join(args.project, "data", "synthetic.jsonl")
        all_pairs = load_pairs(data_path)
        # small internal val slice just to watch generalization
        train_pairs, val_pairs = split_pairs(all_pairs, val_size=1000)
    else:  # gold
        data_path = os.path.join(args.project, "data", "manual_pairs.jsonl")
        all_pairs = load_pairs(data_path)
        train_pairs, val_pairs = split_pairs(all_pairs, val_size=args.val_size)
    print(f"loaded {len(all_pairs)} pairs -> train {len(train_pairs)} | val {len(val_pairs)}")

    optim = AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = math.ceil(len(train_pairs) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        # linear decay to 0 over the rest
        return args.lr * max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    print(f"epochs={args.epochs} steps/epoch={steps_per_epoch} "
          f"total_steps={total_steps} warmup={warmup}")

    # Baseline val loss BEFORE any fine-tuning (great for the writeup).
    base_val = evaluate(model, val_pairs, collator, device, use_bf16, args.batch_size)
    print(f"val_loss @ init (before stage): {base_val:.4f}")

    model.train()
    step = 0
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for batch in batches(train_pairs, args.batch_size, shuffle=True):
            step += 1
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            b = collator(batch)
            input_ids = b["input_ids"].to(device)
            labels = b["labels"].to(device)

            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(input_ids, labels=labels)
            else:
                _, loss = model(input_ids, labels=labels)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running += loss.item()
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step:5d} | loss {running/args.log_every:.4f} "
                      f"| lr {lr_at(step):.2e}")
                running = 0.0

        val = evaluate(model, val_pairs, collator, device, use_bf16, args.batch_size)
        print(f"epoch {epoch} DONE | val_loss {val:.4f}")

        # Save best-val checkpoint.
        if val < best_val:
            best_val = val
            best_path = os.path.join(out_dir, f"{args.stage}_best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val}, best_path)
            print(f"  new best -> saved {best_path}")

    # Always save a final checkpoint too (used as --init for the next stage).
    final_path = os.path.join(out_dir, f"{args.stage}_final.pt")
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "val_loss": val}, final_path)
    print(f"saved {final_path}")
    print(f"done. best val_loss this stage: {best_val:.4f} (started at {base_val:.4f})")


if __name__ == "__main__":
    main()
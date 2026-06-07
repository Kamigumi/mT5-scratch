"""
train.py
========
Local smoke-test pretraining loop for the RTX 4060. The goal here is NOT a good
model -- it's to confirm the whole pipeline runs end to end and the loss goes
DOWN. Once this works locally, the same code scales to the Colab A100 by:
  - swapping MT5Config.tiny() for a larger profile
  - pointing --data at the full Drive corpus
  - raising --steps / --batch_size

Pipeline: text file -> lines -> SpanCorruptionCollator -> (input_ids, labels)
          -> MT5 forward -> cross-entropy loss -> AdamW step.

USAGE (local):
    python train.py --data data/sample.txt --tokenizer tokenizer/spm.model \
        --steps 200 --batch_size 8
"""

import argparse
import random
import torch
from torch.optim import AdamW

from model.config import MT5Config
from model.mt5 import build_model
from data.collator import SpanCorruptionCollator


def line_batches(path, batch_size, max_lines=None):
    """Yield batches of raw text lines, cycling the file."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if len(line) > 40:           # skip very short lines
                lines.append(line)
            if max_lines and len(lines) >= max_lines:
                break
    if not lines:
        raise ValueError(f"No usable lines in {path}")
    print(f"loaded {len(lines)} lines from {path}")
    while True:
        random.shuffle(lines)
        for i in range(0, len(lines) - batch_size, batch_size):
            yield lines[i: i + batch_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/spm.model")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_input_length", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    cfg = MT5Config.tiny()
    model = build_model(cfg).to(device)
    print(f"params: {model.num_params():,}")

    collator = SpanCorruptionCollator(
        args.tokenizer, cfg, max_input_length=args.max_input_length, seed=0
    )
    optim = AdamW(model.parameters(), lr=args.lr)

    model.train()
    batches = line_batches(args.data, args.batch_size)
    running = 0.0
    for step in range(1, args.steps + 1):
        texts = next(batches)
        batch = collator(texts)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, labels=labels)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        running += loss.item()
        if step % args.log_every == 0:
            avg = running / args.log_every
            print(f"step {step:4d} | loss {avg:.4f}")
            running = 0.0

    print("done. loss should have dropped well below the ~10.4 init value.")


if __name__ == "__main__":
    main()
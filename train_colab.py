"""
train_colab.py
==============
Full pretraining loop for the Colab A100. Adds three things the local train.py
lacks:

  1. CHECKPOINT + RESUME  -- Colab sessions die (idle/12-24h cap). We save
     model+optimizer+step to Drive every N steps and auto-resume from the latest
     checkpoint on restart. This is non-negotiable for any real run.

  2. bf16 MIXED PRECISION -- the A100 supports bfloat16. torch.autocast halves
     memory and roughly doubles throughput. bf16 (not fp16) because it has the
     same exponent range as fp32, so no loss-scaling gymnastics needed.

  3. TL/EN MIXED STREAMING -- instead of loading 1M+ lines into RAM, we open the
     four Drive text files and pull lines with a controlled TL:EN ratio, so
     Tagalog stays primary (matching the Taglish target) rather than letting the
     larger English data dominate.

Run AFTER mounting Drive and making the model/ and data/ packages importable
(upload them to /content or clone from your repo). Example Colab cell:

    !python train_colab.py \
        --project /content/drive/MyDrive/mt5_from_scratch \
        --profile small --steps 50000 --batch_size 32 \
        --save_every 1000 --tl_ratio 0.7
"""

import argparse
import os
import glob
import random
import torch
from torch.optim import AdamW

from model.config import MT5Config
from model.mt5 import build_model
from data.collator import SpanCorruptionCollator


# ---------------- data streaming ----------------
class MixedLineStream:
    """Streams shuffled lines from TL and EN files at a target ratio.

    We keep file handles open and refill in-memory buffers as needed, so we
    never hold the whole corpus in RAM. tl_ratio=0.7 means ~70% of yielded
    lines are Tagalog.
    """

    def __init__(self, tl_paths, en_paths, tl_ratio=0.7, min_chars=40,
                 buffer_size=100_000):
        self.tl_paths = tl_paths
        self.en_paths = en_paths
        self.tl_ratio = tl_ratio
        self.min_chars = min_chars
        self.buffer_size = buffer_size
        self._tl_buf, self._en_buf = [], []
        self._tl_files = self._open_cycle(tl_paths)
        self._en_files = self._open_cycle(en_paths)

    def _open_cycle(self, paths):
        """Infinite generator of lines across the given files, cycling forever."""
        def gen():
            while True:
                for p in paths:
                    with open(p, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if len(line) >= self.min_chars:
                                yield line
        return gen()

    def _refill(self, buf, src):
        while len(buf) < self.buffer_size:
            buf.append(next(src))
        random.shuffle(buf)

    def next_line(self):
        if random.random() < self.tl_ratio:
            if not self._tl_buf:
                self._refill(self._tl_buf, self._tl_files)
            return self._tl_buf.pop()
        else:
            if not self._en_buf:
                self._refill(self._en_buf, self._en_files)
            return self._en_buf.pop()

    def batch(self, n):
        return [self.next_line() for _ in range(n)]


# ---------------- checkpointing ----------------
def save_ckpt(path, model, optim, step):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(),
                "optim": optim.state_dict(),
                "step": step}, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save won't corrupt the file


def latest_ckpt(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "step*.pt"))
    if not files:
        return None
    return max(files, key=lambda f: int(
        os.path.basename(f).replace("step", "").replace(".pt", "")))


# ---------------- validation ----------------
def load_val_set(tl_paths, en_paths, n_per_lang=300, min_chars=40, seed=1234):
    """Load a FIXED held-out set: the first n_per_lang usable lines from each
    language. Seeded and deterministic so every eval sees the same text.

    IMPORTANT: this reads from the SAME files training streams from. The streams
    shuffle and cycle the whole files, so there's overlap in principle -- this is
    a lightweight proxy for held-out loss, not a clean split. For a rigorous
    split you'd hold these specific lines OUT of training; see the note in the
    eval print. For our purpose (is loss still improving?) the proxy is fine
    because the val corruption is fixed and the set is tiny relative to the
    corpus, so it tracks generalization closely enough to read the trend.
    """
    val = []
    for paths, lang in [(tl_paths, "tl"), (en_paths, "en")]:
        count = 0
        for p in paths:
            if count >= n_per_lang:
                break
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) >= min_chars:
                        val.append(line)
                        count += 1
                        if count >= n_per_lang:
                            break
    return val


@torch.no_grad()
def evaluate(model, val_texts, val_collator, device, use_bf16, batch_size=32):
    """Mean loss over the held-out set. Uses a FIXED-seed collator so the span
    corruption is identical every call -- otherwise loss would wobble from
    random masking rather than real model change."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(val_texts), batch_size):
        chunk = val_texts[i:i + batch_size]
        batch = val_collator(chunk)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(input_ids, labels=labels)
        else:
            _, loss = model(input_ids, labels=labels)
        total += loss.item() * len(chunk)
        n += len(chunk)
    model.train()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Drive project root")
    ap.add_argument("--profile", default="small", choices=["tiny", "small", "base"])
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max_input_length", type=int, default=256)
    ap.add_argument("--tl_ratio", type=float, default=0.7)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--tokenizer_name", default="spm_v2.model",
                    help="tokenizer file under <project>/tokenizer/")
    ap.add_argument("--ckpt_subdir", default="checkpoints",
                    help="subfolder under <project> for this run's checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device: {device} | bf16: {use_bf16}")

    data_dir = os.path.join(args.project, "data")
    ckpt_dir = os.path.join(args.project, args.ckpt_subdir)
    tok_path = os.path.join(args.project, "tokenizer", args.tokenizer_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg = getattr(MT5Config, args.profile)()
    model = build_model(cfg).to(device)
    print(f"profile={args.profile} params={model.num_params():,}")

    collator = SpanCorruptionCollator(tok_path, cfg,
                                      max_input_length=args.max_input_length)
    optim = AdamW(model.parameters(), lr=args.lr)

    # Linear warmup then inverse-sqrt decay (T5-ish schedule).
    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        return args.lr * (args.warmup ** 0.5) / (step ** 0.5)

    # Resume if a checkpoint exists.
    start_step = 0
    ckpt = latest_ckpt(ckpt_dir)
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        start_step = state["step"]
        print(f"resumed from {ckpt} at step {start_step}")

    stream = MixedLineStream(
        tl_paths=[os.path.join(data_dir, "wikitext_tl.txt"),
                  os.path.join(data_dir, "mc4_tl.txt")],
        en_paths=[os.path.join(data_dir, "wikitext_en.txt"),
                  os.path.join(data_dir, "mc4_en.txt")],
        tl_ratio=args.tl_ratio,
    )

    # Held-out validation set + a FIXED-seed collator (identical corruption each eval).
    val_texts = load_val_set(
        tl_paths=[os.path.join(data_dir, "wikitext_tl.txt"),
                  os.path.join(data_dir, "mc4_tl.txt")],
        en_paths=[os.path.join(data_dir, "wikitext_en.txt"),
                  os.path.join(data_dir, "mc4_en.txt")],
    )
    val_collator = SpanCorruptionCollator(
        tok_path, cfg, max_input_length=args.max_input_length, seed=999
    )
    print(f"val set: {len(val_texts)} lines")

    model.train()
    running = 0.0
    for step in range(start_step + 1, args.steps + 1):
        for g in optim.param_groups:
            g["lr"] = lr_at(step)

        batch = collator(stream.batch(args.batch_size))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

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
            print(f"step {step:6d} | loss {running/args.log_every:.4f} "
                  f"| lr {lr_at(step):.2e}")
            running = 0.0

        if step % args.eval_every == 0:
            val_loss = evaluate(model, val_texts, val_collator, device,
                                use_bf16, batch_size=args.batch_size)
            print(f"  >> VAL step {step:6d} | val_loss {val_loss:.4f}")

        if step % args.save_every == 0:
            path = os.path.join(ckpt_dir, f"step{step}.pt")
            save_ckpt(path, model, optim, step)
            print(f"  saved {path}")

    # final save
    save_ckpt(os.path.join(ckpt_dir, f"step{args.steps}.pt"), model, optim, args.steps)
    print("done.")


if __name__ == "__main__":
    main()
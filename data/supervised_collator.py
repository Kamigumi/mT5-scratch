"""
supervised_collator.py
======================
Supervised seq2seq collator for normalization. Far simpler than the span
corruption collator: NO masking, NO sentinels. It just tokenizes the noisy
`input` as encoder input and the clean `target` as labels, pads both, and sets
-100 on label pad positions so they're ignored in the loss.

The model's existing forward(input_ids, labels=...) already does the rest:
right-shifts labels to make decoder inputs, runs encoder+decoder, computes
cross-entropy with ignore_index=-100. So this collator is the only new piece
needed for fine-tuning.

Data format (both synthetic.jsonl and manual_pairs.jsonl): one JSON object per
line with at least:
    {"input": "<noisy text>", "target": "<clean text>", ...}
Extra keys (ops, source, sentence_id, source_platform) are ignored.

EOS: we append eos to BOTH input and target. T5 conventionally appends eos to
the target (so the decoder learns to stop) and to the input (so the encoder
sees a clear end-of-sequence). The tokenizer's encode() does NOT add eos
automatically, so we do it here.
"""

import json
import random
import torch
from typing import List, Dict

import sentencepiece as spm

from model.config import MT5Config


def load_pairs(path: str) -> List[Dict[str, str]]:
    """Read a .jsonl file into a list of {'input':..., 'target':...} dicts."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" in obj and "target" in obj:
                pairs.append({"input": obj["input"], "target": obj["target"]})
    return pairs


def split_pairs(pairs, val_size, seed=42):
    """Deterministic shuffle + split into (train, val). Used for the gold set so
    we can measure whether fine-tuning helped on held-out real data."""
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    val_idx = set(idx[:val_size])
    train = [pairs[i] for i in idx if i not in val_idx]
    val = [pairs[i] for i in idx if i in val_idx]
    return train, val


class SupervisedCollator:
    def __init__(self, sp_model_path: str, config: MT5Config,
                 max_input_length: int = 256, max_target_length: int = 256):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.config = config
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.eos_id = config.eos_token_id
        self.pad_id = config.pad_token_id

    def _encode(self, text: str, max_len: int) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        ids = ids[: max_len - 1]          # leave room for eos
        ids.append(self.eos_id)
        return ids

    def __call__(self, batch: List[Dict[str, str]]):
        """List of {'input','target'} -> padded input_ids and labels tensors."""
        enc, tgt = [], []
        for ex in batch:
            enc.append(self._encode(ex["input"], self.max_input_length))
            tgt.append(self._encode(ex["target"], self.max_target_length))

        max_in = max(len(x) for x in enc)
        max_tg = max(len(x) for x in tgt)

        input_ids = torch.full((len(enc), max_in), self.pad_id, dtype=torch.long)
        labels = torch.full((len(tgt), max_tg), -100, dtype=torch.long)  # -100 = ignore
        for r, (e, t) in enumerate(zip(enc, tgt)):
            input_ids[r, : len(e)] = torch.tensor(e, dtype=torch.long)
            labels[r, : len(t)] = torch.tensor(t, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


if __name__ == "__main__":
    import sys
    sp_path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer/spm.model"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data/manual_pairs.jsonl"

    cfg = MT5Config.small()
    pairs = load_pairs(data_path)
    print(f"loaded {len(pairs)} pairs from {data_path}")

    train, val = split_pairs(pairs, val_size=min(120, len(pairs) // 5))
    print(f"split -> train {len(train)} | val {len(val)}")

    col = SupervisedCollator(sp_path, cfg)
    batch = col(train[:4])
    print(f"input_ids: {tuple(batch['input_ids'].shape)}")
    print(f"labels   : {tuple(batch['labels'].shape)}")

    # Show one decoded round-trip to confirm input/target line up.
    ex = train[0]
    print(f"\ninput : {ex['input'][:80]}...")
    print(f"target: {ex['target'][:80]}...")
    enc_ids = col._encode(ex["input"], col.max_input_length)
    print(f"input re-decoded: {col.sp.decode(enc_ids)[:80]}...")
    print("OK: supervised collator works")
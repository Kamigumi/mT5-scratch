"""
collator.py
===========
T5/mT5 span-corruption objective. Turns a sequence of token ids into a
(corrupted_input, target) pair:

    original: the quick brown fox jumps over the lazy dog
    input:    the quick <X_0> jumps over <X_1> dog
    target:   <X_0> brown fox <X_1> the lazy </s>

RULES (matching the T5 paper defaults):
  - ~15% of tokens are corrupted (noise_density)
  - corrupted tokens form contiguous spans of mean length 3 (mean_span_length)
  - each span -> ONE sentinel in input; target lists sentinel + dropped tokens
  - sentinels are consumed in order: <extra_id_0>, <extra_id_1>, ...
  - target ends with eos

The math: with noise_density=0.15 and mean_span=3, the number of corrupted
spans = round(seq_len * 0.15 / 3). We pick which tokens are "noise" by sampling
span structure, then walk the sequence emitting input/target.
"""

import random
import torch
from typing import List

import sentencepiece as spm

from model.config import MT5Config


class SpanCorruptionCollator:
    def __init__(self, sp_model_path: str, config: MT5Config,
                 noise_density: float = 0.15, mean_span_length: float = 3.0,
                 max_input_length: int = 512, seed: int = None):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.config = config
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length
        self.max_input_length = max_input_length
        self.eos_id = config.eos_token_id
        self.pad_id = config.pad_token_id
        # Sentinel ids, looked up by name (never hardcoded).
        self.sentinel_ids = [
            self.sp.piece_to_id(f"<extra_id_{i}>") for i in range(100)
        ]
        if seed is not None:
            random.seed(seed)

    def _spans_noise_mask(self, length: int) -> List[bool]:
        """Return a per-token boolean: True = corrupted (noise).
        Builds contiguous noise spans of ~mean_span_length until the noise
        budget (length * noise_density) is filled."""
        num_noise = max(1, round(length * self.noise_density))
        num_noise = min(num_noise, length - 1)  # keep at least 1 non-noise token
        num_spans = max(1, round(num_noise / self.mean_span_length))

        # Distribute num_noise tokens across num_spans spans (each >= 1).
        noise_span_lengths = self._random_segment(num_noise, num_spans)
        # Distribute the remaining (non-noise) tokens into num_spans+? gaps.
        num_nonnoise = length - num_noise
        nonnoise_span_lengths = self._random_segment(num_nonnoise, num_spans)

        # Interleave: nonnoise, noise, nonnoise, noise, ... starting with nonnoise.
        mask = []
        for i in range(num_spans):
            mask.extend([False] * nonnoise_span_lengths[i])
            mask.extend([True] * noise_span_lengths[i])
        # Any leftover length (rounding) -> non-noise tail.
        while len(mask) < length:
            mask.append(False)
        return mask[:length]

    @staticmethod
    def _random_segment(total: int, num_segments: int) -> List[int]:
        """Split `total` items into `num_segments` positive integers, randomly."""
        if num_segments <= 0:
            return []
        if total < num_segments:
            # Not enough to give each >=1; pad with zeros where needed.
            seg = [1] * total + [0] * (num_segments - total)
            random.shuffle(seg)
            return seg
        # Place num_segments-1 cut points among total-num_segments "extra" units.
        cuts = sorted(random.sample(range(1, total), num_segments - 1)) if num_segments > 1 else []
        seg, prev = [], 0
        for c in cuts:
            seg.append(c - prev)
            prev = c
        seg.append(total - prev)
        return seg

    def corrupt(self, token_ids: List[int]):
        """Apply span corruption to one token-id list -> (input_ids, target_ids)."""
        length = len(token_ids)
        mask = self._spans_noise_mask(length)

        input_ids, target_ids = [], []
        sentinel_idx = 0
        i = 0
        while i < length:
            if mask[i]:
                # Start of a noise span: emit one sentinel to input, and to
                # target the sentinel followed by all tokens in this span.
                sentinel = self.sentinel_ids[sentinel_idx]
                sentinel_idx += 1
                input_ids.append(sentinel)
                target_ids.append(sentinel)
                while i < length and mask[i]:
                    target_ids.append(token_ids[i])
                    i += 1
            else:
                input_ids.append(token_ids[i])
                i += 1

        target_ids.append(self.eos_id)
        return input_ids, target_ids

    def encode_text(self, text: str) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        return ids[: self.max_input_length]

    def __call__(self, batch_texts: List[str]):
        """Collate a list of raw strings into padded tensors for the model.
        Returns dict with input_ids and labels (labels use -100 for pad)."""
        inputs, targets = [], []
        for text in batch_texts:
            ids = self.encode_text(text)
            if len(ids) < 2:
                continue
            inp, tgt = self.corrupt(ids)
            inputs.append(inp)
            targets.append(tgt)

        max_in = max(len(x) for x in inputs)
        max_tg = max(len(x) for x in targets)

        input_ids = torch.full((len(inputs), max_in), self.pad_id, dtype=torch.long)
        labels = torch.full((len(targets), max_tg), -100, dtype=torch.long)
        for r, (inp, tgt) in enumerate(zip(inputs, targets)):
            input_ids[r, : len(inp)] = torch.tensor(inp, dtype=torch.long)
            labels[r, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


if __name__ == "__main__":
    import sys
    sp_path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer/spm.model"
    cfg = MT5Config.tiny()
    col = SpanCorruptionCollator(sp_path, cfg, seed=0)

    text = "Ang kape ay sikat na inumin sa buong mundo at maraming tao ang umiinom nito araw-araw"
    ids = col.encode_text(text)
    inp, tgt = col.corrupt(ids)
    print(f"original ({len(ids)} tok): {col.sp.decode(ids)}")
    print(f"\ninput   ({len(inp)} tok): {col.sp.decode(inp)}")
    print(f"target  ({len(tgt)} tok): {col.sp.decode(tgt)}")

    # Verify a batch collates to correct tensor shapes.
    batch = col(["Magandang umaga sa inyong lahat",
                 "The quick brown fox jumps over the lazy dog today"])
    print(f"\nbatch input_ids: {tuple(batch['input_ids'].shape)}")
    print(f"batch labels   : {tuple(batch['labels'].shape)}")
    print("OK: span corruption + batch collation works")
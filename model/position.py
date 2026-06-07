"""
position.py
===========
T5's signature mechanism: RELATIVE POSITION BIAS.

T5/mT5 use NO absolute or sinusoidal positional encodings. Instead, a learned
scalar bias is ADDED to the attention logits, depending only on the relative
distance (key_position - query_position). Two query positions the same distance
apart from their keys get the same bias, regardless of where they sit in the
sequence.

THREE IDEAS, in order:

1. BUCKETING (_relative_position_bucket)
   We can't learn a distinct bias for every possible distance (sequences vary in
   length). So we map each integer distance to one of `num_buckets` buckets:
     - small distances  -> exact, one bucket each (fine resolution nearby)
     - large distances  -> grouped logarithmically (coarse resolution far away)
   The model needs to tell "1 away" from "2 away" precisely, but not "101" from
   "102". Logarithmic bucketing encodes that intuition.

2. DIRECTIONALITY (bidirectional flag)
   - Encoder is BIDIRECTIONAL: it matters whether a key is to the LEFT or RIGHT
     of the query, so we split buckets in half -- one half for each sign.
   - Decoder self-attention is CAUSAL: it only ever attends left, so all
     relative positions are one-signed and we use the full bucket range for
     that single direction.

3. SHARED-ACROSS-LAYERS
   Real T5 computes this bias ONCE (in the first attention layer) and every
   subsequent layer reuses the same tensor. In our design, ONE
   RelativePositionBias module lives in the encoder and one in the decoder, and
   each produces the bias that all blocks of that stack share. The Embedding
   table here is the only learned parameter: shape (num_buckets, num_heads),
   i.e. one learned scalar per (bucket, head).
"""

import torch
import torch.nn as nn

from .config import MT5Config


class RelativePositionBias(nn.Module):
    """Produces an additive attention bias of shape (1, num_heads, q_len, k_len).

    bidirectional=True  -> for the encoder (keys on both sides of the query)
    bidirectional=False -> for the decoder self-attention (causal, left only)
    """

    def __init__(self, config: MT5Config, bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.num_heads = config.num_heads
        # One learned bias scalar per (bucket, head). Looked up by bucket id,
        # giving a (q_len, k_len, num_heads) tensor we then permute.
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor,
                                  bidirectional: bool,
                                  num_buckets: int,
                                  max_distance: int) -> torch.Tensor:
        """Map integer relative positions -> bucket indices.

        This is a direct port of the logic in the T5 reference implementation.
        relative_position = key_pos - query_pos  (can be negative).
        """
        relative_buckets = torch.zeros_like(relative_position)

        if bidirectional:
            # Half the buckets for each direction. The sign of the distance
            # selects which half (offset by num_buckets//2 for positives).
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # Causal: only non-positive relative positions occur (keys to the
            # left). Clamp positives to 0 and flip sign so distance is >= 0.
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )

        # Now relative_position is >= 0.
        # Half the (remaining) buckets are for exact small distances...
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # ...the other half map larger distances logarithmically up to max_distance.
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / torch.log(torch.tensor(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def forward(self, query_length: int, key_length: int,
                device: torch.device = None) -> torch.Tensor:
        """Return additive bias of shape (1, num_heads, query_length, key_length)."""
        if device is None:
            device = self.relative_attention_bias.weight.device

        # context (query) positions down the rows, memory (key) positions across.
        context_position = torch.arange(query_length, dtype=torch.long,
                                        device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long,
                                       device=device)[None, :]
        relative_position = memory_position - context_position  # (q_len, k_len)

        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )  # (q_len, k_len), values in [0, num_buckets)

        # Look up: (q_len, k_len, num_heads) -> permute -> (1, num_heads, q, k)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values


if __name__ == "__main__":
    # Smoke test: `python -m model.position`
    cfg = MT5Config.tiny()

    enc_bias = RelativePositionBias(cfg, bidirectional=True)
    dec_bias = RelativePositionBias(cfg, bidirectional=False)

    b_enc = enc_bias(query_length=8, key_length=8)
    b_dec = dec_bias(query_length=8, key_length=8)
    print(f"encoder bias shape: {tuple(b_enc.shape)}")  # (1, num_heads, 8, 8)
    print(f"decoder bias shape: {tuple(b_dec.shape)}")
    assert b_enc.shape == (1, cfg.num_heads, 8, 8)
    assert b_dec.shape == (1, cfg.num_heads, 8, 8)

    # Sanity: in the CAUSAL (decoder) bias, the bucket for a key to the RIGHT of
    # the query (relative_position > 0) collapses to the same bucket as
    # distance 0, because causal attention never looks right. We can at least
    # confirm the two biases differ (bidirectional vs causal bucketing).
    print("encoder == decoder bias?", torch.allclose(b_enc, b_dec))
    print("OK: relative position bias produces correct shapes")
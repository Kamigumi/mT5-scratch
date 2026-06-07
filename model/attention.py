"""
attention.py
============
Multi-head attention, T5/mT5 style. ONE class serves all three roles:

  - encoder self-attention   : bidirectional, uses relative position bias, no causal mask
  - decoder self-attention   : causal mask, uses relative position bias
  - decoder cross-attention  : queries from decoder, keys/values from encoder
                               output; NO position bias

KEY T5 QUIRKS (the things people get wrong reimplementing from memory):

1. NO 1/sqrt(d_kv) SCALING.
   Standard attention computes scores = QK^T / sqrt(d_kv). T5 deliberately
   omits this division and instead compensates in the weight initialization
   (q is initialized smaller). So our scores are raw QK^T. If you add the
   sqrt scaling "to be safe", you diverge from real T5.

2. BIAS-FREE PROJECTIONS.
   q/k/v/o linear layers have no bias, like the rest of T5.

3. POSITION BIAS IS ADDED TO THE LOGITS.
   The (1, num_heads, q_len, k_len) tensor from position.py is added directly to
   the QK^T scores before softmax. Self-attention only; cross-attention passes
   position_bias=None.

4. inner_dim MAY DIFFER FROM d_model.
   Projections map d_model -> inner_dim (= num_heads * d_kv) and back. We never
   assume they're equal even though they are in mt5-base.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MT5Config


class MultiHeadAttention(nn.Module):
    def __init__(self, config: MT5Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_kv = config.d_kv
        self.inner_dim = config.inner_dim  # num_heads * d_kv

        # Bias-free projections. q/k/v: d_model -> inner_dim. o: inner_dim -> d_model.
        self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout_rate)

    def _shape(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        # (batch, seq, inner_dim) -> (batch, num_heads, seq, d_kv)
        return x.view(batch, -1, self.num_heads, self.d_kv).transpose(1, 2)

    def forward(self,
                hidden_states: torch.Tensor,
                key_value_states: torch.Tensor = None,
                mask: torch.Tensor = None,
                position_bias: torch.Tensor = None) -> torch.Tensor:
        """
        hidden_states    : (batch, q_len, d_model) -- source of QUERIES
        key_value_states : (batch, kv_len, d_model) -- source of KEYS/VALUES for
                           cross-attention. If None, self-attention (kv from
                           hidden_states).
        mask             : (batch, 1, q_len, kv_len) additive mask (0 keep,
                           -inf/large-negative drop). Used for padding and/or causality.
        position_bias    : (1, num_heads, q_len, kv_len) from position.py, or None
                           for cross-attention.
        """
        batch = hidden_states.size(0)
        is_cross = key_value_states is not None
        kv_input = key_value_states if is_cross else hidden_states

        q = self._shape(self.q(hidden_states), batch)   # (b, h, q_len, d_kv)
        k = self._shape(self.k(kv_input), batch)         # (b, h, kv_len, d_kv)
        v = self._shape(self.v(kv_input), batch)         # (b, h, kv_len, d_kv)

        # Raw QK^T -- NO division by sqrt(d_kv). (T5 design choice.)
        scores = torch.matmul(q, k.transpose(-1, -2))    # (b, h, q_len, kv_len)

        # Add relative position bias (self-attention only).
        if position_bias is not None:
            scores = scores + position_bias

        # Add mask (padding and/or causal). Broadcast over heads.
        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)                  # (b, h, q_len, d_kv)
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.inner_dim)
        return self.o(context)                           # (b, q_len, d_model)


if __name__ == "__main__":
    # Smoke test: `python -m model.attention`
    from .position import RelativePositionBias
    cfg = MT5Config.tiny()
    attn = MultiHeadAttention(cfg)

    x = torch.randn(2, 8, cfg.d_model)

    # self-attention with position bias
    enc_bias = RelativePositionBias(cfg, bidirectional=True)(8, 8)
    out_self = attn(x, position_bias=enc_bias)
    print(f"self-attn out: {tuple(out_self.shape)}")
    assert out_self.shape == x.shape

    # cross-attention: queries len 8, keys/values len 12, no position bias
    enc_out = torch.randn(2, 12, cfg.d_model)
    out_cross = attn(x, key_value_states=enc_out)
    print(f"cross-attn out: {tuple(out_cross.shape)}")
    assert out_cross.shape == x.shape

    # causal mask check: upper triangle = large negative
    causal = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
    causal = causal[None, None, :, :]  # (1,1,q,k)
    out_causal = attn(x, mask=causal, position_bias=enc_bias)
    print(f"causal self-attn out: {tuple(out_causal.shape)}")
    assert out_causal.shape == x.shape
    print("OK: self, cross, and causal attention all produce correct shapes")
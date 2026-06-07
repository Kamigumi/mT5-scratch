"""
encoder.py
==========
The encoder: a stack of identical blocks, each = self-attention + feed-forward,
wired in T5's PRE-NORM residual style.

PRE-NORM (T5) vs POST-NORM (original transformer):
    post-norm:  x = LayerNorm(x + Sublayer(x))
    pre-norm:   x = x + Sublayer(RMSNorm(x))     <-- T5/mT5 does this
The normalization happens INSIDE the residual branch, on the input to the
sublayer, and the raw residual is added back un-normalized. This is more stable
to train at depth and is what the reference implementation does.

Each EncoderBlock has TWO sub-layers, each following the pre-norm pattern:
    1. self-attention  (bidirectional, relative position bias, padding mask)
    2. GeGLU feed-forward

The whole stack shares ONE relative position bias (computed once, passed to
every block) and ends with a final RMSNorm + dropout.
"""

import torch
import torch.nn as nn

from .config import MT5Config
from .layers import RMSNorm, GeGLUFeedForward
from .attention import MultiHeadAttention
from .position import RelativePositionBias


class EncoderBlock(nn.Module):
    def __init__(self, config: MT5Config):
        super().__init__()
        # Sub-layer 1: self-attention, with its own pre-norm.
        self.self_attn_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.self_attn = MultiHeadAttention(config)
        # Sub-layer 2: feed-forward, with its own pre-norm.
        self.ff_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.ff = GeGLUFeedForward(config)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, mask=None, position_bias=None):
        # Pre-norm self-attention sub-layer.
        normed = self.self_attn_norm(x)
        attn_out = self.self_attn(normed, mask=mask, position_bias=position_bias)
        x = x + self.dropout(attn_out)            # residual added un-normalized

        # Pre-norm feed-forward sub-layer.
        normed = self.ff_norm(x)
        ff_out = self.ff(normed)
        x = x + self.dropout(ff_out)
        return x


class Encoder(nn.Module):
    def __init__(self, config: MT5Config):
        super().__init__()
        # ONE position-bias module for the whole stack (bidirectional = encoder).
        self.position_bias = RelativePositionBias(config, bidirectional=True)
        self.blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states  : (batch, seq, d_model) -- already embedded tokens
        attention_mask : (batch, 1, 1, seq) additive padding mask (0 keep,
                         large-negative drop), or None.
        """
        seq_len = hidden_states.size(1)
        # Compute the shared bias once: (1, num_heads, seq, seq).
        position_bias = self.position_bias(seq_len, seq_len,
                                            device=hidden_states.device)

        x = self.dropout(hidden_states)
        for block in self.blocks:
            x = block(x, mask=attention_mask, position_bias=position_bias)
        x = self.final_norm(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    # Smoke test: `python -m model.encoder`
    cfg = MT5Config.tiny()
    enc = Encoder(cfg)

    x = torch.randn(2, 8, cfg.d_model)
    out = enc(x)
    print(f"encoder in : {tuple(x.shape)}")
    print(f"encoder out: {tuple(out.shape)}")
    assert out.shape == x.shape

    # With a padding mask (drop last 2 positions of sequence 0).
    mask = torch.zeros(2, 1, 1, 8)
    mask[0, :, :, -2:] = float("-inf")
    out_masked = enc(x, attention_mask=mask)
    print(f"encoder out (masked): {tuple(out_masked.shape)}")
    assert out_masked.shape == x.shape
    print("OK: encoder stack produces correct shapes")
"""
decoder.py
==========
The decoder. Structurally it is the EncoderBlock PLUS the two things that make a
decoder a decoder:

    EncoderBlock          DecoderBlock
    ------------          ------------
    1. self-attention     1. self-attention  (now CAUSAL)
                          2. CROSS-attention  (NEW: reads encoder output)
    2. feed-forward       3. feed-forward

So the decoder block has THREE pre-norm sub-layers instead of two.

THE TWO DIFFERENCES, explicitly:

1. CAUSAL SELF-ATTENTION
   Decoder position i may only attend to positions <= i (it generates
   left-to-right, so it must not peek at future tokens). We enforce this with a
   causal mask: an upper-triangular matrix of large-negative values added to the
   self-attention logits. Still uses relative position bias (bidirectional=False
   in the decoder's RelativePositionBias).

2. CROSS-ATTENTION
   A whole extra sub-layer. Queries come from the decoder hidden states; keys
   and values come from the ENCODER OUTPUT. This is the bridge by which the
   decoder conditions on the source sequence. It uses NO relative position bias
   (positions across two different sequences have no meaningful T5 distance) and
   uses the ENCODER's padding mask (so the decoder ignores source padding).
"""

import torch
import torch.nn as nn

from .config import MT5Config
from .layers import RMSNorm, GeGLUFeedForward
from .attention import MultiHeadAttention
from .position import RelativePositionBias


class DecoderBlock(nn.Module):
    def __init__(self, config: MT5Config):
        super().__init__()
        # Sub-layer 1: CAUSAL self-attention.
        self.self_attn_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.self_attn = MultiHeadAttention(config)
        # Sub-layer 2: CROSS-attention (queries=decoder, kv=encoder output).
        self.cross_attn_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.cross_attn = MultiHeadAttention(config)
        # Sub-layer 3: feed-forward.
        self.ff_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.ff = GeGLUFeedForward(config)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, encoder_hidden_states,
                self_mask=None, cross_mask=None, position_bias=None):
        # 1. Pre-norm CAUSAL self-attention.
        normed = self.self_attn_norm(x)
        attn_out = self.self_attn(normed, mask=self_mask, position_bias=position_bias)
        x = x + self.dropout(attn_out)

        # 2. Pre-norm CROSS-attention: queries from x, keys/values from encoder.
        #    NO position bias here.
        normed = self.cross_attn_norm(x)
        cross_out = self.cross_attn(
            normed,
            key_value_states=encoder_hidden_states,
            mask=cross_mask,
            position_bias=None,
        )
        x = x + self.dropout(cross_out)

        # 3. Pre-norm feed-forward.
        normed = self.ff_norm(x)
        ff_out = self.ff(normed)
        x = x + self.dropout(ff_out)
        return x


class Decoder(nn.Module):
    def __init__(self, config: MT5Config):
        super().__init__()
        # Decoder self-attention position bias is CAUSAL (bidirectional=False).
        self.position_bias = RelativePositionBias(config, bidirectional=False)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_decoder_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    @staticmethod
    def _causal_mask(seq_len, device):
        """Upper-triangular large-negative mask: (1, 1, seq, seq).
        Position i (row) may not attend to j > i (cols to the right)."""
        m = torch.full((seq_len, seq_len), -1e9, device=device)
        m = torch.triu(m, diagonal=1)
        return m[None, None, :, :]

    def forward(self, hidden_states, encoder_hidden_states,
                decoder_padding_mask=None, encoder_attention_mask=None):
        """
        hidden_states          : (batch, tgt_len, d_model) embedded decoder input
        encoder_hidden_states  : (batch, src_len, d_model) encoder output
        decoder_padding_mask   : (batch, 1, 1, tgt_len) additive, or None
        encoder_attention_mask : (batch, 1, 1, src_len) additive, for cross-attn
        """
        tgt_len = hidden_states.size(1)
        device = hidden_states.device

        # Causal mask, optionally combined with decoder padding mask.
        self_mask = self._causal_mask(tgt_len, device)        # (1,1,t,t)
        if decoder_padding_mask is not None:
            self_mask = self_mask + decoder_padding_mask       # broadcast add

        position_bias = self.position_bias(tgt_len, tgt_len, device=device)

        x = self.dropout(hidden_states)
        for block in self.blocks:
            x = block(
                x,
                encoder_hidden_states=encoder_hidden_states,
                self_mask=self_mask,
                cross_mask=encoder_attention_mask,
                position_bias=position_bias,
            )
        x = self.final_norm(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    # Smoke test: `python -m model.decoder`
    cfg = MT5Config.tiny()
    dec = Decoder(cfg)

    tgt = torch.randn(2, 6, cfg.d_model)        # decoder input, len 6
    enc_out = torch.randn(2, 10, cfg.d_model)   # encoder output, len 10

    out = dec(tgt, enc_out)
    print(f"decoder in       : {tuple(tgt.shape)}")
    print(f"encoder context  : {tuple(enc_out.shape)}")
    print(f"decoder out      : {tuple(out.shape)}")
    assert out.shape == tgt.shape

    # With an encoder padding mask (ignore last 3 source positions).
    enc_mask = torch.zeros(2, 1, 1, 10)
    enc_mask[:, :, :, -3:] = float("-inf")
    out2 = dec(tgt, enc_out, encoder_attention_mask=enc_mask)
    print(f"decoder out (masked cross): {tuple(out2.shape)}")
    assert out2.shape == tgt.shape
    print("OK: decoder stack (causal self-attn + cross-attn + FF) shapes correct")
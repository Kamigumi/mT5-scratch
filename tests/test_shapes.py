"""
test_shapes.py
==============
Toy-tensor sanity checks for every module in model/. Run two ways:

    pytest tests/test_shapes.py -v        # nice green checkmarks for a demo
    python tests/test_shapes.py           # plain run, no pytest needed

Every test uses MT5Config.tiny() and tiny tensors (batch=2, short seqs) so the
whole suite runs in well under a second on CPU. The point is correctness of
shapes and basic numerics (no NaNs, grads flow), not training.
"""

import torch

from model.config import MT5Config
from model.layers import RMSNorm, GeGLUFeedForward
from model.position import RelativePositionBias
from model.attention import MultiHeadAttention
from model.encoder import Encoder
from model.decoder import Decoder
from model.mt5 import build_model

CFG = MT5Config.tiny()
B = 2


def test_rmsnorm_preserves_shape():
    x = torch.randn(B, 8, CFG.d_model)
    out = RMSNorm(CFG.d_model)(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_geglu_preserves_shape():
    x = torch.randn(B, 8, CFG.d_model)
    out = GeGLUFeedForward(CFG)(x)
    assert out.shape == x.shape


def test_position_bias_shapes_and_directionality():
    enc = RelativePositionBias(CFG, bidirectional=True)(8, 8)
    dec = RelativePositionBias(CFG, bidirectional=False)(8, 8)
    assert enc.shape == (1, CFG.num_heads, 8, 8)
    assert dec.shape == (1, CFG.num_heads, 8, 8)
    # Bidirectional and causal bucketing must differ.
    assert not torch.allclose(enc, dec)


def test_attention_self_cross_causal():
    attn = MultiHeadAttention(CFG)
    x = torch.randn(B, 8, CFG.d_model)
    bias = RelativePositionBias(CFG, bidirectional=True)(8, 8)
    # self
    assert attn(x, position_bias=bias).shape == x.shape
    # cross (different kv length)
    kv = torch.randn(B, 12, CFG.d_model)
    assert attn(x, key_value_states=kv).shape == x.shape
    # causal mask
    causal = torch.triu(torch.full((8, 8), -1e9), diagonal=1)[None, None]
    assert attn(x, mask=causal, position_bias=bias).shape == x.shape


def test_encoder_stack():
    enc = Encoder(CFG)
    x = torch.randn(B, 8, CFG.d_model)
    assert enc(x).shape == x.shape
    mask = torch.zeros(B, 1, 1, 8)
    mask[0, :, :, -2:] = -1e9
    assert enc(x, attention_mask=mask).shape == x.shape


def test_decoder_stack():
    dec = Decoder(CFG)
    tgt = torch.randn(B, 6, CFG.d_model)
    enc_out = torch.randn(B, 10, CFG.d_model)
    assert dec(tgt, enc_out).shape == tgt.shape


def test_full_model_forward_backward():
    model = build_model(CFG)
    input_ids = torch.randint(3, CFG.vocab_size, (B, 10))
    labels = torch.randint(3, CFG.vocab_size, (B, 7))
    input_ids[0, -2:] = CFG.pad_token_id
    labels[1, -1] = -100

    logits, loss = model(input_ids, labels=labels)
    assert logits.shape == (B, 7, CFG.vocab_size)
    assert not torch.isnan(loss), "loss is NaN"
    # init sanity: loss should be near ln(vocab) for an untrained model
    import math
    assert 0.5 * math.log(CFG.vocab_size) < loss.item() < 2.0 * math.log(CFG.vocab_size)

    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
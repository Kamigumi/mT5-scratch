"""
layers.py
=========
The two building blocks that make this an mT5 rather than a vanilla transformer:

  1. RMSNorm  -- T5's normalization (NOT standard LayerNorm)
  2. GeGLU FF -- mT5's gated feed-forward (NOT original-T5's ReLU FF)

Both are deliberately small and self-contained so they can be unit-tested on
toy tensors (see tests/test_shapes.py) and pointed at individually during a
presentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MT5Config


class RMSNorm(nn.Module):
    """T5-style Root Mean Square LayerNorm.

    Difference from nn.LayerNorm:
      - NO mean subtraction (not re-centered)
      - NO learned bias
      - divides by RMS = sqrt(mean(x^2) + eps), then scales by a learned weight

    Formula:
        rms = sqrt( mean(x^2, last_dim) + eps )
        out = (x / rms) * weight

    Why T5 does this: it's cheaper than LayerNorm and empirically works as well
    for transformers. The absence of the bias and mean-centering is the part
    people get wrong when reimplementing from memory.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # learned scale only
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability even under fp16/bf16,
        # which is exactly what the real T5 implementation does.
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)   # mean of squares
        x = x * torch.rsqrt(variance + self.eps)          # divide by RMS
        return self.weight * x.to(input_dtype)


class GeGLUFeedForward(nn.Module):
    """mT5's gated-GELU feed-forward block.

    Original T5:   FF(x) = Dropout( ReLU(x @ Wi) ) @ Wo
    mT5 (GeGLU):   FF(x) = Dropout( GELU(x @ Wg) * (x @ Wi) ) @ Wo
                                    \_______/   \______/
                                      gate       linear

    So there are THREE weight matrices, not two:
      - wi_0 (gate)  : d_model -> d_ff, passed through GELU
      - wi_1 (linear): d_model -> d_ff, NOT activated
      - wo           : d_ff   -> d_model
    The elementwise product of the GELU-activated gate and the linear branch is
    what "gated" means. This is the headline architectural difference from
    original T5 and the reason config.feed_forward_proj == "gated-gelu".

    Naming (wi_0 / wi_1 / wo) mirrors HuggingFace's T5DenseGatedActDense so the
    code is easy to cross-reference with the reference implementation.
    """

    def __init__(self, config: MT5Config):
        super().__init__()
        # T5 uses bias-free linear layers throughout.
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)  # gate
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)  # linear
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)    # back down
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.wi_0(x))     # GELU-activated gate branch
        linear = self.wi_1(x)           # un-activated linear branch
        x = gate * linear               # the "gated" elementwise product
        x = self.dropout(x)
        x = self.wo(x)                  # project back to d_model
        return x


if __name__ == "__main__":
    # Quick smoke test: `python -m model.layers`
    cfg = MT5Config.tiny()
    x = torch.randn(2, 8, cfg.d_model)  # (batch=2, seq=8, d_model)

    norm = RMSNorm(cfg.d_model, cfg.layer_norm_epsilon)
    ff = GeGLUFeedForward(cfg)

    n = norm(x)
    y = ff(n)
    print(f"input : {tuple(x.shape)}")
    print(f"RMSNorm out: {tuple(n.shape)}")
    print(f"GeGLU  out : {tuple(y.shape)}")
    assert n.shape == x.shape
    assert y.shape == x.shape
    print("OK: shapes preserved")
"""
config.py
=========
Single source of truth for every architectural decision in our from-scratch mT5.

Every other module (layers, attention, encoder, decoder, mt5) reads from an
instance of MT5Config. Nothing hardcodes a dimension. When we move from the
local RTX 4060 smoke-test to the Colab A100 run, we change values *here* and
nowhere else.

The defaults below mirror the REAL google/mt5-base config.json (not the class
defaults shown in the HF docs, which differ). The real base model is:
    d_model=768, d_ff=2048, num_layers=12, num_heads=12, vocab_size=250112
We also provide two helper constructors:
    MT5Config.base()  -> the real mt5-base shape (reference / aspiration)
    MT5Config.tiny()  -> a tiny shape that fits an 8GB 4060 for smoke-testing
"""

from dataclasses import dataclass


@dataclass
class MT5Config:
    # ---- Vocabulary ----
    # Real mt5-base is 250112 (250k mc4 vocab, rounded to a multiple of 128 for
    # TPU efficiency + 100 sentinel/extra tokens + a couple specials). WE will
    # train our OWN SentencePiece on TL+EN, so OUR vocab_size will be much
    # smaller (e.g. 32000). This default is the real value for reference; the
    # tokenizer step will override it.
    vocab_size: int = 250112

    # ---- Model width / depth ----
    d_model: int = 768       # hidden size (embedding + residual stream width)
    d_kv: int = 64           # size of each attention head's q/k/v projection
    d_ff: int = 2048         # inner dimension of the feed-forward block
    num_layers: int = 12     # number of ENCODER blocks
    num_decoder_layers: int = 12  # number of DECODER blocks (mt5: same as enc)
    num_heads: int = 12      # attention heads per attention layer

    # ---- Relative position bias (T5's signature mechanism) ----
    # T5/mT5 use NO absolute/sinusoidal positions. Instead a learned scalar bias
    # is added to attention logits, bucketed by relative distance. These two
    # control the bucketing; explained fully in position.py.
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

    # ---- Regularization / numerics ----
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6  # epsilon inside RMSNorm
    initializer_factor: float = 1.0   # scales weight init (kept 1.0 normally)

    # ---- Feed-forward type ----
    # mt5 uses 'gated-gelu' (GeGLU): gelu(W_gate @ x) * (W_in @ x).
    # Original T5 used plain ReLU. This flag documents the choice; layers.py
    # implements the gated variant.
    feed_forward_proj: str = "gated-gelu"

    # ---- Special token ids ----
    # These match T5/mT5 convention. pad=0, eos=1. T5 has NO bos; the decoder is
    # kick-started with decoder_start_token_id (which equals pad id, 0).
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0

    # ---- Misc ----
    tie_word_embeddings: bool = False  # mt5 does NOT tie input/output embeddings

    # -- derived / convenience --
    @property
    def inner_dim(self) -> int:
        """Total attention dim across all heads = num_heads * d_kv.
        NOTE: in mt5-base this is 12*64 = 768 = d_model, but T5 does NOT require
        inner_dim == d_model. The q/k/v projections map d_model -> inner_dim and
        the output projection maps inner_dim -> d_model, so they can differ."""
        return self.num_heads * self.d_kv

    def __post_init__(self):
        # Light sanity checks — cheap insurance against silent shape bugs later.
        assert self.d_model > 0 and self.d_ff > 0
        assert self.num_heads > 0 and self.d_kv > 0
        assert self.num_layers > 0 and self.num_decoder_layers > 0
        assert self.feed_forward_proj == "gated-gelu", (
            "This build implements only the gated-gelu (GeGLU) variant, "
            "matching mt5. Original-T5 ReLU FF is intentionally not supported."
        )

    # ---------- named profiles ----------
    @classmethod
    def base(cls) -> "MT5Config":
        """The REAL google/mt5-base shape. ~580M params. Reference target —
        will NOT fit comfortably on an 8GB card for training."""
        return cls(
            vocab_size=250112,
            d_model=768, d_kv=64, d_ff=2048,
            num_layers=12, num_decoder_layers=12, num_heads=12,
        )

    @classmethod
    def small(cls, vocab_size: int = 32000) -> "MT5Config":
        """A mid-size profile for the Colab A100 run. ~bigger than tiny, but
        trains comfortably in 40GB with bf16 + decent batch size. Roughly the
        shape of a scaled-down mt5-small."""
        return cls(
            vocab_size=vocab_size,
            d_model=512, d_kv=64, d_ff=1024,
            num_layers=8, num_decoder_layers=8, num_heads=8,
        )

    @classmethod
    def tiny(cls, vocab_size: int = 32000) -> "MT5Config":
        """A tiny shape for LOCAL smoke-testing on the RTX 4060 (8GB).
        Big enough to exercise every code path, small enough to train a few
        thousand steps quickly. We pass our own (smaller) vocab_size here since
        we'll train a 32k SentencePiece on TL+EN rather than use the 250k vocab.
        """
        return cls(
            vocab_size=vocab_size,
            d_model=256, d_kv=32, d_ff=512,
            num_layers=4, num_decoder_layers=4, num_heads=8,
        )


if __name__ == "__main__":
    # Quick manual inspection: `python -m model.config`
    for name, cfg in [("tiny", MT5Config.tiny()), ("base", MT5Config.base())]:
        print(f"--- {name} ---")
        print(f"  d_model={cfg.d_model}  d_ff={cfg.d_ff}  "
              f"layers={cfg.num_layers}  heads={cfg.num_heads}  "
              f"inner_dim={cfg.inner_dim}  vocab={cfg.vocab_size}")
"""
mt5.py
======
The full mT5 model, assembling everything:

    shared token embedding ─┬─> Encoder ─> encoder_hidden_states
                            └─> Decoder (cross-attends to encoder) ─> lm_head ─> logits

KEY ASSEMBLY DECISIONS:

1. SHARED INPUT EMBEDDING
   One nn.Embedding feeds BOTH encoder inputs and decoder inputs. The output
   projection (lm_head) is SEPARATE: mt5 does NOT tie input/output embeddings
   (config.tie_word_embeddings=False).

2. DECODER START TOKEN (no BOS in T5)
   To produce decoder INPUTS from target labels, we shift the targets right by
   one and prepend decoder_start_token_id (which equals pad_id=0 in T5). See
   _shift_right. The model learns to predict token t from tokens < t.

3. T5 WEIGHT INITIALIZATION
   Because attention omits the 1/sqrt(d_kv) scaling (see attention.py), T5 uses
   a specific init scheme rather than default PyTorch init. We replicate the
   important parts: embeddings ~ N(0, factor*1.0); the FF and attention
   projections get std values derived from their fan-in / d_model / d_kv. This
   matters: default init + no-sqrt-scaling trains badly.

4. LOSS
   Standard cross-entropy over decoder logits vs labels, ignoring pad positions
   (ignore_index = pad_token_id). Teacher forcing is implicit in the right-shift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MT5Config
from .encoder import Encoder
from .decoder import Decoder


class MT5(nn.Module):
    def __init__(self, config: MT5Config):
        super().__init__()
        self.config = config

        # Shared token embedding (encoder + decoder inputs).
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Separate output projection (NOT tied to `shared`).
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    # ---------- T5-style initialization ----------
    def _init_weights(self, module):
        factor = self.config.initializer_factor
        d_model = self.config.d_model
        d_ff = self.config.d_ff
        d_kv = self.config.d_kv
        n_heads = self.config.num_heads

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, nn.Linear):
            # Default for the lm_head and any unmatched linear.
            module.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            if module.bias is not None:
                module.bias.data.zero_()

    def _init_module_specific(self):
        """Apply T5's per-projection std values. Called after submodules exist.
        We walk named modules so we can use their ROLE (q/k/v/o, wi/wo)."""
        factor = self.config.initializer_factor
        d_model = self.config.d_model
        d_ff = self.config.d_ff
        d_kv = self.config.d_kv
        n_heads = self.config.num_heads

        for name, p in self.named_parameters():
            if name.endswith("self_attn.q.weight") or name.endswith("cross_attn.q.weight"):
                # q is scaled down by (d_model*d_kv)^-0.5 to compensate for the
                # missing 1/sqrt(d_kv) in the attention scores.
                p.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            elif name.endswith(".k.weight") or name.endswith(".v.weight"):
                p.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            elif name.endswith(".o.weight"):
                p.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))
            elif name.endswith("wi_0.weight") or name.endswith("wi_1.weight"):
                p.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            elif name.endswith("wo.weight"):
                p.data.normal_(mean=0.0, std=factor * (d_ff ** -0.5))

    # ---------- right-shift to build decoder inputs ----------
    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """Prepend decoder_start_token_id and drop the last token.
        labels: (batch, tgt_len) -> decoder_input_ids: (batch, tgt_len)."""
        start = self.config.decoder_start_token_id
        pad = self.config.pad_token_id

        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = start
        # Replace any -100 (loss-ignore) in the shifted inputs with pad, so the
        # embedding lookup never sees a negative index.
        shifted.masked_fill_(shifted == -100, pad)
        return shifted

    # ---------- mask helpers ----------
    def _padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(batch, seq) ids -> (batch, 1, 1, seq) additive mask.
        Positions equal to pad_token_id get large-negative; others 0."""
        pad = self.config.pad_token_id
        mask = (input_ids == pad)  # True where padding
        additive = torch.zeros_like(mask, dtype=torch.float32)
        # Large negative (not -inf): -inf makes fully-masked softmax rows NaN.
        additive.masked_fill_(mask, -1e9)
        return additive[:, None, None, :]

    # ---------- forward ----------
    def forward(self, input_ids, labels=None, attention_mask_ids=None):
        """
        input_ids : (batch, src_len) encoder token ids
        labels    : (batch, tgt_len) target token ids (use -100 to ignore in loss)
        Returns (logits, loss) where loss is None if labels not given.
        """
        # Encoder padding mask from the source ids (or a provided id tensor).
        enc_mask = self._padding_mask(
            attention_mask_ids if attention_mask_ids is not None else input_ids
        )

        enc_emb = self.shared(input_ids)
        encoder_hidden = self.encoder(enc_emb, attention_mask=enc_mask)

        loss = None
        if labels is not None:
            decoder_input_ids = self._shift_right(labels)
            dec_emb = self.shared(decoder_input_ids)
            decoder_padding_mask = self._padding_mask(decoder_input_ids)
            dec_out = self.decoder(
                dec_emb,
                encoder_hidden_states=encoder_hidden,
                decoder_padding_mask=decoder_padding_mask,
                encoder_attention_mask=enc_mask,
            )
            logits = self.lm_head(dec_out)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return logits, loss

        # Inference path without labels: caller drives the decoder separately.
        return encoder_hidden, None

    # ---------- inference ----------
    @torch.no_grad()
    def encode(self, input_ids):
        """Run the encoder once. Returns (encoder_hidden, encoder_mask)."""
        enc_mask = self._padding_mask(input_ids)
        enc_emb = self.shared(input_ids)
        encoder_hidden = self.encoder(enc_emb, attention_mask=enc_mask)
        return encoder_hidden, enc_mask

    @torch.no_grad()
    def decode_step(self, decoder_input_ids, encoder_hidden, encoder_mask):
        """Given the decoder ids SO FAR, return logits for every position.
        We recompute the full decoder each step (no KV cache) -- simple and
        correct; KV caching is an optimization we can add later. The logits for
        the LAST position are what we use to pick the next token."""
        dec_emb = self.shared(decoder_input_ids)
        dec_pad_mask = self._padding_mask(decoder_input_ids)
        dec_out = self.decoder(
            dec_emb,
            encoder_hidden_states=encoder_hidden,
            decoder_padding_mask=dec_pad_mask,
            encoder_attention_mask=encoder_mask,
        )
        return self.lm_head(dec_out)  # (batch, cur_len, vocab)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=0.0,
                 top_k=None, eos_token_id=None):
        """Autoregressive decoding.

        temperature=0.0  -> greedy (argmax). >0 -> sampling with that temperature.
        top_k            -> if set (with temperature>0), restrict sampling to the
                            top-k logits.
        Stops a sequence once it emits eos; returns ids WITHOUT the leading
        decoder_start token, one row per input.
        """
        self.eval()
        device = input_ids.device
        batch = input_ids.size(0)
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        encoder_hidden, encoder_mask = self.encode(input_ids)

        # Start every sequence with decoder_start_token_id.
        dec = torch.full((batch, 1), self.config.decoder_start_token_id,
                         dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decode_step(dec, encoder_hidden, encoder_mask)
            next_logits = logits[:, -1, :]  # (batch, vocab)

            if temperature and temperature > 0.0:
                next_logits = next_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch,1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # greedy

            # Once finished, keep emitting pad so shapes stay aligned.
            next_token = next_token.masked_fill(finished.unsqueeze(1),
                                                self.config.pad_token_id)
            dec = torch.cat([dec, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == eos)
            if finished.all():
                break

        return dec[:, 1:]  # drop the leading start token

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model(config: MT5Config = None) -> MT5:
    """Factory: build an MT5 and apply the role-specific T5 init."""
    config = config or MT5Config.tiny()
    model = MT5(config)
    model._init_module_specific()
    return model


if __name__ == "__main__":
    # Smoke test: `python -m model.mt5`
    cfg = MT5Config.tiny()
    model = build_model(cfg)
    print(f"params: {model.num_params():,}")

    B, S, T = 2, 10, 7
    input_ids = torch.randint(3, cfg.vocab_size, (B, S))   # avoid specials in input
    labels = torch.randint(3, cfg.vocab_size, (B, T))
    # Throw in some padding + ignore positions.
    input_ids[0, -2:] = cfg.pad_token_id
    labels[1, -1] = -100

    logits, loss = model(input_ids, labels=labels)
    print(f"logits: {tuple(logits.shape)}  (expect (2, 7, {cfg.vocab_size}))")
    print(f"loss  : {loss.item():.4f}")
    assert logits.shape == (B, T, cfg.vocab_size)

    # A backward pass to confirm gradients flow.
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"all params have grads: {grad_ok}")
    print("OK: full mT5 forward + backward works")
"""
train_tokenizer.py
==================
Trains our own SentencePiece tokenizer on a balanced TL+EN sample, with the
T5/mT5 special-token layout:

    id 0 -> <pad>
    id 1 -> </s>   (eos)
    id 2 -> <unk>
    then normal learned subword pieces ...
    then 100 sentinels: <extra_id_0> ... <extra_id_99>

WHY THIS MATTERS
----------------
The sentinels are the hinge of T5 span-corruption pretraining. The collator
(built next) will mask spans of input and replace each with a sentinel, then
ask the decoder to reconstruct "<extra_id_0> <dropped span> <extra_id_1> ...".
So the sentinels MUST exist as dedicated, single-piece tokens in the vocab
BEFORE we train the model. We reserve them here as user_defined_symbols.

DESIGN CHOICES (and how to present them)
----------------------------------------
- vocab_size=32000: tailored to TL+EN only, far smaller than mt5's 250k. The
  embedding table is vocab_size * d_model, so this keeps the model small enough
  to train on an 8GB 4060.
- We never hardcode sentinel ids. We ask the trained tokenizer for them by name
  (sp.piece_to_id("<extra_id_0>")). Robust + self-documenting.
- model_type="unigram": T5/mT5 use the unigram LM model, not BPE. We match it.
- character_coverage=1.0: TL+EN are Latin-script with small alphabets, so we can
  afford full character coverage (unlike CJK, which uses ~0.9995).
- <unk> is NOT passed in user_defined_symbols (SentencePiece forbids it; unk is
  always special). We only set its id/piece via the dedicated flags.

USAGE
-----
    python train_tokenizer.py --input data/tok_sample.txt --out tokenizer/spm

Produces tokenizer/spm.model and tokenizer/spm.vocab
"""

import argparse
import os
import sentencepiece as spm

NUM_SENTINELS = 100  # T5 reserves 100 extra_id tokens


def build_sentinels(n: int = NUM_SENTINELS):
    """The sentinel pieces, in T5's naming. extra_id_0 .. extra_id_(n-1)."""
    return [f"<extra_id_{i}>" for i in range(n)]


def train(input_path: str, model_prefix: str, vocab_size: int = 32000,
          character_coverage: float = 1.0, input_sentence_size: int = 2_000_000):
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Training corpus not found: {input_path}\n"
            f"Run the data step first to produce a balanced TL+EN sample."
        )

    sentinels = build_sentinels()

    # We fix the special-token ids explicitly to the T5 layout.
    #   pad=0, eos=1, unk=2, and NO bos (bos_id=-1 disables it).
    # The sentinels go in user_defined_symbols: each is guaranteed to be a single
    # piece and never split. They'll be appended to the vocab by SentencePiece.
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",                # T5/mT5 use unigram, not bpe
        character_coverage=character_coverage,
        # --- special tokens / T5 id layout ---
        pad_id=0, pad_piece="<pad>",
        eos_id=1, eos_piece="</s>",
        unk_id=2, unk_piece="<unk>",
        bos_id=-1,                            # T5 has no BOS
        user_defined_symbols=sentinels,       # the 100 sentinels
        # --- practical knobs ---
        input_sentence_size=input_sentence_size,  # subsample huge corpora
        shuffle_input_sentence=True,
        normalization_rule_name="nmt_nfkc_cf",     # T5-ish unicode normalization
        train_extremely_large_corpus=False,
    )
    print(f"\nTrained: {model_prefix}.model / {model_prefix}.vocab")
    return f"{model_prefix}.model"


def inspect(model_file: str):
    """Load and sanity-check the trained tokenizer. Great to run live in a demo."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)

    print(f"\n--- tokenizer report ---")
    print(f"vocab_size : {sp.get_piece_size()}")
    print(f"pad id     : {sp.pad_id()}  piece: {sp.id_to_piece(sp.pad_id())!r}")
    print(f"eos id     : {sp.eos_id()}  piece: {sp.id_to_piece(sp.eos_id())!r}")
    print(f"unk id     : {sp.unk_id()}  piece: {sp.id_to_piece(sp.unk_id())!r}")

    s0 = sp.piece_to_id("<extra_id_0>")
    s99 = sp.piece_to_id("<extra_id_99>")
    print(f"<extra_id_0>  id: {s0}")
    print(f"<extra_id_99> id: {s99}")
    assert s0 != sp.unk_id(), "sentinel resolved to <unk> — not reserved!"

    # Round-trip on a Taglish sample — the exact thing we care about.
    sample = "Nag-jowa na sila omg sobrang cute parang movie lang"
    ids = sp.encode(sample, out_type=int)
    pieces = sp.encode(sample, out_type=str)
    print(f"\nsample      : {sample}")
    print(f"pieces      : {pieces}")
    print(f"n_pieces    : {len(pieces)}")
    print(f"round-trip  : {sp.decode(ids)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to TL+EN training text")
    ap.add_argument("--out", default="tokenizer/spm", help="model_prefix")
    ap.add_argument("--vocab_size", type=int, default=32000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model_file = train(args.input, args.out, vocab_size=args.vocab_size)
    inspect(model_file)
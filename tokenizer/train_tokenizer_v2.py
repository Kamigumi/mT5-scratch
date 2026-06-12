"""
train_tokenizer_v2.py
=====================
Version 2 of the tokenizer, with TWO changes that fix the emoji/unk problem:

  1. byte_fallback=True
     Any character not in the learned vocab (emoji, rare Unicode, unusual
     punctuation) is encoded as its raw UTF-8 BYTE tokens instead of collapsing
     to <unk>. This GUARANTEES lossless round-trip: 💖 -> [byte tokens] -> 💖.
     SentencePiece adds 256 byte pieces (<0x00>..<0xFF>) to the vocab for this.

  2. Social text in the training corpus
     v1 trained only on Wikipedia + mC4 (clean), so emoji/slang were rare and
     got no dedicated pieces. v2 ADDS the normalization data's own text (both
     noisy inputs and clean targets from synthetic + gold) so the tokenizer sees
     the actual distribution it must handle. Even with byte_fallback, having
     frequent emoji as learned pieces is more efficient than always falling back.

Everything else matches v1: unigram model, T5 id layout (pad=0,eos=1,unk=2),
100 sentinels as user_defined_symbols, nmt_nfkc_cf normalization.

IMPORTANT: byte_fallback changes vocab_size's composition (adds 256 byte
pieces). We keep the TARGET vocab_size at 32000; SentencePiece fits learned
pieces + specials + sentinels + 256 bytes within it.

USAGE:
    python train_tokenizer_v2.py --input data/tok_sample_v2.txt --out tokenizer/spm_v2
"""

import argparse
import os
import sentencepiece as spm

NUM_SENTINELS = 100


def build_sentinels(n=NUM_SENTINELS):
    return [f"<extra_id_{i}>" for i in range(n)]


def train(input_path, model_prefix, vocab_size=32000,
          character_coverage=1.0, input_sentence_size=2_000_000):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Corpus not found: {input_path}")

    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        # --- THE KEY CHANGE ---
        byte_fallback=True,            # unknown chars -> UTF-8 byte tokens, not <unk>
        # --- T5 id layout (unchanged from v1) ---
        pad_id=0, pad_piece="<pad>",
        eos_id=1, eos_piece="</s>",
        unk_id=2, unk_piece="<unk>",
        bos_id=-1,
        user_defined_symbols=build_sentinels(),
        # --- practical knobs ---
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        normalization_rule_name="nmt_nfkc",  # case-PRESERVING (no _cf casefold)
        # With byte_fallback, we want the model to also keep frequent whole
        # symbols; max_sentencepiece_length default is fine.
    )
    print(f"\nTrained: {model_prefix}.model / {model_prefix}.vocab")
    return f"{model_prefix}.model"


def inspect(model_file):
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    print(f"\n--- tokenizer v2 report ---")
    print(f"vocab_size : {sp.get_piece_size()}")
    print(f"pad/eos/unk: {sp.pad_id()}/{sp.eos_id()}/{sp.unk_id()}")
    print(f"<extra_id_0> : {sp.piece_to_id('<extra_id_0>')}")
    print(f"<extra_id_99>: {sp.piece_to_id('<extra_id_99>')}")

    # THE CRITICAL TEST: emoji must round-trip, NOT become <unk>.
    for sample in [
        "Nag-jowa na sila omg sobrang cute 💖😍 parang movie lang 😭",
        "di ko inexpect yung prices grabe huhu 🤣",
    ]:
        ids = sp.encode(sample, out_type=int)
        decoded = sp.decode(ids)
        unk_count = ids.count(sp.unk_id())
        # detect byte pieces in the piece view
        pieces = sp.encode(sample, out_type=str)
        n_byte = sum(1 for p in pieces if p.startswith("<0x"))
        print(f"\nsample  : {sample}")
        print(f"decoded : {decoded}")
        print(f"<unk> count: {unk_count}  (MUST be 0)")
        print(f"byte pieces used: {n_byte}")
        print(f"round-trip exact: {decoded == sample}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="tokenizer/spm_v2")
    ap.add_argument("--vocab_size", type=int, default=32000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model_file = train(args.input, args.out, vocab_size=args.vocab_size)
    inspect(model_file)
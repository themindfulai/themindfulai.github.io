"""Build character-level corpus + vocabulary from unified JSONL.

Input default:
  training/datasets/processed/unified_train.jsonl
Outputs:
  data/corpus_full.txt
  data/char_vocab.json   (list of unique chars)
"""

from __future__ import annotations
import json, argparse, random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "training" / "datasets" / "processed" / "unified_train.jsonl"
DATA_DIR = REPO_ROOT / "data"


def iter_formatted(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            txt = obj.get("formatted")
            if not txt:
                u = obj.get("user","")
                a = obj.get("assistant","")
                if u and a:
                    txt = f"<|user|>\n{u}\n<|assistant|>\n{a}"
            if txt:
                yield txt.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="Unified JSONL path")
    ap.add_argument("--max-chars", type=int, default=2_000_000, help="Cap total characters")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle samples before truncation")
    args, _ = ap.parse_known_args()  # tolerate extra Jupyter args

    src = Path(args.input)
    if not src.is_file():
        raise SystemExit(f"[error] Input not found: {src}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    samples = list(iter_formatted(src))
    if not samples:
        raise SystemExit("[error] No samples extracted.")

    if args.shuffle:
        random.shuffle(samples)

    pieces = []
    total = 0
    for s in samples:
        seg = s + "\n\n"
        if total + len(seg) > args.max_chars:
            seg = seg[: max(0, args.max_chars - total)]
        pieces.append(seg)
        total += len(seg)
        if total >= args.max_chars:
            break

    corpus = "".join(pieces)
    (DATA_DIR / "corpus_full.txt").write_text(corpus, encoding="utf-8")
    vocab = sorted(set(corpus))
    with (DATA_DIR / "char_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"[ok] chars={len(corpus)} unique={len(vocab)} -> data/corpus_full.txt, data/char_vocab.json")

if __name__ == "__main__":
    main()

"""Extract chosen responses from Anthropic/hh-rlhf subsets you selected.

Subsets handled (matching your screenshot):
  - helpful-base
  - harmless-base
  - helpful-rejection-sampled
  - helpful-online (optionally downsampled)
  - red-team-attempts (USER PROMPTS ONLY -> evaluation file)

Outputs (JSONL) written to raw/:
  hh_helpful_train.jsonl                (helpful-base)
  hh_harmless_train.jsonl               (harmless-base)
  hh_helpful_rejection_train.jsonl      (helpful-rejection-sampled)
  hh_helpful_online_train.jsonl         (sampled helpful-online)
  red_team_prompts.txt                  (one prompt per line, evaluation only)

Usage examples:
  python extract_hh.py                  # default fractions
  python extract_hh.py --online-fraction 0.25 --oversample-rejection 0.4

You can disable subsets:
  python extract_hh.py --no-online

Requires: datasets
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover
    raise SystemExit("datasets library not installed. Run: pip install datasets") from e

RAW_DIR = Path(__file__).parent / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

SUBSET_CONFIGS = {
    'helpful-base': ('hh_helpful_train.jsonl', True),
    'harmless-base': ('hh_harmless_train.jsonl', True),
    'helpful-rejection-sampled': ('hh_helpful_rejection_train.jsonl', True),
    'helpful-online': ('hh_helpful_online_train.jsonl', True),
    'red-team-attempts': ('red_team_prompts.txt', False),  # eval only
}


def extract_chosen(dataset, include_rejected=False):
    for ex in dataset:
        chosen = ex.get('chosen')
        if not chosen:
            continue
        lines = chosen.split('\n')
        if len(lines) < 2:
            continue
        user = lines[0].strip()
        assistant = '\n'.join(lines[1:]).strip()
        if not user or not assistant:
            continue
        yield {'user': user, 'assistant': assistant}


def write_jsonl(path: Path, rows):
    count = 0
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-online', action='store_true', help='Skip helpful-online subset')
    ap.add_argument('--no-rejection', action='store_true', help='Skip helpful-rejection-sampled subset')
    ap.add_argument('--fallback-default', action='store_true', help='If individual configs missing, load default combined dataset')
    ap.add_argument('--online-fraction', type=float, default=0.3, help='Fraction of helpful-online to keep (0-1)')
    ap.add_argument('--oversample-rejection', type=float, default=0.3, help='Fractional oversample of rejection-sampled (0.3 => +30%)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit-per-subset', type=int, default=0,
                help='If >0, cap number of chosen rows per subset AFTER sampling/oversample.')
    args = ap.parse_args()

    random.seed(args.seed)

    # Always load base & harmless
    subsets = ['helpful-base', 'harmless-base']
    if not args.no_rejection:
        subsets.append('helpful-rejection-sampled')
    if not args.no_online and args.online_fraction > 0:
        subsets.append('helpful-online')
    subsets.append('red-team-attempts')  # always gather prompts for eval

    summary = []
    red_team_prompts = []

    loaded_any = False
    for cfg in subsets:
        out_name, train_flag = SUBSET_CONFIGS[cfg]
        print(f"[info] Loading subset {cfg} ...")
        try:
            ds = load_dataset('Anthropic/hh-rlhf', cfg, split='train')
        except Exception as e:
            print(f"[warn] Failed to load {cfg}: {e}")
            continue
        loaded_any = True
        if cfg == 'red-team-attempts':
            for ex in ds:
                prompt = ex.get('prompt') or ex.get('chosen') or ''
                if prompt:
                    red_team_prompts.append(prompt.strip().split('\n')[0])
            continue
        rows = list(extract_chosen(ds))
        original_len = len(rows)
        if cfg == 'helpful-online' and 0 < args.online_fraction < 1:
            keep = int(len(rows) * args.online_fraction)
            rows = random.sample(rows, keep) if keep < len(rows) else rows
        if cfg == 'helpful-rejection-sampled' and args.oversample_rejection > 0:
            extra = int(len(rows) * args.oversample_rejection)
            if extra > 0 and rows:
                rows.extend(random.sample(rows, min(extra, len(rows))))
        # NEW: enforce limit
        if args.limit_per_subset > 0 and len(rows) > args.limit_per_subset:
            rows = random.sample(rows, args.limit_per_subset)
        out_path = RAW_DIR / out_name
        count = write_jsonl(out_path, rows)
        summary.append((cfg, count, original_len))
        print(f"[ok] {cfg}: wrote {count} rows -> {out_name} (raw before sampling {original_len})")

    if args.fallback_default and not loaded_any:
        print('[info] Attempting fallback: load default combined dataset ...')
        try:
            ds_all = load_dataset('Anthropic/hh-rlhf', split='train')
            rows = []
            for ex in ds_all:
                chosen = ex.get('chosen')
                if not chosen:
                    continue
                parts = chosen.split('\n')
                if len(parts) < 2:
                    continue
                user = parts[0].strip()
                assistant = '\n'.join(parts[1:]).strip()
                if user and assistant:
                    rows.append({'user': user, 'assistant': assistant})
            out_path = RAW_DIR / 'hh_default_train.jsonl'
            count = write_jsonl(out_path, rows)
            summary.append(('default', count, count))
            print(f"[ok] default: wrote {count} rows -> hh_default_train.jsonl")
        except Exception as e:
            print(f"[error] Fallback default load failed: {e}")

    # Write red-team prompts file
    if red_team_prompts:
        rt_path = RAW_DIR / 'red_team_prompts.txt'
        with rt_path.open('w', encoding='utf-8') as f:
            for p in red_team_prompts:
                f.write(p + '\n')
        print(f"[ok] red-team prompts -> {rt_path} ({len(red_team_prompts)} prompts)")

    print('\nSummary:')
    for cfg, kept, orig in summary:
        print(f"  {cfg:26s} kept={kept} original={orig}")
    print("Done.")

if __name__ == '__main__':
    main()

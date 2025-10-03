"""
Colab end-to-end:
1. Sample Open-Orca streaming
2. (Optional) Distill with teacher
3. Build corpus
4. Train tiny MiniGPT
5. Export weights
6. Commit & push

Usage (inside Colab):
python training/colab_auto_pipeline.py \
  --repo-user <user> --repo-name themindfulai.github.io \
  --token <gh_pat> --sample 4000 --distill 3000
"""
import argparse, json, subprocess, sys, random, os
from pathlib import Path

def run(cmd):
    print(f"[run] {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"[error] Command failed: {cmd}")
        sys.exit(r.returncode)

def stream_openorca(out_file: Path, limit: int):
    from datasets import load_dataset
    stream = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    written=0
    with out_file.open("w", encoding="utf-8") as f:
        for ex in stream:
            if written >= limit:
                break
            u = (ex.get("question") or "").strip()
            a = (ex.get("response") or "").strip()
            if u and a:
                rec = {
                    "source":"orca",
                    "user":u,
                    "assistant":a,
                    "formatted":f"<|user|>\n{u}\n<|assistant|>\n{a}"
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    print(f"[ok] Open-Orca streamed: {written}")

def distill(in_file: Path, out_file: Path, sample: int, teacher: str):
    run(
        f"python training/distill_teacher.py "
        f"--teacher-model {teacher} "
        f"--unified-file {in_file} "
        f"--out-file {out_file} "
        f"--sample {sample} --max-new-tokens 160 --temperature 0.8"
    )
    # Replace unified with distilled
    in_file.unlink(missing_ok=True)
    out_file.rename(in_file)
    print("[ok] Distillation replaced unified file")

def git_commit_push(user, email, token, repo_user, repo_name, message):
    run(f'git config user.name "{user}"')
    run(f'git config user.email "{email}"')
    run("git add weights/manifest.json weights/weights.bin data/char_vocab.json || true")
    run("git add training/build_corpus.py training/train_mini_transformer.py training/model_minigpt.py training/export_weights.py || true")
    # Avoid empty commit
    status = subprocess.check_output("git status --porcelain", shell=True).decode().strip()
    if not status:
        print("[info] Nothing new to commit.")
        return
    run(f'git commit -m "{message}"')
    run(f'git remote set-url origin https://{token}@github.com/{repo_user}/{repo_name}.git')
    run("git push origin master")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-user", required=True)
    ap.add_argument("--repo-name", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--git-user", default="colab-bot")
    ap.add_argument("--git-email", default="colab-bot@example.com")
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--distill", type=int, default=0, help="If >0, number of samples to distill")
    ap.add_argument("--teacher", default="Qwen/Qwen1.5-0.5B")
    ap.add_argument("--chars", type=int, default=1_800_000)
    ap.add_argument("--steps", type=int, default=1200)
    args, _ = ap.parse_known_args()

    unified = Path("training/datasets/processed/unified_train.jsonl")
    if not unified.exists():
        stream_openorca(unified, args.sample)

    if args.distill > 0:
        distill(unified, Path("training/datasets/processed/distilled_pairs.jsonl"),
                args.distill, args.teacher)

    run(f"python training/build_corpus.py --max-chars {args.chars}")
    run(f"python training/train_mini_transformer.py --steps {args.steps} --context 256")
    run("python training/export_weights.py --out-dir weights")

    size = Path("weights/weights.bin").stat().st_size / (1024*1024)
    print(f"[info] weights.bin size: {size:.2f} MB")
    if size > 95:
        print("[warn] weights larger than ~95MB; GitHub might reject without LFS.")

    git_commit_push(
        args.git_user, args.git_email, args.token,
        args.repo_user, args.repo_name,
        f"Auto train mini model (steps={args.steps}, distill={args.distill})"
    )
    print("[done] Pipeline complete.")

if __name__ == "__main__":
    main()
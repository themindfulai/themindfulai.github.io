# The MindfulAI (Local, Privacy-First Prototype)

This project is a fully static, privacy-first mental wellness companion. No external model calls, no backend database. All logic runs in-browser and data stays in `localStorage` on the user's device.

## IMPORTANT DISCLAIMER
This app is NOT a substitute for professional diagnosis, therapy, or crisis intervention. If you (or someone you know) are in immediate danger or considering self‑harm, contact local emergency services or a crisis hotline immediately.

## Features
- 100% static hosting (GitHub Pages compatible)
- Local heuristic retrieval model (lexicon + lightweight retrieval + templates + incremental vocabulary learning)
- Optional local generative trigram character model (MODE='gen') – mood-conditioned, no external calls
- Mood detection (positive, negative, stress, neutral)
- Chat history persisted locally (never uploaded)
- Extensible datasets under `data/`
 - Mini transformer prototype (MODE='gpt') with optional automated training & weight export

## Structure
```
index.html        Landing page (unchanged UI)
app.html          Chat interface (unchanged UI, local model wiring)
config.js         Global config (MODE=local)
js/model.js       Local model + mood detection
data/responses.json  Intent templates & patterns
data/lexicon.json     Mood lexicon
static/favicon.png    (add your icon file)
```

## How the Local Model Works
1. Tokenize user message.
2. Mood scoring via weighted lexicon.
3. Retrieval: cosine similarity (TF-IDF) against pattern groups in `responses.json`.
4. If a best intent passes threshold, choose a random template; else fallback supportive template with reflective wording.
5. Incremental vocabulary learning (stored frequencies in `localStorage`) for potential future tuning.

## Extending the Model
- Retrieval path (MODE='local'):
	- Add new intents: append objects in `data/responses.json` with `intent`, `patterns`, `templates`.
	- Add emotional words: extend categories in `data/lexicon.json` (keep weights ~0.8–1.6).
	- Bump `MODEL_VERSION` in `config.js` if you later add caching logic.
- Generative path (MODE='gen'):
	- Edit / append mood‑tagged lines in `data/corpus.txt` format: `[mood]|Your supportive sentence here`.
	- Supported mood tags are free-form; unknown map to neutral distribution.
	- (Optional) Precompute trigram JSON with `training/build_trigram.py` and embed for faster load.

## Deployment (GitHub Pages)
1. Create a GitHub repo and push these files to the `master` branch.
2. In repo Settings > Pages: choose `Deploy from branch`, branch `master`, folder `/ (root)`.
3. Wait for Pages to build; site appears at `https://<username>.github.io/<repo>/`.
4. Share only after adding your real favicon to `static/favicon.png`.

## Optional GitHub Actions Automation
Add `.github/workflows/deploy.yml` (already provided if you create one) to ensure Pages redeploys on push.

## Local Testing
You can open `index.html` directly (file://) OR run a tiny local server:
```
python -m http.server 8080
```
Then visit http://localhost:8080/app.html

### Switch to Generative Mode
Edit `config.js` and set:
```
MODE: 'gen'
```
Reload `app.html` – responses will now come from the trigram character model.

### Switch to Mini GPT Mode
```
MODE: 'gpt'
```
If `weights/manifest.json` + `weights/weights.bin` are present they load automatically (see Automation section).

### Improving Generative Quality
1. Expand `data/corpus.txt` with more high-quality, concise supportive lines (avoid personal data!).
2. Keep each line focused and non-repetitive.
3. Add mood variants (e.g., `[stress]|...`, `[negative]|...`, `[positive]|...`).
4. Avoid sensitive or identifying info; keep general and supportive.
5. (Optional offline) run:
```
python training/build_trigram.py --corpus data/corpus.txt --out data/trigram_model.json
```
	Then adapt `js/gen_model.js` to load the JSON directly instead of rebuilding.

## Automated Mini GPT Training & Export
Workflow file: `.github/workflows/train-model.yml` (manual trigger) performs:
1. Dataset expansion from `training/raw/seed_samples.txt` → `training/build/dataset.txt`.
2. Tiny transformer training (char-level) aligning with browser architecture.
3. Weight export to `weights/manifest.json` + `weights/weights.bin`.
4. Auto commit back to `master`.

Run it via GitHub Actions → "Train Mini Model" → Run workflow. After completion set `MODE: 'gpt'` and refresh.

Customize by editing:
- Seeds: `training/raw/seed_samples.txt`
- Hyperparams: `training/train_mini_transformer.py`
- Paraphraser: `training/prepare_dataset.py`

Safety: Crisis lines remain static; code overrides model output for suicidal cues.

## Automated Web Mini Model Pipeline
In addition to the earlier `train-model.yml`, a lighter end‑to‑end pipeline is provided:

Files:
- `training/run_full_pipeline.py` – orchestrates unify → corpus → train → export.
- `training/build_corpus.py` – caps unified instruction data into a character corpus.
- `.github/workflows/train-web-model.yml` – Action you can trigger manually.

Trigger (GitHub UI → Actions → "Train Web Mini Model") – it will:
1. Run `prepare_unified.py` (with a small ORCA cap) if present.
2. Build `data/corpus_full.txt` (default 2M chars) + `data/char_vocab.json`.
3. Train the tiny transformer (`train_mini_transformer.py`) for the configured steps.
4. Export weights to `weights/manifest.json` + `weights/weights.bin`.
5. Commit updated artifacts.

Local manual run example:
```
python training/run_full_pipeline.py --train-steps 1500 --context 256 --max-chars 2000000
```

Then set in `config.js`:
```
MODE: 'gpt'
```
Refresh `app.html` – new weights will be fetched by `js/gpt_weights_loader.js`.

Tuning knobs:
- `--max-chars` smaller → faster train & smaller vocab.
- `--train-steps` more steps → potentially better fit (risk overfit if corpus tiny).
- `--context` raise if your conversations get longer (costs memory/time).

Outputs to watch:
- `weights/manifest.json` (tensor shapes & metadata)
- `weights/weights.bin` (flat binary array)
- `data/char_vocab.json` (character set for reproducibility)

If weights fail to load in browser: check console for fetch errors (CORS / path) and ensure they are committed on `master`.

### Large Dataset / Model Too Big? Use Colab Distillation
If the upstream dataset (e.g. Open-Orca) or a teacher model is too large for local/GitHub:

1. Open Google Colab and select a GPU runtime (A100 preferable, T4 acceptable).
2. Clone this repo (without committing huge datasets):
	```
	!git clone https://github.com/<you>/<repo>.git
	%cd <repo>
	```
3. Install dependencies for a *teacher* model:
	```
	!pip install transformers accelerate bitsandbytes safetensors sentencepiece
	```
4. Download / stream only the portion of Open-Orca you need inside Colab using `datasets`:
	```python
	from datasets import load_dataset
	ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
	# Collect a manageable sample
	sample = []
	for i, ex in enumerate(ds):
		 if i>=5000: break
		 sample.append(ex)
	# Convert to a temporary jsonl if needed
	import json
	with open("training/datasets/processed/unified_train.jsonl","w",encoding="utf-8") as f:
		 for ex in sample:
			  user=ex.get('question','').strip(); assistant=ex.get('response','').strip()
			  if user and assistant:
					formatted=f"<|user|>\n{user}\n<|assistant|>\n{assistant}"
					f.write(json.dumps({'source':'orca_sample','user':user,'assistant':assistant,'formatted':formatted},ensure_ascii=False)+"\n")
	```
5. Distill with a teacher (e.g. Qwen1.5 0.5B) to produce a compact synthetic dataset:
	```
	!python training/distill_teacher.py \
		 --teacher-model Qwen/Qwen1.5-0.5B \
		 --unified-file training/datasets/processed/unified_train.jsonl \
		 --out-file training/datasets/processed/distilled_pairs.jsonl \
		 --sample 3000 --max-new-tokens 160
	```
6. (Optional) Replace unified file with distilled pairs or merge them:
	```
	!mv training/datasets/processed/distilled_pairs.jsonl training/datasets/processed/unified_train.jsonl
	```
7. Run local mini pipeline (still in Colab):
	```
	!python training/build_corpus.py --max-chars 2000000
	!python training/train_mini_transformer.py --steps 1500 --context 256
	!python training/export_weights.py --out-dir weights
	```
8. Download the small weights bundle:
	```
	from google.colab import files
	files.download('weights/weights.bin')
	files.download('weights/manifest.json')
	```
9. Add the small weight files to your repo and push (they should remain <<100MB total). Ensure `MODE: 'gpt'`.

This approach keeps gigantic source datasets & teacher weights off GitHub while delivering a compact student model for the browser.


## Privacy Notes
- All chat data keys: `TheMindfulAI_*` in `localStorage`.
- To erase: open DevTools > Application > Local Storage > Clear, or run:
```
localStorage.clear()
```

## Roadmap Ideas
- Add service worker for offline support.
- Add export/import of chats (JSON file) fully local.
- Sentiment timeline visualization.
- Safeguard phrase detection (e.g., escalate to show crisis resources) – be careful & responsible.

## License
Specify your intended license (e.g., MIT) here.

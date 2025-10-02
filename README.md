# The MindfulAI (Local, Privacy-First Prototype)

This project is a fully static, privacy-first mental wellness companion. No external model calls, no backend database. All logic runs in-browser and data stays in `localStorage` on the user's device.

## IMPORTANT DISCLAIMER
This app is NOT a substitute for professional diagnosis, therapy, or crisis intervention. If you (or someone you know) are in immediate danger or considering self‑harm, contact local emergency services or a crisis hotline immediately.

## Features
- 100% static hosting (GitHub Pages compatible)
- Local heuristic "model" (lexicon + lightweight retrieval + templates + incremental vocabulary learning)
- Mood detection (positive, negative, stress, neutral)
- Chat history persisted locally (never uploaded)
- Extensible datasets under `data/`

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
- Add new intents: append objects in `data/responses.json` with `intent`, `patterns`, `templates`.
- Add emotional words: extend categories in `data/lexicon.json` (keep weights ~0.8–1.6).
- Bump `MODEL_VERSION` in `config.js` if you later add caching logic.

## Deployment (GitHub Pages)
1. Create a GitHub repo and push these files to the `main` branch.
2. In repo Settings > Pages: choose `Deploy from branch`, branch `main`, folder `/ (root)`.
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

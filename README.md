# AuthenticWrite

AI-generated text detection system. Paste or upload any document and AuthenticWrite tells you how likely it was written by AI — with a confidence score, writing-quality breakdown, and a downloadable PDF report.

Built as a final-year project at **Federal University Oye-Ekiti**, Department of Computer Science.

---

## What It Does

- **AI Detection** — Uses the open-source RoBERTa-based `Hello-SimpleAI/chatgpt-detector-roberta` transformer, trained on the HC3 human-vs-ChatGPT dataset
- **File Upload** — PDF, DOCX, TXT, CSV, XLSX, code files, and more
- **Batch Analysis** — Upload up to 20 files at once
- **Writing Quality** — Readability (Flesch-Kincaid, Gunning Fog), vocabulary richness, style analysis
- **Fact Checking** — Extracts and verifies factual claims
- **PDF Reports** — Download a detailed analysis report
- **Browser Extension** — Detect AI content on any webpage directly from Chrome

---

## Project Structure

```
AuthenticWrite/
├── backend/                  # Flask API + transformer detector
│   ├── app.py                # API server (all endpoints)
│   ├── requirements.txt      # Python dependencies
│   └── uploads/              # Runtime upload dir (git-ignored)
├── frontend/authenticwrite/  # React + TypeScript web app (CRA + Tailwind)
│   ├── src/                  # App source
│   ├── .env                  # Dev server port (PORT=7000)
│   └── package.json
├── browser-extension/        # Chrome extension (Manifest V3)
└── docs/                     # Design specs
```

---

## Prerequisites

Install these **before** starting:

| Tool       | Minimum version | Notes                                                   |
|------------|-----------------|---------------------------------------------------------|
| Python     | 3.10+           | 3.13 tested                                             |
| Node.js    | 18+             | Bundled npm                                             |
| Git        | any             |                                                         |
| Disk space | ~2 GB free      | PyTorch CPU wheel + RoBERTa model + node_modules        |
| RAM        | 4 GB+           | Detector needs ~1.5 GB resident                         |
| Internet   | on first run    | To download ~500 MB model from Hugging Face            |
| Chrome     | optional        | Required only for the browser extension                 |

---

## Deployment Guide

### 1. Clone the repo

```bash
git clone https://github.com/lakunzy7/AuthenticWrite.git
cd AuthenticWrite
```

### 2. Set up the backend (Flask + transformer)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate            # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt     # Installs PyTorch CPU + transformers
```

First install takes **5–10 minutes** (large wheels). On slow networks use:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

### 3. Download NLTK data (one-time)

The backend uses NLTK for sentence tokenization. Download its data once:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 4. Start the backend

```bash
python app.py
```

Expected output:

```
Loading detector model: Hello-SimpleAI/chatgpt-detector-roberta ...
Detector model ready.
* Running on http://127.0.0.1:5000
```

**First run only:** the model (~500 MB) downloads from Hugging Face into `~/.cache/huggingface/`. Subsequent starts take ~5 seconds.

Keep this terminal open and move to the frontend.

### 5. Set up the frontend (React)

In a new terminal:

```bash
cd frontend/authenticwrite
npm install
npm start
```

`npm install` takes 2–5 minutes the first time. The app then opens at **http://localhost:7000** (port set in `frontend/authenticwrite/.env`).

### 6. (Optional) Install the browser extension

1. Copy the `browser-extension/` folder somewhere permanent (e.g. your Desktop).
2. Open Chrome → `chrome://extensions/`
3. Turn on **Developer mode** (top-right toggle).
4. Click **Load unpacked** → select the copied folder.
5. Pin the AuthentiWrite icon via the 🧩 menu.
6. On any webpage, **select text** (50+ characters), click the extension icon, and hit **Analyze Selected Text**. The backend must be running.

> **WSL note:** Load the extension from the *Windows* filesystem, not from `/home/...` — Chrome on Windows cannot reliably load extensions from inside WSL. Copy the folder over once:
> ```bash
> cp -r browser-extension /mnt/c/Users/<your-windows-user>/Desktop/AuthentiWrite-Extension
> ```

---

## Daily Startup (After Initial Setup)

The easiest way is the bundled script, which starts both services in the background, waits until they're healthy, and prints the URLs:

```bash
./start.sh
```

Stop them again with:

```bash
./stop.sh
```

Logs go to `logs/backend.log` and `logs/frontend.log`. Tail them in real time:

```bash
tail -f logs/backend.log logs/frontend.log
```

### Manual two-terminal alternative

If you'd rather see the output live in your terminals:

**Terminal 1 — backend:**
```bash
cd backend
source venv/bin/activate    # Windows: venv\Scripts\activate
python app.py
```

**Terminal 2 — frontend:**
```bash
cd frontend/authenticwrite
npm start
```

Stop each with `Ctrl+C`.

If port 7000 or 5000 is stuck on an old process:
```bash
./stop.sh    # cleans both ports
```

---

## Verifying It Works

Once both services are running:

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Your long text goes here, at least 50 characters."}'
```

You should get JSON containing `ai_probability`, `classification`, and `confidence`.

---

## Using the App

### Web app

1. Open **http://localhost:7000**
2. Paste text or upload a file (PDF, DOCX, TXT, etc.)
3. Click **Analyze**
4. Review:
   - AI probability percentage (0–100)
   - Classification: Likely AI-Generated / Possibly AI-Generated / Uncertain / Possibly Human-Written / Likely Human-Written
   - Confidence level
   - Optional writing-quality and fact-check sections
   - Download a PDF report

### Classification thresholds

| AI probability | Classification              | Confidence |
|----------------|-----------------------------|------------|
| ≥ 80%          | Likely AI-Generated         | High       |
| 60–80%         | Possibly AI-Generated       | Medium     |
| 40–60%         | Uncertain                   | Low        |
| 20–40%         | Possibly Human-Written      | Medium     |
| < 20%          | Likely Human-Written        | High       |

---

## API Reference

All endpoints run on `http://localhost:5000`.

| Method | Endpoint            | Description                                             |
|--------|---------------------|---------------------------------------------------------|
| POST   | `/analyze`          | Analyze raw text — returns AI probability + metadata    |
| POST   | `/upload`           | Upload a file and extract its text                      |
| GET    | `/sample-text`      | Get a sample AI or human text (`?type=ai` / `?type=human`) |
| POST   | `/fact-check`       | Extract and verify factual claims                       |
| POST   | `/quality-analysis` | Readability + style metrics                             |
| POST   | `/batch-analyze`    | Analyze multiple files at once                          |
| POST   | `/generate-report`  | Produce a downloadable PDF report                       |

### `/analyze` request

```json
{ "text": "Your text to analyze (minimum 50 characters)" }
```

### `/analyze` response

```json
{
  "ai_probability": 98.72,
  "classification": "Likely AI-Generated",
  "confidence": "High",
  "text_length": 987,
  "word_count": 152,
  "sentence_count": 8,
  "patterns": ["Overly formal language structure"],
  "ml_model_used": true,
  "model": "chatgpt-detector-roberta",
  "features": {}
}
```

---

## Model Details

- **Model:** [`Hello-SimpleAI/chatgpt-detector-roberta`](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta)
- **Architecture:** RoBERTa-base fine-tuned for binary classification (Human vs ChatGPT)
- **Training data:** HC3 dataset (Human ChatGPT Comparison Corpus)
- **Token limit:** 512 tokens per window; longer inputs are chunked with 64-token overlap and results averaged
- **Runs on:** CPU (GPU optional and much faster if available)
- **Inference time:** ~0.5–2 s per request on modern CPU

Short inputs (< ~100 words) tend to score low even for AI text — the model was trained on full-length responses. For best accuracy, submit at least one paragraph.

---

## Production Deployment

The built-in Flask dev server is **not** for production. For real deployment:

1. **Use a WSGI server** (Gunicorn recommended):
   ```bash
   pip install gunicorn
   gunicorn -w 2 -b 0.0.0.0:5000 --timeout 60 app:app
   ```
   Use only 1–2 workers — each one loads its own copy of the model (~1.5 GB RAM each).

2. **Front with a reverse proxy** (nginx/Caddy) for HTTPS and static serving.

3. **Build the frontend** for production:
   ```bash
   cd frontend/authenticwrite
   npm run build
   ```
   Serve the `build/` directory via nginx, or copy it into the Flask app and serve with `send_from_directory`.

4. **Lock down CORS** in `backend/app.py` — replace `CORS(app)` with a specific origin list once you know your deployment domain.

5. **Set `FLASK_DEBUG=0`** (don't run Flask's reloader in prod).

---

## Troubleshooting

| Symptom                                                             | Fix                                                                                  |
|---------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `OSError: No such file or directory: '.../nltk_data/tokenizers/punkt/PY3_tab'` | `pip install --upgrade 'nltk>=3.9'`                                                 |
| Model download hangs on first run                                   | Check internet; optionally set `HF_TOKEN` env var for higher HF rate limits          |
| `Something is already running on port 7000`                         | Kill the process: `lsof -ti:7000 \| xargs kill`, or change `PORT` in `frontend/authenticwrite/.env` |
| Extension says "Could not connect to AuthentiWrite API"             | Make sure backend is running on port 5000 and CORS isn't blocking the extension origin |
| Backend kills itself right after "Loading detector"                 | Out of RAM. Close other apps or use a machine with 4 GB+ free RAM                    |
| Chrome can't load extension from `/home/...` (WSL)                  | Copy `browser-extension/` to a Windows path first (e.g. `C:\Users\you\Desktop\`)     |

---

## License

MIT

# AuthenticWrite

An AI-generated text detection system built for academic integrity. Paste or upload any document and AuthenticWrite tells you how likely it was written by AI — with a confidence score, stylometric breakdown, and a downloadable PDF report.

Built as a final-year project at **Federal University Oye-Ekiti**, Department of Computer Science.

---

## What It Does

- **AI Detection** — Analyzes text using 28 stylometric features + TF-IDF with an ensemble ML model (Random Forest + Gradient Boosting + Logistic Regression)
- **File Upload** — Supports PDF, DOCX, TXT, CSV, XLSX, code files, and more
- **Batch Analysis** — Upload up to 20 files at once
- **Writing Quality** — Readability scores (Flesch-Kincaid, Gunning Fog, etc.), vocabulary richness, and style analysis
- **Fact Checking** — Extracts and verifies factual claims in text
- **PDF Reports** — Download a detailed analysis report
- **Browser Extension** — Detect AI-generated content on any webpage directly from Chrome

---

## Project Structure

```
AuthenticWrite/
├── backend/                  # Flask API + ML model
│   ├── app.py                # API server (all endpoints)
│   ├── features.py           # 28 stylometric feature extraction
│   ├── train_model.py        # Model training script
│   ├── training_data.py      # Training dataset generator
│   ├── training_dataset.csv  # Training data
│   └── requirements.txt      # Python dependencies
├── frontend/
│   └── authenticwrite/       # React + TypeScript app
│       ├── src/
│       │   ├── App.tsx       # Main app component
│       │   └── index.tsx     # Entry point
│       ├── package.json
│       └── tailwind.config.js
└── browser-extension/        # Chrome extension (Manifest V3)
    ├── manifest.json
    ├── popup.html / popup.js # Extension popup UI
    ├── content.js            # Content script (runs on webpages)
    ├── background.js         # Service worker
    └── styles.css
```

---

## How to Set Up (Step by Step)

### What You Need First

Before you start, make sure you have these installed on your computer:

| Tool | Version | How to check | Where to get it |
|------|---------|-------------|-----------------|
| **Python** | 3.9 or higher | `python --version` | [python.org/downloads](https://www.python.org/downloads/) |
| **Node.js** | 16 or higher | `node --version` | [nodejs.org](https://nodejs.org/) |
| **npm** | comes with Node.js | `npm --version` | Installed with Node.js |
| **Git** | any recent version | `git --version` | [git-scm.com](https://git-scm.com/) |
| **Google Chrome** | any recent version | — | For the browser extension |

> **Not sure if you have these?** Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run the "How to check" commands above. If you see a version number, you're good. If you see an error, you need to install that tool.

---

### Step 1: Get the Code

Open your terminal and run:

```bash
git clone https://github.com/lakunzy7/AuthenticWrite.git
cd AuthenticWrite
```

---

### Step 2: Set Up the Backend (Python/Flask)

#### 2a. Create a virtual environment

This keeps the project's packages separate from your system Python. Think of it as a sandbox.

**On Windows:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

> You'll know it worked when you see `(venv)` at the start of your terminal line.

#### 2b. Install Python packages

```bash
pip install -r requirements.txt
```

This downloads all the libraries the backend needs (Flask, scikit-learn, NLTK, etc.). It may take a minute or two.

#### 2c. Train the ML model

```bash
python train_model.py
```

This reads the training data, builds the AI detection model, and saves three files: `model.pkl`, `vectorizer.pkl`, and `scaler.pkl`. You only need to do this once.

> **What if it shows errors?** Make sure your virtual environment is activated (you see `(venv)` in your terminal). If NLTK data download fails, check your internet connection.

#### 2d. Start the backend server

```bash
python app.py
```

You should see output like:
```
 * Running on http://0.0.0.0:5000
```

**Leave this terminal open** — the backend needs to stay running. Open a new terminal for the next steps.

---

### Step 3: Set Up the Frontend (React)

Open a **new terminal window** and run:

```bash
cd frontend/authenticwrite
npm install
```

This downloads all the JavaScript packages. It may take a few minutes the first time.

Then start the development server:

```bash
npm start
```

Your browser should automatically open `http://localhost:3000` with the app running.

> **If port 3000 is busy**, the terminal will ask if you want to use a different port. Type `Y` and press Enter.

---

### Step 4: Set Up the Browser Extension (Optional)

This lets you analyze text on any webpage without leaving the page.

1. Open Chrome and go to `chrome://extensions/`
2. Turn on **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Navigate to the `browser-extension` folder inside this project and select it
5. The AuthenticWrite icon should appear in your browser toolbar

> **Important:** The extension talks to the backend at `http://localhost:5000`, so the backend must be running for it to work.

---

## Using the App

### Web App (Frontend)

1. Open `http://localhost:3000` in your browser
2. Paste text into the text box, or upload a file (PDF, DOCX, TXT, etc.)
3. Click **Analyze** — you'll see:
   - AI probability percentage
   - Classification (Likely AI-Generated, Possibly AI-Generated, Possibly Human-Written, Likely Human-Written)
   - Confidence level
   - Stylometric feature breakdown
   - Detected patterns
4. Use the **Quality Analysis** tab for readability scores and writing style metrics
5. Use the **Fact Check** tab to verify claims in the text
6. Click **Download Report** to get a PDF of the analysis

### Browser Extension

1. Highlight text on any webpage
2. Click the AuthenticWrite icon in your toolbar
3. The extension will analyze the selected text and show results in a popup

### API (For Developers)

The backend exposes a REST API. Key endpoints:

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/analyze` | POST | Analyze text — send `{"text": "..."}` |
| `/upload` | POST | Upload a file for text extraction |
| `/quality-analysis` | POST | Get readability and style scores |
| `/batch-analyze` | POST | Analyze up to 20 files at once |
| `/fact-check` | POST | Verify factual claims in text |
| `/generate-report` | POST | Generate a PDF analysis report |
| `/health` | GET | Check if the API is running |
| `/api/docs` | GET | Full API documentation |

**Example with curl:**

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze goes here..."}'
```

---

## How the AI Detection Works

1. **Feature Extraction** — The system extracts 28 stylometric features from the text:
   - Sentence structure (length, variation, uniformity)
   - Vocabulary richness (diversity, Yule's K, hapax ratio)
   - Writing style (personal pronouns, contractions, formal phrases)
   - Readability (Flesch Reading Ease, Gunning Fog Index)
   - Entropy measures (Shannon entropy, bigram entropy)
   - Pattern detection (passive voice, transition word density)

2. **TF-IDF Vectorization** — Converts the text into a numerical representation based on word importance

3. **Ensemble Model** — Three models vote on the result:
   - Random Forest
   - Gradient Boosting
   - Logistic Regression

4. **Classification** — The combined prediction gives an AI probability score from 0–100%

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` command not found | Try `python3` instead (common on Mac/Linux) |
| `pip` command not found | Try `pip3` or `python -m pip` |
| "Module not found" errors | Make sure your virtual environment is activated |
| Backend won't start | Check that port 5000 is free. Try `python app.py` again |
| Frontend won't start | Delete `node_modules` folder and run `npm install` again |
| Model files missing | Run `python train_model.py` in the backend folder |
| NLTK download fails | Check your internet connection and retry |
| Extension not working | Make sure the backend is running on port 5000 |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3, Flask 3.0, scikit-learn, NLTK, textstat |
| Frontend | React 19, TypeScript, Tailwind CSS |
| ML Model | Random Forest + Gradient Boosting + Logistic Regression ensemble |
| Document Parsing | PyPDF2, python-docx, pandas, openpyxl |
| Browser Extension | Chrome Manifest V3, vanilla JS |
| Reports | ReportLab (PDF generation) |

---

## Author

**Owofola Olakunle** (FTP/CSC/25/0121610)
Department of Computer Science, Federal University Oye-Ekiti

---

## License

This project was built for academic purposes. Feel free to use it as a reference or starting point for your own work.

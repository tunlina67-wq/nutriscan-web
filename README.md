---
title: NutriScan AI
emoji: 🥗
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🥗 NutriScan AI

**Snap a label. Understand what you eat.**
A web application that scans a nutrition label photo, reads it with AI, and gives a personalized A–E health grade with advice.

---

## 1. Quick start (10 minutes)

You need **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/); on Windows tick *"Add Python to PATH"*).

```bash
# 1. open a terminal in this folder, then create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. install libraries (EasyOCR + YOLO download PyTorch, about 1 GB, first time only)
pip install -r requirements.txt

# 3. run
python app.py
```

Open **http://localhost:5000** and click **"Try the demo label"**.

> The first scan takes longer (~30 s) because EasyOCR downloads its Thai + English models once.

### Open it on your phone
Connect your phone and laptop to the **same Wi-Fi**, find your laptop's IP address
(`ipconfig` on Windows, `ifconfig` on macOS), then open `http://<laptop-ip>:5000` on the phone.
**Scan Label** opens the phone camera.

---

## 2. How it works (matches the proposal slides)

```
Photo ─► Preprocess ─► YOLOv8 crop ─► EasyOCR ─► Extract nutrients ─► Expert rules + ML ─► LLM advice
         (OpenCV)      (our model)    (Thai+Eng)   (NLP + regex)        (grade A–E)          (personal)
```

| Slide technique | File | What it does |
|---|---|---|
| Preprocess (OpenCV) | `nutriscan/ocr.py` → `preprocess()` | resize, grayscale, CLAHE contrast |
| Detect table (CNN · YOLOv8) | `nutriscan/detector.py` | crops the nutrition table using **our trained `best.pt`** |
| OCR (EasyOCR) | `nutriscan/ocr.py` | reads Thai + English, groups words into lines |
| Information extraction | `nutriscan/parser.py` | keyword + regex parsing, serving size, sanity checks (energy ≈ 4C + 4P + 9F) |
| Expert system | `nutriscan/scoring.py` | personal limits (Mifflin–St Jeor, Thai RDI, WHO) → risk score → grade |
| ML classifier (Random Forest) | `nutriscan/classifier.py`, `train_classifier.py` | predicts Nutri-Score grade, trained on Open Food Facts |
| Large language model | `nutriscan/advice.py` | friendly personalized advice (template fallback without a key) |
| Web UI | `static/` + `app.py` | Scan → Nutrients Found → Your Result, plus profile |

The home screen shows which **AI modules** are active (green = on).

---

## 3. Turn on each AI module

### 🟢 YOLO table detector
Train it with the Colab notebook **`NutriScan_YOLO_Training.ipynb`**, download `best.pt`, and put it in `models/best.pt`. Restart `app.py`.

### 🟢 OCR (Thai + English)
Turns on automatically after `pip install -r requirements.txt`.

### 🟢 ML grade classifier
```bash
python train_classifier.py                 # downloads Open Food Facts (first 400k products), ~5–10 min
python train_classifier.py --rows 100000   # faster, smaller
```
Creates `models/grade_rf.joblib` plus `models/classifier_report.txt` and a confusion matrix image. **Put both in your report.**
If the download is slow, get the CSV from https://world.openfoodfacts.org/data and run
`python train_classifier.py --csv path/to/file.csv`.

### 🟢 LLM advice
Get an API key from https://console.anthropic.com and set it before running:
```bash
set ANTHROPIC_API_KEY=sk-ant-...          # Windows (cmd)
$env:ANTHROPIC_API_KEY="sk-ant-..."       # Windows (PowerShell)
export ANTHROPIC_API_KEY=sk-ant-...       # macOS / Linux
python app.py
```
Without a key the app uses rule-based template advice, so the demo always works.
Never commit your key to GitHub.

---

## 4. How the grade is calculated

1. **Daily needs:** energy from Mifflin–St Jeor × activity (±goal), sugar ≤ 24 g per 2,000 kcal (Thai 6 tsp), sodium ≤ 2,000 mg (1,500 for *Reduce salt*), sat fat ≤ 20 g per 2,000 kcal. Without a profile: 2,000 kcal Thai RDI.
2. **Expert rules:** risk = 0.45·sugar% + 0.25·sodium% + 0.15·sat-fat% + 0.15·energy% (each capped at 150), −2/−4 for protein. A < 8 ≤ B < 18 ≤ C < 40 ≤ D < 65 ≤ E. Any nutrient ≥ 100% → at least D.
3. **ML model:** Random Forest predicts a grade from per-100 g/ml values.
4. **Final grade:** average of rules and ML, rounded to the more cautious side.

Example, Royal-D Grape 330 ml: sugar 16 g = 67% → risk 34.2 → **grade C**.

---

## 5. Project structure

```
nutriscan-web/
├── app.py                  Flask server + API (/api/scan, /api/analyze, /api/status)
├── train_classifier.py     trains the Random Forest grade model
├── requirements.txt
├── nutriscan/
│   ├── detector.py         YOLOv8 table detection + crop
│   ├── ocr.py              preprocessing + EasyOCR
│   ├── parser.py           nutrient extraction + validation
│   ├── scoring.py          expert system + personal daily needs
│   ├── classifier.py       ML grade prediction
│   └── advice.py           LLM / template advice
├── models/                 put best.pt and grade_rf.joblib here
└── static/                 web UI (index.html, style.css, app.js, sample-label.jpg)
```

### Suggested team split
- **Image & OCR:** `detector.py`, `ocr.py`, YOLO notebook
- **Data & Scoring:** `parser.py`, `scoring.py`
- **AI & Model:** `classifier.py`, `train_classifier.py`, `advice.py`
- **UI & Design:** `static/`

---

## 6. Put it online (GitHub → Hugging Face Spaces, free)

The app runs on a free Hugging Face Space (2 vCPU, 16 GB RAM). Every push to GitHub's `main`
branch uploads it to the Space through `.github/workflows/sync-to-huggingface.yml`, and the Space rebuilds itself.

1. **Hugging Face:** create a free account at https://huggingface.co → **New Space** → SDK **Docker** (blank template), hardware **CPU basic (free)**.
2. **Token:** Hugging Face → Settings → **Access Tokens** → create a token with **Write** permission.
3. **GitHub:** create a repository and push this folder (models included, `venv/` is ignored).
4. In the GitHub repo → **Settings → Secrets and variables → Actions**:
   - **Secrets** tab → `HF_TOKEN` = the token from step 2
   - **Variables** tab → `HF_SPACE` = `your-hf-username/your-space-name`
5. GitHub → **Actions** tab → *Sync to Hugging Face Space* → **Run workflow** (later pushes run it automatically).
6. Watch the build in the Space's **Logs** tab (first build ~10 min). The app appears at
   `https://huggingface.co/spaces/your-hf-username/your-space-name`.
7. Optional, for LLM advice: Space → **Settings → Variables and secrets** → add secret `ANTHROPIC_API_KEY`.

A free Space sleeps after ~48 h without visitors; the next visit wakes it (a few minutes).

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Reinstall Python and tick "Add to PATH", or try `py` / `python3` |
| OCR chip stays grey | `pip install easyocr`, then restart `app.py` |
| "Scan failed" / wrong numbers | Values are editable: fix them and press Analyze. A trained `best.pt` improves accuracy a lot |
| Phone can't open the page | Same Wi-Fi? Allow Python through the firewall. Use the laptop IP, not `localhost` |
| Port 5000 busy (macOS AirPlay) | `PORT=5050 python app.py` |

*NutriScan gives general nutrition information, not medical advice.*

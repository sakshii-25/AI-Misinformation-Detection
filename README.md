# Debunkr — AI Misinformation Detector

A web dashboard that detects misinformation using:
- **XGBoost ML model** (trained on your dissertation data — 95% accuracy)
- **Google Fact Check Tools API**
- **Snopes** cross-verification
- **FactCheck.org** cross-verification

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data file
Put your `data.csv` in the project root (same folder as `app.py`).

### 3. Set your Google API key
```bash
export GOOGLE_API_KEY=your_key_here
```

### 4. Run the app
```bash
python app.py
```

Visit: **http://localhost:5000**

## First Run
On first run, the app will **automatically train and save** the ML model from your `data.csv`.
This takes a few minutes. Subsequent runs load from disk instantly.

## Project Structure
```
debunkr/
├── app.py                      # Flask backend + API routes
├── requirements.txt
├── Procfile                    # For Railway deployment
├── data.csv                    # Your training data (add this!)
├── model/
│   ├── predict.py              # ML prediction pipeline
│   └── *.pkl                   # Saved models (auto-generated)
├── fact_checkers/
│   ├── google_api.py           # Google Fact Check API
│   ├── snopes_scraper.py       # Snopes.com scraper
│   ├── factcheck_org.py        # FactCheck.org scraper
│   └── url_fetcher.py          # Article URL text extractor
└── templates/
    └── index.html              # Dashboard frontend
```

## Verdict Logic
All 4 sources vote on the verdict:
- ML model gets **2× weight**
- Google, Snopes, FactCheck.org each get **1× weight**
- Majority verdict wins: **REAL**, **FAKE**, or **UNCERTAIN**

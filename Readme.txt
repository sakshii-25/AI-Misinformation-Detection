# Debunkr — AI Misinformation Detector 🔍

> Evaluating the Usefulness of AI/ML in Misinformation Detection and Mitigation in Social Media

## 🌐 Live Demo
**[https://debunkr-0ovp.onrender.com](https://debunkr-0ovp.onrender.com)**

---

## 📌 Overview
Debunkr is a machine learning-powered web application that classifies real and fake news using textual data from tweets. It combines multiple feature extraction techniques and ML algorithms with the Google Fact-Check API to cross-verify news authenticity and enhance prediction accuracy.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python, Flask, SQLAlchemy |
| ML / NLP | scikit-learn, XGBoost, Gensim (Word2Vec), NLTK |
| Feature Extraction | TF-IDF, LSA (Truncated SVD), Word2Vec |
| External API | Google Fact-Check API |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Deployment | Render (Free Tier) |

---

## 🧠 Machine Learning Models
The following classifiers were trained and evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Decision Tree
- XGBoost

Models were compared using **accuracy, precision, recall, and F1-score**, with bar graphs and ROC curves generated for visual comparison.

---

## 🔬 Feature Extraction Pipeline
1. **TF-IDF** — Term frequency-inverse document frequency vectorisation
2. **LSA** — Latent Semantic Analysis via Truncated SVD
3. **Word2Vec** — Word embeddings trained using Gensim

---

## 📁 Project Structure
```
AI-Misinformation-Detection/
├── debunkr/                  # Web application
│   ├── app.py                # Flask app entry point
│   ├── templates/            # HTML templates
│   ├── model/                # Trained ML models
│   ├── fact_checkers/        # Google Fact-Check API integration
│   ├── requirements.txt      # Python dependencies
│   └── Procfile              # Deployment config
├── Sakshi's_Final_dissertation.ipynb   # Full dissertation notebook
├── sakshi's_final_dissertation.py      # Python script version
└── data.csv                  # Tweet dataset
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- pip

### Steps
```bash
# Clone the repository
git clone https://github.com/sakshii-25/AI-Misinformation-Detection.git

# Navigate to the web app
cd AI-Misinformation-Detection/debunkr

# Install dependencies
pip install -r requirements.txt

# Add your environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Run the app
flask run
```

Then visit `http://localhost:5000` in your browser.

---

## 🔑 Environment Variables
Create a `.env` file in the `debunkr/` folder with:
```
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 📊 Running the Dissertation Notebook

### On Google Colab
```python
from google.colab import files
uploaded = files.upload()  # Upload data.csv

import pandas as pd
df = pd.read_csv('data.csv')
```

### Locally
```python
import pandas as pd
df = pd.read_csv('data.csv')
```

---

## 📈 Evaluation Metrics
Models are evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC Curves**

---

## 👩‍💻 Author
**Sakshi Chauhan**  
Final Year Dissertation Project  
[GitHub](https://github.com/sakshii-25)

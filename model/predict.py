"""
predict.py — wraps your trained XGBoost pipeline for inference.

First run trains and saves the model objects to disk.
Subsequent runs load from disk for fast prediction.
"""

import os
import re
import pickle
import numpy as np

# ── NLTK lazy-download ──────────────────────────────────────────
import nltk
for pkg in ('punkt', 'stopwords', 'wordnet', 'punkt_tab'):
    try:
        nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

MODEL_DIR  = os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.pkl')
TFIDF_PATH = os.path.join(MODEL_DIR, 'tfidf.pkl')
LSA_PATH   = os.path.join(MODEL_DIR, 'lsa.pkl')
W2V_PATH   = os.path.join(MODEL_DIR, 'w2v_model.pkl')

_model    = None
_tfidf    = None
_lsa      = None
_w2v      = None

# ── Preprocessing (mirrors your notebook exactly) ───────────────
def preprocess_text(text: str) -> list[str]:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOP_WORDS]
    return tokens

def get_avg_w2v(tokens, model, vector_size=100):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# ── Model loading / training ────────────────────────────────────
def _load_or_train():
    global _model, _tfidf, _lsa, _w2v

    if all(os.path.exists(p) for p in [MODEL_PATH, TFIDF_PATH, LSA_PATH, W2V_PATH]):
        with open(MODEL_PATH, 'rb') as f:  _model = pickle.load(f)
        with open(TFIDF_PATH, 'rb') as f:  _tfidf = pickle.load(f)
        with open(LSA_PATH,   'rb') as f:  _lsa   = pickle.load(f)
        with open(W2V_PATH,   'rb') as f:  _w2v   = pickle.load(f)
        return

    # ── Train from scratch ──
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from gensim.models import Word2Vec
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data.csv')
    df = pd.read_csv(DATA_PATH).dropna()

    df['clean_tweet'] = df['tweet'].apply(lambda t: ' '.join(preprocess_text(str(t))))

    # TF-IDF
    _tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tfidf = _tfidf.fit_transform(df['clean_tweet'])

    # LSA
    _lsa = TruncatedSVD(n_components=300, random_state=42)
    X_lsa = _lsa.fit_transform(X_tfidf)

    # Word2Vec
    tokenized = [word_tokenize(t) for t in df['clean_tweet']]
    _w2v = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2)
    X_w2v = np.array([get_avg_w2v(t, _w2v) for t in tokenized])

    X = np.hstack([X_lsa, X_w2v])
    y = df['BinaryNumTarget']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    _model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    _model.fit(X_train, y_train)

    # Save
    with open(MODEL_PATH, 'wb') as f: pickle.dump(_model, f)
    with open(TFIDF_PATH, 'wb') as f: pickle.dump(_tfidf, f)
    with open(LSA_PATH,   'wb') as f: pickle.dump(_lsa,   f)
    with open(W2V_PATH,   'wb') as f: pickle.dump(_w2v,   f)

# ── Public API ──────────────────────────────────────────────────
def predict_text(text: str) -> tuple[str, float]:
    """Returns (verdict, confidence) where verdict ∈ {'REAL','FAKE','UNCERTAIN'}."""
    if _model is None:
        _load_or_train()

    tokens = preprocess_text(text)
    if not tokens:
        return 'UNCERTAIN', 0.5

    tfidf_vec = _tfidf.transform([' '.join(tokens)])
    lsa_vec   = _lsa.transform(tfidf_vec)
    w2v_vec   = get_avg_w2v(tokens, _w2v).reshape(1, -1)
    combined  = np.hstack([lsa_vec, w2v_vec])

    proba      = _model.predict_proba(combined)[0]
    pred_class = int(_model.predict(combined)[0])
    confidence = float(proba[pred_class])

    if confidence < 0.60:
        return 'UNCERTAIN', confidence
    return ('REAL' if pred_class == 1 else 'FAKE'), confidence

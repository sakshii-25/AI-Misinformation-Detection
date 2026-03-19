from flask import Flask, render_template, request, jsonify
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///checks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── Database Model ──────────────────────────────────────────────
class Check(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    input_type   = db.Column(db.String(20))   # 'text', 'url', 'file'
    content      = db.Column(db.Text)
    verdict      = db.Column(db.String(20))    # 'REAL' | 'FAKE' | 'UNCERTAIN'
    confidence   = db.Column(db.Float)
    ml_verdict   = db.Column(db.String(20))
    google_result= db.Column(db.Text)
    snopes_result= db.Column(db.Text)
    factcheck_result = db.Column(db.Text)
    timestamp    = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ── Routes ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/check', methods=['POST'])
def check():
    from model.predict import predict_text
    from fact_checkers.google_api import check_google
    from fact_checkers.snopes_scraper import check_snopes
    from fact_checkers.factcheck_org import check_factcheck_org

    data = request.get_json()
    input_type = data.get('type', 'text')   # 'text' | 'url'
    content    = data.get('content', '').strip()

    if not content:
        return jsonify({'error': 'No content provided'}), 400

    # If URL input, fetch article text first
    if input_type == 'url':
        from fact_checkers.url_fetcher import fetch_url_text
        content = fetch_url_text(content)
        if not content:
            return jsonify({'error': 'Could not fetch article from URL'}), 400

    # ── Layer 1: ML Model ──
    ml_verdict, confidence = predict_text(content)

    # ── Layer 2 & 3: Fact-Check APIs ──
    google_result   = check_google(content)
    snopes_result   = check_snopes(content)
    factcheck_result = check_factcheck_org(content)

    # ── Final Verdict: Smart override logic ──
    # Priority:
    # 1. If 2+ external sources agree → overrides ML model
    # 2. If 1 external source clear + ML agrees → that verdict wins
    # 3. If 1 external source clear + ML is uncertain → external wins
    # 4. No external data or conflict → trust ML model

    external_verdicts = []
    for result in [google_result, snopes_result, factcheck_result]:
        if result:
            v = result.get("verdict", "UNCERTAIN")
            if v in ("REAL", "FAKE"):
                external_verdicts.append(v)

    external_real = external_verdicts.count("REAL")
    external_fake = external_verdicts.count("FAKE")

    if external_real >= 2:
        final_verdict = "REAL"
    elif external_fake >= 2:
        final_verdict = "FAKE"
    elif external_real == 1 and external_fake == 0 and ml_verdict != "FAKE":
        final_verdict = "REAL"
    elif external_fake == 1 and external_real == 0 and ml_verdict != "REAL":
        final_verdict = "FAKE"
    elif external_real == 1 and ml_verdict == "REAL":
        final_verdict = "REAL"
    elif external_fake == 1 and ml_verdict == "FAKE":
        final_verdict = "FAKE"
    else:
        final_verdict = ml_verdict

    # ── Save to DB ──
    check_record = Check(
        input_type=input_type,
        content=content[:1000],
        verdict=final_verdict,
        confidence=confidence,
        ml_verdict=ml_verdict,
        google_result=str(google_result),
        snopes_result=str(snopes_result),
        factcheck_result=str(factcheck_result)
    )
    db.session.add(check_record)
    db.session.commit()

    return jsonify({
        'id': check_record.id,
        'verdict': final_verdict,
        'confidence': round(confidence * 100, 1),
        'sources': {
            'ml_model': {'verdict': ml_verdict, 'confidence': round(confidence * 100, 1)},
            'google':   google_result,
            'snopes':   snopes_result,
            'factcheck_org': factcheck_result
        },
        'timestamp': check_record.timestamp.isoformat()
    })

@app.route('/api/history')
def history():
    page  = request.args.get('page', 1, type=int)
    items = Check.query.order_by(Check.timestamp.desc()).paginate(page=page, per_page=10)
    return jsonify({
        'checks': [{
            'id': c.id,
            'input_type': c.input_type,
            'content': c.content[:120] + '…' if len(c.content) > 120 else c.content,
            'verdict': c.verdict,
            'confidence': round(c.confidence * 100, 1) if c.confidence else None,
            'timestamp': c.timestamp.isoformat()
        } for c in items.items],
        'total': items.total,
        'pages': items.pages,
        'current_page': items.page
    })

@app.route('/api/history/<int:check_id>')
def history_detail(check_id):
    c = Check.query.get_or_404(check_id)
    return jsonify({
        'id': c.id,
        'input_type': c.input_type,
        'content': c.content,
        'verdict': c.verdict,
        'confidence': round(c.confidence * 100, 1) if c.confidence else None,
        'ml_verdict': c.ml_verdict,
        'google_result': c.google_result,
        'snopes_result': c.snopes_result,
        'factcheck_result': c.factcheck_result,
        'timestamp': c.timestamp.isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    content = f.read().decode('utf-8', errors='ignore')
    return jsonify({'content': content[:5000]})

# ── Clear all history ──────────────────────────────────────────
@app.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    try:
        num = db.session.query(Check).delete()
        db.session.commit()
        return jsonify({'success': True, 'deleted': num})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

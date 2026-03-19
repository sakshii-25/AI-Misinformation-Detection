"""
google_api.py — Google Fact Check Tools API integration.
"""

import os
import requests

def check_google(query: str) -> dict | None:
    """
    Query the Google Fact Check Tools API.
    Returns a structured result dict or None if no results found.
    """
    # Read key fresh every call so .env is always picked up
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == 'YOUR_GOOGLE_API_KEY_HERE':
        return {'verdict': 'UNCERTAIN', 'source': 'Google Fact Check', 'detail': 'API key not configured', 'claims': []}

    query_short = query[:200]
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {'key': GOOGLE_API_KEY, 'query': query_short, 'languageCode': 'en'}

    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return None

        data = resp.json()
        claims = data.get('claims', [])
        if not claims:
            return None

        parsed_claims = []
        fake_signals = {'false', 'pants on fire', 'incorrect', 'misleading', 'fake', 'debunked', 'untrue'}
        real_signals = {'true', 'correct', 'accurate', 'verified', 'confirmed'}
        verdict_votes = {'FAKE': 0, 'REAL': 0}

        for claim in claims[:5]:
            for review in claim.get('claimReview', []):
                rating = review.get('textualRating', '').lower()
                publisher = review.get('publisher', {}).get('name', 'Unknown')
                url_link  = review.get('url', '')

                parsed_claims.append({
                    'text': claim.get('text', '')[:200],
                    'rating': review.get('textualRating', ''),
                    'publisher': publisher,
                    'url': url_link
                })

                if any(s in rating for s in fake_signals):
                    verdict_votes['FAKE'] += 1
                elif any(s in rating for s in real_signals):
                    verdict_votes['REAL'] += 1

        if not parsed_claims:
            return None

        if verdict_votes['FAKE'] > verdict_votes['REAL']:
            verdict = 'FAKE'
        elif verdict_votes['REAL'] > verdict_votes['FAKE']:
            verdict = 'REAL'
        else:
            verdict = 'UNCERTAIN'

        return {
            'verdict': verdict,
            'source': 'Google Fact Check',
            'claims': parsed_claims
        }

    except Exception as e:
        return {'verdict': 'UNCERTAIN', 'source': 'Google Fact Check', 'detail': str(e), 'claims': []}

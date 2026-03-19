"""
snopes_scraper.py — Cross-verifies claims by searching Snopes.
Uses Snopes search endpoint and scrapes result ratings.
"""

import requests
from bs4 import BeautifulSoup
import urllib.parse

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

FAKE_RATINGS = {'false', 'mostly false', 'mixture', 'outdated', 'scam', 'legend', 'miscaptioned'}
REAL_RATINGS = {'true', 'mostly true', 'correct attribution'}

def check_snopes(query: str) -> dict | None:
    """
    Search Snopes for the claim and return a structured verdict.
    """
    # Use key phrases from query (first 80 chars)
    search_q = urllib.parse.quote(query[:80])
    search_url = f"https://www.snopes.com/?s={search_q}"

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')

        results = []
        # Snopes search results have article cards
        articles = soup.select('article.media-object, div.article-card, .search-result')
        if not articles:
            articles = soup.select('article')

        for art in articles[:5]:
            title_el  = art.select_one('h2, h3, .title')
            rating_el = art.select_one('.rating-label, .label, [class*="rating"]')
            link_el   = art.select_one('a[href*="snopes.com"]') or art.select_one('a')

            if not title_el:
                continue

            title  = title_el.get_text(strip=True)
            rating = rating_el.get_text(strip=True).lower() if rating_el else ''
            link   = link_el.get('href', '') if link_el else ''

            results.append({'title': title, 'rating': rating, 'url': link})

        if not results:
            return None

        # Vote on verdict
        votes = {'FAKE': 0, 'REAL': 0}
        for r in results:
            rat = r['rating']
            if any(s in rat for s in FAKE_RATINGS):
                votes['FAKE'] += 1
            elif any(s in rat for s in REAL_RATINGS):
                votes['REAL'] += 1

        if votes['FAKE'] > votes['REAL']:
            verdict = 'FAKE'
        elif votes['REAL'] > votes['FAKE']:
            verdict = 'REAL'
        else:
            verdict = 'UNCERTAIN'

        return {
            'verdict': verdict,
            'source': 'Snopes',
            'results': results[:3]
        }

    except Exception as e:
        return {'verdict': 'UNCERTAIN', 'source': 'Snopes', 'detail': str(e), 'results': []}

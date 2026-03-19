"""
factcheck_org.py — Cross-verifies claims against FactCheck.org search results.
"""

import requests
from bs4 import BeautifulSoup
import urllib.parse

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

FAKE_SIGNALS = {'false', 'misleading', 'inaccurate', 'no evidence', 'exaggerated', 'distorted', 'incorrect'}
REAL_SIGNALS = {'true', 'accurate', 'correct', 'confirmed', 'verified'}

def check_factcheck_org(query: str) -> dict | None:
    """
    Search FactCheck.org and return structured results.
    """
    search_q = urllib.parse.quote(query[:80])
    search_url = f"https://www.factcheck.org/?s={search_q}"

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')

        results = []
        articles = soup.select('article, .article-item, .entry, .post')

        for art in articles[:5]:
            title_el   = art.select_one('h2, h3, h4, .entry-title')
            excerpt_el = art.select_one('p, .entry-content, .excerpt')
            link_el    = art.select_one('a[href*="factcheck.org"]') or art.select_one('a')

            if not title_el:
                continue

            title   = title_el.get_text(strip=True)
            excerpt = excerpt_el.get_text(strip=True)[:200] if excerpt_el else ''
            link    = link_el.get('href', '') if link_el else ''

            # Infer verdict from title/excerpt keywords
            combined = (title + ' ' + excerpt).lower()
            if any(s in combined for s in FAKE_SIGNALS):
                inferred = 'FAKE'
            elif any(s in combined for s in REAL_SIGNALS):
                inferred = 'REAL'
            else:
                inferred = 'UNCERTAIN'

            results.append({
                'title': title,
                'excerpt': excerpt,
                'inferred': inferred,
                'url': link
            })

        if not results:
            return None

        votes = {'FAKE': 0, 'REAL': 0, 'UNCERTAIN': 0}
        for r in results:
            votes[r['inferred']] += 1

        verdict = max(votes, key=votes.get)
        if votes[verdict] == 0:
            return None

        return {
            'verdict': verdict,
            'source': 'FactCheck.org',
            'results': results[:3]
        }

    except Exception as e:
        return {'verdict': 'UNCERTAIN', 'source': 'FactCheck.org', 'detail': str(e), 'results': []}

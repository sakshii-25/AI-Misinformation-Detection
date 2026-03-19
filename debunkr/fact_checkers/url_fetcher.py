"""
url_fetcher.py — Fetches and extracts clean text from article URLs.
"""

import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def fetch_url_text(url: str) -> str | None:
    """Fetch a URL and extract the main article text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Remove script, style, nav etc.
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
            tag.decompose()

        # Try to find main article content
        article = soup.find('article') or soup.find(attrs={'class': lambda c: c and 'article' in c.lower()})
        if article:
            text = article.get_text(separator=' ', strip=True)
        else:
            # Fallback: largest <p> block
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text(strip=True) for p in paragraphs)

        # Trim to reasonable size for processing
        return text[:3000] if text else None

    except Exception:
        return None

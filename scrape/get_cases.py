import requests
from tqdm import tqdm
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Oyez case list is loaded by JS; use the public API which returns JSON.
API_BASE = "https://api.oyez.org/cases"
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
})


def format_title(title):
    if not title:
        return ""
    title = title.replace('\n', '')
    while '  ' in title:
        title = title.replace('  ', ' ')
    return title


def get_case_links():
    cases_path = os.path.join(DATA_DIR, 'cases.json')
    try:
        with open(cases_path, 'r') as f:
            data = f.read().strip() or '[]'
            existing = json.loads(data)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []
    if not isinstance(existing, list):
        existing = []

    # Keep only cases outside the range we will fetch from API
    want_terms = {str(y) for y in range(2019, 2026)}
    cases = [c for c in existing if isinstance(c, dict) and str(c.get('year', '')) not in want_terms]

    # Paginate API and collect cases for 2019-2025
    per_page = 100
    page = 0
    seen_ids = set()
    with tqdm(desc='API pages') as pbar:
        while True:
            try:
                r = SESSION.get(API_BASE, params={'per_page': per_page, 'page': page}, timeout=30)
                r.raise_for_status()
            except requests.RequestException as e:
                print(f"API request failed: {e}")
                break
            try:
                batch = r.json()
            except json.JSONDecodeError:
                print("API returned non-JSON")
                break
            if not isinstance(batch, list) or len(batch) == 0:
                break
            for c in batch:
                term = c.get('term')
                if term not in want_terms:
                    continue
                cid = c.get('ID')
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                href = c.get('href') or ''
                # API href: https://api.oyez.org/cases/1966/510 -> www URL
                main_url = href.replace('api.oyez.org', 'www.oyez.org') if href else ''
                if not main_url:
                    continue
                cases.append({
                    'name': format_title(c.get('name') or ''),
                    'main_url': main_url,
                    'year': int(term),
                })
            pbar.update(1)
            if len(batch) < per_page:
                break
            page += 1

    with open(cases_path, 'w') as f:
        json.dump(cases, f)
    print(f"Wrote {len(cases)} cases to {cases_path}")


get_case_links()

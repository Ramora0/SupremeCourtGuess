"""
Fetch oral argument transcripts from the Oyez API (no Selenium).
Uses transcript_url from basic.json (API case_media URLs from get_basic.py).
Output: same format as before — list of {"speaker": "...", "text": "..."} in data/raw_cases/.
"""
import json
import os
import re

import requests
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_CASES_DIR = os.path.join(DATA_DIR, 'raw_cases')

SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
})


def safe_filename_from_url(url: str) -> str:
    """Build a filesystem-safe filename from transcript API URL."""
    s = url.replace('/', '-').strip()
    s = re.sub(r'[<>:"|?*]', '_', s)
    return s or 'unknown'


def api_transcript_to_statements(api_data: dict) -> list[dict]:
    """
    Parse Oyez case_media/oral_argument_audio JSON into list of {speaker, text}.
    Matches the format previously produced by the Selenium HTML scraper.
    """
    statements = []
    transcript = api_data.get('transcript')
    if not transcript or not isinstance(transcript, dict):
        return statements

    sections = transcript.get('sections') or []
    for section in sections:
        if not isinstance(section, dict):
            continue
        for turn in (section.get('turns') or []):
            if not isinstance(turn, dict):
                continue
            speaker_obj = turn.get('speaker')
            if isinstance(speaker_obj, dict):
                speaker = (speaker_obj.get('last_name') or speaker_obj.get('name') or '').strip()
            else:
                speaker = str(speaker_obj or '').strip()

            text_parts = []
            for block in (turn.get('text_blocks') or []):
                if isinstance(block, dict) and block.get('text'):
                    text_parts.append(block['text'].strip())
            text = ' '.join(text_parts).strip()

            if speaker or text:
                statements.append({'speaker': speaker, 'text': text})

    return statements


def get_transcript(transcript_url: str, timeout: int = 30) -> bool:
    """
    Fetch transcript from API and save as JSON in raw_cases/.
    Returns True if saved, False if skipped (exists or no transcript).
    """
    file_name = safe_filename_from_url(transcript_url) + '.json'
    file_path = os.path.join(RAW_CASES_DIR, file_name)

    if os.path.exists(file_path):
        return False

    try:
        r = SESSION.get(transcript_url, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return False

    try:
        data = r.json()
    except json.JSONDecodeError:
        return False

    statements = api_transcript_to_statements(data)
    if not statements:
        return False

    os.makedirs(RAW_CASES_DIR, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(statements, f, ensure_ascii=False)

    return True


def main():
    with open(os.path.join(DATA_DIR, 'basic.json'), 'r') as f:
        raw = json.load(f)
    main_list = raw if isinstance(raw, list) else []

    cases_with_transcripts = [c for c in main_list if isinstance(c, dict) and c.get('transcript_url')]
    cases_without_transcripts = [c for c in main_list if isinstance(c, dict) and not c.get('transcript_url')]

    print(f"Total cases: {len(main_list)}")
    print(f"Cases with transcript URLs: {len(cases_with_transcripts)}")
    print(f"Cases without transcript URLs: {len(cases_without_transcripts)}")

    if not cases_with_transcripts:
        print("No cases with transcript URLs found. Exiting.")
        return

    saved = 0
    skipped = 0
    failed = 0

    for case in tqdm(cases_with_transcripts, desc="Processing transcripts"):
        transcript_url = case.get('transcript_url')
        try:
            if get_transcript(transcript_url):
                saved += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"Error for {transcript_url}: {e}")

    print(f"Saved: {saved}, Skipped (existing or empty): {skipped}, Failed: {failed}")


if __name__ == '__main__':
    main()

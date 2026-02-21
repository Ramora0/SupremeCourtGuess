"""
Fetch oral argument transcripts from the Oyez API (no Selenium).
Uses transcript_url from basic.json (API case_media URLs from get_basic.py).
Output: one JSON per case in data/raw_cases/ with all data needed for case_transcripts_cleaned:
  - statements: list of {"speaker": "...", "text": "..."} where advocates are labeled
    as "Petitioner"/"Respondent" and justices keep their last name.
  - name, votes (with side: Petitioner/Respondent), win_side (1/0/2/-1) from basic.json.
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


def _build_advocate_label_map(case: dict | None) -> dict[str, str]:
    """
    Build a mapping from advocate name (lowercased) to label ('Petitioner' or 'Respondent').
    Uses the advocates list from basic.json which has side info (1=petitioner, 0=respondent).
    """
    label_map = {}
    if not case:
        return label_map
    for adv in (case.get('advocates') or []):
        if not isinstance(adv, dict):
            continue
        name = (adv.get('name') or '').strip()
        side = adv.get('side')
        if not name:
            continue
        if side == 1:
            label_map[name.lower()] = 'Petitioner'
        elif side == 0:
            label_map[name.lower()] = 'Respondent'
        else:
            label_map[name.lower()] = 'Unknown'
    return label_map


def api_transcript_to_statements(api_data: dict, advocate_labels: dict[str, str] | None = None) -> list[dict]:
    """
    Parse Oyez case_media/oral_argument_audio JSON into list of {speaker, text}.
    Advocates are labeled as 'Petitioner' or 'Respondent' using advocate_labels map.
    Justices keep their last name.
    """
    if advocate_labels is None:
        advocate_labels = {}
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
                last_name = (speaker_obj.get('last_name') or '').strip()
                full_name = (speaker_obj.get('name') or '').strip()
                # Check if this speaker is a known advocate
                label = None
                for name_variant in (full_name.lower(), last_name.lower()):
                    if name_variant and name_variant in advocate_labels:
                        label = advocate_labels[name_variant]
                        break
                if label is None and last_name:
                    for adv_name, adv_label in advocate_labels.items():
                        if last_name.lower() in adv_name:
                            label = adv_label
                            break
                # If matched as advocate, use side label; otherwise keep last name
                speaker = label if label else (last_name or full_name)
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


def get_transcript(transcript_url: str, case: dict | None = None, timeout: int = 30) -> bool:
    """
    Fetch transcript from API and save as JSON in raw_cases/.
    If case is provided (from basic.json), saves name, votes, win_side so the file
    has all data needed to generate case_transcripts_cleaned format.
    Advocates are labeled as 'Petitioner'/'Respondent' in statements.
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

    advocate_labels = _build_advocate_label_map(case)
    statements = api_transcript_to_statements(data, advocate_labels)
    if not statements:
        return False

    payload = {
        'statements': statements,
    }
    if isinstance(case, dict):
        payload['name'] = case.get('name') or ''
        payload['votes'] = case.get('votes') or []
        payload['win_side'] = case.get('win_side', -1)

    os.makedirs(RAW_CASES_DIR, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)

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
            if get_transcript(transcript_url, case=case):
                saved += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"Error for {transcript_url}: {e}")

    print(f"Saved: {saved}, Skipped (existing or empty): {skipped}, Failed: {failed}")


if __name__ == '__main__':
    main()

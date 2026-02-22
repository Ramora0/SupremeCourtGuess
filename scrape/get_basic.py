"""
Fetch basic case information from the Oyez API (no Selenium).
Uses the same API approach as get_cases.py so it works in WSL/headless environments.
"""
import json
import os
import random
import re
from html import unescape

import requests
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
})

with open(os.path.join(DATA_DIR, 'cases.json'), 'r') as file:
    cases = json.load(file)


def _api_url_from_main_url(main_url: str) -> str:
    """Convert www.oyez.org case URL to api.oyez.org case URL."""
    return main_url.replace('www.oyez.org', 'api.oyez.org')


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode entities."""
    if not html or not isinstance(html, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return unescape(text)


def get_basic(case: dict, timeout: int = 30) -> dict | None:
    """
    Fetch basic case information from the Oyez API.

    Args:
        case: Dict with 'main_url', 'name', 'year'.
        timeout: Request timeout in seconds.

    Returns:
        Dict with transcript_url, petitioners, facts, question, votes, majority, parties;
        None if case was vacated and remanded or API request failed.
    """
    url = case['main_url']
    api_url = _api_url_from_main_url(url)

    try:
        r = SESSION.get(api_url, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return None

    try:
        data = r.json()
    except json.JSONDecodeError:
        return None

    # Check conclusion - return None if vacated and remanded
    conclusion = (data.get('conclusion') or '') if isinstance(data.get('conclusion'), str) else ''
    if conclusion and 'Vacated and remanded' in _strip_html(conclusion):
        return None

    # Transcript URL: first oral argument media link (API href) or empty
    transcript_url = ''
    oral = data.get('oral_argument_audio') or []
    if oral and isinstance(oral, list) and len(oral) > 0:
        first = oral[0]
        if isinstance(first, dict) and first.get('href'):
            transcript_url = first['href']
        elif isinstance(first, dict) and first.get('id'):
            transcript_url = f"https://api.oyez.org/case_media/oral_argument_audio/{first['id']}"

    # Advocates with side info (petitioner=1, respondent=0)
    # The advocate_description field contains strings like "for petitioner" or "for respondent"
    advocates = []
    for item in (data.get('advocates') or []):
        if not isinstance(item, dict):
            continue
        adv = item.get('advocate') or {}
        name = (adv.get('name') or '').strip() if isinstance(adv, dict) else ''
        desc = (item.get('advocate_description') or '').strip().lower()
        if not name:
            continue
        if 'petitioner' in desc or 'appellant' in desc:
            side = 1
        elif 'respondent' in desc or 'appellee' in desc:
            side = 0
        else:
            side = -1  # unknown
        advocates.append({'name': name, 'side': side})

    # Facts and question (strip HTML)
    facts = _strip_html(data.get('facts_of_the_case') or '')
    question = _strip_html(data.get('question') or '')

    # Parties (extracted early so win_side logic can use them)
    parties = {}
    fp = (data.get('first_party') or '').strip()
    sp = (data.get('second_party') or '').strip()
    fpl = (data.get('first_party_label') or '').strip()
    spl = (data.get('second_party_label') or '').strip()
    if fpl and fp:
        parties[fpl] = fp
    if spl and sp:
        parties[spl] = sp

    # Votes and majority from first decision
    votes = []
    majority = ''
    win_side = -1  # 1=petitioner, 0=respondent, 2=unclear, -1=unknown
    decisions = data.get('decisions') or []
    if decisions and isinstance(decisions[0], dict):
        dec = decisions[0]
        majority = (dec.get('winning_party') or '').strip()
        for v in (dec.get('votes') or []):
            if not isinstance(v, dict):
                continue
            member = v.get('member') or {}
            name = (member.get('last_name') or member.get('name') or '').strip() if isinstance(member, dict) else ''
            vote = (v.get('vote') or '').strip().lower()
            if vote not in ('majority', 'minority'):
                vote = 'majority'
            if name:
                votes.append({'name': name, 'vote': vote})

    # Determine win_side: compare winning_party to first_party/second_party
    def _normalize(s: str) -> str:
        """Lowercase, strip punctuation differences like et al./et al, Corp./Corporation."""
        return re.sub(r'[.,;]', '', s.lower()).strip()

    def _party_matches(winner: str, party: str) -> bool:
        if not winner or not party:
            return False
        w, p = _normalize(winner), _normalize(party)
        if w == p or w in p or p in w:
            return True
        # Check if any word in winner appears in party (handles last names)
        w_words = w.split()
        p_words = p.split()
        if len(w_words) == 1 and any(w_words[0] == pw for pw in p_words):
            return True
        return False

    majority_lower = majority.lower().strip()
    if majority_lower in ('petitioner', 'appellant'):
        win_side = 1
    elif majority_lower in ('respondent', 'appellee'):
        win_side = 0
    elif majority_lower in ('dismissal', 'both', 'neither party', 'all parties',
                            'vacatur', 'n/a', 'questions certified'):
        win_side = 2
    elif majority:
        fp_match = _party_matches(majority, fp)
        sp_match = _party_matches(majority, sp)
        if fp_match and not sp_match:
            if fpl.lower() in ('petitioner', 'appellant'):
                win_side = 1
            elif fpl.lower() in ('respondent', 'appellee'):
                win_side = 0
            else:
                win_side = 2
        elif sp_match and not fp_match:
            if spl.lower() in ('petitioner', 'appellant'):
                win_side = 1
            elif spl.lower() in ('respondent', 'appellee'):
                win_side = 0
            else:
                win_side = 2
        else:
            win_side = 2  # ambiguous or no match

    # Convert votes from majority/minority to petitioner/respondent side
    # If a justice voted "majority" and petitioner won (win_side=1), they voted Petitioner
    votes_with_side = []
    for v in votes:
        name = v.get('name', '')
        vote = v.get('vote', '')
        if win_side == 1:
            side = 'Petitioner' if vote == 'majority' else 'Respondent'
        elif win_side == 0:
            side = 'Respondent' if vote == 'majority' else 'Petitioner'
        else:
            side = 'Unknown'
        votes_with_side.append({'name': name, 'side': side})

    # Filter out cases missing critical data
    if not majority:
        return None
    if not transcript_url:
        return None

    return {
        'name': case['name'],
        'main_url': url,
        'year': case['year'],
        'transcript_url': transcript_url,
        'advocates': advocates,
        'facts': facts,
        'question': question,
        'votes': votes_with_side,
        'win_side': win_side,
        'majority': majority,
        'parties': parties,
    }


# Configuration: Set to True to re-run previously failed cases
RERUN_BAD_CASES = False

with open(os.path.join(DATA_DIR, 'basic.json'), 'r') as file:
    raw = json.load(file)
main = raw if isinstance(raw, list) else []

with open(os.path.join(DATA_DIR, 'bad_cases.json'), 'r') as file:
    raw = json.load(file)
bad_cases = raw if isinstance(raw, list) else []

# Filter out cases that are already processed
existing_case_names = {case['name'] for case in main if isinstance(case, dict) and case.get('name')}
bad_case_names = {case['name'] for case in bad_cases if isinstance(case, dict) and case.get('name')}

if RERUN_BAD_CASES:
    cases_to_process = [
        case for case in cases if case['name'] not in existing_case_names]
else:
    cases_to_process = [
        case for case in cases
        if case['name'] not in existing_case_names and case['name'] not in bad_case_names
    ]

# Shuffle so each run picks different cases (useful when debugging a single case)
random.shuffle(cases_to_process)

# If nothing new to process, pick a random already-processed case to refresh
if not cases_to_process and main:
    refresh_case = random.choice(cases)
    # Find the matching source case from cases list
    print(f"Nothing to process. Refreshing random case: {refresh_case['name']}")
    try:
        case_data = get_basic(refresh_case)
        if case_data is not None:
            # Replace the existing entry (don't duplicate)
            main = [c for c in main if c.get('name') != refresh_case['name']]
            main.append(case_data)
            with open(os.path.join(DATA_DIR, 'basic.json'), 'w') as file:
                json.dump(main, file)
            print(f"Refreshed case: {refresh_case['name']}")
        else:
            print(f"Refresh returned None for: {refresh_case['name']}")
    except Exception as e:
        print(f"Error refreshing case {refresh_case['name']}: {e}")

print(
    f"Total cases: {len(cases)}, Already processed: {len(existing_case_names)}, "
    f"Bad cases: {len(bad_case_names)}, To process: {len(cases_to_process)}, "
    f"Rerun bad cases: {RERUN_BAD_CASES}"
)

for case in tqdm(cases_to_process, desc="Processing cases"):
    try:
        case_data = get_basic(case)
        if case_data is not None:
            main.append(case_data)

            if RERUN_BAD_CASES and case['name'] in bad_case_names:
                bad_cases = [bc for bc in bad_cases if bc['name'] != case['name']]
                with open(os.path.join(DATA_DIR, 'bad_cases.json'), 'w') as file:
                    json.dump(bad_cases, file)
                print(f"Successfully processed previously failed case: {case['name']}")

            with open(os.path.join(DATA_DIR, 'basic.json'), 'w') as file:
                json.dump(main, file)
        else:
            # Track cases that returned None (vacated/remanded, API failure, etc.)
            print(f"Case returned None, adding to bad_cases: {case['name']}")
            if case['name'] not in bad_case_names:
                bad_cases.append(case)
                bad_case_names.add(case['name'])
            with open(os.path.join(DATA_DIR, 'bad_cases.json'), 'w') as file:
                json.dump(bad_cases, file)
    except requests.RequestException as e:
        print(f"Request failed for case {case['name']}: {e}")
        if case['name'] not in bad_case_names:
            bad_cases.append(case)
        with open(os.path.join(DATA_DIR, 'bad_cases.json'), 'w') as file:
            json.dump(bad_cases, file)
        continue
    except Exception as e:
        print(f"Error processing case {case['main_url']}: {e}")
        break

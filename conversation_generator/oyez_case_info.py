"""
Case-level legal info for the Oyez pipeline only (2019–2026).

Use this only for cases that come from your Oyez scrape (data/cases.json +
data/basic.json). Do NOT use for ConvoKit corpus cases (1940–2019); use
corpus_case_info.py and ConvoKit cases.jsonl for those.
"""

import json
import os
import re
from html import unescape

# Optional: only used if fetch_from_oyez=True
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Default path to basic.json (Oyez-derived case data)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BASIC_JSON = os.path.join(DATA_DIR, "basic.json")


def _strip_html(html: str) -> str:
    if not html or not isinstance(html, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return unescape(text)


def _oyez_key_from_main_url(main_url: str) -> str | None:
    """Extract (year_docket) key from Oyez main_url, e.g. cases/2019/18-280 -> 2019_18-280."""
    if not main_url or not isinstance(main_url, str):
        return None
    # .../cases/2019/18-280 or .../cases/1955/71
    m = re.search(r"/cases/(\d{4})/([^/?#]+)", main_url)
    if not m:
        return None
    year, docket = m.group(1), m.group(2).strip()
    return f"{year}_{docket}"


def build_basic_index(basic_path: str | None = None) -> dict[str, dict]:
    """
    Build case_id -> case_legal_info from data/basic.json.
    Keys are year_docket (e.g. 2019_18-280) to align with ConvoKit convo.meta["case_id"].
    """
    path = basic_path or BASIC_JSON
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    entries = raw if isinstance(raw, list) else []
    index = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        main_url = entry.get("main_url")
        key = _oyez_key_from_main_url(main_url)
        if key is None:
            continue
        index[key] = {
            "name": entry.get("name") or "",
            "facts": entry.get("facts") or "",
            "question": entry.get("question") or "",
            "year": entry.get("year"),
            "main_url": main_url or "",
            "majority": entry.get("majority") or "",
            "parties": entry.get("parties") or {},
        }
    return index


# Module-level cache (call build_basic_index once per process)
_basic_index: dict[str, dict] | None = None


def get_case_legal_info(
    case_id: str,
    basic_path: str | None = None,
    fetch_from_oyez: bool = False,
    timeout: int = 15,
) -> dict | None:
    """
    Get case-level legal info for a ConvoKit case_id (e.g. "1955_71", "2019_18-280").

    Args:
        case_id: From convo.meta["case_id"].
        basic_path: Path to basic.json; default is data/basic.json.
        fetch_from_oyez: If True and case_id not in basic.json, try Oyez API.
        timeout: Request timeout when fetching from Oyez.

    Returns:
        Dict with keys: name, facts, question, year, main_url, majority, parties;
        or None if not found and fetch_from_oyez False or API failed.
    """
    global _basic_index
    if _basic_index is None:
        _basic_index = build_basic_index(basic_path)

    case_id = (case_id or "").strip()
    if not case_id:
        return None

    # Direct lookup (ConvoKit case_id often matches our key)
    if case_id in _basic_index:
        return _basic_index[case_id]

    # Try alternate key: "1955_71" -> some Oyez URLs use "55-71" for docket
    if "_" in case_id:
        year_str, docket = case_id.split("_", 1)
        if year_str.isdigit() and len(year_str) == 4 and docket.isdigit():
            alt_docket = f"{year_str[2:]}-{docket}"
            alt_key = f"{year_str}_{alt_docket}"
            if alt_key in _basic_index:
                return _basic_index[alt_key]

    if not fetch_from_oyez or not HAS_REQUESTS:
        return None

    # Try Oyez API: e.g. GET https://api.oyez.org/cases/1955/71 or cases/2019/18-280
    if "_" in case_id:
        year_str, docket = case_id.split("_", 1)
        for docket_path in (docket, f"{year_str[2:]}-{docket}" if year_str.isdigit() and len(year_str) == 4 else None):
            if not docket_path:
                continue
            url = f"https://api.oyez.org/cases/{year_str}/{docket_path}"
            try:
                r = requests.get(
                    url,
                    timeout=timeout,
                    headers={"User-Agent": "SupremeCourtGuess/1.0", "Accept": "application/json"},
                )
                if r.status_code != 200:
                    continue
                data = r.json()
            except Exception:
                continue
            name = (data.get("name") or "").strip() or (data.get("case_name") or "").strip()
            if not name and data.get("first_party") and data.get("second_party"):
                name = f"{data.get('first_party', '')} v. {data.get('second_party', '')}"
            info = {
                "name": name,
                "facts": _strip_html(data.get("facts_of_the_case") or ""),
                "question": _strip_html(data.get("question") or ""),
                "year": int(year_str) if year_str.isdigit() else None,
                "main_url": f"https://www.oyez.org/cases/{year_str}/{docket_path}",
                "majority": "",
                "parties": {},
            }
            dec = (data.get("decisions") or [None])[0]
            if isinstance(dec, dict):
                info["majority"] = (dec.get("winning_party") or "").strip()
            # Cache for next time
            _basic_index[case_id] = info
            return info

    return None

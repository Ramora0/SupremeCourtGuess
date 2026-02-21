"""
Case-level legal info for the ConvoKit Supreme Court corpus (1940–2019).

The corpus itself (conversations.json, utterances.jsonl) does NOT contain
case name, citation, or decision date. That data lives in a separate file
from the same ConvoKit dataset: cases.jsonl, hosted by Cornell. This module
downloads that file (once) and provides lookup by ConvoKit case_id.

Use this for corpus cases. For 2019–2026 case data from your Oyez scrape,
use the Oyez pipeline (e.g. data/basic.json) separately — do not mix the two.
"""

import json
import os

# ConvoKit cases.jsonl URL (same dataset as supreme-corpus, not included in the zip)
CONVOKIT_CASES_JSONL_URL = "https://zissou.infosci.cornell.edu/convokit/datasets/supreme-corpus/cases.jsonl"

# Local path: project data dir (corpus lives in ~/.convokit, cases.jsonl we keep in project)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CASES_JSONL_PATH = os.path.join(DATA_DIR, "convokit_cases.jsonl")

# In-memory index: case_id (e.g. "1955_71") -> case row from cases.jsonl
_index: dict | None = None


def download_cases_jsonl(path: str | None = None, url: str | None = None) -> str:
    """
    Download ConvoKit cases.jsonl to the project data dir. Idempotent.
    Returns the path where the file was written.
    """
    out_path = path or CASES_JSONL_PATH
    url = url or CONVOKIT_CASES_JSONL_URL
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "SupremeCourtGuess/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(out_path, "wb") as f:
                f.write(resp.read())
    except Exception as e:
        raise RuntimeError(f"Failed to download cases.jsonl from {url}: {e}") from e
    return out_path


def _load_index(path: str | None = None) -> dict:
    """Load cases.jsonl and index by id. Cached in _index (use path=None to use cache)."""
    global _index
    target = path or CASES_JSONL_PATH
    if _index is not None and path is None:
        return _index
    if not os.path.isfile(target):
        _index = {} if path is None else _index
        return _index if _index is not None else {}
    _index = {}
    with open(target) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cid = row.get("id")
                if cid is not None:
                    _index[str(cid)] = row
            except json.JSONDecodeError:
                continue
    return _index


def get_case_legal_info_from_corpus(case_id: str, cases_path: str | None = None) -> dict | None:
    """
    Get case-level legal info for a ConvoKit case_id (e.g. "1955_71").

    Data comes from ConvoKit cases.jsonl (same dataset as the corpus).
    Returns dict with: id, title, citation, decided_date, petitioner, respondent,
    docket_no, court, url, year, and vote-related fields. Does NOT include
    facts of the case or question presented (not in the ConvoKit dataset).

    Returns None if cases.jsonl is missing or case_id not found.
    """
    idx = _load_index(cases_path)
    case_id = (case_id or "").strip()
    if not case_id:
        return None
    row = idx.get(case_id)
    if row is None:
        return None
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "citation": row.get("citation"),
        "decided_date": row.get("decided_date"),
        "petitioner": row.get("petitioner"),
        "respondent": row.get("respondent"),
        "docket_no": row.get("docket_no"),
        "court": row.get("court"),
        "url": row.get("url"),
        "year": row.get("year"),
    }


if __name__ == "__main__":
    # Download cases.jsonl so corpus case lookups work.
    path = download_cases_jsonl()
    print(f"Downloaded ConvoKit cases.jsonl to {path}")
    n = len(_load_index(path))
    print(f"Indexed {n} cases.")

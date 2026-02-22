#!/usr/bin/env python3
"""
Match SCDB (Supreme Court Database) records to ConvoKit transcript case_ids.

Builds a mapping from ConvoKit case_id (e.g. "1955_71", "1980_79-1709") to the
corresponding SCDB row(s), using term + docket number as the join key.

The SCDB CSVs must be downloaded first into data/scdb/:
  - SCDB_2025_01_caseCentered_Docket.csv
  - SCDB_2025_01_justiceCentered_Docket.csv

Usage:
  python scdb_matcher.py                  # report match rates
  python scdb_matcher.py --verbose        # show unmatched cases
"""

import os
import re

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCDB_DIR = os.path.join(DATA_DIR, "scdb")
SCDB_CASE_CSV = os.path.join(SCDB_DIR, "SCDB_2025_01_caseCentered_Docket.csv")
SCDB_JUSTICE_CSV = os.path.join(SCDB_DIR, "SCDB_2025_01_justiceCentered_Docket.csv")
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "case_transcripts_cleaned")


# ---------------------------------------------------------------------------
# Docket normalisation
# ---------------------------------------------------------------------------

def _normalize_docket(raw: str) -> str:
    """
    Normalize a docket string for matching.

    Handles the various formats across SCDB and ConvoKit:
      SCDB:     "11 ORIG", "No. 12, Original", "294 M", "75-245"
      ConvoKit: "43-orig", "79-1709", "71"

    Strategy: lowercase, strip whitespace, collapse "No. N, Original" and
    "N ORIG" patterns to "N-orig", drop trailing "M" suffix, etc.
    """
    s = str(raw).strip().lower()

    # "no. 12, original" -> "12-orig"
    m = re.match(r"no\.\s*(\d+),?\s*original", s)
    if m:
        return f"{m.group(1)}-orig"

    # "11 orig" or "11orig" -> "11-orig"
    m = re.match(r"(\d+)\s*orig(?:inal)?$", s)
    if m:
        return f"{m.group(1)}-orig"

    # Already "43-orig" -> keep
    if s.endswith("-orig") or s.endswith("-original"):
        s = re.sub(r"-original$", "-orig", s)
        return s

    # "294 M" or "294M" -> "294m" (misc docket)
    m = re.match(r"(\d+)\s*m$", s)
    if m:
        return f"{m.group(1)}m"

    # "22O65" (2020 term uses this format) -> keep as-is lowered
    # Standard hyphenated "75-245" or plain "71" -> keep
    return s


def _build_scdb_key(term: int, docket: str) -> str:
    """Build a normalized match key from SCDB term + docket."""
    return f"{term}_{_normalize_docket(docket)}"


def _convokit_case_id_to_key(case_id: str) -> str:
    """
    Normalize a ConvoKit case_id for matching.

    ConvoKit case_ids look like "1955_71" or "1980_79-1709" or "1970_43-orig".
    The transcript filenames are the same, optionally with a _CONVOID suffix.
    """
    s = str(case_id).strip()
    if "_" not in s:
        return s.lower()
    parts = s.split("_", 1)
    term_str, docket = parts[0], parts[1]
    return f"{term_str}_{_normalize_docket(docket)}"


# ---------------------------------------------------------------------------
# SCDB loading
# ---------------------------------------------------------------------------

def load_scdb_case(path: str | None = None) -> pd.DataFrame:
    """Load case-centered SCDB CSV and add match_key column."""
    path = path or SCDB_CASE_CSV
    df = pd.read_csv(path, encoding="latin-1")
    df["match_key"] = df.apply(
        lambda r: _build_scdb_key(r["term"], str(r["docket"])), axis=1
    )
    return df


def load_scdb_justice(path: str | None = None) -> pd.DataFrame:
    """Load justice-centered SCDB CSV and add match_key column."""
    path = path or SCDB_JUSTICE_CSV
    df = pd.read_csv(path, encoding="latin-1")
    df["match_key"] = df.apply(
        lambda r: _build_scdb_key(r["term"], str(r["docket"])), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# ConvoKit case_id collection
# ---------------------------------------------------------------------------

def get_convokit_case_ids() -> list[str]:
    """
    Get unique ConvoKit case_ids from the corpus.

    Loads the full corpus to extract case_id from conversation metadata.
    """
    from convokit import Corpus, download

    path = download("supreme-corpus")
    corpus = Corpus(filename=path)
    seen = set()
    case_ids = []
    for cid in corpus.get_conversation_ids():
        c = corpus.get_conversation(cid)
        case_id = str(c.meta.get("case_id", cid))
        if case_id not in seen:
            seen.add(case_id)
            case_ids.append(case_id)
    return case_ids


def get_transcript_case_ids(transcripts_dir: str | None = None) -> list[str]:
    """
    Get unique case_ids from transcript filenames in case_transcripts_cleaned/.

    Filename format: {term}_{docket}.txt or {term}_{docket}_{convoid}.txt
    We strip the optional _convoid suffix and .txt extension.
    """
    d = transcripts_dir or TRANSCRIPTS_DIR
    if not os.path.isdir(d):
        return []
    seen = set()
    case_ids = []
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".txt"):
            continue
        base = fname[:-4]  # strip .txt
        # Could be "1955_71" or "1955_71_13092" (with convo_id suffix)
        # The case_id is term_docket. We split and check: if last segment is
        # all digits and the preceding parts form a valid term_docket, strip it.
        parts = base.split("_")
        if len(parts) >= 3:
            # Check if last part is a bare convo_id (all digits, no hyphen)
            # Docket parts can have hyphens (79-1709) but convo_ids don't
            last = parts[-1]
            if last.isdigit() and len(last) >= 4:
                # Likely a convo_id suffix — drop it
                base = "_".join(parts[:-1])
        if base not in seen:
            seen.add(base)
            case_ids.append(base)
    return case_ids


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_cases(
    convokit_ids: list[str],
    scdb_df: pd.DataFrame,
) -> dict:
    """
    Match ConvoKit case_ids to SCDB rows.

    Returns dict with:
        matched:   {case_id: match_key}  -- successful matches
        unmatched: [case_id, ...]         -- ConvoKit ids with no SCDB match
        scdb_only: [match_key, ...]       -- SCDB keys with no ConvoKit match (in overlapping year range)
    """
    # Build set of SCDB match keys
    scdb_keys = set(scdb_df["match_key"].unique())

    matched = {}
    unmatched = []
    for case_id in convokit_ids:
        key = _convokit_case_id_to_key(case_id)
        if key in scdb_keys:
            matched[case_id] = key
        else:
            unmatched.append(case_id)

    # Find SCDB keys in the ConvoKit year range with no match
    convokit_keys = {_convokit_case_id_to_key(cid) for cid in convokit_ids}
    years = set()
    for cid in convokit_ids:
        if "_" in cid:
            try:
                years.add(int(cid.split("_")[0]))
            except ValueError:
                pass
    min_year = min(years) if years else 1955
    max_year = max(years) if years else 2019
    scdb_in_range = scdb_df[(scdb_df["term"] >= min_year) & (scdb_df["term"] <= max_year)]
    scdb_keys_in_range = set(scdb_in_range["match_key"].unique())
    scdb_only = sorted(scdb_keys_in_range - convokit_keys)

    return {
        "matched": matched,
        "unmatched": unmatched,
        "scdb_only": scdb_only,
    }


# ---------------------------------------------------------------------------
# SCDB code -> human-readable label maps
# ---------------------------------------------------------------------------

ISSUE_AREA_LABELS = {
    1: "Criminal Procedure",
    2: "Civil Rights",
    3: "First Amendment",
    4: "Due Process",
    5: "Privacy",
    6: "Attorneys",
    7: "Unions",
    8: "Economic Activity",
    9: "Judicial Power",
    10: "Federalism",
    11: "Interstate Relations",
    12: "Federal Taxation",
    13: "Miscellaneous",
    14: "Private Action",
}

# Petitioner / respondent codes share the same scheme. We map every code
# from the SCDB codebook to a readable label.
_PARTY_LABELS = {
    1: "U.S. Attorney General",
    2: "state board of education",
    3: "city or local government",
    4: "state commission or authority",
    5: "county government",
    6: "court or judicial district",
    7: "state department or agency",
    8: "government employee",
    9: "female government employee",
    10: "minority government employee",
    11: "minority female government employee",
    12: "unlisted government agency",
    13: "retired government employee",
    14: "U.S. House of Representatives",
    15: "interstate compact",
    16: "judge",
    17: "state legislature",
    18: "local government unit",
    19: "government official",
    20: "state or U.S. supreme court",
    21: "local school district",
    22: "U.S. Senate",
    23: "U.S. senator",
    24: "foreign nation",
    25: "state or local taxpayer",
    26: "state college or university",
    27: "United States",
    28: "state government",
    100: "person accused of crime",
    101: "advertising business",
    102: "agent or fiduciary",
    103: "airplane manufacturer",
    104: "airline",
    105: "alcohol distributor",
    106: "alien or denaturalization subject",
    107: "American Medical Association",
    108: "Amtrak",
    109: "amusement establishment",
    110: "arrested person or pretrial detainee",
    111: "attorney or bar applicant",
    112: "author or copyright holder",
    113: "bank or financial institution",
    114: "bankrupt person or business",
    115: "liquor establishment",
    116: "water transportation or stevedore",
    117: "bookstore or printer",
    118: "brewery or distillery",
    119: "broker or securities firm",
    120: "construction company",
    121: "bus or passenger carrier",
    122: "business or corporation",
    123: "buyer or purchaser",
    124: "cable TV",
    125: "car dealer",
    126: "person convicted of crime",
    127: "tangible property owner",
    128: "chemical company",
    129: "child or children",
    130: "religious organization or person",
    131: "private club",
    132: "coal company or mine operator",
    133: "computer business",
    134: "consumer",
    135: "creditor",
    136: "allegedly criminally insane person",
    137: "defendant",
    138: "debtor",
    139: "real estate developer",
    140: "disabled person or disability claimant",
    141: "distributor",
    142: "selective service subject",
    143: "drug manufacturer",
    144: "pharmacist or pharmacy",
    145: "employee or job applicant",
    146: "employee trust or health fund",
    147: "electric equipment manufacturer",
    148: "electric or hydroelectric utility",
    149: "eleemosynary institution",
    150: "environmental organization",
    151: "employer",
    152: "farmer or farm worker",
    153: "father",
    154: "female employee or job applicant",
    155: "female",
    156: "movie or theatrical production",
    157: "fisherman or fishing company",
    158: "food or meat packing company",
    159: "foreign nongovernmental entity",
    160: "franchiser",
    161: "franchisee",
    162: "LGBT person or organization",
    163: "guarantor",
    164: "handicapped individual",
    165: "health organization or nursing home",
    166: "heir or beneficiary",
    167: "hospital or medical center",
    168: "husband or ex-husband",
    169: "involuntarily committed patient",
    170: "Indian or Indian tribe",
    171: "insurance company or surety",
    172: "inventor or patent holder",
    173: "investor",
    174: "nonphysically injured person",
    175: "juvenile",
    176: "government contractor",
    177: "license or permit holder",
    178: "magazine",
    179: "male",
    180: "medical or Medicaid claimant",
    181: "medical supply company",
    182: "minority employee or job applicant",
    183: "minority female employee or job applicant",
    184: "manufacturer",
    185: "management or executive officer",
    186: "military personnel or dependent",
    187: "mining company or miner",
    188: "mother",
    189: "auto manufacturer",
    190: "newspaper or journal",
    191: "radio and television network",
    192: "nonprofit organization",
    193: "nonresident",
    194: "nuclear power facility",
    195: "owner or landlord",
    196: "shareholders (tender offer target)",
    197: "tender offeror",
    198: "oil or natural gas company",
    199: "elderly person or organization",
    200: "out-of-state noncriminal defendant",
    201: "political action committee",
    202: "parent or parents",
    203: "parking lot or service",
    204: "patient",
    205: "telephone or telecom company",
    206: "physician or medical society",
    207: "public interest organization",
    208: "physically injured person",
    209: "pipeline company",
    210: "package or container",
    211: "political candidate or party",
    212: "indigent or welfare recipient",
    213: "indigent defendant",
    214: "private person",
    215: "prisoner or inmate",
    216: "professional",
    217: "probationer or parolee",
    218: "protester or demonstrator",
    219: "public utility",
    220: "publisher",
    221: "radio station",
    222: "racial or ethnic minority",
    223: "person protesting segregation",
    224: "minority student or applicant",
    225: "realtor",
    226: "journalist or news media member",
    227: "resident",
    228: "restaurant or food vendor",
    229: "mentally incompetent person",
    230: "retired or former employee",
    231: "railroad",
    232: "private school or university",
    233: "seller or vendor",
    234: "shipper or exporter",
    235: "shopping center",
    236: "spouse or former spouse",
    237: "stockholder or bondholder",
    238: "retail business",
    239: "student or admissions applicant",
    240: "federal taxpayer",
    241: "tenant or lessee",
    242: "theater or studio",
    243: "forest products or lumber company",
    244: "person wishing to travel abroad",
    245: "trucking company or motor carrier",
    246: "television station",
    247: "union member",
    248: "unemployed person",
    249: "union or labor organization",
    250: "veteran",
    251: "voter or prospective voter",
    252: "wholesale trade",
    253: "wife or ex-wife",
    254: "witness or person under subpoena",
    255: "network",
    256: "slave",
    257: "slave owner",
    258: "Bank of the United States",
    259: "timber company",
    260: "U.S. job applicant or employee",
    301: "Army and Air Force Exchange Service",
    302: "Atomic Energy Commission",
    303: "U.S. Air Force",
    304: "Department of Agriculture",
    305: "Alien Property Custodian",
    306: "U.S. Army",
    307: "Board of Immigration Appeals",
    308: "Bureau of Indian Affairs",
    310: "Bonneville Power Administration",
    311: "Benefits Review Board",
    312: "Civil Aeronautics Board",
    313: "Bureau of the Census",
    314: "Central Intelligence Agency",
    315: "Commodity Futures Trading Commission",
    316: "Department of Commerce",
    317: "Comptroller of Currency",
    318: "Consumer Product Safety Commission",
    319: "Civil Rights Commission",
    320: "Civil Service Commission",
    321: "Customs Service",
    322: "Defense Base Closure Commission",
    323: "Drug Enforcement Agency",
    324: "Department of Defense",
    325: "Department of Energy",
    326: "Department of the Interior",
    327: "Department of Justice",
    328: "Department of State",
    329: "Department of Transportation",
    330: "Department of Education",
    331: "Employees' Compensation Commission",
    332: "Equal Employment Opportunity Commission",
    333: "Environmental Protection Agency",
    334: "Federal Aviation Administration",
    335: "Federal Bureau of Investigation",
    336: "Federal Bureau of Prisons",
    337: "Farm Credit Administration",
    338: "Federal Communications Commission",
    339: "Federal Credit Union Administration",
    340: "Food and Drug Administration",
    341: "Federal Deposit Insurance Corporation",
    342: "Federal Energy Administration",
    343: "Federal Election Commission",
    344: "Federal Energy Regulatory Commission",
    345: "Federal Housing Administration",
    346: "Federal Home Loan Bank Board",
    347: "Federal Labor Relations Authority",
    348: "Federal Maritime Board",
    349: "Federal Maritime Commission",
    350: "Farmers Home Administration",
    351: "Federal Parole Board",
    352: "Federal Power Commission",
    353: "Federal Railroad Administration",
    354: "Federal Reserve Board",
    355: "Federal Reserve System",
    356: "Federal Savings and Loan Insurance Corporation",
    357: "Federal Trade Commission",
    358: "Federal Works Administration",
    359: "General Accounting Office",
    360: "Comptroller General",
    361: "General Services Administration",
    362: "Department of Health, Education and Welfare",
    363: "Department of Health and Human Services",
    364: "Department of Housing and Urban Development",
    366: "Interstate Commerce Commission",
    367: "Indian Claims Commission",
    368: "Immigration and Naturalization Service",
    369: "Internal Revenue Service",
    370: "Information Security Oversight Office",
    371: "Department of Labor",
    372: "Loyalty Review Board",
    373: "Legal Services Corporation",
    374: "Merit Systems Protection Board",
    375: "Multistate Tax Commission",
    376: "NASA",
    377: "U.S. Navy",
    378: "National Credit Union Administration",
    379: "National Endowment for the Arts",
    380: "National Enforcement Commission",
    381: "National Highway Traffic Safety Administration",
    382: "National Labor Relations Board",
    383: "National Mediation Board",
    384: "National Railroad Adjustment Board",
    385: "Nuclear Regulatory Commission",
    386: "National Security Agency",
    387: "Office of Economic Opportunity",
    388: "Office of Management and Budget",
    389: "Office of Price Administration",
    390: "Office of Personnel Management",
    391: "Occupational Safety and Health Administration",
    392: "OSHA Review Commission",
    393: "Office of Workers' Compensation",
    394: "Patent Office",
    395: "Pay Board",
    396: "Pension Benefit Guaranty Corporation",
    397: "U.S. Public Health Service",
    398: "Postal Rate Commission",
    399: "Provider Reimbursement Review Board",
    400: "Renegotiation Board",
    401: "Railroad Adjustment Board",
    402: "Railroad Retirement Board",
    403: "Subversive Activities Control Board",
    404: "Small Business Administration",
    405: "Securities and Exchange Commission",
    406: "Social Security Administration",
    407: "Selective Service System",
    408: "Department of the Treasury",
    409: "Tennessee Valley Authority",
    410: "U.S. Forest Service",
    411: "U.S. Parole Commission",
    412: "U.S. Postal Service",
    413: "U.S. Sentencing Commission",
    414: "Veterans' Administration",
    415: "War Production Board",
    416: "Wage Stabilization Board",
    417: "General Land Office",
    418: "Transportation Security Administration",
    419: "Surface Transportation Board",
    420: "U.S. Shipping Board",
    421: "Reconstruction Finance Corp.",
    422: "Department of Homeland Security",
    501: "unidentifiable",
    600: "international entity",
}

JURISDICTION_LABELS = {
    1: "cert",
    2: "appeal",
    3: "bail",
    4: "certification",
    5: "docketing fee",
    6: "rehearing or reargument",
    7: "injunction",
    8: "mandamus",
    9: "original",
    10: "prohibition",
    12: "stay",
    13: "writ of error",
    14: "writ of habeas corpus",
    15: "other",
}

CERT_REASON_LABELS = {
    1: "case did not arise on cert or cert not granted",
    2: "federal court conflict",
    3: "federal court conflict and important question",
    4: "putative conflict",
    5: "conflict between federal and state court",
    6: "state court conflict",
    7: "federal court confusion or uncertainty",
    8: "state court confusion or uncertainty",
    9: "federal and state court confusion or uncertainty",
    10: "to resolve important or significant question",
    11: "to resolve question presented",
    12: "no reason given",
    13: "other reason",
}

LC_DISPOSITION_LABELS = {
    1: "stay, petition, or motion granted",
    2: "affirmed",
    3: "reversed",
    4: "reversed and remanded",
    5: "vacated and remanded",
    6: "affirmed and reversed in part",
    7: "affirmed and reversed in part and remanded",
    8: "vacated",
    9: "petition denied or appeal dismissed",
    10: "modified",
    11: "remanded",
    12: "unusual disposition",
}

LC_DISPOSITION_DIRECTION_LABELS = {
    1: "conservative",
    2: "liberal",
    3: "unspecifiable",
}

LAW_TYPE_LABELS = {
    1: "Constitution",
    2: "constitutional amendment",
    3: "federal statute",
    4: "court rules",
    5: "other",
    6: "infrequently litigated statute",
    8: "state or local law",
    9: "no legal provision",
}


def _label(mapping: dict, code, fallback: str = "unknown") -> str:
    """Look up a numeric SCDB code in a label dict, handling NaN/None."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return fallback
    return mapping.get(int(code), fallback)


# ---------------------------------------------------------------------------
# Header formatting (for prepending to transcripts)
# ---------------------------------------------------------------------------

def format_scdb_header(case_id: str) -> str | None:
    """
    Build the SCDB metadata header block for a ConvoKit case_id.

    Returns a multi-line string like:
        ISSUE AREA: Criminal Procedure
        PETITIONER: United States
        RESPONDENT: person accused of crime
        ...
    or None if the case has no SCDB match.
    """
    row = get_scdb_case_row(case_id)
    if row is None:
        return None

    lines = []

    ia = _label(ISSUE_AREA_LABELS, row.get("issueArea"))
    lines.append(f"ISSUE AREA: {ia}")

    pet = _label(_PARTY_LABELS, row.get("petitioner"))
    lines.append(f"PETITIONER TYPE: {pet}")

    resp = _label(_PARTY_LABELS, row.get("respondent"))
    lines.append(f"RESPONDENT TYPE: {resp}")

    lcd = _label(LC_DISPOSITION_LABELS, row.get("lcDisposition"))
    lines.append(f"LOWER COURT DISPOSITION: {lcd}")

    lcdir = _label(LC_DISPOSITION_DIRECTION_LABELS, row.get("lcDispositionDirection"))
    lines.append(f"LOWER COURT DIRECTION: {lcdir}")

    lcd_disagree = row.get("lcDisagreement")
    if lcd_disagree is not None and not (isinstance(lcd_disagree, float) and pd.isna(lcd_disagree)):
        lines.append(f"LOWER COURT DISAGREEMENT: {'yes' if int(lcd_disagree) == 1 else 'no'}")

    cr = _label(CERT_REASON_LABELS, row.get("certReason"))
    lines.append(f"CERT REASON: {cr}")

    jur = _label(JURISDICTION_LABELS, row.get("jurisdiction"))
    lines.append(f"JURISDICTION: {jur}")

    lt = _label(LAW_TYPE_LABELS, row.get("lawType"))
    lines.append(f"LAW TYPE: {lt}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lookup helper (for use by other modules)
# ---------------------------------------------------------------------------

_scdb_case_index: dict[str, pd.Series] | None = None
_scdb_justice_index: dict[str, pd.DataFrame] | None = None


def get_scdb_case_row(case_id: str) -> pd.Series | None:
    """Look up the SCDB case-centered row for a ConvoKit case_id. Returns None if no match."""
    global _scdb_case_index
    if _scdb_case_index is None:
        if not os.path.isfile(SCDB_CASE_CSV):
            return None
        df = load_scdb_case()
        _scdb_case_index = {}
        for _, row in df.iterrows():
            _scdb_case_index[row["match_key"]] = row
    key = _convokit_case_id_to_key(case_id)
    return _scdb_case_index.get(key)


def get_scdb_justice_rows(case_id: str) -> pd.DataFrame | None:
    """Look up SCDB justice-centered rows for a ConvoKit case_id. Returns None if no match."""
    global _scdb_justice_index
    if _scdb_justice_index is None:
        if not os.path.isfile(SCDB_JUSTICE_CSV):
            return None
        df = load_scdb_justice()
        _scdb_justice_index = {}
        for key, group in df.groupby("match_key"):
            _scdb_justice_index[key] = group
    key = _convokit_case_id_to_key(case_id)
    return _scdb_justice_index.get(key)


# ---------------------------------------------------------------------------
# CLI: match report
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Match SCDB records to ConvoKit transcripts.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show unmatched case details")
    parser.add_argument(
        "--source",
        choices=["corpus", "files"],
        default="files",
        help="Source of ConvoKit case_ids: 'files' (from transcript filenames, fast) or 'corpus' (load full ConvoKit corpus, slow)",
    )
    args = parser.parse_args()

    if not os.path.isfile(SCDB_CASE_CSV):
        print(f"ERROR: SCDB case CSV not found at {SCDB_CASE_CSV}")
        print("Download from https://scdb.la.psu.edu/data/2025-release-01/")
        return

    print("Loading SCDB case-centered data...", flush=True)
    scdb_df = load_scdb_case()
    print(f"  {len(scdb_df)} SCDB rows, {scdb_df['match_key'].nunique()} unique docket keys")
    print(f"  Term range: {scdb_df['term'].min()}-{scdb_df['term'].max()}")

    if args.source == "corpus":
        print("Loading ConvoKit corpus (this may take a while)...", flush=True)
        convokit_ids = get_convokit_case_ids()
    else:
        print(f"Reading transcript filenames from {TRANSCRIPTS_DIR}...", flush=True)
        convokit_ids = get_transcript_case_ids()

    if not convokit_ids:
        print("ERROR: No ConvoKit case_ids found.")
        return

    print(f"  {len(convokit_ids)} unique ConvoKit case_ids")

    # Year range
    years = []
    for cid in convokit_ids:
        if "_" in cid:
            try:
                years.append(int(cid.split("_")[0]))
            except ValueError:
                pass
    if years:
        print(f"  Year range: {min(years)}-{max(years)}")

    print("\nMatching...", flush=True)
    result = match_cases(convokit_ids, scdb_df)

    n_matched = len(result["matched"])
    n_unmatched = len(result["unmatched"])
    n_total = n_matched + n_unmatched
    n_scdb_only = len(result["scdb_only"])
    pct = 100 * n_matched / n_total if n_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  MATCH RESULTS")
    print(f"{'='*60}")
    print(f"  ConvoKit cases:       {n_total:>6}")
    print(f"  Matched to SCDB:      {n_matched:>6}  ({pct:.1f}%)")
    print(f"  Unmatched:            {n_unmatched:>6}  ({100-pct:.1f}%)")
    print(f"  SCDB-only (no transcript, in year range): {n_scdb_only}")
    print(f"{'='*60}")

    if n_unmatched > 0:
        # Break down unmatched by decade
        unmatched_by_decade: dict[int, int] = {}
        for cid in result["unmatched"]:
            if "_" in cid:
                try:
                    decade = (int(cid.split("_")[0]) // 10) * 10
                    unmatched_by_decade[decade] = unmatched_by_decade.get(decade, 0) + 1
                except ValueError:
                    pass
        if unmatched_by_decade:
            print("\n  Unmatched by decade:")
            for decade in sorted(unmatched_by_decade):
                print(f"    {decade}s: {unmatched_by_decade[decade]}")

    if args.verbose and n_unmatched > 0:
        print(f"\n  Unmatched ConvoKit case_ids ({n_unmatched}):")
        for cid in sorted(result["unmatched"]):
            key = _convokit_case_id_to_key(cid)
            print(f"    {cid}  (normalized: {key})")

    if args.verbose and n_scdb_only > 0:
        print(f"\n  SCDB-only keys in year range (first 30 of {n_scdb_only}):")
        for key in result["scdb_only"][:30]:
            print(f"    {key}")


if __name__ == "__main__":
    main()

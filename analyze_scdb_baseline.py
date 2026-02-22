"""Test SCDB features + justice priors as logistic regression baseline.

Trains on pre-2019 data, evaluates on 2019. Reports vote-level and case-level accuracy.
"""

import glob
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Add project root for scdb_matcher import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "conversation_generator"))
from scdb_matcher import (
    load_scdb_case, get_scdb_case_row, _convokit_case_id_to_key,
    get_transcript_case_ids, ISSUE_AREA_LABELS,
)

DATA_DIR = "case_transcripts_cleaned"
VOTES_DELIMITER = "\n---\nJUSTICE VOTES:\n"
EVAL_YEAR = "2019"

KNOWN_JUSTICE_NAMES = {
    "White", "Jr.", "Scalia", "Stevens", "Frankfurter", "Rehnquist",
    "Stewart", "Black", "Burger", "Warren", "Marshall", "Breyer",
    "Ginsburg", "Kennedy", "O'Connor", "Souter", "Sotomayor", "Fortas",
    "Blackmun", "Kagan", "Whittaker", "Douglas", "Goldberg", "Clark",
    "Gorsuch", "Reed", "Kavanaugh", "Burton", "Thomas", "Minton", "II",
    "Harlan",
}
PARTY_LABELS = {"Petitioner", "Respondent"}
ALL_SPEAKER_LABELS = KNOWN_JUSTICE_NAMES | PARTY_LABELS | {"Unknown"}


@dataclass
class Turn:
    speaker_label: str
    text: str
    speaker_type: str


def load_transcripts(data_dir):
    samples = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path) as f:
            text = f.read()
        if VOTES_DELIMITER not in text:
            continue
        idx = text.index(VOTES_DELIMITER)
        transcript = text[:idx]
        votes_text = text[idx + len(VOTES_DELIMITER):].strip()
        votes = {}
        for line in votes_text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("OUTCOME:"):
                continue
            if ": " not in line:
                continue
            name, side = line.rsplit(": ", 1)
            side = side.strip().lower()
            if side == "petitioner":
                votes[name.strip()] = 0
            elif side == "respondent":
                votes[name.strip()] = 1
        if votes:
            samples.append({
                "transcript": transcript,
                "votes": votes,
                "filename": os.path.basename(path),
            })
    return samples


def classify_speaker(label):
    if label == "Petitioner":
        return "petitioner"
    elif label == "Respondent":
        return "respondent"
    elif label in KNOWN_JUSTICE_NAMES:
        return "justice"
    return "unknown"


def parse_turns(transcript):
    lines = transcript.split("\n")
    turns = []
    current_speaker = None
    current_text_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in ALL_SPEAKER_LABELS:
            if current_speaker is not None and current_text_lines:
                turns.append(Turn(current_speaker, " ".join(current_text_lines),
                                  classify_speaker(current_speaker)))
            current_speaker = stripped
            current_text_lines = []
        else:
            current_text_lines.append(stripped)
    if current_speaker is not None and current_text_lines:
        turns.append(Turn(current_speaker, " ".join(current_text_lines),
                          classify_speaker(current_speaker)))
    return turns


def split_phases(turns):
    for i, turn in enumerate(turns):
        if turn.speaker_type == "respondent" and len(turn.text.split()) >= 20:
            return turns[:i], turns[i:]
    return turns, []


def case_result(votes):
    pet = sum(1 for v in votes.values() if v == 0)
    resp = sum(1 for v in votes.values() if v == 1)
    if pet > resp:
        return 0
    elif resp > pet:
        return 1
    return None


def filename_to_case_id(filename):
    """Convert transcript filename to case_id for SCDB matching."""
    base = filename.replace(".txt", "")
    parts = base.split("_")
    # Strip trailing convo_id if present
    if len(parts) >= 3:
        last = parts[-1]
        if last.isdigit() and len(last) >= 4:
            base = "_".join(parts[:-1])
    return base


def get_scdb_features(case_id):
    """Extract numeric SCDB features for a case. Returns dict or None."""
    row = get_scdb_case_row(case_id)
    if row is None:
        return None

    def safe_int(val, default=-1):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return int(val)

    return {
        "issueArea": safe_int(row.get("issueArea")),
        "petitioner": safe_int(row.get("petitioner")),
        "respondent": safe_int(row.get("respondent")),
        "jurisdiction": safe_int(row.get("jurisdiction")),
        "certReason": safe_int(row.get("certReason")),
        "lcDisposition": safe_int(row.get("lcDisposition")),
        "lcDispositionDirection": safe_int(row.get("lcDispositionDirection")),
        "lcDisagreement": safe_int(row.get("lcDisagreement")),
        "lawType": safe_int(row.get("lawType")),
    }


def build_rows(samples, justice_prior, include_scdb=True, include_qcounts=True):
    """Build feature rows: one per (case, justice) pair.

    Returns list of dicts with features + 'vote' label.
    """
    rows = []
    scdb_miss = 0
    scdb_hit = 0

    for s in samples:
        case_id = filename_to_case_id(s["filename"])

        # SCDB features
        scdb = get_scdb_features(case_id) if include_scdb else None
        if include_scdb and scdb is None:
            scdb_miss += 1
            continue  # skip cases without SCDB match
        if scdb is not None:
            scdb_hit += 1

        # Question counts
        turns = parse_turns(s["transcript"])
        pet_turns, resp_turns = split_phases(turns)

        for name, vote in s["votes"].items():
            row = {
                "justice": name,
                "vote": vote,
                "filename": s["filename"],
                # Justice prior
                "justice_prior": justice_prior.get(name, 0.5),
            }

            if include_scdb and scdb is not None:
                row.update(scdb)

            if include_qcounts:
                pet_q = sum(1 for t in pet_turns if t.speaker_label == name)
                resp_q = sum(1 for t in resp_turns if t.speaker_label == name)
                total_q = pet_q + resp_q
                row["pet_qcount"] = pet_q
                row["resp_qcount"] = resp_q
                row["total_qcount"] = total_q
                row["qcount_ratio"] = (pet_q / total_q) if total_q > 0 else 0.5

            rows.append(row)

    if include_scdb:
        print(f"  SCDB match: {scdb_hit} cases matched, {scdb_miss} missed")
    return rows


def evaluate_case_accuracy(df):
    """From a df with filename, pred, vote columns, compute case-level greedy accuracy."""
    correct = 0
    total = 0
    for fname, group in df.groupby("filename"):
        true_votes = dict(zip(group["justice"], group["vote"]))
        pred_votes = dict(zip(group["justice"], group["pred"]))
        true_result = case_result(true_votes)
        pred_result = case_result(pred_votes)
        if true_result is not None and pred_result is not None:
            correct += int(true_result == pred_result)
            total += 1
    return correct, total


def run_experiment(name, train_rows, eval_rows, feature_cols):
    """Train logistic regression and evaluate."""
    train_df = pd.DataFrame(train_rows)
    eval_df = pd.DataFrame(eval_rows)

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["vote"].values
    X_eval = eval_df[feature_cols].values.astype(float)
    y_eval = eval_df["vote"].values

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)

    # Train
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_eval)
    vote_acc = accuracy_score(y_eval, y_pred)

    eval_df = eval_df.copy()
    eval_df["pred"] = y_pred
    case_correct, case_total = evaluate_case_accuracy(eval_df)
    case_acc = case_correct / case_total if case_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Vote accuracy: {vote_acc:.3f} ({int(vote_acc * len(y_eval))}/{len(y_eval)})")
    print(f"  Case accuracy: {case_acc:.3f} ({case_correct}/{case_total})")

    # Feature importances
    if len(feature_cols) <= 20:
        coefs = sorted(zip(feature_cols, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Feature coefficients (toward Respondent):")
        for feat, coef in coefs:
            print(f"    {feat:30s}: {coef:+.4f}")

    return vote_acc, case_acc


def main():
    print("Loading transcripts...")
    samples = load_transcripts(DATA_DIR)
    train = [s for s in samples if not s["filename"].startswith(EVAL_YEAR)]
    eval_ = [s for s in samples if s["filename"].startswith(EVAL_YEAR)]
    print(f"Train: {len(train)} cases, Eval: {len(eval_)} cases")

    # Justice priors from training set
    justice_pet = defaultdict(int)
    justice_total = defaultdict(int)
    for s in train:
        for name, vote in s["votes"].items():
            justice_total[name] += 1
            if vote == 0:
                justice_pet[name] += 1
    justice_prior = {n: justice_pet[n] / justice_total[n] for n in justice_total}

    scdb_feature_cols = [
        "issueArea", "petitioner", "respondent", "jurisdiction",
        "certReason", "lcDisposition", "lcDispositionDirection",
        "lcDisagreement", "lawType",
    ]
    qcount_cols = ["pet_qcount", "resp_qcount", "total_qcount", "qcount_ratio"]

    # ── Experiment 1: Justice prior only ──
    print("\nBuilding rows: prior only...")
    train_rows = build_rows(train, justice_prior, include_scdb=False, include_qcounts=False)
    eval_rows = build_rows(eval_, justice_prior, include_scdb=False, include_qcounts=False)
    run_experiment("Justice Prior Only", train_rows, eval_rows, ["justice_prior"])

    # ── Experiment 2: Justice prior + question counts ──
    print("\nBuilding rows: prior + question counts...")
    train_rows = build_rows(train, justice_prior, include_scdb=False, include_qcounts=True)
    eval_rows = build_rows(eval_, justice_prior, include_scdb=False, include_qcounts=True)
    run_experiment("Prior + Question Counts", train_rows, eval_rows,
                   ["justice_prior"] + qcount_cols)

    # ── Experiment 3: SCDB features only ──
    print("\nBuilding rows: SCDB only...")
    train_rows = build_rows(train, justice_prior, include_scdb=True, include_qcounts=False)
    eval_rows = build_rows(eval_, justice_prior, include_scdb=True, include_qcounts=False)
    run_experiment("SCDB Features Only", train_rows, eval_rows, scdb_feature_cols)

    # ── Experiment 4: Justice prior + SCDB ──
    print("\nBuilding rows: prior + SCDB...")
    train_rows = build_rows(train, justice_prior, include_scdb=True, include_qcounts=False)
    eval_rows = build_rows(eval_, justice_prior, include_scdb=True, include_qcounts=False)
    run_experiment("Prior + SCDB", train_rows, eval_rows,
                   ["justice_prior"] + scdb_feature_cols)

    # ── Experiment 5: Justice prior + SCDB + question counts ──
    print("\nBuilding rows: prior + SCDB + question counts...")
    train_rows = build_rows(train, justice_prior, include_scdb=True, include_qcounts=True)
    eval_rows = build_rows(eval_, justice_prior, include_scdb=True, include_qcounts=True)
    run_experiment("Prior + SCDB + Question Counts", train_rows, eval_rows,
                   ["justice_prior"] + scdb_feature_cols + qcount_cols)

    # ── Experiment 6: Per-justice model with SCDB ──
    # One-hot encode justice identity so the model learns per-justice interactions with SCDB
    print("\nBuilding rows: justice identity + SCDB + qcounts...")
    train_rows = build_rows(train, justice_prior, include_scdb=True, include_qcounts=True)
    eval_rows = build_rows(eval_, justice_prior, include_scdb=True, include_qcounts=True)

    train_df = pd.DataFrame(train_rows)
    eval_df = pd.DataFrame(eval_rows)

    # One-hot justice names (only justices seen in both train and eval)
    eval_justices = set(eval_df["justice"].unique())
    train_justices = set(train_df["justice"].unique())
    common_justices = sorted(eval_justices & train_justices)
    print(f"  Common justices (train & eval): {common_justices}")

    justice_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    justice_ohe.fit(train_df[["justice"]])

    X_train_justice = justice_ohe.transform(train_df[["justice"]])
    X_eval_justice = justice_ohe.transform(eval_df[["justice"]])

    all_feature_cols = ["justice_prior"] + scdb_feature_cols + qcount_cols
    X_train_feats = train_df[all_feature_cols].values.astype(float)
    X_eval_feats = eval_df[all_feature_cols].values.astype(float)

    scaler = StandardScaler()
    X_train_feats = scaler.fit_transform(X_train_feats)
    X_eval_feats = scaler.transform(X_eval_feats)

    X_train_full = np.hstack([X_train_justice, X_train_feats])
    X_eval_full = np.hstack([X_eval_justice, X_eval_feats])

    y_train = train_df["vote"].values
    y_eval = eval_df["vote"].values

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train_full, y_train)

    y_pred = clf.predict(X_eval_full)
    vote_acc = accuracy_score(y_eval, y_pred)

    eval_df = eval_df.copy()
    eval_df["pred"] = y_pred
    case_correct, case_total = evaluate_case_accuracy(eval_df)
    case_acc = case_correct / case_total if case_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Justice OHE + SCDB + Prior + QCounts")
    print(f"{'='*60}")
    print(f"  Vote accuracy: {vote_acc:.3f} ({int(vote_acc * len(y_eval))}/{len(y_eval)})")
    print(f"  Case accuracy: {case_acc:.3f} ({case_correct}/{case_total})")

    # Show top features
    justice_names = list(justice_ohe.get_feature_names_out())
    all_names = justice_names + all_feature_cols
    coefs = sorted(zip(all_names, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Top 15 features (toward Respondent):")
    for feat, coef in coefs[:15]:
        print(f"    {feat:30s}: {coef:+.4f}")

    # ── Experiment 7: Sweep regularization ──
    print(f"\n{'='*60}")
    print(f"  Regularization sweep (Justice OHE + SCDB + Prior + QCounts)")
    print(f"{'='*60}")
    for C in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs")
        clf.fit(X_train_full, y_train)
        y_pred = clf.predict(X_eval_full)
        va = accuracy_score(y_eval, y_pred)
        eval_df_tmp = eval_df.copy()
        eval_df_tmp["pred"] = y_pred
        cc, ct = evaluate_case_accuracy(eval_df_tmp)
        ca = cc / ct if ct > 0 else 0
        print(f"  C={C:>6.3f}: vote={va:.3f}, case={ca:.3f} ({cc}/{ct})")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Always-petitioner case acc:          0.618")
    print(f"  (check experiments above for improvements)")


if __name__ == "__main__":
    main()

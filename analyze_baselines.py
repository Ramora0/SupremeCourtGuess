"""Analyze how predictive question counts + justice priors are for case outcomes."""

import glob
import os
from collections import defaultdict
from dataclasses import dataclass

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


def main():
    samples = load_transcripts(DATA_DIR)
    train = [s for s in samples if not s["filename"].startswith(EVAL_YEAR)]
    eval_ = [s for s in samples if s["filename"].startswith(EVAL_YEAR)]
    print(f"Train: {len(train)}, Eval: {len(eval_)}")

    # ── 1. Justice priors from training set ──
    justice_pet_votes = defaultdict(int)  # times justice voted petitioner
    justice_total_votes = defaultdict(int)
    for s in train:
        for name, vote in s["votes"].items():
            justice_total_votes[name] += 1
            if vote == 0:
                justice_pet_votes[name] += 1

    justice_prior = {}  # P(petitioner) per justice
    for name in justice_total_votes:
        justice_prior[name] = justice_pet_votes[name] / justice_total_votes[name]

    print("\n── Justice priors (P(petitioner)) from training set ──")
    for name in sorted(justice_prior, key=justice_prior.get, reverse=True):
        n = justice_total_votes[name]
        if n >= 10:  # only show justices with enough data
            print(f"  {name:15s}: {justice_prior[name]:.3f}  (n={n})")

    # ── 2. Evaluate different strategies on eval set ──

    # Strategy A: Pure justice prior (predict petitioner if prior > 0.5)
    prior_vote_correct = 0
    prior_vote_total = 0
    prior_case_correct = 0
    prior_case_total = 0

    for s in eval_:
        true_result = case_result(s["votes"])
        if true_result is None:
            continue
        pred_votes = {}
        for name, true_vote in s["votes"].items():
            p_pet = justice_prior.get(name, 0.5)
            pred = 0 if p_pet > 0.5 else 1
            pred_votes[name] = pred
            prior_vote_correct += int(pred == true_vote)
            prior_vote_total += 1
        pred_result = case_result(pred_votes)
        if pred_result is not None:
            prior_case_correct += int(pred_result == true_result)
            prior_case_total += 1

    print(f"\n── Strategy A: Justice prior only ──")
    print(f"  Vote accuracy: {prior_vote_correct}/{prior_vote_total} = "
          f"{prior_vote_correct/prior_vote_total:.3f}")
    print(f"  Case accuracy: {prior_case_correct}/{prior_case_total} = "
          f"{prior_case_correct/prior_case_total:.3f}")

    # Strategy B: Always predict petitioner (majority class?)
    pet_case = 0
    resp_case = 0
    for s in eval_:
        r = case_result(s["votes"])
        if r == 0:
            pet_case += 1
        elif r == 1:
            resp_case += 1
    print(f"\n── Eval set class balance ──")
    print(f"  Petitioner wins: {pet_case}, Respondent wins: {resp_case}")
    print(f"  Always-petitioner case acc: {pet_case}/{pet_case+resp_case} = "
          f"{pet_case/(pet_case+resp_case):.3f}")

    # Strategy C: Question count ratio → vote prediction
    # Research: more questions to side X → vote against X
    # So if justice asks more during pet phase → votes respondent (against pet)
    qcount_vote_correct = 0
    qcount_vote_total = 0
    qcount_case_correct = 0
    qcount_case_total = 0
    qcount_applicable = 0  # justices where we had count signal

    for s in eval_:
        true_result = case_result(s["votes"])
        if true_result is None:
            continue
        turns = parse_turns(s["transcript"])
        pet_turns, resp_turns = split_phases(turns)

        pred_votes = {}
        for name, true_vote in s["votes"].items():
            pet_count = sum(1 for t in pet_turns if t.speaker_label == name)
            resp_count = sum(1 for t in resp_turns if t.speaker_label == name)

            if pet_count != resp_count:
                # More questions to pet side → vote respondent (against pet)
                pred = 1 if pet_count > resp_count else 0
                qcount_applicable += 1
            else:
                # Tie: fall back to justice prior
                p_pet = justice_prior.get(name, 0.5)
                pred = 0 if p_pet > 0.5 else 1

            pred_votes[name] = pred
            qcount_vote_correct += int(pred == true_vote)
            qcount_vote_total += 1

        pred_result = case_result(pred_votes)
        if pred_result is not None:
            qcount_case_correct += int(pred_result == true_result)
            qcount_case_total += 1

    print(f"\n── Strategy C: Question count (more Qs → vote against) + prior fallback ──")
    print(f"  Vote accuracy: {qcount_vote_correct}/{qcount_vote_total} = "
          f"{qcount_vote_correct/qcount_vote_total:.3f}")
    print(f"  Case accuracy: {qcount_case_correct}/{qcount_case_total} = "
          f"{qcount_case_correct/qcount_case_total:.3f}")
    print(f"  Justices with count signal: {qcount_applicable}/{qcount_vote_total}")

    # Strategy D: Question count only (no prior fallback — random on ties)
    import random
    random.seed(42)
    qonly_vote_correct = 0
    qonly_vote_total = 0
    qonly_case_correct = 0
    qonly_case_total = 0

    for s in eval_:
        true_result = case_result(s["votes"])
        if true_result is None:
            continue
        turns = parse_turns(s["transcript"])
        pet_turns, resp_turns = split_phases(turns)

        pred_votes = {}
        for name, true_vote in s["votes"].items():
            pet_count = sum(1 for t in pet_turns if t.speaker_label == name)
            resp_count = sum(1 for t in resp_turns if t.speaker_label == name)

            if pet_count != resp_count:
                pred = 1 if pet_count > resp_count else 0
            else:
                pred = random.randint(0, 1)

            pred_votes[name] = pred
            qonly_vote_correct += int(pred == true_vote)
            qonly_vote_total += 1

        pred_result = case_result(pred_votes)
        if pred_result is not None:
            qonly_case_correct += int(pred_result == true_result)
            qonly_case_total += 1

    print(f"\n── Strategy D: Question count only (random on ties) ──")
    print(f"  Vote accuracy: {qonly_vote_correct}/{qonly_vote_total} = "
          f"{qonly_vote_correct/qonly_vote_total:.3f}")
    print(f"  Case accuracy: {qonly_case_correct}/{qonly_case_total} = "
          f"{qonly_case_correct/qonly_case_total:.3f}")

    # Strategy E: Combined — weighted score from prior + question count
    print(f"\n── Strategy E: Weighted prior + question count ──")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        comb_vote_correct = 0
        comb_vote_total = 0
        comb_case_correct = 0
        comb_case_total = 0

        for s in eval_:
            true_result = case_result(s["votes"])
            if true_result is None:
                continue
            turns = parse_turns(s["transcript"])
            pet_turns, resp_turns = split_phases(turns)

            pred_votes = {}
            for name, true_vote in s["votes"].items():
                # Prior score: P(petitioner)
                p_pet_prior = justice_prior.get(name, 0.5)

                # Question count score
                pet_count = sum(1 for t in pet_turns if t.speaker_label == name)
                resp_count = sum(1 for t in resp_turns if t.speaker_label == name)
                total_q = pet_count + resp_count
                if total_q > 0:
                    # Higher ratio of pet questions → more likely resp vote
                    # So P(petitioner) from counts = resp_count / total
                    p_pet_count = resp_count / total_q
                else:
                    p_pet_count = 0.5

                # Weighted combination
                p_pet = alpha * p_pet_count + (1 - alpha) * p_pet_prior
                pred = 0 if p_pet > 0.5 else 1

                pred_votes[name] = pred
                comb_vote_correct += int(pred == true_vote)
                comb_vote_total += 1

            pred_result = case_result(pred_votes)
            if pred_result is not None:
                comb_case_correct += int(pred_result == true_result)
                comb_case_total += 1

        print(f"  alpha={alpha:.2f} (count weight): "
              f"vote={comb_vote_correct/comb_vote_total:.3f}, "
              f"case={comb_case_correct/comb_case_total:.3f}")

    # ── Detailed per-case breakdown for eval ──
    print(f"\n── Per-case breakdown (eval, Strategy C) ──")
    for s in eval_:
        true_result = case_result(s["votes"])
        if true_result is None:
            continue
        turns = parse_turns(s["transcript"])
        pet_turns, resp_turns = split_phases(turns)

        correct_votes = 0
        total_votes = len(s["votes"])
        details = []
        for name, true_vote in s["votes"].items():
            pet_count = sum(1 for t in pet_turns if t.speaker_label == name)
            resp_count = sum(1 for t in resp_turns if t.speaker_label == name)
            if pet_count != resp_count:
                pred = 1 if pet_count > resp_count else 0
            else:
                p_pet = justice_prior.get(name, 0.5)
                pred = 0 if p_pet > 0.5 else 1
            ok = pred == true_vote
            correct_votes += int(ok)
            vote_str = "Pet" if true_vote == 0 else "Resp"
            pred_str = "Pet" if pred == 0 else "Resp"
            mark = "✓" if ok else "✗"
            details.append(f"    {name:12s}: pet_q={pet_count} resp_q={resp_count} "
                          f"→ pred={pred_str:4s} actual={vote_str:4s} {mark}")

        result_str = "Petitioner" if true_result == 0 else "Respondent"
        print(f"\n  {s['filename']} — Winner: {result_str} "
              f"({correct_votes}/{total_votes} votes correct)")
        for d in details:
            print(d)


if __name__ == "__main__":
    main()

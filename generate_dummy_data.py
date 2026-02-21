"""Generate dummy transcript data for memory testing."""

import os
import random

OUTPUT_DIR = "data/transcripts"
NUM_FILES = 20

JUSTICES = [
    "Roberts", "Thomas", "Alito", "Sotomayor", "Kagan",
    "Gorsuch", "Kavanaugh", "Barrett", "Jackson",
]
SPEAKERS = JUSTICES + ["Petitioner Counsel", "Respondent Counsel"]
SIDES = ["petitioner", "respondent"]

# Filler sentences to bulk up transcripts
FILLER = [
    "Could you clarify your position on that point?",
    "I think the precedent in this case is quite clear.",
    "But what about the constitutional implications?",
    "The lower court ruled differently on this matter.",
    "I'd like to return to the question of standing.",
    "How do you reconcile that with the statute?",
    "That seems to be a very broad interpretation.",
    "Let me ask you about the remedy you're seeking.",
    "The government's position is that this falls under federal jurisdiction.",
    "I'm not sure the record supports that conclusion.",
    "We believe the text of the statute is unambiguous.",
    "Your Honor, the facts of this case are distinguishable.",
    "Can you point to where in the record that appears?",
    "The amicus brief raises an interesting point here.",
    "I think we need to consider the practical consequences.",
]


def generate_transcript(target_words: int) -> str:
    lines = []
    word_count = 0
    while word_count < target_words:
        speaker = random.choice(SPEAKERS)
        # Each turn is 1-4 sentences
        num_sentences = random.randint(1, 4)
        sentences = " ".join(random.choice(FILLER) for _ in range(num_sentences))
        line = f"{speaker}: {sentences}"
        lines.append(line)
        word_count += len(line.split())

    transcript = "<transcript>\n" + "\n".join(lines) + "\n</transcript>"

    votes = []
    for justice in JUSTICES:
        side = random.choice(SIDES)
        votes.append(f"{justice} voted for the {side}")
    vote_block = "\n".join(votes)

    return f"{transcript}\n<votes>\n{vote_block}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ~12k words ≈ ~16k tokens, good for testing near the 16384 limit
    target_words = 12000

    for i in range(NUM_FILES):
        text = generate_transcript(target_words)
        path = os.path.join(OUTPUT_DIR, f"dummy_case_{i:03d}.txt")
        with open(path, "w") as f:
            f.write(text)
        word_count = len(text.split())
        print(f"Wrote {path} ({word_count} words)")

    print(f"\nGenerated {NUM_FILES} dummy files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

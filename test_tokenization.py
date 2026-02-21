"""Tokenization statistics for all transcript .txt files.

Uses the same Qwen2.5-7B tokenizer from lees-branch train.py to tokenize
every file, then displays a comprehensive matplotlib dashboard of statistics.
"""

import glob
import os
import sys
import time

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# ── Config (mirrors lees-branch train.py) ─────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-7B"
TOKENS_PER_10S = 16_000  # training throughput estimate
MAX_USABLE_TOKENS = 15_000  # examples above this are discarded

DATA_DIRS = [
    "case_transcripts"
]


# ── Collect files ─────────────────────────────────────────────────────────────


def collect_txt_files(dirs: list[str]) -> list[str]:
    """Gather all .txt file paths from the given directories."""
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[WARN] Directory not found, skipping: {d}")
            continue
        found = sorted(glob.glob(os.path.join(d, "*.txt")))
        print(f"  Found {len(found):,} files in {d}/")
        paths.extend(found)
    if not paths:
        print("ERROR: No .txt files found in any directory.")
        sys.exit(1)
    return paths


# ── Tokenize ──────────────────────────────────────────────────────────────────


def tokenize_files(paths: list[str], tokenizer, batch_size: int = 256) -> dict:
    """Tokenize files in batches for speed."""
    # Read all files first
    texts = []
    char_counts = []
    filenames = []
    skipped = 0

    print("  Reading files ...")
    for path in paths:
        with open(path) as f:
            text = f.read()
        if not text.strip():
            skipped += 1
            continue
        texts.append(text)
        char_counts.append(len(text))
        filenames.append(os.path.basename(path))

    # Batch tokenize
    token_counts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing",
                  unit="batch"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False,
                            return_attention_mask=False)
        token_counts.extend(len(ids) for ids in encoded["input_ids"])

    print(f"  Tokenized {len(token_counts):,} files ({skipped} empty/skipped)")
    return {
        "token_counts": np.array(token_counts),
        "char_counts": np.array(char_counts),
        "filenames": filenames,
    }


# ── Statistics ────────────────────────────────────────────────────────────────


def compute_stats(arr: np.ndarray) -> dict:
    """Compute comprehensive statistics on an array."""
    q = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "count": len(arr),
        "total": int(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p5": float(q[0]),
        "p10": float(q[1]),
        "q1": float(q[2]),
        "q2": float(q[3]),
        "q3": float(q[4]),
        "p90": float(q[5]),
        "p95": float(q[6]),
        "p99": float(q[7]),
        "iqr": float(q[4] - q[2]),
        "range": int(arr.max() - arr.min()),
    }


def fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


# ── Dashboard ─────────────────────────────────────────────────────────────────


def build_dashboard(stats: dict, token_counts: np.ndarray, char_counts: np.ndarray,
                    filenames: list[str]):
    """Build a matplotlib figure with tokenization stats."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Discard stats
    n_discard = int(np.sum(token_counts > MAX_USABLE_TOKENS))
    pct_discard = n_discard / len(token_counts) * 100
    tokens_discarded = int(token_counts[token_counts > MAX_USABLE_TOKENS].sum())
    kept_counts = token_counts[token_counts <= MAX_USABLE_TOKENS]
    kept_total = int(kept_counts.sum())

    total_tokens = stats["total"]
    est_all_sec = total_tokens / TOKENS_PER_10S * 10
    est_kept_sec = kept_total / TOKENS_PER_10S * 10

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        "Supreme Court Transcript — Tokenization Analysis\n"
        f"(Qwen2.5-7B tokenizer  ·  {stats['count']:,} files)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.35,
                  height_ratios=[0.8, 2, 2])

    # ── 0. TRAINING TIME BANNER (top row, full width) ─────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)

    # Big background box
    from matplotlib.patches import FancyBboxPatch
    bg = FancyBboxPatch((0.02, 0.05), 0.96, 0.90, boxstyle="round,pad=0.02",
                        facecolor="#1a1a2e", edgecolor="#16213e", linewidth=2)
    ax0.add_patch(bg)

    ax0.text(0.5, 0.82, "ESTIMATED TRAINING TIME", ha="center", va="center",
             fontsize=14, fontweight="bold", color="#aaa", family="monospace")

    # All files
    ax0.text(0.25, 0.55, "ALL FILES", ha="center", va="center",
             fontsize=10, color="#888", fontweight="bold")
    ax0.text(0.25, 0.32, f"1 epoch: {fmt_time(est_all_sec)}",
             ha="center", va="center", fontsize=18, fontweight="bold", color="#e0e0e0")
    ax0.text(0.25, 0.13, f"3 epochs: {fmt_time(est_all_sec * 3)}",
             ha="center", va="center", fontsize=14, color="#999")

    # Separator
    ax0.plot([0.5, 0.5], [0.12, 0.72], color="#444", lw=1.5, ls="--")

    # Kept files only
    ax0.text(0.75, 0.55, f"KEPT ONLY (<{MAX_USABLE_TOKENS:,} tok)", ha="center",
             va="center", fontsize=10, color="#888", fontweight="bold")
    ax0.text(0.75, 0.32, f"1 epoch: {fmt_time(est_kept_sec)}",
             ha="center", va="center", fontsize=18, fontweight="bold", color="#4fc3f7")
    ax0.text(0.75, 0.13, f"3 epochs: {fmt_time(est_kept_sec * 3)}",
             ha="center", va="center", fontsize=14, color="#81d4fa")

    # Rate annotation
    ax0.text(0.5, 0.02, f"@ {TOKENS_PER_10S:,} tokens / 10s",
             ha="center", va="center", fontsize=9, color="#666", style="italic")

    # ── 1. Histogram of token counts ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0:2])
    n_bins = min(100, max(30, stats["count"] // 50))
    ax1.hist(token_counts, bins=n_bins, color="#4C72B0", edgecolor="black",
             linewidth=0.3, alpha=0.85)
    ax1.axvline(stats["mean"], color="red", ls="--", lw=1.5, label=f"Mean: {stats['mean']:,.0f}")
    ax1.axvline(stats["median"], color="orange", ls="--", lw=1.5, label=f"Median: {stats['median']:,.0f}")
    ax1.axvline(MAX_USABLE_TOKENS, color="black", ls="-", lw=2,
                label=f"Cutoff: {MAX_USABLE_TOKENS:,} ({pct_discard:.1f}% discarded)")
    ax1.axvspan(MAX_USABLE_TOKENS, token_counts.max() * 1.02, alpha=0.15, color="red")
    ax1.set_title("Token Count Distribution", fontweight="bold")
    ax1.set_xlabel("Tokens per file")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)

    # ── 2. Top 20 longest files ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2:4])
    top_idx = np.argsort(token_counts)[::-1][:20]
    top_tokens = token_counts[top_idx]
    top_labels = [os.path.splitext(filenames[i])[0] for i in top_idx]
    bar_colors = ["#C44E52" if v > MAX_USABLE_TOKENS else "#DD8452"
                  for v in top_tokens]
    bars = ax2.barh(range(len(top_tokens)), top_tokens, color=bar_colors,
                    edgecolor="black", linewidth=0.3)
    ax2.axvline(MAX_USABLE_TOKENS, color="black", ls="-", lw=2, label=f"Cutoff: {MAX_USABLE_TOKENS:,}")
    ax2.set_yticks(range(len(top_tokens)))
    ax2.set_yticklabels(top_labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_title("Top 20 Longest Files (red = discarded)", fontweight="bold")
    ax2.set_xlabel("Tokens")
    ax2.legend(fontsize=8)
    for i, (bar, val) in enumerate(zip(bars, top_tokens)):
        ax2.text(val + token_counts.max() * 0.01, i, f"{val:,}", va="center", fontsize=7)

    # ── 3. Cases by year ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0:2])
    years = [os.path.splitext(f)[0].split("_")[0] for f in filenames]
    unique_years = sorted(set(years))
    year_counts = [years.count(y) for y in unique_years]
    # Color bars by mean token count per year
    year_mean_tokens = []
    for y in unique_years:
        mask = np.array([yr == y for yr in years])
        year_mean_tokens.append(token_counts[mask].mean())
    norm = plt.Normalize(vmin=min(year_mean_tokens), vmax=max(year_mean_tokens))
    bar_colors = plt.cm.RdYlGn_r(norm(year_mean_tokens))
    ax3.bar(unique_years, year_counts, color=bar_colors, edgecolor="black", linewidth=0.3)
    # Show every Nth label to avoid crowding
    step = max(1, len(unique_years) // 15)
    ax3.set_xticks(unique_years[::step])
    ax3.tick_params(axis="x", rotation=45, labelsize=7)
    ax3.set_title("Cases by Year (color = mean tokens)", fontweight="bold")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Number of Cases")
    ax3.grid(axis="y", alpha=0.3)

    # ── 4. Summary statistics table ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 2:4])
    ax4.axis("off")

    table_data = [
        ["Total Files", f"{stats['count']:,}"],
        ["Total Tokens (all)", f"{stats['total']:,}"],
        ["Mean", f"{stats['mean']:,.1f}"],
        ["Std Dev", f"{stats['std']:,.1f}"],
        ["Median", f"{stats['median']:,.1f}"],
        ["Min", f"{stats['min']:,}"],
        ["Max", f"{stats['max']:,}"],
        ["IQR (Q3-Q1)", f"{stats['iqr']:,.1f}"],
        ["Q1 (25th)", f"{stats['q1']:,.1f}"],
        ["Q3 (75th)", f"{stats['q3']:,.1f}"],
        ["P95", f"{stats['p95']:,.1f}"],
        ["P99", f"{stats['p99']:,.1f}"],
        ["", ""],
        [f"Cutoff", f"{MAX_USABLE_TOKENS:,} tokens"],
        ["Discarded Files", f"{n_discard:,} / {stats['count']:,}"],
        ["% Discarded", f"{pct_discard:.2f}%"],
        ["Tokens Discarded", f"{tokens_discarded:,}"],
        ["Kept Files", f"{stats['count'] - n_discard:,}"],
        ["Kept Tokens", f"{kept_total:,}"],
    ]

    table = ax4.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
        colWidths=[0.45, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[i, j].set_facecolor("#f0f0f0")
            else:
                table[i, j].set_facecolor("#ffffff")
    # Highlight discard rows in red
    for i in range(14, len(table_data) + 1):
        for j in range(2):
            table[i, j].set_facecolor("#fdd")

    ax4.set_title("Summary Statistics & Training Estimates", fontweight="bold",
                  pad=15)

    plt.savefig("tokenization_stats.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to tokenization_stats.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  Tokenization Analysis — Supreme Court Transcripts")
    print("=" * 60)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"  Tokenizer loaded in {time.time() - t0:.1f}s")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    print(f"\nCollecting .txt files ...")
    paths = collect_txt_files(DATA_DIRS)
    print(f"  Total: {len(paths):,} files")

    print(f"\nTokenizing ...")
    t0 = time.time()
    result = tokenize_files(paths, tokenizer)
    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.1f}s "
          f"({len(paths) / elapsed:.0f} files/sec)")

    token_counts = result["token_counts"]
    char_counts = result["char_counts"]

    print(f"\nComputing statistics ...")
    stats = compute_stats(token_counts)

    # Print summary to terminal too
    print(f"\n{'─' * 50}")
    print(f"  Files:       {stats['count']:>10,}")
    print(f"  Total tokens:{stats['total']:>10,}")
    print(f"  Mean:        {stats['mean']:>10,.1f}")
    print(f"  Std:         {stats['std']:>10,.1f}")
    print(f"  Median:      {stats['median']:>10,.1f}")
    print(f"  Min:         {stats['min']:>10,}")
    print(f"  Max:         {stats['max']:>10,}")
    print(f"  Q1:          {stats['q1']:>10,.1f}")
    print(f"  Q3:          {stats['q3']:>10,.1f}")
    print(f"  IQR:         {stats['iqr']:>10,.1f}")
    print(f"  P95:         {stats['p95']:>10,.1f}")
    print(f"  P99:         {stats['p99']:>10,.1f}")
    n_discard = int(np.sum(token_counts > MAX_USABLE_TOKENS))
    pct_discard = n_discard / len(token_counts) * 100
    kept_total = int(token_counts[token_counts <= MAX_USABLE_TOKENS].sum())
    est_all_sec = stats["total"] / TOKENS_PER_10S * 10
    est_kept_sec = kept_total / TOKENS_PER_10S * 10
    print(f"{'─' * 50}")
    print(f"  Cutoff:      {MAX_USABLE_TOKENS:>10,} tokens")
    print(f"  Discarded:   {n_discard:>10,} files ({pct_discard:.2f}%)")
    print(f"  Kept files:  {stats['count'] - n_discard:>10,}")
    print(f"  Kept tokens: {kept_total:>10,}")
    print(f"{'─' * 50}")
    print(f"  Training estimate @ {TOKENS_PER_10S:,} tok / 10s:")
    print(f"    ALL FILES:  1 epoch = {fmt_time(est_all_sec)},  3 epochs = {fmt_time(est_all_sec * 3)}")
    print(f"    KEPT ONLY:  1 epoch = {fmt_time(est_kept_sec)},  3 epochs = {fmt_time(est_kept_sec * 3)}")
    print(f"{'─' * 50}")

    print(f"\nBuilding dashboard ...")
    build_dashboard(stats, token_counts, char_counts, result["filenames"])


if __name__ == "__main__":
    main()

"""
Step 6: Build the golden evaluation set review sheet.

Strategy for speed: rather than labeling from a blank sheet, we stratified-sample
across the already-predicted intent buckets (so rare intents aren't missed), and
PRE-FILL both the intent and an escalation guess using simple heuristics. Your job
becomes reviewing and correcting ~150-200 rows in a spreadsheet, not writing labels
from scratch. You are still the ground truth -- every correction you make (or don't
make) is your genuine judgment call, which is what makes this a real golden set.

Usage:
    python 06_build_golden_set.py --input data/classified.csv --target-n 180 \
        --out data/golden_review.csv
"""

import argparse
import pandas as pd

# crude, fast heuristic to pre-fill an escalation suggestion -- YOU are the final judge,
# this just saves typing for the obvious cases
ESCALATE_INTENTS = {"security_scam_report", "support_process_complaint"}
ANGRY_MARKERS = ["!!", "worst", "pathetic", "furious", "ridiculous", "thugs", "scam",
                  "unacceptable", "disgusting", "never again", "cheated"]


def guess_escalate(row):
    text = str(row["customer_text"]).lower()
    if row["llm_intent"] in ESCALATE_INTENTS:
        return "Y"
    if any(marker in text for marker in ANGRY_MARKERS):
        return "Y"
    if row["llm_intent"] == "other_uncategorized":
        return "Y"  # unclear/non-English -- safer to default to human review
    return "N"


def stratified_sample(df, target_n):
    intents = df["llm_intent"].unique()
    n_buckets = len(intents)
    per_bucket_target = max(1, target_n // n_buckets)

    parts = []
    for intent in intents:
        bucket = df[df["llm_intent"] == intent]
        n = min(len(bucket), per_bucket_target)
        parts.append(bucket.sample(n=n, random_state=3))

    sampled = pd.concat(parts)

    # if under target (small buckets), top up with random extra rows from what's left
    remaining_needed = target_n - len(sampled)
    if remaining_needed > 0:
        leftover = df.drop(sampled.index)
        top_up = leftover.sample(n=min(remaining_needed, len(leftover)), random_state=4)
        sampled = pd.concat([sampled, top_up])

    return sampled.sample(frac=1, random_state=5).reset_index(drop=True)  # shuffle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--target-n", type=int, default=180)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"[info] loaded {len(df)} classified messages")

    sampled = stratified_sample(df, args.target_n)
    print(f"[info] stratified sample: {len(sampled)} rows across {sampled['llm_intent'].nunique()} intents")
    print(sampled["llm_intent"].value_counts())

    review = pd.DataFrame({
        "thread_id": sampled["thread_id"],
        "customer_text": sampled["customer_text"],
        "brand_text": sampled["brand_text"],
        "llm_predicted_intent": sampled["llm_intent"],
        "gold_intent": sampled["llm_intent"],  # pre-filled with LLM guess -- correct if wrong
        "gold_escalate": sampled.apply(guess_escalate, axis=1),  # pre-filled heuristic guess
        "labeler_notes": "",  # optional: why you changed something, or anything odd
    })

    review.to_csv(args.out, index=False)
    print(f"[done] wrote {len(review)} rows -> {args.out}")
    print("\n[next step] Open this CSV in Excel. For each row:")
    print("  1. Read customer_text (and brand_text for context on how it was actually resolved)")
    print("  2. Check if gold_intent (pre-filled) is actually correct -- if not, type the right")
    print("     intent from your final_taxonomy.md")
    print("  3. Check if gold_escalate (Y/N, pre-filled heuristically) matches your real judgment")
    print("  4. Leave labeler_notes blank unless something is genuinely ambiguous/interesting")
    print("Most rows should take a few seconds -- you're confirming, not writing from scratch.")


if __name__ == "__main__":
    main()

"""
Step 8: Evaluation harness.

Scores the agent's predictions against your hand-labeled golden set:
  1. Intent classification: accuracy + macro F1 vs gold_intent
  2. Escalation decision: precision/recall/F1 vs gold_escalate
  3. Reply quality: LLM-as-judge rubric score (1-5) on grounded/relevant/toned/actionable
  4. Judge validation: you hand-score a subset yourself; this reports agreement between
     you and the LLM judge (required by the assignment -- "evidence of how well your
     judge agrees with a human")

Also computes the TWO baselines required by the assignment:
  - Trivial baseline: always predict the majority intent class, always "auto_handle"
  - Simple baseline: TF-IDF + Logistic Regression intent classifier (trained on golden
    set labels via cross-validation, since that's the only labeled data available)

Usage:
    # Step A: score intents + escalation + generate judge scores for all rows
    python 08_eval_harness.py score \
        --golden data/golden_review.csv --predictions data/agent_predictions.csv \
        --out data/eval_results.csv

    # Step B: after you've hand-scored ~30-40 rows yourself in a 'human_reply_score' column
    # (1-5) in a copy of eval_results.csv, check judge agreement:
    python 08_eval_harness.py judge-agreement --scored data/eval_results_human_scored.csv
"""

import argparse
import os
import time

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_MODEL = "openai/gpt-oss-120b"


def groq_call(system, user, api_key, model, max_tokens=300, max_retries=6):
    for attempt in range(max_retries):
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "reasoning_effort": "low",
            },
            timeout=60,
        )
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("Gave up after repeated rate limiting.")


def judge_reply(customer_text, drafted_reply, api_key):
    """LLM-as-judge rubric: grounded, relevant, correct tone, actionable. Score 1-5 + reason."""
    system = (
        "You are grading a customer support reply on a 1-5 scale. Criteria: (1) relevant to "
        "the customer's actual message, (2) actionable -- gives the customer a clear next step, "
        "(3) appropriate apologetic/professional tone for a support context, (4) does not invent "
        "policy details or promises that sound fabricated. "
        "Respond in EXACTLY this format: 'SCORE: <1-5>\\nREASON: <one short sentence>'"
    )
    user = f"Customer message: {customer_text}\n\nDrafted reply: {drafted_reply}"
    raw = groq_call(system, user, api_key, JUDGE_MODEL, max_tokens=150)

    score = 3  # safe default if parsing fails
    reason = raw
    for line in raw.split("\n"):
        if line.strip().upper().startswith("SCORE"):
            digits = "".join(c for c in line if c.isdigit())
            if digits:
                score = min(5, max(1, int(digits[0])))
        if line.strip().upper().startswith("REASON"):
            reason = line.split(":", 1)[-1].strip()
    return score, reason


def trivial_baseline(golden_df):
    majority_intent = golden_df["gold_intent"].mode()[0]
    preds_intent = [majority_intent] * len(golden_df)
    preds_escalate = ["N"] * len(golden_df)  # trivial: never escalate
    return preds_intent, preds_escalate


def simple_baseline(golden_df):
    """TF-IDF + LogReg, evaluated via 5-fold cross-validation since golden labels are
    the only labeled data we have (using them both to train and test directly would be
    circular otherwise)."""
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vec.fit_transform(golden_df["customer_text"].astype(str))
    y = golden_df["gold_intent"]
    preds = cross_val_predict(LogisticRegression(max_iter=1000, class_weight="balanced"), X, y, cv=5)
    return list(preds)


def cmd_score(args):
    golden = pd.read_csv(args.golden)
    preds = pd.read_csv(args.predictions)

    merged = golden.merge(preds, on="thread_id", suffixes=("_gold", "_pred"))
    print(f"[info] matched {len(merged)} rows between golden set and predictions")

    # --- Main system metrics ---
    intent_acc = accuracy_score(merged["gold_intent"], merged["predicted_intent"])
    intent_f1 = f1_score(merged["gold_intent"], merged["predicted_intent"], average="macro", zero_division=0)

    gold_esc_bin = (merged["gold_escalate"].str.upper() == "Y").astype(int)
    pred_esc_bin = (merged["escalate_decision"] == "escalate").astype(int)
    esc_precision = precision_score(gold_esc_bin, pred_esc_bin, zero_division=0)
    esc_recall = recall_score(gold_esc_bin, pred_esc_bin, zero_division=0)
    esc_f1 = f1_score(gold_esc_bin, pred_esc_bin, zero_division=0)

    print("\n=== MAIN SYSTEM ===")
    print(f"Intent accuracy: {intent_acc:.3f}  |  macro F1: {intent_f1:.3f}")
    print(f"Escalation precision: {esc_precision:.3f}  recall: {esc_recall:.3f}  F1: {esc_f1:.3f}")

    # --- Baselines ---
    triv_intent, triv_esc = trivial_baseline(golden)
    triv_acc = accuracy_score(golden["gold_intent"], triv_intent)
    triv_esc_bin = [0] * len(golden)
    triv_esc_f1 = f1_score(gold_esc_bin, triv_esc_bin, zero_division=0)
    print("\n=== TRIVIAL BASELINE (majority class, never escalate) ===")
    print(f"Intent accuracy: {triv_acc:.3f}")
    print(f"Escalation F1: {triv_esc_f1:.3f}")

    simple_preds = simple_baseline(golden)
    simple_acc = accuracy_score(golden["gold_intent"], simple_preds)
    simple_f1 = f1_score(golden["gold_intent"], simple_preds, average="macro", zero_division=0)
    print("\n=== SIMPLE BASELINE (TF-IDF + LogReg, 5-fold CV) ===")
    print(f"Intent accuracy: {simple_acc:.3f}  |  macro F1: {simple_f1:.3f}")

    # --- LLM-as-judge for reply quality ---
    api_key = None if args.dry_run else os.environ.get("GROQ_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("GROQ_API_KEY not set.")

    judge_scores, judge_reasons = [], []
    for i, row in merged.iterrows():
        if args.dry_run:
            score, reason = 4, "[dry-run stub]"
        else:
            score, reason = judge_reply(row["customer_text_gold"], row["drafted_reply"], api_key)
            time.sleep(0.3)
        judge_scores.append(score)
        judge_reasons.append(reason)
        if (i + 1) % 20 == 0:
            print(f"[info] judged {i+1}/{len(merged)}")

    merged["judge_score"] = judge_scores
    merged["judge_reason"] = judge_reasons

    print(f"\n=== REPLY QUALITY (LLM judge, 1-5) ===")
    print(f"Mean: {merged['judge_score'].mean():.2f}  |  Median: {merged['judge_score'].median():.1f}")

    merged.to_csv(args.out, index=False)
    print(f"\n[done] wrote full eval results -> {args.out}")
    print("\n[next step] Hand-score 30-40 of these rows yourself in a new 'human_reply_score'")
    print("column (1-5), save as a new file, then run 'judge-agreement' on it. This is a")
    print("REQUIRED deliverable: evidence of how well your judge agrees with a human.")


def cmd_judge_agreement(args):
    df = pd.read_csv(args.scored)
    scored = df.dropna(subset=["human_reply_score"])
    if len(scored) < 5:
        raise SystemExit(f"Only {len(scored)} rows have human_reply_score filled in -- "
                          "label at least 20-30 for a meaningful agreement estimate.")

    diff = (scored["judge_score"] - scored["human_reply_score"]).abs()
    exact_agreement = (diff == 0).mean()
    within_one = (diff <= 1).mean()
    mean_abs_diff = diff.mean()

    print(f"[info] {len(scored)} rows hand-scored")
    print(f"Exact agreement: {exact_agreement:.1%}")
    print(f"Within +/-1 point agreement: {within_one:.1%}")
    print(f"Mean absolute difference: {mean_abs_diff:.2f}")
    print("\nReport BOTH numbers in your report's judge validation section -- exact agreement")
    print("alone can look artificially low for a 1-5 scale even when the judge is reasonable.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score")
    p_score.add_argument("--golden", required=True)
    p_score.add_argument("--predictions", required=True)
    p_score.add_argument("--out", required=True)
    p_score.add_argument("--dry-run", action="store_true")
    p_score.set_defaults(func=cmd_score)

    p_judge = sub.add_parser("judge-agreement")
    p_judge.add_argument("--scored", required=True)
    p_judge.set_defaults(func=cmd_judge_agreement)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

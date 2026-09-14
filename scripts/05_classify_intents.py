"""
Step 5: Intent classifier + two baselines, as required by the assignment
("results vs. at least two baselines: a trivial one and a simple one").

- Trivial baseline : always predict the majority class
- Simple baseline  : TF-IDF + Logistic Regression
- Main system      : few-shot LLM classification (Groq)

Usage:
    python 05_classify_intents.py --input data/pairs_amazonhelp_subsample.csv \
        --taxonomy data/final_taxonomy.md --n-classify 300 --out data/classified.csv
"""

import argparse
import json
import os
import re
import time

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CLASSIFY_MODEL = "openai/gpt-oss-20b"  # smaller/faster model for high-volume classification
                                        # (see check_groq_models.py for current options)

INTENTS = [
    "delivery_issue", "refund_return", "payment_billing", "account_technical",
    "support_process_complaint", "security_scam_report", "product_info_availability",
    "promo_giftcard", "other_uncategorized",
]


def call_groq_classify(text, api_key, max_retries=6):
    system = (
        "Classify the customer support message into EXACTLY ONE of these intents: "
        f"{', '.join(INTENTS)}. "
        "delivery_issue = late/lost/misdelivered package. refund_return = wants refund/return/"
        "cancellation/compensation. payment_billing = charges/card/subscription/Prime billing. "
        "account_technical = login/app/device technical issues. support_process_complaint = "
        "complaint specifically about support itself (no response, refused escalation) not the "
        "underlying issue. security_scam_report = fraud/phishing/security concern. "
        "product_info_availability = pre-purchase info question, no problem. promo_giftcard = "
        "genuine promotion/contest/gift-card question. other_uncategorized = spam, off-topic, "
        "pure thanks, unclear, or non-English. "
        "Respond with ONLY the intent label, nothing else -- no reasoning, no explanation."
    )
    for attempt in range(max_retries):
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": CLASSIFY_MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                "temperature": 0,
                "max_tokens": 500,  # gpt-oss is a reasoning model -- needs room for thinking
                                     # tokens before it writes the final label, or it gets cut off
                "reasoning_effort": "low",  # minimize thinking tokens spent (gpt-oss supports this)
            },
            timeout=60,
        )
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            print(f"[warn] rate limited, waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return extract_label(content)
    raise RuntimeError(f"Gave up after {max_retries} retries due to repeated rate limiting.")


def extract_label(raw_text):
    """
    Robust to reasoning models that wrap the label in extra text (thinking traces,
    punctuation, restated instructions). Searches for any known intent label
    appearing anywhere in the response rather than requiring an exact match.
    """
    text_lower = raw_text.lower()
    for intent in INTENTS:
        if intent in text_lower:
            return intent
    return "other_uncategorized"


def trivial_baseline_predict(n, majority_class="delivery_issue"):
    return [majority_class] * n


def simple_baseline_train_predict(train_texts, train_labels, test_texts):
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train_labels)
    return clf.predict(X_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n-classify", type=int, default=300,
                     help="how many messages to run the LLM classifier on (costs API calls)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    n = min(args.n_classify, len(df))
    sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    print(f"[info] classifying {n} messages")

    api_key = None if args.dry_run else os.environ.get("GROQ_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("GROQ_API_KEY not set.")

    llm_labels = []
    try:
        for i, text in enumerate(sample["customer_text"]):
            if args.dry_run:
                label = INTENTS[i % len(INTENTS)]
            else:
                label = call_groq_classify(text, api_key)
                time.sleep(0.3)  # small pause between calls to stay under per-minute limits
            llm_labels.append(label)
            if (i + 1) % 25 == 0:
                print(f"[info] classified {i+1}/{n}")
                # periodic save so a crash mid-run doesn't lose progress
                partial = sample.iloc[:len(llm_labels)].copy()
                partial["llm_intent"] = llm_labels
                partial.to_csv(args.out, index=False)
    except Exception as e:
        print(f"[error] stopped early after {len(llm_labels)}/{n}: {e}")
        if llm_labels:
            partial = sample.iloc[:len(llm_labels)].copy()
            partial["llm_intent"] = llm_labels
            partial.to_csv(args.out, index=False)
            print(f"[info] saved partial progress ({len(llm_labels)} rows) -> {args.out}")
            print("[info] re-run with the same command -- it will start over, but progress up")
            print("       to here is safe in the CSV if you want to stop and use a smaller n.")
        raise

    sample["llm_intent"] = llm_labels
    sample.to_csv(args.out, index=False)
    print(f"[done] wrote {len(sample)} classified messages -> {args.out}")

    print("\n[info] intent distribution from LLM classifier:")
    print(sample["llm_intent"].value_counts())

    print("\n[next step] This CSV of LLM-labeled messages becomes the training signal for the")
    print("TF-IDF simple baseline once you have your hand-labeled golden set (Day 4) to actually")
    print("SCORE these three systems against. Classifier output here is not yet 'ground truth' --")
    print("it's your main system's predictions, to be evaluated against golden labels next.")


if __name__ == "__main__":
    main()

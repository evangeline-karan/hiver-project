"""
Step 7: The actual AI support agent. For each incoming customer message:
  1. Classify intent (LLM, few-shot against the taxonomy)
  2. Retrieve similar historically-resolved cases (TF-IDF cosine similarity over the
     corpus) as grounding for the reply
  3. Draft a reply grounded in those retrieved precedents
  4. Decide auto-handle vs escalate, WITH a stated reason

Design choice (decision log entry): retrieval uses TF-IDF cosine similarity rather than
dense embeddings (e.g. sentence-transformers). Given the time budget, TF-IDF needs no
model download, is fast, and is fully deterministic -- a reasonable trade against the
better semantic matching dense embeddings would give. Worth naming as a "what I'd do
with one more week" item in the report.

Leakage prevention: the retrieval corpus EXCLUDES any thread_id present in the golden
eval set, so the agent never gets to "cheat" by retrieving its own eval answer.

Usage:
    python 07_agent_pipeline.py \
        --corpus data/pairs_amazonhelp_subsample.csv \
        --golden data/golden_review.csv \
        --out data/agent_predictions.csv \
        --top-k 3
"""

import argparse
import os
import time

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CLASSIFY_MODEL = "openai/gpt-oss-20b"
DRAFT_MODEL = "openai/gpt-oss-120b"  # bigger model for the actual reply -- quality here matters
                                      # more than for classification

INTENTS = [
    "delivery_issue", "refund_return", "payment_billing", "account_technical",
    "support_process_complaint", "security_scam_report", "product_info_availability",
    "promo_giftcard", "other_uncategorized",
]

HIGH_RISK_INTENTS = {"security_scam_report", "support_process_complaint"}
ANGRY_MARKERS = ["!!", "worst", "pathetic", "furious", "ridiculous", "thugs", "scam",
                  "unacceptable", "disgusting", "never again", "cheated"]


def groq_call(system, user, api_key, model, max_tokens=500, max_retries=6):
    for attempt in range(max_retries):
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
                "reasoning_effort": "low",
            },
            timeout=60,
        )
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            print(f"[warn] rate limited, waiting {wait:.1f}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("Gave up after repeated rate limiting.")


def extract_label(raw_text):
    text_lower = raw_text.lower()
    for intent in INTENTS:
        if intent in text_lower:
            return intent
    return "other_uncategorized"


def classify_intent(text, api_key):
    system = (
        "Classify the customer support message into EXACTLY ONE of these intents: "
        f"{', '.join(INTENTS)}. Respond with ONLY the intent label, no reasoning."
    )
    raw = groq_call(system, text, api_key, CLASSIFY_MODEL, max_tokens=300)
    return extract_label(raw)


def build_retrieval_index(corpus_df):
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    matrix = vec.fit_transform(corpus_df["customer_text"].astype(str))
    return vec, matrix


def retrieve_similar(query_text, vec, matrix, corpus_df, top_k=3):
    q_vec = vec.transform([query_text])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        results.append({
            "customer_text": corpus_df.iloc[idx]["customer_text"],
            "brand_text": corpus_df.iloc[idx]["brand_text"],
            "similarity": float(sims[idx]),
        })
    return results


def draft_reply(query_text, retrieved, api_key):
    context_blocks = "\n\n".join(
        f"Past customer message: {r['customer_text']}\nHow it was resolved: {r['brand_text']}"
        for r in retrieved
    )
    system = (
        "You are drafting a customer support reply for AmazonHelp's Twitter support account. "
        "Match the brand's typical tone: apologetic where warranted, concise, action-oriented "
        "(usually directing the customer to DM order details). Ground your reply in how "
        "similar past issues were actually resolved, shown below. Do not invent policy details "
        "not supported by the examples. Keep it to 1-3 sentences, Twitter-reply length."
    )
    user = f"Similar past resolved cases:\n{context_blocks}\n\nNew customer message:\n{query_text}"
    return groq_call(system, user, api_key, DRAFT_MODEL, max_tokens=600)


def decide_escalation(intent, text, retrieved):
    """Rule-based, with a stated reason -- required by the assignment."""
    text_lower = str(text).lower()
    max_similarity = max((r["similarity"] for r in retrieved), default=0.0)

    if intent in HIGH_RISK_INTENTS:
        return "escalate", f"high-risk intent ({intent}) always routed to a human"
    if any(marker in text_lower for marker in ANGRY_MARKERS):
        return "escalate", "message contains strong negative-sentiment markers"
    if intent == "other_uncategorized":
        return "escalate", "intent unclear or message outside handled scope (e.g. non-English)"
    if max_similarity < 0.15:
        return "escalate", f"low grounding confidence (best retrieval similarity={max_similarity:.2f}), no close historical precedent"
    return "auto_handle", f"routine {intent} issue with a close historical precedent (similarity={max_similarity:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus_df = pd.read_csv(args.corpus)
    golden_df = pd.read_csv(args.golden)

    # leakage prevention: drop any golden thread_id from the retrieval corpus
    before = len(corpus_df)
    corpus_df = corpus_df[~corpus_df["thread_id"].isin(golden_df["thread_id"])].reset_index(drop=True)
    print(f"[info] retrieval corpus: {before} -> {len(corpus_df)} rows after excluding golden set")

    vec, matrix = build_retrieval_index(corpus_df)
    print(f"[info] built TF-IDF retrieval index over {len(corpus_df)} historical pairs")

    api_key = None if args.dry_run else os.environ.get("GROQ_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("GROQ_API_KEY not set.")

    rows = []
    for i, row in golden_df.iterrows():
        text = row["customer_text"]
        retrieved = retrieve_similar(text, vec, matrix, corpus_df, top_k=args.top_k)

        if args.dry_run:
            intent = INTENTS[i % len(INTENTS)]
            reply = f"[dry-run stub reply for: {text[:40]}]"
        else:
            intent = classify_intent(text, api_key)
            reply = draft_reply(text, retrieved, api_key)
            time.sleep(0.3)

        decision, reason = decide_escalation(intent, text, retrieved)

        rows.append({
            "thread_id": row["thread_id"],
            "customer_text": text,
            "predicted_intent": intent,
            "drafted_reply": reply,
            "escalate_decision": decision,
            "escalate_reason": reason,
            "top_retrieval_similarity": round(max((r["similarity"] for r in retrieved), default=0.0), 3),
            "retrieved_example_1": retrieved[0]["customer_text"] if retrieved else "",
        })

        if (i + 1) % 20 == 0:
            print(f"[info] processed {i+1}/{len(golden_df)}")
            pd.DataFrame(rows).to_csv(args.out, index=False)  # periodic save

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"[done] wrote {len(out_df)} predictions -> {args.out}")
    print("\n[info] escalation decision distribution:")
    print(out_df["escalate_decision"].value_counts())


if __name__ == "__main__":
    main()

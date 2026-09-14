"""
Step 4: LLM-assisted intent taxonomy proposal.

Instead of reading hundreds of messages by hand, we show the LLM a sample of
real customer messages and ask it to propose 6-10 recurring intent categories
with short definitions and example message numbers. YOU then review, rename,
merge, or split these -- the taxonomy is only defensible if you can justify
every category yourself, so don't skip the review step, just compress it.

Requires: GROQ_API_KEY environment variable set (see earlier setup).

Usage:
    python 04_propose_taxonomy.py --input data/pairs_amazonhelp_subsample.csv \
        --n-sample 150 --out data/proposed_taxonomy.md
"""

import argparse
import json
import os
import random

import pandas as pd
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"  # confirmed available via check_groq_models.py on 2026-09-13
                                 # if this model name is retired, re-run check_groq_models.py
                                 # and swap in whatever current large model is listed there


def call_groq(messages, api_key, temperature=0.2):
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages, "temperature": temperature},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def build_prompt(sample_texts):
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(sample_texts))
    system = (
        "You are helping design an intent taxonomy for a customer support AI agent. "
        "You will be shown real customer messages sent to a company's support Twitter account. "
        "Propose 6 to 10 mutually-exclusive intent categories that cover the recurring issue "
        "types you see. For each category give: a short_name (snake_case), a one-sentence "
        "definition, and a list of the message numbers (from the numbered list) that best "
        "exemplify it. Respond ONLY with valid JSON: a list of objects with keys "
        "'short_name', 'definition', 'example_message_numbers'. No prose, no markdown fences."
    )
    user = f"Customer messages:\n{numbered}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_markdown(taxonomy, sample_texts):
    lines = ["# Proposed intent taxonomy (LLM draft -- REVIEW BEFORE USING)\n"]
    lines.append("Review checklist: does every category make sense to a human? are any two "
                  "categories really the same thing? does anything important NOT fit any "
                  "category (may need an 'other' bucket, or a missed category)?\n")
    for cat in taxonomy:
        lines.append(f"## {cat['short_name']}")
        lines.append(f"{cat['definition']}\n")
        lines.append("Examples:")
        for n in cat.get("example_message_numbers", []):
            idx = n - 1
            if 0 <= idx < len(sample_texts):
                lines.append(f"- ({n}) {sample_texts[idx]}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n-sample", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true",
                     help="skip the real API call and use a stub response (for testing the script itself)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    n = min(args.n_sample, len(df))
    sample = df.sample(n=n, random_state=args.seed)["customer_text"].tolist()
    print(f"[info] sampled {len(sample)} customer messages")

    if args.dry_run:
        print("[info] dry-run mode: skipping real API call")
        raw = json.dumps([
            {"short_name": "delivery_issue", "definition": "Package late, lost, or not arrived.",
             "example_message_numbers": [1, 2] if len(sample) >= 2 else [1]},
            {"short_name": "refund_request", "definition": "Customer wants money back.",
             "example_message_numbers": [1]},
        ])
    else:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("GROQ_API_KEY not set. See setup instructions.")
        messages = build_prompt(sample)
        raw = call_groq(messages, api_key)

    try:
        taxonomy = json.loads(raw)
    except json.JSONDecodeError:
        print("[error] model did not return valid JSON. Raw output was:\n")
        print(raw)
        raise SystemExit(1)

    md = format_markdown(taxonomy, sample)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[done] wrote proposed taxonomy -> {args.out}")
    print(f"[info] {len(taxonomy)} categories proposed. Open the file and review/edit now.")


if __name__ == "__main__":
    main()

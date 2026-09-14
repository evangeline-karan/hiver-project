"""
Quick interactive human-scoring tool. Shows you one message + reply at a time,
you type a 1-5 score, it saves automatically. No Excel needed.

Usage:
    python quick_human_score.py --input data/eval_results.csv \
        --out data/eval_results_human_scored.csv --n 35
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=35, help="how many NEW rows to score this run")
    args = ap.parse_args()

    # resume support: if the output file already exists (has your previous scores),
    # load THAT instead of starting fresh from --input
    import os
    if os.path.exists(args.out):
        df = pd.read_csv(args.out)
        print(f"[info] resuming from {args.out} ({df['human_reply_score'].notna().sum()} already scored)")
    else:
        df = pd.read_csv(args.input)
        if "human_reply_score" not in df.columns:
            df["human_reply_score"] = pd.NA

    # only sample from rows that DON'T already have a score
    unscored = df[df["human_reply_score"].isna()]
    if len(unscored) == 0:
        print("[info] everything is already scored!")
        return

    sample = unscored.sort_values("judge_score").iloc[
        list(range(0, len(unscored), max(1, len(unscored) // min(args.n, len(unscored)))))
    ].head(args.n)

    print(f"Scoring {len(sample)} messages. For each: read the exchange, type 1-5, Enter.")
    print("Type 's' to skip a row, 'q' to stop early and save what you have.\n")

    for idx in sample.index:
        row = df.loc[idx]
        print("=" * 70)
        print(f"CUSTOMER: {row.get('customer_text_gold', row.get('customer_text', ''))}")
        print(f"\nDRAFTED REPLY: {row['drafted_reply']}")
        print(f"\n(judge gave this a {row['judge_score']}/5 -- don't just copy that, form your own view)")

        while True:
            ans = input("\nYour score (1-5, s=skip, q=quit): ").strip().lower()
            if ans == "q":
                df.to_csv(args.out, index=False)
                print(f"\n[saved] {df['human_reply_score'].notna().sum()} scores -> {args.out}")
                return
            if ans == "s":
                break
            if ans in {"1", "2", "3", "4", "5"}:
                df.loc[idx, "human_reply_score"] = int(ans)
                break
            print("Please type 1-5, s, or q.")

        df.to_csv(args.out, index=False)  # save after every row, never lose progress

    print(f"\n[done] {df['human_reply_score'].notna().sum()} scores -> {args.out}")
    print("Now run: python scripts/08_eval_harness.py judge-agreement --scored " + args.out)


if __name__ == "__main__":
    main()

"""
Fills the two remaining REPORT.md gaps and sanity-checks the golden set.

Outputs:
  1. Low-similarity retrieval examples (for failure mode #3)
  2. Count of likely non-English messages in the golden set (for failure mode #5)
  3. Whether golden labels differ from the LLM pre-fills (evidence you actually reviewed them)

Usage:
    python report_gaps.py
"""
import pandas as pd

print("=" * 70)
print("1. LOW-SIMILARITY RETRIEVAL EXAMPLES (for failure mode #3)")
print("=" * 70)
preds = pd.read_csv("data/agent_predictions.csv")
low_sim = preds.nsmallest(5, "top_retrieval_similarity")
for _, row in low_sim.iterrows():
    print(f"\n--- similarity: {row['top_retrieval_similarity']:.3f} | intent: {row['predicted_intent']} ---")
    print(f"CUSTOMER: {str(row['customer_text'])[:200]}")
    print(f"RETRIEVED (best match): {str(row['retrieved_example_1'])[:200]}")
    print(f"DECISION: {row['escalate_decision']} -- {row['escalate_reason']}")

print("\n" + "=" * 70)
print("2. NON-ENGLISH ESTIMATE (for failure mode #5)")
print("=" * 70)
golden = pd.read_csv("data/golden_review.csv")


def likely_non_english(text):
    """Crude heuristic: non-ASCII script chars, or common non-English function words."""
    t = str(text)
    # CJK / Japanese / Arabic / Cyrillic ranges
    if any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff"
            or "\u0600" <= c <= "\u06ff" or "\u0400" <= c <= "\u04ff" for c in t):
        return True
    markers = [" je ", " le ", " les ", " pas ", " vous ", " pour ", " qui ",
                " el ", " la ", " que ", " por ", " para ", " con ", " una ",
                " nao ", " não ", " voce ", " você ", " eu ", " meu ",
                " der ", " die ", " und ", " ich ", " nicht "]
    t_lower = f" {t.lower()} "
    return any(m in t_lower for m in markers)


golden["likely_non_english"] = golden["customer_text"].apply(likely_non_english)
n_non_eng = golden["likely_non_english"].sum()
print(f"Likely non-English: {n_non_eng} / {len(golden)} rows ({n_non_eng/len(golden):.1%})")
print("\nSample of flagged messages:")
for t in golden[golden["likely_non_english"]]["customer_text"].head(5):
    print(f"  - {str(t)[:120]}")
print("\n(Heuristic only -- eyeball these to confirm before quoting the % in the report.)")

print("\n" + "=" * 70)
print("3. GOLDEN SET LABELING CHECK")
print("=" * 70)
changed = (golden["gold_intent"] != golden["llm_predicted_intent"]).sum()
print(f"Rows where your gold_intent differs from the LLM pre-fill: {changed} / {len(golden)}")
if changed == 0:
    print("\n  WARNING: zero corrections means either (a) the LLM was perfect (unlikely),")
    print("  or (b) the rows were never actually reviewed. If (b), your 'hand-labelled'")
    print("  claim is not accurate -- review at least a portion before submitting.")
else:
    print(f"  ({changed/len(golden):.1%} correction rate -- evidence of genuine review)")

blank_esc = golden["gold_escalate"].isna().sum()
print(f"Rows with blank gold_escalate: {blank_esc}")
print(f"\nEscalation label distribution:\n{golden['gold_escalate'].value_counts()}")

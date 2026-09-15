# Decision log

1. **Brand: AmazonHelp.** Chosen over airlines/other brands via a data-driven survey
   (scripts/02_brand_survey.py) — highest volume (169k+ resolvable reply pairs) and a
   domain (orders/delivery/refunds) I could personally judge reply quality on.

2. **Single-hop pairs, not full thread reconstruction.** Built (customer message →
   brand's direct reply) pairs only, not full multi-turn conversations. Faster to
   build and sufficient for retrieval grounding at this project's scale; loses context
   for issues that unfold over multiple exchanges.

3. **9 intents, not the LLM's first-draft 8.** Used an LLM to propose a first-pass
   taxonomy from 150 real messages, then hand-reviewed and edited it: dropped a noisy
   `promotional_giftcard` catch-all, added an explicit `other_uncategorized` bucket
   (genuine noise/spam/thanks needs somewhere to go), and tightened
   `support_process_complaint`'s definition to avoid overlapping with `delivery_issue`.

4. **Non-English messages: escalate, don't attempt generation.** A meaningful share of
   AmazonHelp messages are French/Japanese/Spanish/Portuguese. Given the time budget,
   these route to `other_uncategorized` → escalate rather than attempting reply quality
   I have no way to evaluate.

5. **TF-IDF retrieval instead of dense embeddings.** No model download required, fast,
   deterministic. Traded off semantic match quality on paraphrased issues — named
   explicitly as a "next week" item rather than silently accepted.

6. **openai/gpt-oss-20b for classification, openai/gpt-oss-120b for drafting/judging.**
   Smaller/faster model for the high-volume classification step (300+ calls), larger
   model where output quality matters more and call volume is lower.

7. **Discovered gpt-oss models need reasoning token budget.** Initial classification
   runs with max_tokens=20 silently failed — the model spent its whole budget on
   internal reasoning before the label ever appeared, so every response fell through
   to the `other_uncategorized` fallback. Fixed by raising max_tokens and making label
   extraction search for known labels anywhere in the response rather than requiring
   an exact match — a concrete example of why silent fallbacks need visibility (I only
   caught this because the output distribution was implausibly 100% one class).

8. **Rule-based escalation, not a learned classifier.** Given 180 golden labels split
   across 9 intents, there isn't enough data to train a reliable escalation model.
   Used explicit rules (high-risk intent, angry-language keywords, low retrieval
   similarity, unclear intent) — each decision comes with a stated reason, as required.

9. **Golden set stratified-sampled from the LLM's own intent predictions, not raw
   messages.** This was a time-saving choice (guarantees coverage of rare intents,
   avoids blind random sampling that might miss them entirely) but introduces a
   circularity flagged explicitly in the report's "what's misleading" section: the
   golden set's intent distribution reflects what the LLM tends to predict, not
   necessarily ground truth.

10. **Pre-filled the golden-set review sheet with the LLM's own guesses (intent +
    escalation) rather than a blank sheet.** Sped up hand-labeling substantially under
    time pressure, at the cost of some risk of anchoring bias (a labeler may be more
    likely to accept a plausible-looking pre-filled answer than generate one from
    scratch). Mitigated partly by explicitly reviewing/correcting rather than just
    confirming.

11. **LLM-as-judge uses the same model family as the reply drafter
    (gpt-oss-120b for both).** Flagged as a likely source of judge self-preferencing
    rather than treated as a neutral evaluator — confirmed via a small (n=10)
    human-agreement check showing the judge scores ~1.2 points higher on average than
    a human reviewer.

12. **Human-agreement sample capped at 10 rows instead of the suggested 30-40**, due to
    time constraints. Reported honestly as a limitation on the agreement estimate
    itself, rather than presented as equivalent-strength evidence to a full 30-40 row
    check.

13. **Subsampled 5,000 of 153,004 available pairs for the working corpus**, and ran the
    full pipeline against a 300-message classified sample / 180-message golden-set
    candidate pool — consistent with the assignment's explicit expectation that a
    subsample is used, not the full dataset.

14. **Discarded a first golden-labeling attempt rather than reporting it.** An initial
    pass using a terminal-based labeling tool produced a 92.5% correction rate against
    the LLM's predictions — implausibly high, and inspection showed the corrections
    were essentially unrelated to message content (e.g. "I love you" relabeled as
    `refund_return`), indicating the tool's UX led to fast, non-genuine responses rather
    than real review. That data was discarded and 80 examples were relabeled properly
    (message read in full before deciding), producing a much more plausible 32.5%
    correction rate. Reported the smaller, trustworthy sample rather than the larger,
    unreliable one.

15. **Golden-set sample size (n=80) came in below the assignment's 150-250 target**,
    a direct consequence of decision #14 above and the overall one-day time budget.
    Disclosed explicitly in the report rather than padded with unreviewed rows.

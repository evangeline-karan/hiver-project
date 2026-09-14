# Final intent taxonomy — AmazonHelp support agent

Reviewed and edited from the LLM-proposed draft (see data/proposed_taxonomy.md for the
raw draft). Changes made: dropped `promotional_giftcard` as a catch-all (too noisy),
added explicit `other_uncategorized`, tightened `support_process_complaint` definition
to avoid overlap with `delivery_issue`, resolved the duplicate example (3) by assigning
it to `refund_return` only (the scam angle is a secondary detail, refund is the actionable
ask — single-label classification, ties broken by primary actionable intent).

1. **delivery_issue** — Package is late, lost, misdelivered, or delivery status/tracking
   is unclear. About the physical delivery event itself.

2. **refund_return** — Customer wants a refund, return, cancellation, or compensation for
   a wrong, missing, or defective item.

3. **payment_billing** — Problems with charges, card details, subscription/Prime membership
   billing or terms.

4. **account_technical** — Login failures, app/device bugs, can't access account or content.
   About the technical system, not a specific order.

5. **support_process_complaint** — Complaint specifically about the SUPPORT INTERACTION
   itself: no response, long wait, refused escalation, rude agent, repeated unresolved
   contact. NOT the same as being upset about a late delivery (that's delivery_issue even
   if angry) — this is specifically "your support process failed me."

6. **security_scam_report** — Suspicion of fraud, phishing, or a security incident
   (unauthorized charges framed as a security concern, suspicious messages claiming to
   be from Amazon).

7. **product_info_availability** — Pre-purchase questions: availability, release date,
   specs, "where do I find X" — no problem being reported, just information-seeking.

8. **promo_giftcard** — Genuine questions about active promotions, contests, or gift cards
   (narrowed from the draft — noise like off-topic links and thank-you notes moved to
   other_uncategorized).

9. **other_uncategorized** — Spam, off-topic, pure thanks/positive feedback with no request,
   unclear intent, or non-English messages your pipeline isn't set up to handle (see scope
   note below).

## Scope decision (write this in the decision log)
A meaningful fraction of AmazonHelp messages are non-English (French, Japanese, Spanish,
Portuguese seen in the sample). Given the time budget, intent classification and reply
drafting target English messages; non-English messages are still classified but routed to
`other_uncategorized` -> escalate, rather than attempting non-English generation quality
we can't properly evaluate. This is a real, stated limitation, not a silent gap.

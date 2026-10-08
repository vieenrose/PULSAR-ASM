# K2 quality probe: base Q4 (our engine) vs meeting-agent fine-tune (LiteRT q4)

Fixture: `window.txt` (6-turn zh-TW meeting slice: budget NUMBER, online-signup
PROPOSAL + objection, DECISION with deadline, next-meeting time, venue-fee
reminder), full `system_prompt.txt` verbatim, IFM chat markers. Both sides
temp 0.2 (their harness default), 150 tokens, same text (own tokenizers).

## Fine-tune (`finetune_q4_t02.txt`, LiteRT-LM q4, carved signatures)

One stray opening line, then locks into protocol — 4/4 notes verifiable
against the transcript (all cited times genuine):
- NUMBER 1200萬/+300萬 ✓ / PROPOSAL 線上報名/省80萬 ✓ /
  DECISION 照案通過 ✓ / ACTION S2+下週五+書面報告 ✓
13% of its tokens are extended-vocab (why base-tokenizer reads were garbage).
Cut at 150 before 場地費/NEXT (longer run would finish it).

## Base (`base_q4_t02.txt`, our Q4 engine)

150 tokens of English meta-deliberation about what the task might be; zero
NOTE lines. Never trained on the protocol — exactly the gap SFT+DPO closes.

## Verdict

The fine-tune's value is behavioral, not just weights: same prompt, same
temp, base deliberates while the agent executes. Matches the published
eval direction (21% contradicted / 0.97 coverage vs base's 0 usable notes
here). Smoke-grade (one window, no judge) — the 13-session rig stays theirs.

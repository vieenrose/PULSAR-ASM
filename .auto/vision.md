# Vision compliance (from tomtsai28/PULSAR-ASM README) — standing directives

Original authors: Kuo Ting Tsai (@tomtsai28), Shin Rung Tsai; AI pair: Antigravity.
Motto: 「虛胖的 AI 活在雲端，純粹的 AI 走向實體。」

## Non-negotiables for the deliverable
1. Pure assembly core, 0 CRT, 0 third-party libs, 0 compiler abstraction.
   Current port: `pulsar_arm/asm/*.S`, `as` + `ld -static`, SVC syscalls only.
   No Python / C in the inference path. Old `runtime/`+`kernels/` tree is
   REFERENCE ONLY (parity oracle), never part of the deliverable.
2. Bit-exactness as definition of done: same math, same order, same bits.
   Gates: self-determinism (run twice, `cmp`), module agreement vs HF,
   identical-logits A/B on every change.
3. Bandwidth saturation is THE metric: report achieved GB/s vs measured
   streaming ceiling (we are at 3.84/3.93 = 98%). No mocks, no edited text.
4. Honest status: disclose divergences with root causes (cf. his HF-bf16
   drift analysis, llama.cpp write-up, "~1.05x as wired" MTP note).
   Our disclosures: L17 chaos (~3e11 gain), Pi thermal drift (~18%).
5. Real-text demos over token IDs: BPE detokenization in asm (decided),
   sampler (temp/top-k/top-p, xorshift) after greedy is solid.
6. No benchmark gaming: steady-state numbers, disclosed conditions
   (SoC temp), same-job A/B for decisions, page-cache warmth is legitimate
   steady state; taskset-pinning for stability is hygiene, not a result.

## Conscious, disclosed deviations
- Pressure vehicle is 270M on Pi (his: 2B on PC). Justified by his own
  pillar 1: stress-test scale decouples from landing scale; small model
  suits the ARM stepping stone toward Cortex-M/RVV.
- Greedy first, sampler later (his engine has both; sequencing only).
- Single core, no spin-wait pool: measured +4.2% loss (bus saturated).
  His MESI pool doctrine assumes compute headroom; physics here says no.
  Revisit only if traffic/weight-format assumptions change.

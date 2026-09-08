# Mechanistic Validity Audit: the lookback mechanism for belief tracking

**Claim audited.** Prakash, Shapira, Sen Sharma, Riedl, Belinkov, Rott Shaham, Bau and Geiger
(ICLR 2026, arXiv:2505.14685) report that language models track beliefs through a *lookback
mechanism* — pointer dereference via attention — with three sub-mechanisms: binding (layers
33–38), answer (52–56) and visibility (10–31) in Llama-3-70B-Instruct.

**Framework.** Mechanistic Validity, 36 criteria (6 construct, 7 measurement, 12 internal,
6 external, 5 interpretive). Deposit 10.5281/zenodo.20478480.

**Status of this file.** Scored after our pre-registered experiments, not before. Nine of the
fifteen registered experiments have results; three overturned our own predictions, two of them
in the audited claim's favor. Where an assessment changed because of a result, the change is
stated. `mechval_audit_v2_retired.md` is the pre-experiment version, on the retired
25-criterion scheme.

---

## Verdict

**Causally Suggestive**, capped by specificity (I4).

The ladder is conjunctive, so a rung is reached only when every criterion it names is
established:

| Rung | Requires | Status |
|---|---|---|
| Proposed | C1, C2 | met |
| Causally Suggestive | I1, M2 | **met** — M2 closed by EXP2 |
| Mechanistically Supported | I2, I4, E1 | I2 met, E1 met, **I4 not** |
| Triangulated | C3, C4, I5, I6, I7, (E2 or E4) | C3, C4, I5, I6 all untested |
| Validated | the remaining 22 | far off |

The single blocking criterion is **I4 (specificity)**. Two experiments bear on it and they
disagree: cross-task transfer is partial (EXP11: L38 = 0.529, L52 = 0.0), and the mechanism is
fragile to distractor insertion (EXP13, n = 0 at mid and late layers) where we predicted
robustness. Neither settles whether the lookback is specific to belief tracking or is generic
attention-mediated retrieval.

### What changed from the pre-experiment audit

- **M2 (baseline separation) moves from Not met to Met.** The pre-experiment audit called the
  missing random-subspace control "the critical missing baseline". EXP2 ran it: 350 matched-rank
  random subspaces across the six identified subspaces, and **not one reached the identified
  IIA**. M2 gates every causal rung, so this is the result that lets the claim reach Causally
  Suggestive at all.
- **Two predictions overturned in the claim's favor.** We predicted the answer subspaces would
  not transfer to stories where belief diverges from reality (EXP14) or to observability-mediated
  knowledge (EXP15), at IIA < 0.30. They transferred at 0.975 and 0.988. The subspaces are doing
  more than the degenerate CausalToM generator can explain.
- **One overturned against it.** EXP13 predicted robustness to distractors and found fragility.
- **I9 (epistatic interaction) is now tested,** and exactly rather than by estimation. The
  cross-model mismatch test is a complete two-player coalition function, so its Walsh
  decomposition is exact: the binding protocol is a parity function at layers 24–34 (both
  order-1 coefficients exactly zero, all non-constant energy at order 2); the answer protocol
  is a conjunction at 32–38, w₂ = +0.249 against a ceiling of +0.25.

---

## Scorecard

`C` confirmed · `PC` partially confirmed · `I` inconclusive · `U` untested · `D` disconfirmed ·
`N/A` structurally inapplicable

| | Criterion | | Basis |
|---|---|:--:|---|
| **C1** | Falsifiability | PC | Named layer ranges and directional IIA predictions that could fail. No disconfirming condition stated in advance. |
| **C2** | Structural plausibility | PC | The causal model (Fig. 3) is explicit and each step carries a predicted intervention outcome. No weight-level account: no head is named, no QK/OV product examined. |
| **C3** | Convergent validity | **U** | Every result comes from one instrument. DCM is interchange intervention with a learned mask, so it shares the primitive rather than testing it. No probe, no weight analysis, no attention-pattern analysis. |
| **C4** | Discriminant validity | **U** | No neighboring construct is localized, so nothing separates "belief tracking" from entity tracking or generic in-context retrieval. |
| **C5** | Nomological validity | PC | Connected to ordering IDs, entity tracking and variable binding. The connection is asserted rather than measured — whether the lookback reuses the same OIs is untested. |
| **C6** | Complementation validity | **U** | Three sub-mechanisms are named. They are never ablated together, so the subdivision is a labeling convention rather than a demonstrated one. |
| **M1** | Reliability | PC | EXP1 replicates the identification within ±0.05 at L34 and higher at L38/L52. L53 shows high seed-to-seed variance and is excluded from verdicts. No interval accompanies any origin IIA. |
| **M2** | Baseline separation | **C** | EXP2. Matched-rank random subspaces, 50 draws each (100 at L34). Identified IIA 0.550–0.963; random maxima 0.000–0.613; **0 of 350 draws reached the identified value**. p = 0.020 per subspace, 0.010 at L34. |
| **M3** | Stability | U | Subspace rank is selected, not swept. No threshold sensitivity reported. |
| **M4** | Calibration | PC | IIA is bounded on [0,1] with an interpretable zero. The origin reports no interval, and at n = 80 the standard error on a binary outcome is ≈5.6 points. |
| **M5** | Sensitivity | U | Controls are known-negatives. No mechanism of known extent is planted and recovered, so the instrument's floor is unmeasured. |
| **M6** | Invariance | PC | Three models at origin, at different layer ranges and subspace dimensions. Our cross-model test (Qwen2.5-14B) finds the answer lookback's dependency structure transfers and the binding mechanism does not. |
| **M7** | Selection correction | **D** | All 80 stories are pre-filtered to cases the model answers correctly — selection on the dependent variable. The layer-wise search is uncorrected across ~80 layers × 6 experiments. |
| **I1** | Necessity | PC | Patching at the predicted layers reverses the output. The rescue experiment gives an 8% accuracy drop on removal and recovery to 1.0 on restoration. The intervention swaps a large subspace, so it transfers more than the hypothesized pointer. |
| **I2** | Sufficiency | PC | Patching the identified subspace at the predicted layers produces the predicted output. Sufficiency is for the estimand, not for a capability. |
| **I4** | Specificity | **I** | **The capping criterion.** EXP11 gives partial cross-task transfer (L38 = 0.529, L52 = 0.0); EXP13 finds the mechanism fragile to distractor insertion. The two point in opposite directions and neither is decisive. |
| **I3** | Minimality | U | No subspace is removed singly and the remainder rescored against a matched control. EXP17 is registered and unrun. |
| **I5** | Rival mechanism exclusion | U | The simplest rival — attention to the most recently mentioned state token — is never tested. Nor is the positional-regularity account: CausalToM fixes character, object and state tokens at nearly constant positions. |
| **I6** | Double dissociation | U | One mechanism, one behavior. No second mechanism is shown intact under an ablation that breaks the lookback. |
| **I7** | Confound control | PC | The visibility condition is a designed contrast. The full-vector confound is unaddressed: an interchange intervention transfers everything in the swapped subspace, not only the hypothesized content. |
| **I8** | Confounding sensitivity | U | No bound on how strong an unmeasured confounder would need to be. Our own analysis is exploratory. |
| **I9** | Epistatic interaction | **PC** | Tested exactly, not estimated. The cross-model mismatch test is a complete two-player coalition function; its order-2 Walsh coefficient is parity at binding (order-1 terms exactly 0) and conjunction at answer (w₂ = +0.249, ceiling +0.25). A marginal-contribution summary assigns the binding run two zeros and misses the structure entirely. |
| **I10** | Rescue reversibility | PC | Run. Removal costs 8% accuracy; restoring the subspace recovers to 1.0. One corruption method, one restoration method. |
| **I11** | Onset coupling | U | No checkpoints released, so onset cannot be timed. |
| **I12** | Offset coupling | U | No interval over which the capability lapses is observed. |
| **E1** | Intervention reach | PC | Interchange intervention at four granularities plus DCM masking. All share one counterfactual construction, so the families are not independent. |
| **E2** | Prompt generalization | PC | EXP10 gives cross-template transfer at 0.957. EXP12 finds framing modulates it: belief questions IIA = 1.0, reality-state questions 0.82–0.92 over the same pairs. Broad within the CausalToM frame; the frame is held fixed. |
| **E3** | Cross-task generalization | **I** | EXP11: L38 = 0.529, L52 = 0.0. Transfer at one layer and none at another, on the same task pair. |
| **E4** | Cross-model recurrence | PC | Three models at origin at different layer ranges. Our Qwen test: the answer lookback's dependency structure transfers (singles near zero, pair at 1.0); the binding mechanism shows partial transport that full-residual patching cannot detect. |
| **E5** | Graded response | U | Patching is all-or-nothing. No interpolation between clean and counterfactual activation, so no dose-response curve exists. |
| **E6** | Novel prediction | PC | EXP14 and EXP15 are novel predictions that the origin's account could have failed and did not — they were *our* predictions of failure, and the mechanism survived both at 0.975 and 0.988. |
| **V1** | Level declaration | PC | The unit is declared exactly (residual-stream subspace at named layers). The level is not: "pointer dereference" is an algorithmic claim, the evidence is implementational-topographic. |
| **V2** | Level-evidence match | PC | Flow and topography are supported at the level claimed. The algorithmic reading — that this is dereference rather than attention-weighted retrieval — has no evidence at its own level. |
| **V3** | Alternative level | U | The positional-artifact reading is not stated. CausalToM's fixed token positions mean the result is compatible with positional regularity. |
| **V4** | Unlicensed labeling | **D** | "Pointer dereference" imports a specific operation — allocate a reference, store an address, follow it — from programming languages. What is measured is attention-mediated information flow. "Address" is a key vector, "payload" a value, "dereference" attention. Stripped of the metaphor the finding is that attention retrieves information. |
| **V5** | Scope declaration | PC | Models and dataset are stated. The abstract and conclusion generalize past them: "pervasive", "a fundamental role in in-context reasoning", "mapped the end-to-end underlying mechanism". |

**Tally.** 1 confirmed · 18 partially confirmed · 2 inconclusive · 13 untested · 2 disconfirmed. (36)

---

## The three readings

The evidence supports different readings to different depths, and separating them is what the
audit is for.

| Reading | Verdict | What blocks it |
|---|---|---|
| Interchange interventions at the named layer ranges transfer belief-relevant content, and not by chance | **Causally Suggestive** | I4 — specificity is inconclusive |
| The three sub-mechanisms are one mechanism | **Underdetermined** | C6 — never ablated together; they differ in layer range, token type and rank |
| The mechanism is pointer dereference | **Disconfirmed** | V4 — the label names an operation the method cannot detect; nothing distinguishes it from attention-weighted retrieval |

We propose **perspective-indexed entity binding** as a label the evidence licenses.

---

## What our experiments settled, and what they did not

**Settled.**

- The subspaces are not artifacts of subspace search (M2, EXP2).
- They carry belief content that diverges from reality, and observability-mediated knowledge
  (EXP14, EXP15) — contrary to our own prediction, and contrary to what the degenerate
  CausalToM generator alone would license.
- Question framing modulates transfer without abolishing it (EXP12).
- The answer lookback's dependency structure recurs cross-model; the binding mechanism's does
  not (cross-model test).
- The interaction between token groups is exact and differs by protocol: parity at binding,
  conjunction at answer (I9, EXP19–EXP20).

**Not settled, with the registered experiment that would settle it.**

| Open criterion | Registered experiment | Status |
|---|---|---|
| I4 specificity, I3 minimality | **EXP23** — decompose the binding span; the span contains the answer readout position, so patching it may reduce to overwriting that site | not run |
| C4 discriminant, I5 rivals, V4 labeling | **EXP16** — three conditions separating belief from observation history from reality | not run |
| I3 minimality, I9 interaction | **EXP17** — joint ablation, redundancy against dispensability | not run |
| M6 invariance | **EXP21** — directional symmetry of the mismatch test | not run |
| I4 specificity | **EXP22** — omitted-region localization (exploratory; settles nothing alone) | not run |
| E2 prompt generalization | **EXP18** — fill-completion framing at n = 250 | not run |
| M2 second arm | **EXP3** — shuffled-label control | not run |

EXP23 is the one that could change a verdict already recorded: if patching the binding span
reduces to overwriting the readout site, the interaction structure reported under I9 is about
the readout rather than about binding.

---

## Provenance

Experiments were registered in timestamped amendments before results
(`AMENDMENTS.md`, tags `prereg-amendment-1` onward). Results are in `results/`. Quotations from
the audited paper are pinned with SHA-256 in `claims/` and resolve under `citations verify`.
Compute estimates are in `COMPUTE_ESTIMATE.md`.

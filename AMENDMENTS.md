# Pre-registration Amendments

Original pre-registration: commit `c9bb775`, tag `prereg-v1`.

All amendments below follow inspection of the released implementation at
https://github.com/Nix07/belief_tracking (commit `0579347e`) but precede
any model execution. No experimental results informed these changes.

---

## Amendment 1: Concept/stage labeling correction

**Problem.** The original scripts used ambiguous names like `binding_L35`
and populated them with values from `answer_lookback/pointer` — the answer
pointer concept evaluated at binding-stage layers. This conflated two
distinct subspaces: the paper's binding mechanism
(`binding_lookback/address_and_payload`) and the answer pointer mechanism
(`answer_lookback/pointer`) which happens to show high IIA at the same layers.

**Correction.** Every subspace entry is now keyed by
`{lookback_type}_{concept}_L{layer}` (e.g., `binding_addr_payload_L35`,
`answer_pointer_L52`). Each entry records its `lookback_type` and `concept`
fields to prevent future conflation.

**Rank changes.** Using the correct concept for each mechanism:

| Entry | Old (incorrect) | New (correct) | Source |
|-------|----------------|---------------|--------|
| binding L34 | not included | rank=3, IIA=0.9625 | binding_lookback/address_and_payload/34.json |
| binding L35 | rank=7 (accidental match) | rank=7, IIA=0.7375 | binding_lookback/address_and_payload/35.json |
| binding L36 | rank=5 (from answer_pointer) | rank=8, IIA=0.7500 | binding_lookback/address_and_payload/36.json |
| binding L38 | rank=3 (from answer_pointer) | removed from binding, reassigned to answer_pointer | answer_lookback/pointer/38.json |
| answer L38 | not included | rank=3, IIA=0.925 | answer_lookback/pointer/38.json |
| answer L52 | rank=18 | rank=18, IIA=0.775 (unchanged) | answer_lookback/pointer/52.json |
| answer L53 | rank=10 (incorrect) | rank=19, IIA=0.55 | answer_lookback/pointer/53.json |
| answer L54 | rank=8 (incorrect) | rank=19, IIA=0.4375 | answer_lookback/pointer/54.json |

**Why.** The original placeholders were guesses before the released results
were extracted. Several were wrong (L53: 10→19, L54: 8→19), and the
"binding" entries came from the wrong concept type entirely.

---

## Amendment 2: Train/eval split clarification

**Problem.** The paper draft stated that "subspace identification and
reported results both use the training split."

**Correction.** Inspection of `run_patching_exp_utils.py:465` and
`run_single_layer_patching_exps.py:559-585` shows that the binary mask IS
trained on one split (80 stories) and IIA IS evaluated on a separate split
(80 stories). However, the SVD basis (`svd/` directory, not committed) is
pre-computed offline with no documented held-out procedure, so the
evaluation is out-of-sample for the mask but likely in-sample for the
feature space. Additionally, all 160 stories are pre-filtered to
model-correct cases (`filter_dataset_on_lm`, line 394-409).

**Why.** The original claim overstated the train/eval overlap. The corrected
version is more precise about what is and is not held out.

---

## Amendment 3: Visibility rank range

**Problem.** The paper table listed visibility rank as "varies" (provisional).

**Correction.** Extracted from `visibility_lookback/source` results: ranks
range from 160 to 320 (out of 500 SVD components). This is 32-64% of the
truncated candidate basis, though only 2-4% of the full 8192-dimensional
residual stream. The other two visibility concepts (address_and_pointer,
payload) have only full-rank results in the released data (no learned mask).

At early layers, the selected subspace achieves higher IIA than the full
500-component basis (e.g., 0.90 vs 0.175 at layer 7), suggesting that
including the remaining SVD components destroys counterfactual behavior.

**Why.** Real numbers replace provisional estimates.

---

## Amendment 4: Experiment priority reordering

**Problem.** The original pre-registration listed five experiments with
equal priority: random subspace baseline, shuffled-label control, non-ToM
control, failure case analysis, unity coupling test.

**Correction.** Reorder to three primary experiments, two secondary:

**Primary (run first):**
1. Held-out basis validation — recompute SVD on a disjoint story pool,
   fit mask on training stories, evaluate on held-out. Tests whether the
   SVD basis (not just the mask) generalizes out of sample. The mask is
   already validated out-of-sample in the original code.
2. Matched mask controls — compare real-label mask, shuffled-label mask, and
   random matched-rank subspace, all evaluated on the same held-out set.
   Shuffled mask must be trained on shuffled assignments but evaluated on
   real counterfactual targets.
3. Task-specificity controls — factorial design: copy/echo (no binding, no
   belief), entity tracking (binding, no belief), first-order false belief
   (binding + belief), true-belief matched stories.

**Secondary (run only if primary experiments succeed):**
4. Failure case analysis — unchanged.
5. Cross-stage mediation (replaces super-additivity unity test) — intervene
   on binding representation, measure effect on later pointer
   representation, test whether pointer mediates binding's effect on output.

**Why.** The held-out basis validation is the most consequential experiment: a
subspace whose SVD basis overfits the story pool is not an identified mechanism
regardless of whether it beats random geometry. The super-additivity test is
replaced because independent stages in a serial pipeline can produce
super-additive loss, making the test non-diagnostic for mechanism unity.

---

## Amendment 5: Pre-committed interpretation clauses

**Problem.** The original pre-registration specified what outcomes would
disconfirm each claim but not what outcomes would confirm it. This makes
confirmatory results read as failed attacks rather than completed audits.

**Correction.** Each experiment now includes a "Pre-committed interpretations"
paragraph specifying what we conclude under both confirmatory and
disconfirmatory outcomes. Key clauses:

- If real-label subspaces beat both random and shuffled-label baselines on
  held-out data, M2 is satisfied and the belief-specificity concern narrows
  to the task-generality question (C4).
- If the lookback pattern is absent on echo/copy but present on entity
  tracking, the mechanism is entity-state retrieval---a correction of the
  label, not a refutation of the finding.
- If held-out basis validation succeeds, the SVD-basis-leakage concern is
  closed and we report M3 as satisfied for the feature space.

**Why.** An audit that can only disconfirm is a critique. Pre-committed
confirmatory interpretations convert the paper into a genuine validity audit.

---

---

## Amendment 6: Experiment 6 (compositional generalization) and audit fixes

**Problem.** The original pre-registration had five experiments. Cognitive science
review (Vegner et al. 2025, Apperly & Butterfill 2009, Schuwerk et al. 2015)
identified an untested assumption: the paper infers representational
systematicity from behavioral systematicity without testing compositional
generalization.

**Correction.** Added Experiment 6 (compositional generalization) with three axes:
1. Character scaling: 3, 4, 5 characters (distractor characters present but
   non-interacting, holding interacting characters' positions stable)
2. Nested beliefs: second-order false belief
3. Content selectivity: location, existence, intentions, knowledge states

Additionally, the following audit fixes are applied:
- Random subspace baseline (#1): changed from Grassmannian QR sampling to
  random r-of-500 SVD index selection (matching the paper's binary mask method)
- SE calculation (#2): added "(at most)" for the upper-bound SE at p=0.5
- Multiple comparisons (#4): added BH correction across all pre-registered tests
- Contracted claim (#9): stripped "two-stage retrieval process" (untested)
- Third-answer steelman (#10): expanded with QK/OV explanation, pre-committed
  quantitative criterion for "coherent" third answers

**Why.** CausalToM uses exactly two characters, first-order beliefs, and beliefs
about object location only. A compositional mechanism should generalize; a
template-matched circuit will degrade.

---

## Files changed

- `experiments/random_subspace_baseline.py` — labeling fix, null distribution fix
- `experiments/shuffled_label_control.py` — labeling fix
- `experiments/non_tom_control.py` — labeling fix
- `experiments/failure_case_analysis.py` — labeling fix
- `experiments/unity_coupling_test.py` — labeling fix
- `experiments/compositional_generalization.py` — created (Experiment 6)
- `reference/extracted_results_llama70b.json` — created (full extraction)
- `paper/lookback_audit.tex` — rank corrections, train/eval clarification,
  visibility finding, experiment rewrite with pre-committed interpretations,
  discussion rewrite for audit framing, audit fixes (SE, BH, contracted claim,
  third-answer steelman, random baseline method)
- `COMPUTE_ESTIMATE.md` — created (GPU-hours breakdown)
- `.gitignore` — added LaTeX build artifacts

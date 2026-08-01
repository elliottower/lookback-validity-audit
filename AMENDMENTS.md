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

## Amendment 7: Model substitution and scope clarification

**Problem 1 (model).** Prakash et al. used Llama-3-70B-Instruct. NDIF
provides Llama-3.1-70B-Instruct. These are different weights with
potentially different layer-wise geometry. All rank and IIA values in
Amendment 1 were extracted from the paper's released Llama-3.0 results,
but our interventions run on Llama-3.1.

**Correction.** We retain Llama-3.1 (3.0 is not available on NDIF) and
add a **same-model replication** as a positive control before running
any pre-registered experiments:

1. Reproduce the paper's headline IIA values on Llama-3.1 using
   our extracted SVD bases and top-r component selection.
2. If we replicate within 0.10 of the reported values for the
   primary subspaces (binding_addr_payload_L34, answer_pointer_L52),
   the model substitution is validated and all downstream results
   are interpretable on 3.1.
3. If we cannot replicate (delta > 0.10), this is itself a cross-model
   generalization result (analogous to E4) and we report it as such.
   In that case, experiments testing SVD basis generalization (E1)
   become uninterpretable for the overfitting question specifically,
   though the shuffled-label and task-specificity controls remain valid.

This replication runs first, before any pre-registered experiment.

**Problem 2 (visibility scope).** The SVD extraction covers binding
(state_tokens, layers 34-38) and answer (last_token, layers 34-54).
It does not cover the visibility lookback, which operates on visibility
sentence tokens at early layers (7-24) with ranks of 160-320 out of
500 components.

**Correction.** Visibility is explicitly out of scope for this audit.
The cross-stage mediation test (Experiment 5) is scoped to
binding-vs-answer coupling only. Visibility has the weakest evidence
in the original paper (ranks consuming 32-64% of the SVD basis suggest
the "subspace" is most of the representation), and characterizing it
would require a separate extraction targeting early-layer visibility
sentence positions.

**Why.** An audit on different weights than the original must establish
that the finding transfers before testing whether it generalizes. The
positive control is the minimum requirement for interpretability. The
visibility scoping prevents an apparent omission from reading as a gap.

---

## Amendment 8: Five severity tests (specificity, necessity, localization, dimensionality, discriminant validity)

**Problem.** Experiments 1--6 test whether the paper's claimed subspaces
beat appropriate null baselines (random, shuffled, non-ToM) and replicate
across seeds and compositions. They do not test five additional properties
that a genuine mechanistic account requires:

1. Whether the subspace is **necessary** for correct behavior (not just
   sufficient for interchange steering).
2. Whether the effect is **localized** to the claimed layers or broadly
   recoverable at adjacent layers.
3. Whether the full reported rank is needed or whether a **single SVD
   direction** captures the mechanism.
4. Whether the subspace transfers across **prompt templates and entity
   names** (robustness to surface features).
5. Whether the same subspace steers **unrelated tasks** (discriminant
   validity against a generic answer-pointer interpretation).

**Correction.** Five additional experiments, numbered E7--E11. No results
from these experiments have been computed or inspected at time of writing
(result directories exist but are empty).

### Experiment 7: Necessity test (zero-ablation)

**Question.** Is the identified subspace necessary for correct model
behavior, or merely correlated with it?

**Protocol.** For each of the 6 subspaces: (1) collect clean activations
at the target layer, (2) zero-ablate the subspace component at position -1
(answer subspaces) or all positions (binding subspaces):
`x_ablated = x - x @ P` where `P = V_sel @ V_sel.T`, (3) measure
post-ablation accuracy on the same 80 CausalToM pairs used in the
positive control.

**Metrics.** Clean accuracy, ablated accuracy, accuracy drop
(clean - ablated) per subspace.

**Predictions.**
- Primary answer subspaces (L38, L52): accuracy drop > 0.30. These are
  the paper's strongest subspaces and should be necessary for correct
  output if they implement the claimed mechanism.
- Secondary subspaces (L35, L36, L53): accuracy drop > 0.10 but
  potentially smaller than primary, consistent with redundant coding.
- Binding subspaces (L34--36, full-sequence ablation): accuracy drop
  > 0.15. Binding information feeds the answer pointer, so disrupting
  it should propagate.

**Pre-committed interpretations.**
- If ablated accuracy remains above 0.80 for all subspaces, the
  subspaces are not necessary and the finding reduces to "these
  directions correlate with belief information but the model does not
  rely on them." This would be a significant qualification of the
  paper's mechanistic claims.
- If ablated accuracy drops below 0.50 for L38 and L52, necessity is
  confirmed for the primary subspaces and the mechanism claim is
  strengthened beyond what interchange intervention alone establishes.
- Intermediate drops (0.50--0.80) are consistent with partial necessity
  with redundant pathways.

### Experiment 8: Layer spread test

**Question.** Is the interchange effect localized to the paper's claimed
layers (34--36, 38, 52--53) or recoverable at adjacent layers?

**Protocol.** For each of the 3 answer subspaces (L38, L52, L53), compute
IIA at layers L-2 through L+2 (5 layers each). At each adjacent layer,
use the top-k SVD directions (same rank as the paper's subspace at that
layer) computed from the same 80 evaluation pairs' activations. At the
paper's own layer, use the paper's mask.

**Metrics.** IIA per (subspace, layer) combination. The "localization
ratio": IIA at the paper's layer divided by the mean IIA at adjacent
layers.

**Predictions.**
- Paper layers should show IIA > 0.80 (matching positive control).
- Adjacent layers (offset +/-1) should show IIA < 0.50. The paper claims
  layer-specific processing stages; if information is equally accessible
  one layer away, the "stage" claim weakens to "this information is
  distributed across middle/late layers."
- Layers at offset +/-2 should show IIA < 0.30.
- Localization ratio should be > 2.0 for each subspace.

**Pre-committed interpretations.**
- If localization ratio > 2.0 for all three answer subspaces, the
  layer-specificity claim is supported.
- If localization ratio < 1.5 for any primary subspace (L38 or L52),
  the processing-stage framing is overstated: the model represents
  belief information broadly rather than computing it at discrete
  layers. This weakens the two-stage pipeline claim without
  invalidating the finding that belief-relevant subspaces exist.
- If adjacent layers show IIA > 0.80, we report this as a failure of
  layer-specificity regardless of the paper layer's IIA.

### Experiment 9: Rank-1 decomposition

**Question.** Does each SVD direction in the paper's mask contribute
meaningfully to IIA, or does one direction do most of the work?

**Protocol.** For each of the 3 answer subspaces: (1) measure full-mask
IIA (baseline), (2) measure IIA for each individual mask direction
(rank-1 projections), (3) measure leave-one-out IIA (full mask minus
each direction).

**Metrics.** Per subspace: full_mask_iia, individual direction IIAs,
leave-one-out IIAs. Derived: single-to-full ratio (best individual
IIA / full IIA), effectively_rank_1 flag (ratio > 0.90).

**Predictions.**
- Low-rank subspaces (L38, rank 3): no single direction should exceed
  0.70 of the full-mask IIA. With only 3 directions, each should
  contribute meaningfully.
- High-rank subspaces (L52 rank 18, L53 rank 19): the best single
  direction is more likely to capture a large fraction, but full IIA
  should still require multiple directions. Best single direction
  predicted at 0.40--0.60 of full.
- Leave-one-out should show small individual drops for most directions,
  with at most 2--3 directions whose removal drops IIA by > 0.10.

**Pre-committed interpretations.**
- If the best single direction achieves > 0.90 of full IIA for any
  subspace, the multi-rank claim for that subspace is inflated: the
  "subspace" is functionally a single feature direction, which is a
  qualitatively different (and less surprising) finding.
- If all individual directions contribute < 0.50 of full IIA and
  leave-one-out drops are distributed across multiple directions, the
  multi-dimensional subspace claim is supported.
- For L52/L53 specifically: if the top 3 directions together capture
  > 0.90 of IIA, the effective rank is 3, not 18--19, even though the
  mask selects more components.

### Experiment 10: Surface heuristic / adversarial robustness

**Question.** Does the subspace transfer across prompt templates and
entity names, or does it exploit surface-level features of the CausalToM
dataset?

**Protocol.** Four conditions:
1. **Template 2 baseline**: standard CausalToM template-2 pairs (should
   match positive control).
2. **Template transfer**: re-tell the same stories using template 1 (or
   a manually varied template). IIA should be preserved if the subspace
   encodes beliefs, not template tokens.
3. **Entity-name swap**: replace character names with novel names not in
   CausalToM. IIA should be preserved.
4. **Recency confound**: construct stories where the most recently
   mentioned substance before the question is NOT the belief-correct
   answer. If IIA tracks the heuristic answer rather than the belief
   answer, the subspace encodes recency, not belief.

For conditions 2--4, filtering and IIA evaluation use the **intersection
of pairs that pass behavioral filtering under both the original and
modified prompts**, so that IIA differences reflect template/name
sensitivity rather than different surviving subsets.

**Metrics.** IIA per (condition, subspace). Heuristic match rate for
condition 4 (fraction of intervened predictions matching the recency
target rather than the belief target).

**Predictions.**
- Template 2 baseline: IIA > 0.90 (matching positive control).
- Template transfer: IIA > 0.70. Some degradation is expected because
  the SVD basis was computed on template 2, but the belief content is
  identical.
- Entity-name swap: IIA > 0.85. Names are surface tokens; the
  subspace should be robust.
- Recency confound: heuristic match rate < 0.20. If > 0.50, the
  subspace tracks surface position rather than belief content.

**Pre-committed interpretations.**
- If template transfer IIA < 0.30, the subspace is template-specific
  and the "belief tracking" label is overstated: it would be more
  accurately described as "template-2 answer retrieval."
- If entity-swap IIA < 0.50, the subspace relies on specific token
  identities, suggesting memorization rather than compositional
  representation.
- If heuristic match rate > 0.50, the subspace encodes "last mentioned
  substance" rather than "believed substance," which would be a
  fundamental recharacterization of what the mechanism computes.
- If all conditions show IIA > 0.70, the subspace is genuinely robust
  to surface variation and the template-specificity concern is closed.

### Experiment 11: Cross-task contamination (discriminant validity)

**Question.** Does the belief-tracking subspace affect unrelated tasks?
If so, it may encode a generic answer-retrieval mechanism rather than
belief-specific information.

**Protocol.** Apply all 6 subspaces (answer subspaces at position -1,
binding subspaces skipped) to three non-belief task types:
1. **Factual recall**: "The capital of France is" -> "Paris" vs
   "The capital of Japan is" -> "Tokyo". 40 pairs (world capitals,
   chemical elements, animal facts).
2. **Simple arithmetic**: "15 + 23 =" -> "38" vs "42 + 17 =" -> "59".
   40 pairs.
3. **Property association**: "Grass is typically the color" -> "green" vs
   "The sky is typically the color" -> "blue". 40 pairs.

Filter on model accuracy. Compute IIA using `compute_iia_answer_flex`
(handles variable token lengths). The key question: does swapping the
belief subspace at L52 make the model say "Tokyo" when prompted about
France?

**Metrics.** IIA per (task, subspace). Chance baseline is 0.0 (swapping
should have no effect on unrelated tasks).

**Predictions.**
- All three tasks: IIA < 0.15 for all answer subspaces. The belief
  subspace should not steer factual recall, arithmetic, or property
  association.
- If binding subspaces are tested (full-sequence ablation variant):
  IIA < 0.10.

**Pre-committed interpretations.**
- If IIA < 0.15 across all tasks and subspaces, the belief-specificity
  claim is supported: the subspace does not function as a generic
  answer pointer.
- If IIA > 0.40 on factual recall but < 0.15 on arithmetic, the
  subspace encodes "entity-associated retrieval" (a broader category
  than belief tracking but narrower than generic answer pointing).
  This would be a meaningful refinement of the mechanism's scope.
- If IIA > 0.50 on all three tasks, the subspace is a generic
  last-token answer mechanism and the "belief tracking" label is
  incorrect. This would be the strongest possible falsification of
  the paper's specificity claims.

---

**Methodological notes applying to all five experiments.**

1. All experiments use the same 80 model-filtered CausalToM pairs as the
   positive control (seed 42, 240 generated, filtered to 80).
2. All answer-subspace IIA uses the last-token intervention protocol
   (`compute_iia_answer` or `compute_iia_answer_flex`).
3. All projections use our reconstructed top-r SVD masks (see Amendment 7
   caveat: these approximate but do not exactly reproduce the paper's
   learned binary masks).
4. Results are checkpointed per subspace/condition to survive NDIF
   instability.
5. BH correction is applied across all tests within each experiment.

**Design fix (E10).** An earlier draft independently filtered original
and entity-swapped pairs, making the IIA comparison unpaired. The
registered version evaluates on the intersection of surviving pairs
under both conditions.

**Why.** Sufficiency (interchange works) is the weakest form of causal
evidence. These five experiments test whether the identified subspaces
are also necessary, localized, multi-dimensional, robust, and
belief-specific. Together with E1--E6, they constitute a comprehensive
severity test of the mechanistic claim.

---

## Files changed (Amendments 1--7)

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

## Files changed (Amendment 8)

- `experiments/necessity_test.py` — created (Experiment 7)
- `experiments/layer_spread_test.py` — created (Experiment 8)
- `experiments/rank_decomposition_test.py` — created (Experiment 9)
- `experiments/adversarial_heuristic_test.py` — created (Experiment 10)
- `experiments/cross_task_contamination.py` — created (Experiment 11)

---

## Amendment 9: Belief-vs-binding dissociation experiments (E12--E15)

**Date.** 2026-07-31, prior to any results from E12--E15.

**Motivation.** Inspection of the CausalToM data generation code
(`src/dataset.py`, `story_templates.json`) reveals that the paper's
experimental task does not test false beliefs. The `causal_event` and
`event_noticed` template fields (which describe drink swaps and whether
a character noticed) are defined in the template JSON but never included
in generated stories: `set_story()` uses only `template["context"]`.
The paper always queries a character about the container they themselves
filled (`set_character=rc, set_container=rc`), so the belief-correct
answer always equals the ground-truth state. No story contains a state
change the character fails to observe.

The task therefore reduces to entity binding: tracking which character
put which drink in which container. A subspace that achieves high IIA on
this task need not encode epistemic state; encoding {character, container,
drink} associations suffices. This observation motivates four experiments
that dissociate belief tracking from entity binding.

### Experiment 12: Question-framing control (belief vs non-belief)

**Question.** Does the subspace encode "what the character believes" or
"what drink is associated with this character and container"? Since the
paper's task uses stories where belief = reality (the character is always
asked about their own container), a subspace could achieve perfect IIA by
tracking entity-state bindings without representing epistemic state.

**Protocol.** Use the same clean/CF story pairs from the positive control
(template 2, seed 42, 80 model-filtered pairs). For each pair, generate
four question framings by replacing only the question text:

1. **Belief (original):** "What does [char] believe the [container]
   contains?" (baseline)
2. **Action recall:** "What did [char] put in the [container]?"
3. **Reality state:** "What is inside the [container]?"
4. **Fill completion:** "[Char] filled the [container] with"

All four framings yield the same correct answer because belief = reality
for the queried character's own container. Pairs are filtered for each
framing independently (the model must answer both clean and CF correctly
under the new question). IIA is computed on the intersection of pairs
passing all four framings.

Only answer subspaces (L38, L52, L53) are tested at position -1. Uses
`compute_iia_answer_flex` because different question framings produce
different token lengths.

**Metrics.** IIA per (framing, subspace). Framing transfer ratio:
non-belief IIA / belief IIA.

**Predictions.**
- Belief baseline: IIA > 0.80 (matching positive control).
- Action recall: IIA within 0.10 of belief baseline. The subspace
  encodes the binding regardless of epistemic framing.
- Reality state: IIA within 0.10 of belief baseline.
- Fill completion: IIA within 0.10 of belief baseline.

**Pre-committed interpretations.**
- If all non-belief framings show IIA within 0.10 of the belief
  framing, the subspace encodes entity-state bindings, and the "belief
  tracking" characterization adds no explanatory power beyond "entity
  retrieval." This would not invalidate the subspace's existence but
  would recharacterize what it computes.
- If belief framing IIA exceeds non-belief framings by > 0.25 for at
  least two subspaces, the subspace is sensitive to epistemic question
  framing, supporting the belief-tracking interpretation.
- If the fill-completion framing (which removes the question frame
  entirely) shows IIA > 0.80, the subspace functions as a cloze
  completion mechanism, independent of any question structure.

### Experiment 13: Distractor insertion (positional heuristic test)

**Question.** Does the subspace use the semantic character-drink binding
or the positional location of drink tokens in the prompt?

**Protocol.** Take model-filtered template-2 pairs. For each pair,
create three distractor variants by inserting sentences that mention
drinks in non-binding contexts:

1. **Early distractor:** After the opening sentence ("... busy
   restaurant."), insert: "A customer nearby was drinking [distractor_drink]."
2. **Mid distractor:** Between the two fill actions, insert:
   "The smell of [distractor_drink] wafted from the kitchen."
3. **Late distractor:** After both fill actions but before the question,
   insert: "Another order called for [distractor_drink]."

The distractor drink is sampled from the drink pool, excluding drinks
used in the story. Each variant is re-filtered on the model (must still
answer both clean and CF correctly). IIA is computed on the intersection
of pairs passing the original and each distractor condition.

**Metrics.** IIA per (distractor position, subspace). Distractor drop:
original IIA minus distractor IIA.

**Predictions.**
- All distractor conditions: IIA within 0.10 of the no-distractor
  baseline. The subspace should track semantic binding (who filled
  what), not positional drink-token location.
- If the late distractor (closest to the question) causes the largest
  drop, the subspace has a recency bias toward drink tokens near the
  query position.

**Pre-committed interpretations.**
- If distractor drop > 0.20 for any position, the subspace relies on
  positional features of the prompt. This would indicate that the
  mechanism is a positional lookup ("what drink token appeared at
  position P?") rather than a semantic binding.
- If all distractor drops < 0.10, the subspace is robust to positional
  perturbation.

### Experiment 14: True false-belief test

**Question.** Can the subspace handle stories with genuine false beliefs,
where a character's belief differs from reality?

**Protocol.** Construct stories by extending template 2 with the
causal_event the paper defined but never used:

> "[Char1] and [Char2] are working in a busy restaurant. To complete an
> order, [Char1] grabs an opaque [Container1] and fills it with [State1].
> Then [Char2] grabs another opaque [Container2] and fills it with
> [State2]. While [Char1] was attending to another task, a co-worker
> swapped the [State1] in the [Container1] with [State3]. [Char1] did
> not notice the swap."

Now char1 falsely believes container1 has state1, but it actually
contains state3. The question "What does [Char1] believe [Container1]
contains?" should yield state1 (the false belief).

Clean/CF pairs:
- Clean: char1 believes state1 (pre-swap drink)
- CF: reversed characters, different drinks; char_cf believes state_cf

Generate 240 stories (seed 42), filter on model accuracy (model must
answer with the pre-swap drink, demonstrating false-belief reasoning),
retain up to 80. Compute IIA with the paper's answer subspaces.

**Metrics.** IIA per subspace. Behavioral accuracy: fraction of stories
where the model correctly answers with the false belief (pre-swap drink).

**Predictions.**
- Behavioral accuracy > 0.50. LLama-3.1-70B-Instruct should handle
  simple false-belief stories.
- IIA on false-belief pairs: < 0.30 for all answer subspaces. The
  paper's subspace was trained on entity-binding pairs (belief = reality)
  and should not generalize to genuine epistemic state tracking where
  belief diverges from reality.

**Pre-committed interpretations.**
- If behavioral accuracy < 0.30, the model itself cannot reliably
  perform false-belief reasoning on this story format. Report the
  behavioral failure and note that the subspace cannot be tested.
- If behavioral accuracy > 0.50 AND IIA > 0.60 for at least one
  subspace, the subspace genuinely tracks the character's epistemic
  state even when it diverges from reality. This would be the strongest
  possible evidence FOR the paper's "belief tracking" interpretation.
- If behavioral accuracy > 0.50 AND IIA < 0.30, the subspace trained
  on entity-binding does not generalize to false beliefs. Combined
  with E12 showing framing-invariance, this would support the
  recharacterization: the subspace encodes character-state bindings
  that happen to align with beliefs in the paper's task, but does not
  encode epistemic state per se.

### Experiment 15: Observability dissociation

**Question.** Does the subspace encode who can observe what — a core
component of belief tracking — or only self-action bindings?

**Protocol.** Use template 1, where char1 CAN observe char2's actions
but char2 CANNOT observe char1's. This creates an asymmetry:
- Char1 knows what char2 put in container2 (observed it)
- Char2 does not know what char1 put in container1

Construct pairs asking about char1's knowledge of container2 (the
container char2 filled, which char1 observed):
- Clean: "What does [Char1] believe [Container2] contains?" → state2
- CF: reversed chars, new states → different answer

Also construct a control condition asking about char1's knowledge of
container1 (which char1 filled themselves):
- "What does [Char1] believe [Container1] contains?" → state1

This matches the paper's standard query pattern (self-filled container)
and serves as a positive control: if the subspace works on self-queries,
any deficit on observed-other queries isolates the observability gap.

For both conditions: generate 240 template-1 pairs, filter on model
accuracy, retain up to 80.

**Metrics.** IIA per subspace for both conditions. Dissociation =
self-container IIA minus observed-other-container IIA.

**Predictions.**
- Behavioral accuracy (both conditions): > 0.50.
- IIA (self-container control): within 0.10 of the positive-control
  baseline from E1. Template 1 self-queries match the paper's pattern.
- IIA (observed-other-container): < 0.30. The paper's subspace was
  trained on self-filled-container pairs (set_character=rc,
  set_container=rc). Observability-mediated knowledge of ANOTHER
  character's container is a different binding pattern.

**Pre-committed interpretations.**
- If self-container IIA matches baseline AND observed-other IIA > 0.60,
  the subspace encodes observability-mediated beliefs, supporting the
  "belief tracking" interpretation.
- If self-container IIA matches baseline AND observed-other IIA < 0.30,
  the subspace only encodes self-action bindings. It tracks "what did I
  put where?" but not "what did I see someone else put where?" — the
  latter requires modeling observability, which the subspace does not
  capture.
- If behavioral accuracy < 0.30 for observed-other queries, template 1
  produces unreliable model behavior and the observability test is
  uninformative. Report as inconclusive.

---

**Methodological notes applying to E12--E15.**

1. All experiments use the same SVD-reconstructed projections and
   compute_iia_answer_flex as E7--E11.
2. All filtering requires correct model predictions on both clean and
   CF prompts under the experimental condition.
3. E12 evaluates on the intersection of pairs passing all framings,
   ensuring paired comparison. E13 compares baseline IIA (all
   model-filtered pairs) against each distractor condition IIA (the
   subset of those pairs where the model still answers correctly with
   the distractor inserted). Distractor pairs are a subset of baseline
   pairs by construction.
4. Results are checkpointed per condition/subspace.
5. E14 extends the template-2 context string with the causal_event
   text. Because this changes prompt length, compute_iia_answer_flex
   handles the length mismatch.

**Observation that motivates all four experiments.** The paper's
CausalToM stories never include drink swaps or unobserved state changes.
The `causal_event` and `event_noticed` fields exist in the template JSON
but `Dataset.set_story()` only uses `template["context"]`. The question
always queries a character about the container they filled
(`set_character=rc, set_container=rc`), so belief = reality for every
evaluated pair. The task is entity binding, not false-belief reasoning.
This does not invalidate the paper's subspace findings — the subspaces
exist and achieve high IIA — but it reframes what the subspaces compute.
E12--E15 test whether the "belief" characterization adds explanatory
power beyond "entity-state binding."

## Files changed (Amendment 9)

- `experiments/question_framing_test.py` — created (Experiment 12)
- `experiments/distractor_insertion_test.py` — created (Experiment 13)
- `experiments/false_belief_test.py` — created (Experiment 14)
- `experiments/observability_test.py` — created (Experiment 15)

---

## Amendment 10: Framework version, prompt format fix, and provenance corrections

**Date.** 2026-08-01.

### 10a: Mechanistic validity framework v10 → v11

**Problem.** The audit was designed against version 10 of the mechanistic
validity framework (31 criteria: C1--C5, M1--M7, I1--I10, E1--E6,
V1--V5). Version 11 adds three criteria:

- **I3 (Minimality):** the identified mechanism uses no more components
  than necessary
- **I5 (Rival mechanism exclusion):** no alternative subspace of the
  same rank achieves comparable IIA
- **I11 (Onset-offset coupling):** the mechanism appears and disappears
  with the capability it supports

Of these, I5 was already addressed by Experiment 5 (rival mechanism
test) under a different name. I3 was partially addressed by the rank
decomposition test (E9). I11 requires either training checkpoints
(emergence direction, infeasible for Llama-3.1-70B) or fine-tuning away
the capability (disappearance direction, feasible).

**Correction.** The criterion set is updated from 31 to 34. Results
computed before this amendment (E1--E15, I8 confounding sensitivity)
were designed and executed under v10 numbering. This amendment records
the version change and maps existing experiments to the new criteria:

| v11 criterion | Status | Experiment |
|---------------|--------|-----------|
| I3 Minimality | Covered by E9 (rank decomposition) | rank_decomposition_test.py |
| I5 Rival mechanism exclusion | Covered by new script | rival_mechanism_test.py |
| I11 Onset-offset coupling | New script (disappearance direction only) | onset_offset_test.py |

No existing results are invalidated by this version change. The scoring
table in the paper will use v11 numbering.

### 10b: Compositional generalization prompt format fix

**Problem.** All compositional generalization conditions except
`content_existence` (yes/no answers) produced n_pairs=0 after model
filtering. Root cause: the CausalToM paper's prompts use a structured
format with an instruction prefix and "Answer:" suffix that elicits
single-token responses. The compositional generator used bare narrative
prompts, causing the model to produce multi-token answers that failed
exact-match filtering. For `content_intention`, the expected answers
were multi-word phrases ("cook dinner") that cannot match a single
argmax token.

**Correction.**
1. All compositional generators now wrap prompts in the CausalToM
   structured format: `Instruction: ...\n\nStory: ...\nQuestion: ...\nAnswer:`
2. The instruction text matches the CausalToM paper's instruction for
   substance-name answers, with appropriate variants for yes/no and
   single-word conditions.
3. `content_intention` answers changed from multi-word phrases to
   single-word activities ("cooking", "cleaning", etc.).
4. Added `filter_on_model_diagnostic` that logs pre-filter counts,
   post-filter counts, and sample rejections (first 5) per condition.
5. Stale results from the broken run moved to
   `results/compositional/stale_v1/`.

### 10c: I8 confounding sensitivity labeled exploratory

**Problem.** The E-value confounding sensitivity analysis (I8) ran from
an uncommitted script (`experiments/confounding_sensitivity.py`). The
script was not part of any pre-registration tag at time of execution.

**Correction.** I8 results are labeled exploratory in the paper text.
The script has since been committed and is included in this amendment,
but the results themselves cannot retroactively become pre-registered.
All conclusions drawn from I8 are hedged accordingly.

### 10d: Pre-registration tag provenance note

**Problem.** The tag `prereg-amendment-2` was moved to a different SHA
during development. Git tags are mutable references, and moving a
pre-registration tag destroys the provenance chain.

**Correction.** This cannot be fixed retroactively. Future amendments
use a new tag with `-corrected` suffix rather than moving existing tags.
The moved tag is documented here for transparency.

## Files changed (Amendment 10)

- `experiments/compositional_generalization.py` — prompt format fix,
  diagnostic filter, single-word intention answers
- `AMENDMENTS.md` — this amendment
- `experiments/confounding_sensitivity.py` — now committed (was untracked)
- `experiments/rival_mechanism_test.py` — committed (I5)
- `experiments/sufficiency_test.py` — committed (I2 graded)
- `experiments/rescue_reversibility_test.py` — committed (I10 with
  random-subspace control)
- `experiments/double_dissociation_test.py` — committed (I6, relabeled
  as single dissociation)
- `experiments/convergent_validity_probe.py` — committed (C3)

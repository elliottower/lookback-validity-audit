# Mechanistic Validity Audit V2: Language Models Use Lookbacks to Track Beliefs

Paper: Prakash, Shapira, Sen Sharma, Riedl, Belinkov, Rott Shaham, Bau, Geiger. ICLR 2026. arXiv:2505.14685.

Frameworks applied: MechVal (7-tier verdict), MechRef (transport + failure modes), CVI (cross-view invariance depth).

This is the comprehensive "rip it to shreds" version. V1 was diplomatic; this version states the problems directly.

---

## TL;DR — What This Paper Actually Shows vs. What It Claims

**What it shows:** In Llama-3-70B-Instruct, swapping residual stream vectors at specific layer ranges between two templated stories (from a pool of 80 pre-filtered correct-answer cases) changes the model's belief-tracking output in ways consistent with a hypothesized three-stage retrieval process. The pattern qualitatively replicates in two other models at different layer ranges.

**What it claims:** LMs use "a pervasive computation, the lookback mechanism — pointer dereference via attention — for belief tracking" that "takes a step toward reverse-engineering ToM reasoning in LMs." The mechanism is described as a unified algorithmic pattern that the model employs systematically.

**The gap:** Every word of the claim beyond what the evidence shows is interpretive inflation. "Pervasive" from 80 filtered stories. "Pointer dereference" from residual-stream swaps. "Mechanism" from three dissociable sub-effects at different layers. "ToM reasoning" from a construct that imports cognitive-science vocabulary without licensing it.

---

## MechVal Verdict: Causally Suggestive (Tier 4)

### Construct Validity

**C1 (Construct definition): PARTIALLY MET — with a serious problem.**

"Lookback mechanism" is defined operationally through the interchange interventions that identify it. The construct imports two heavily loaded terms:

1. **"Pointer dereference"** imports computational specificity from programming languages. A pointer dereference is a specific operation: allocate a reference, store an address, follow the reference to retrieve a value at that address. The evidence shows that information flows from state tokens to the final token via attention in certain layers. Calling this "pointer dereference" is like calling every database query a "B-tree traversal" — it imports implementation detail the evidence doesn't speak to.

2. **"Belief tracking"** imports cognitive-science vocabulary (Theory of Mind, Premack & Woodruff 1978) without licensing it. The model passes a behavioral ToM test. The paper then assumes the internal mechanism deserves the cognitive-science label "belief." This is a construct-validity error: behavioral equivalence does not entail mechanistic equivalence. The model could be doing pattern-matching on story templates without anything resembling belief representation.

**C2 (Operationalization): MET.** The causal model (Fig. 3) is explicit. Each step has a specific predicted intervention outcome. The operationalization is clear even if the construct labels are inflated.

**C3 (Convergent evidence): NOT MET.**

All evidence comes from a single method family: interchange interventions on residual stream vectors via NNsight. The DCM subspace identification is a variant of the same method (optimizing a mask to maximize the same IIA metric). There is:
- No weight analysis (which attention heads implement the "lookback"?)
- No attention pattern analysis (do the predicted QK circuits actually form?)
- No independent probing (does a linear probe find the same OI structure DCM identifies?)
- No training dynamics (when does the lookback emerge during pre-training?)
- No ablation studies (what happens when you knock out specific heads?)

Every experiment in the paper is: "swap a vector, measure IIA." The experiments differ in which vector and which token, but the method is always the same.

**C4 (Discriminant validity): NOT MET.**

No test of whether the "lookback" pattern is absent in tasks where it should not appear. This is the single most damaging gap. Without it, "lookback" could be a generic attention-based retrieval pattern present in ALL in-context reasoning, not specific to belief tracking. Candidate negative controls:

- Factual recall without belief tracking (e.g., "What does the bottle contain?" asked of the narrator, not a character)
- Simple copying tasks
- Stories with only one character (where binding should be trivial and the "mechanism" should be much simpler or absent)
- Tasks where ToM is irrelevant (arithmetic, translation)

If the "lookback" pattern appears in all of these, it's attention-mediated retrieval, not a belief-tracking-specific mechanism.

**C5 (Nomological network): PARTIALLY MET.** The paper connects to ordering IDs (Dai et al. 2024), entity tracking (Prakash et al. 2024), variable binding (Feng & Steinhardt 2023). But the relationship is asserted, not tested. Does the lookback reuse the same OIs that entity tracking uses? Or are these labels for overlapping but distinct phenomena?

### Measurement Validity

**M1 (Instrument calibration): NOT MET.**

IIA is the sole metric. No calibration of what IIA value constitutes meaningful alignment. The paper uses visual inspection of line plots — "the IIA goes up at layer X" — without reporting confidence intervals, thresholds, or effect sizes. For n=80, even the standard error on IIA is ~5.6pp (sqrt(0.5*0.5/80)), so values below ~0.60 are within noise of chance for a binary outcome.

**M2 (Baseline control): NOT MET.**

**The critical missing baseline: random subspace control.** The paper identifies DCM subspaces of dimension 14, 20, 33, and 167. Without testing whether a random subspace of the same dimensionality achieves comparable IIA, the subspace identification proves nothing. This is especially concerning for the 167-dimensional subspace (Fig. 6b): in a 5120-dimensional residual stream, 167 dimensions is 3.3% of the space. A random 167-dimensional subspace may well capture enough information to transfer belief-relevant signal.

The Hofmann et al. 2025 result — that unconstrained DAS achieves 100% IIA on random models — is directly relevant. DCM adds sparsity constraints, but the fundamental concern remains: gradient-based optimization over subspaces will find the subset that maximizes IIA regardless of whether that subset has mechanistic significance.

Furthermore: in most figures (4b, 5b, 8b), the full residual stream intervention (gray line) and the subspace intervention achieve nearly identical IIA. This means the subspace adds no information beyond what the full swap already provides. The "identified subspace" is a subset of the full swap, not evidence of targeted identification.

**M3 (Measurement independence): NOT MET — Selection on the Dependent Variable.**

All 80 CausalToM examples were pre-filtered to cases "that the model answers correctly" (p.3). This is selection on the dependent variable: the analysis only includes cases where the model's output matches the expected answer. The problems:

1. The "mechanism" identified may be specific to success cases. Failure cases might use a completely different internal pathway — or the "lookback" might be absent, which would be genuinely informative about whether it's the mechanism vs. a correlated process.
2. The causal model was developed and tested on identical data. No held-out split. Every IIA result is in-sample.
3. n=80 is small enough that the specific stories matter. Different random subsets of 80 correct-answer stories could yield different layer ranges, different subspace dimensions, different IIA profiles. No sensitivity analysis checks this.

**M4 (Effect size reporting): PARTIALLY MET.** IIA values are plotted (0-1 scale), which is informative. But no confidence intervals, no error bars across samples, no statistical tests. For n=80, the uncertainty on each IIA value is substantial.

**M5 (Replication): PARTIALLY MET.** Cross-model replication on Qwen2.5-14B and Llama-3.1-405B (Appendix N). But different layer ranges and different subspace dimensions across models — if the "mechanism" is the same, why does it live in completely different architectural locations?

**M6 (Sensitivity analysis): NOT MET.** No analysis of how results change with different sample sizes, different story structures, different DCM initialization seeds, or different subspace dimensions.

### Internal Validity

**I1 (Necessity): PARTIALLY MET.**

Patching at specific layers changes the output — but the intervention swaps the ENTIRE residual stream vector (or a large subspace of it). This transfers ALL information in the vector, not just the hypothesized "pointer" or "OI." The paper cannot distinguish between:
- (a) The model uses a specific pointer structure, and the swap transfers that pointer
- (b) The model uses any of the information in the residual stream, and the swap happens to include it

Option (b) is consistent with all the data and makes no mechanistic claim beyond "information flows through the residual stream."

**I2 (Sufficiency): MET** for the causal model predictions. When the correct subspace is patched at the predicted layers, the output changes as predicted.

**I3 (Specificity / negative controls): NOT MET.**

- No test on tasks where the lookback should NOT appear
- No test on scrambled/nonsensical stories (does the mechanism persist when the story makes no semantic sense?)
- No test on stories with only one character
- No test on non-ToM tasks that also require multi-step retrieval

**I4 (Alternative mechanisms): NOT MET.**

The paper does not test whether a simpler explanation produces the same results. The simplest alternative: "the model attends to the most recently mentioned state token and retrieves its content." This requires no pointer/address/payload decomposition, no OI structure, no binding — just attention to relevant tokens. The paper presents the pointer/address/payload decomposition as the only candidate mechanism.

**I5 (Confound control): PARTIALLY MET.**

The visibility lookback experiments (Sec. 6) provide a partial confound control by testing a more complex condition. However, the fundamental confound — that interchange interventions transfer ALL information in the swapped vector — is unaddressed.

### External Validity

**E1 (Cross-distribution): NOT MET.**

CausalToM is 80 synthetic, templated stories with uniform structure: two characters, two objects, two states, formulaic sentences. Every story follows the same template:
- "[Character1] and [Character2] are working in a [setting]."
- "[Character1] grabs an opaque [Object1] and fills it with [State1]."
- "[Character2] grabs another opaque [Object2] and fills it with [State2]."
- "Question: What does [Character1] believe [Object2] contains?"

The paper claims the lookback is "a pervasive computation... for belief tracking" and "a fundamental role in in-context reasoning." From 80 fill-in-the-blank exercises.

BigToM generalization is mentioned (Appendix M) but not shown in the main text and not evaluated with the same intervention experiments — only behavioral accuracy.

**E2 (Scope boundaries): PARTIALLY MET.** The paper investigates no-visibility vs. explicit-visibility conditions, which is a genuine scope exploration.

**E3 (Cross-model): PARTIALLY MET.** Three models tested, but with different layer ranges and subspace dimensions. Role-level generalization at best, not structural.

**E4 (Ecological validity): NOT MET.** No naturalistic text. No multi-turn dialogue. No ambiguous pronouns. No more than two characters. No temporal complexity (all stories happen in a single temporal frame). The construct "belief tracking" in the real world involves all of these.

### Interpretive Validity

**V1 (Label accuracy): FAILS.**

"Pointer dereference" implies a specific computational operation from programming languages. The evidence shows attention-mediated information flow. Accurate labels:
- "Lookback" → "attention-mediated retrieval"
- "Pointer" → "query vector that attends to a specific position"
- "Address" → "key vector at the attended position"
- "Payload" → "value vector content at the attended position"
- "Dereference" → "attention"

These relabelings make the finding sound less novel because they reduce to: "the model uses attention to retrieve information." Which is... what attention does.

**V2 (Granularity match): PARTIALLY MET.** Claims are about layer ranges and subspaces; evidence is at that granularity. But "mechanism" implies component-level understanding (specific heads, specific weights) that is absent.

**V3 (Alternative interpretations): NOT MET.** The simpler interpretation — "LMs retrieve information by attending to relevant tokens, and the apparent structure is an artifact of the templated dataset's fixed token positions" — is not discussed. In CausalToM, the character tokens are ALWAYS in positions 1-3, the object tokens in positions 15-20, the state tokens in positions 22-27. The "mechanism" could partly reflect positional regularity in the dataset, not a general computational pattern.

**V4 (Interpretive inflation): FAILS.**

Multiple levels of inflation:
1. "Mechanism" from Role-level evidence (information flow through layers)
2. "Pervasive" from 80 templated stories
3. "Pointer dereference" from residual-stream swaps
4. "Belief tracking" from a behavioral ToM test
5. "Takes a step toward reverse-engineering ToM reasoning" from a single synthetic dataset

The conclusion (p.10): "we have mapped the end-to-end underlying mechanism responsible for the processing of partial knowledge and false beliefs" — this is an extraordinary claim from 80 stories, one method family, no negative controls, and no random baselines.

**V5 (Scope discipline): PARTIALLY MET.** The paper correctly restricts to three models. But the abstract and conclusion generalize far beyond the evidence: "pervasive," "fundamental role in in-context reasoning."

### Score Summary

| Category | Confirmed | Partial | Not Met | Fails |
|----------|:---------:|:-------:|:-------:|:-----:|
| Construct (C1-C5) | 1 | 2 | 2 | 0 |
| Measurement (M1-M6) | 0 | 2 | 4 | 0 |
| Internal (I1-I5) | 1 | 2 | 2 | 0 |
| External (E1-E4) | 0 | 2 | 2 | 0 |
| Interpretive (V1-V5) | 0 | 2 | 1 | 2 |
| **Total** | **2** | **10** | **11** | **2** |

V1 reclassified from V2's partial to clear FAIL because the existing audit was too generous — "pointer dereference" is not a partially accurate label, it's a specifically wrong one that imports computational claims the evidence cannot support.

---

## The Unity Problem: Is "Lookback" One Mechanism or Three?

The paper identifies three "lookback mechanisms":

| Lookback | Layers | Tokens | Subspace dim | What it does |
|----------|--------|--------|:------------:|--------------|
| Binding | 33-38 | State tokens | ≤14 | Transfers OIs from character/object tokens to state tokens |
| Answer | 52-56 | Final token (":") | ≤33 | Retrieves state OI at final token and dereferences to state value |
| Visibility | 10-31 | Visibility sentence | ≤167 | Transfers observed character's OI to observing character's awareness |

These operate at completely different layer ranges (10-31 vs. 33-38 vs. 52-56), on different tokens (visibility sentence vs. state tokens vs. final token), with different subspace dimensions (167 vs. 14 vs. 33). What makes them "the same mechanism"?

The paper's answer: they all involve "lookback" — attention to an earlier token to retrieve information. But that describes attention itself. Every use of the QK circuit to attend to a previous position and retrieve content through the OV circuit is a "lookback" by this definition. Bundling three dissociable sub-effects under one name because they all use attention is like calling every use of multiplication "the same algorithm."

**The critical question:** If you ablate the binding lookback (layers 33-38), does the answer lookback (layers 52-56) still function? If the three lookbacks are genuinely one mechanism, disrupting the early stage should cascade to the later stages. If they're independent processes that happen to all use attention, they should be separately disruptable. The paper never tests this.

**Mechanistic reference diagnosis:** This is the core mechanistic reference problem. "Lookback" as a label stitches together three phenomena that share a naming convention but may not share an underlying computational type. The unity claim is assumed, not tested.

---

## Selection on the Dependent Variable (n=80)

Page 3: "All subsequent experiments are conducted on 80 samples that the model answers correctly."

This is textbook selection bias. The paper:
1. Filters to correct-answer cases only
2. Finds a mechanism in those cases
3. Claims this is "the mechanism" for belief tracking

What's missing:
- **Failure cases**: What does the model do differently on the stories it gets wrong? If the "lookback" is present in failure cases too, it's not the mechanism — it's a correlated process. If the lookback is absent in failure cases, that's actual evidence for necessity (but they don't check).
- **Sample sensitivity**: Would a different 80 correct-answer stories yield the same layer ranges and subspace dimensions? With n=80, the specific stories matter.
- **Selection justification**: The paper says they selected correct-answer cases because "we analyze LMs' ability to reason about characters' beliefs" — but the mechanism question is equally about WHY the model fails. A mechanism that only explains success is incomplete.

**The deeper problem:** By conditioning on correct answers, the paper conditions on the OUTPUT. Any internal process that correlates with correct answers will look like "the mechanism," even if it's one of several redundant pathways. The paper has no way to distinguish "the mechanism" from "a mechanism" from "a correlate of the mechanism."

---

## No Falsification Target

The paper never specifies what result would have falsified the lookback hypothesis. Consider each possible outcome:

| Outcome | Paper's interpretation |
|---------|----------------------|
| High IIA at layers 33-38 | "The binding lookback operates at these layers" |
| Low IIA at layers 33-38 | "The binding lookback doesn't operate at these layers" (no falsification — it could operate elsewhere) |
| High IIA with full residual stream | "The mechanism uses the full residual stream" |
| High IIA with subspace | "The mechanism uses a specific subspace" |
| Subspace IIA ≈ full IIA | "The subspace captures the relevant information" |
| Subspace IIA < full IIA | "Some information is outside the subspace" |

Every outcome is interpreted as supporting the hypothesis. This is not a falsifiable framework. A strong mechanistic claim should specify: "if the lookback hypothesis is wrong, we would expect [specific pattern X]. We test for X and do not find it."

---

## MechRef Analysis: Transport and Failure Modes

### Transport Level Assessment

**Object level:** Established within the narrow scope (80 stories, 3 models). Layer ranges and subspaces are identified.

**Role level:** Three roles claimed (binding/answer/visibility lookback). Partially established — the roles do show qualitatively similar IIA patterns across models. But the roles are so broadly defined ("retrieve information via attention") that almost any model would satisfy them.

**Subspace level:** DCM identifies specific subspaces, but these differ across models in dimensionality and location. No stability analysis across random seeds. Subspace transport: NOT ESTABLISHED.

**Structural level:** NOT ESTABLISHED. No computational graph. No specific attention heads identified. No QK/OV circuit analysis. No gauge-invariance test.

**Process level:** NOT ESTABLISHED. No training dynamics. No checkpoint analysis.

### Failure Mode Classification

**Primary: Claim Laundering (Role → Structural)**

The evidence establishes Role-level findings: "in certain layer ranges, swapping residual stream representations transfers belief-tracking information between runs." The paper's claims are at the Structural level: "LMs implement a pointer dereference mechanism." This is a two-level inferential leap (Role → Subspace → Structural) without the intermediate evidence.

The laundering operates through vocabulary: calling attention "pointer dereference" and calling residual stream content "address/payload" makes Role-level evidence sound like Structural-level understanding. Strip the metaphor and the finding is: "swapping vectors at certain layers changes the output."

**Secondary: Potential Mimic**

Without a random subspace baseline, the identified subspaces could be mimics — gradient-optimized subsets that maximize IIA without mechanistic significance. The 167-dimensional subspace for source reference is especially suspicious: at 3.3% of the residual stream, a random subspace of this size likely captures substantial signal.

**Tertiary: Construct Import**

"Belief tracking" and "Theory of Mind" are imported from cognitive science without validating the transfer. The model passes a behavioral ToM test. The internal mechanism is then labeled with cognitive-science terminology. This creates an appearance of deep understanding ("we found the belief-tracking mechanism") that the evidence doesn't support. The mechanism might be better described as "templated retrieval" or "structured attention routing."

### Contracted Claim (Algorithm 1)

**Original (from abstract + conclusion):** "LMs use a pervasive computation, the lookback mechanism — pointer dereference via attention — for belief tracking. We have mapped the end-to-end underlying mechanism responsible for the processing of partial knowledge and false beliefs."

**Contracted:** "In Llama-3-70B-Instruct, on 80 pre-filtered correct-answer synthetic stories with fixed template structure, interchange interventions on residual stream vectors at state tokens (layers 33-38) and at the final token (layers 52-56) alter belief-tracking outputs in ways consistent with a hypothesized two-stage retrieval process. A third stage (visibility lookback, layers 10-31) operates when explicit visibility conditions are present. Qualitatively similar patterns appear in two other models at different layer ranges and subspace dimensions. No random baseline, negative control, or failure-case analysis was performed."

---

## CVI Analysis: Cross-View Invariance Depth

### Claims Decomposed

**Claim 1: Binding lookback — LMs bind character-object-state triples via OIs at state tokens**
- Support: {Object (interchange intervention at state tokens, 1 family)}
- δ = 1. Single view, single family.
- Realism: No.

**Claim 2: Answer lookback — LMs dereference state OI via pointer at the final token**
- Support: {Object (interchange intervention at final token, same family)}
- δ = 1. Same method, different token. Does not increase δ.
- Realism: No.

**Claim 3: Lookback operates via low-rank subspaces (DCM-identified)**
- Support: {Subspace (DCM mask optimization, same method family as interchange interventions)}
- δ = 1. DCM is interchange interventions with a learned mask — same family.
- Realism: No.

**Claim 4: The three lookbacks constitute a unified mechanism**
- Support: None tested. The paper never tests whether disrupting one lookback affects the others. The unity is assumed from shared vocabulary, not demonstrated.
- δ = 0 for the unity claim.
- Realism: No.

**Claim 5: The mechanism generalizes across models**
- Support: {Object (cross-model, same method)}
- δ = 1 additional (same method, different system). But different layer ranges and subspace dims undermine structural equivalence.
- Realism: No.

### Invariance Depth Summary

Maximum δ across all claims: 1
Realism inference: NOT licensed (requires δ ≥ 3 from 3 distinct families)

### What Would Increase δ

To δ = 2:
- Weight analysis: identify which attention heads implement the lookback. Show QK composition between pointer and address positions. (adds Structural view, independent family)
- Or: independent probing — train a probe to predict OI roles without using interchange interventions. (adds independent measurement family)

To δ = 3 (realism licensed):
- Both of the above, PLUS:
- Training dynamics: show when during pre-training the lookback emerges and whether the three stages develop together (supporting unity) or independently (undermining it). (adds Process view)
- OR: Discriminant validity — show the lookback is absent on non-ToM retrieval tasks. (adds convergent evidence from negative controls)

---

## Sharpest Questions for the Authors

1. **Unity:** How do you know binding lookback, answer lookback, and visibility lookback are the same kind of mechanism rather than three coincidentally similar circuits bundled under one name? They operate at different layers, different tokens, with different subspace dimensions. What experiment would distinguish "one mechanism in three stages" from "three independent attention-routing processes"?

2. **Discriminant validity:** Does the lookback pattern appear in non-ToM tasks that also require multi-step retrieval (e.g., multi-hop factual reasoning, syllogistic inference)? If yes, it's attention-based retrieval generally, not a ToM-specific mechanism.

3. **Selection bias:** What happens in the stories the model gets wrong? Is the lookback present but misaligned? Absent? Different? Conditioning on correct answers makes the analysis silent on the most informative cases.

4. **Random subspace:** Does a random 14/20/33/167-dimensional subspace achieve comparable IIA? Without this baseline, DCM subspace identification is unfalsifiable — gradient optimization will always find some subspace that maximizes IIA.

5. **Label inflation:** If we replace "pointer dereference" with "attention-mediated retrieval," "address" with "key vector," "payload" with "value content," and "lookback" with "attending to a previous token" — what novel claim remains? The computational metaphor from programming languages imports specificity the evidence doesn't support.

6. **Falsification:** What experimental result would have led you to reject the lookback hypothesis? Every outcome in the paper (high IIA, low IIA, full vs. subspace) is interpreted as consistent with the hypothesis. What would inconsistency look like?

7. **Ecological validity:** CausalToM stories follow a single template with fixed token positions. Has the lookback been tested on stories where characters, objects, and states appear at varying positions? Where there are more than two characters? Where the question requires reasoning about nested beliefs?

---

## Proposed Follow-Up Experiments

These are the empirical tests for the validity audit. See
COMPUTE_ESTIMATE.md for authoritative GPU-hour estimates (~30-38
realistic GPU-hours total across all experiments).

### Tier 1: Quick Kills

**Test 1: Random Subspace Control.**
For each DCM-identified subspace of dimension d, sample 100 random d-dimensional subspaces of the residual stream. Compute IIA for each. If the random subspaces achieve comparable IIA, the DCM finding is uninformative.

**Test 2: Non-ToM Retrieval Control.**
Run the same interchange intervention experiments on non-ToM multi-step retrieval tasks (multi-hop QA, syllogistic reasoning with similar template structure). If the "lookback" pattern appears, it's generic retrieval, not ToM-specific.

**Test 3: Failure Case Analysis.**
Expand CausalToM beyond the 80 correct-answer stories. Run the same interventions on cases where the model gets the answer wrong. Is the lookback present? Absent? Different?

**Test 4: Layer-Wise BH Correction.**
Apply Benjamini-Hochberg correction to the layer-wise IIA results. With 80 layers tested per experiment and ~6 experiments, the implicit search space is ~480 layer comparisons. How many survive correction?

### Tier 2: Stronger Tests

**Test 5: Unity Test.**
Ablate the binding lookback (layers 33-38) by mean-ablating those layers, then test whether the answer lookback (layers 52-56) still functions. If the three stages are one mechanism, disrupting the early stage should cascade. If they're independent, each should be separately disruptable.

**Test 6: Position-Shuffled Stories.**
Generate CausalToM stories where the character/object/state tokens appear at varying positions (reorder sentences, add filler). If the lookback depends on fixed token positions, it's a dataset artifact.

### Tier 3: If Time Permits

**Test 7: Attention Head Attribution.**
Identify which specific attention heads implement each "lookback" stage. If binding and answer lookback use completely different heads, the unity claim is undermined. If they share heads, that's genuine evidence for unity.

**Test 8: Cross-Dataset Transfer.**
Train DCM subspaces on CausalToM, test on BigToM (or naturalistic ToM text). If the subspaces don't transfer, the "mechanism" is dataset-specific.

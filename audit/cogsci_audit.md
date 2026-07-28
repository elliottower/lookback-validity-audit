# Cognitive Science Audit: Language Models Use Lookbacks to Track Beliefs

Paper: Prakash, Shapira, Sen Sharma, Riedl, Belinkov, Rott Shaham, Bau, Geiger. ICLR 2026. arXiv:2505.14685.

This audit evaluates the paper's cognitive-science claims separately from its mechanistic-validity issues (see `prakash_lookback_mechval_audit_v2.md` for the MechVal/MechRef/CVI analysis). The focus here: does the paper's framing as a contribution to understanding "how LMs track beliefs" hold up against what cognitive science actually knows about belief tracking?

---

## 1. Behavioral vs. Representational Systematicity

**The critique (Vegner, de Souza, Forch, Lewis, Doumas, ACL 2025):** The field routinely conflates two claims: (a) a model behaves systematically (correct outputs across compositional variations) and (b) the model's internal representations are compositionally structured. Most benchmarks test only (a) while implicitly claiming (b).

**How this applies to Lookback:** The paper shows:
- Behavioral systematicity: the model answers belief-tracking questions correctly across 80 story variations (filtered to correct answers)
- A decodable internal pattern: interchange interventions localize information flow to specific layers and subspaces

It then infers representational systematicity: the OI/pointer scheme constitutes a structured, compositional representation of belief. But this inference is unwarranted without testing whether the representation composes correctly on structurally novel combinations:

- **New numbers of characters:** CausalToM always has exactly 2. Does the OI scheme scale to 3, 4, 5 characters? If the "binding lookback" assigns OI-1 and OI-2, what happens with OI-3? Representational systematicity predicts clean extension; behavioral matching doesn't.
- **Nested beliefs:** CausalToM asks "What does Alice believe Object2 contains?" — first-order belief. Representational systematicity predicts the scheme extends to "What does Alice think Bob believes Object2 contains?" (second-order). The paper never tests this.
- **Novel structural combinations:** Stories with different numbers of objects, asymmetric interactions, temporal reordering. If the OI scheme is a genuine compositional representation, it should generalize. If it's a pattern-matched retrieval circuit tuned to the template, it won't.

**Verdict:** The paper demonstrates behavioral systematicity and infers representational systematicity. Vegner et al.'s taxonomy identifies this as the field's most common conflation error. The gap is testable: run the interchange interventions on structurally novel CausalToM variants and see if the OI scheme composes.

---

## 2. The Single-Mechanism Assumption vs. Human ToM Research

The paper assumes a single unified "lookback mechanism" for belief tracking. Human cognitive science has not resolved this question even for humans.

### The one-system vs. two-system debate

Human false-belief research is split between accounts of belief tracking:

**One-system (implicit/automatic):** Kovacs, Teglas, & Endress (2010) reported that adults automatically track others' beliefs, suggesting a single fast system. This was taken as evidence for a unified belief-tracking mechanism.

**Two-system (implicit + explicit):** Apperly & Butterfill (2009) proposed that humans have two distinct systems — a fast, automatic, low-cost system that tracks simple belief-like states, and a slow, deliberate, flexible system for full propositional belief attribution.

**Replication failures:** Phillips et al. (2015) directly failed to replicate the Kovacs et al. "automatic belief" effect under controlled conditions, finding the original effects were better explained by attention-check timing artifacts rather than automatic belief tracking. This undermines the strongest evidence for a single automatic mechanism.

**Relevance to Lookback:** If cognitive scientists cannot agree that humans have a single unified belief-tracking mechanism, the Lookback paper's confident single-mechanism framing ("the lookback mechanism") for an artificial system is scientifically premature. The paper bundles three dissociable sub-effects (binding, answer, visibility lookback) under one name and calls it "a pervasive computation." The human literature suggests that even if belief tracking exists as a capacity, its implementation may be multi-system and content-selective rather than unified.

### Content selectivity in human belief attribution

Schuwerk, Vuori, & Sodian (2015) showed that human belief tracking is content-selective: people track false beliefs about object presence differently than about object absence. This implies belief tracking is not one clean unified computation even at the behavioral level.

**Relevance:** If "lookback" is meant as an analogue to human belief tracking, content selectivity should appear as different circuit behavior across belief content types. CausalToM only tests one content type (what state an object is in). The paper never tests whether the same circuit handles belief about object location, object existence, character intentions, or character knowledge states. A "pervasive" mechanism for belief tracking should handle all of these; a template-matched retrieval circuit wouldn't need to.

---

## 3. "Theory of Mind" as a Construct Import

### The construct transfer problem

The paper's abstract: "This question lies at the heart of understanding the Theory of Mind (ToM) capabilities of LMs." It positions the work as a contribution to understanding ToM in artificial systems.

But ToM in cognitive science (Premack & Woodruff 1978, Dennett 1981, Wimmer & Perner 1983) refers to a rich capacity: attributing mental states (beliefs, desires, intentions, knowledge) to others, predicting behavior based on those attributions, and updating those attributions based on new information. The Sally-Anne test is a simple probe for ONE aspect of this capacity (false-belief attribution).

The Lookback paper tests an even simpler probe: given a templated story where two characters interact with objects, can the model correctly answer "What does Character1 believe Object2 contains?" This is a single-step false-belief question with no nested beliefs, no intention attribution, no desire reasoning, no knowledge-state tracking beyond presence/absence of visual access.

Calling the circuit that solves this task "the mechanism for ToM" is like finding the circuit that solves 2+3=5 and calling it "the mechanism for mathematics."

### The "re-evaluation" literature

Ruis et al. (2025, "Re-evaluating Theory of Mind evaluation in large language models") argue that most LLM ToM claims conflate behavioral matching with matching the underlying computations, and that current evaluations "deviate from pure measurements of ToM abilities." Their critique applies directly to Lookback:

1. Behavioral matching (correct answers on CausalToM) ≠ cognitive ToM
2. Finding a decodable internal pattern ≠ finding "the ToM mechanism"
3. Template-based evaluation conflates task-specific pattern matching with genuine mental-state attribution

This is a citable, on-the-nose companion to Vegner et al.'s argument, applied specifically to LLM ToM work.

---

## 4. What the Paper Would Need to Justify Its Cognitive-Science Framing

To legitimately claim a contribution to understanding "how LMs track beliefs" (rather than "how LMs solve a specific templated QA task"), the paper would need:

### 4.1 Compositional generalization tests (Vegner et al.)
- More characters (3, 4, 5) — does the OI scheme compose?
- Nested beliefs ("What does A think B believes?") — does the binding extend?
- Novel story structures — does the mechanism generalize beyond the template?

### 4.2 Content selectivity tests (Schuwerk et al.)
- Beliefs about presence vs. absence
- Beliefs about location vs. state
- Beliefs about intentions vs. facts
- If the "mechanism" handles all of these the same way, that's evidence for a unified system. If it handles them differently, "lookback" is a family of circuits, not one mechanism.

### 4.3 Dissociation from simpler retrieval
- Non-ToM retrieval tasks with similar structure (multi-hop QA, syllogistic reasoning)
- If the "lookback" appears on these tasks too, it's generic retrieval, not ToM
- This is the behavioral-vs-representational question applied to the mechanism: is the circuit doing belief tracking, or is it doing information retrieval that happens to produce correct belief-tracking answers?

### 4.4 Failure mode analysis
- Cases where the model gets the ToM question wrong
- Cases where the model succeeds behaviorally but via a different internal route
- Cases where the "lookback" is present but the model fails anyway

### 4.5 Scope boundaries for the cognitive label
- At what complexity does the "lookback" break? 3 characters? Nested beliefs? Ambiguous pronouns?
- The breakpoint tells you the scope of the mechanism. If it breaks at 3 characters, it's a 2-character retrieval circuit, not a "belief-tracking mechanism."

---

## 5. The Strongest Question to Ask

Synthesizing Vegner et al., the two-system debate, and the content-selectivity literature:

> "The field has an unresolved debate about whether even humans have a single belief-tracking mechanism — Apperly & Butterfill's two-system account vs. the one-system view, with Phillips et al. failing to replicate the key automatic-tracking evidence. Your paper identifies three dissociable sub-effects (binding, answer, visibility lookback) at different layers with different subspace dimensions and calls them one 'pervasive mechanism.' Do you think 'lookback' picks out one computational kind, or is it a family of dissociable circuits that share a naming convention? And would Vegner et al.'s behavioral-vs-representational systematicity distinction change how you'd test that?"

This is specific, well-sourced, and forces a real answer. It name-drops Ivan's paper naturally, cites the human ToM debate as independent support, and frames the question constructively rather than as a takedown. The question has no easy deflection: either they haven't tested unity (which is the case), or they have and didn't report it (unlikely).

---

## 6. How This Connects to the Denominator Paper

The cognitive-science audit strengthens the denominator paper in two ways:

1. **The construct import is itself a form of multiplicity.** The paper searches across 80 layers, multiple subspace dimensions, and multiple intervention targets, then labels the surviving results with cognitive-science terminology that imports additional claims the experiments didn't test. This is semantic multiplicity layered on top of statistical multiplicity: the researcher's degrees of freedom include not just which layers to highlight, but which vocabulary to attach to the findings.

2. **The Vegner et al. distinction provides a general principle.** Many MI papers show behavioral alignment (the model's output changes as predicted under intervention) and infer representational alignment (the model's internal structure matches the proposed mechanism). The denominator paper can cite this as a field-wide pattern: behavioral systematicity inflates the apparent significance of mechanistic findings because it makes them seem more specific than they are.

---

## References

- Apperly, I. A., & Butterfill, S. A. (2009). Do humans have two systems to track beliefs and belief-like states? *Psychological Review*, 116(4), 953-970.
- Kovacs, A. M., Teglas, E., & Endress, A. D. (2010). The social sense: Susceptibility to others' beliefs in human infants and adults. *Science*, 330(6012), 1830-1834.
- Phillips, J., Ong, D. C., Surtees, A. D. R., Xin, Y., Williams, S., Saxe, R., & Frank, M. C. (2015). A second look at automatic Theory of Mind: Reconsidering Kovacs, Teglas, and Endress (2010). *Psychological Science*, 26(9), 1353-1367.
- Ruis, L., et al. (2025). Re-evaluating Theory of Mind evaluation in large language models. *Nature Human Behaviour*.
- Schuwerk, T., Vuori, M., & Sodian, B. (2015). Implicit and explicit false-belief tracking: Content selectivity in human belief attribution. *PLOS ONE*, 10(9).
- Vegner, I., de Souza, L., Forch, V., Lewis, R., & Doumas, L. A. A. (2025). Behavioural vs. representational systematicity in end-to-end models. In *Proceedings of the 63rd Annual Meeting of the ACL*.
- Premack, D., & Woodruff, G. (1978). Does the chimpanzee have a theory of mind? *Behavioral and Brain Sciences*, 1(4), 515-526.
- Wimmer, H., & Perner, J. (1983). Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children's understanding of deception. *Cognition*, 13(1), 103-128.

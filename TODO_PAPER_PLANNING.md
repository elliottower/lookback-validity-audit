# Paper Planning

## New England Mechinterp Conference (Aaron Mueller / Bao Lab)

### Title options
1. "A Pre-Registered Mechanistic Validity Audit of Belief Tracking in Llama-3.1-70B"
2. "Does the Lookback Mechanism Track Beliefs? A 15-Experiment Validity Audit"

### Abstract draft

Mechanistic interpretability routinely labels causal findings with vocabulary borrowed from cognitive science without testing whether the evidence supports the specificity these labels imply. We apply the Mechanistic Validity framework (Tower, 2026) to audit a three-part "lookback mechanism" reported to track characters' beliefs in Llama-3.1-70B-Instruct through pointer dereference implemented via attention (Prakash et al., ICLR 2026). A pre-registered assessment identifies six untested assumptions in the original evaluation, most critically that the story-generation code never produces false beliefs: belief always equals reality in every evaluated pair.

We conduct 15 pre-registered experiments targeting these gaps. The results yield a nuanced verdict. The identified subspaces survive every adversarial control (shuffled labels, random baselines, cross-template transfer). Contrary to our pre-registered prediction that the subspaces would fail on genuine false beliefs, the false-belief test achieves IIA = 0.975 at layer 38, and observability-mediated knowledge achieves IIA = 0.988. At the same time, the subspaces are framing-sensitive: belief questions produce IIA = 1.0, but action-recall questions drop to 0.886 at L38 and 0.400 at L52, while reality-state and fill-completion questions produce IIA = 0.0. This dissociation between question framings, combined with the false-belief generalization, suggests the subspaces encode character-state bindings that constitute belief representations rather than encoding belief as a separate epistemic primitive. We upgrade the claim from "Causally Suggestive" to "Mechanistically Supported" and propose the recharacterization "belief-constitutive entity binding." All protocols, pre-registration hashes, and code are publicly deposited.

### Submission details
- Prefer poster
- Paper link: Zenodo preprint (to be uploaded)
- Framework (mechval) not yet peer-reviewed; expected TMLR soon
- Preprint on Zenodo since mechval itself isn't peer-reviewed yet

### TODO
- [ ] Upload preprint to Zenodo
- [ ] Get Zenodo DOI for mechval framework paper
- [ ] Submit abstract to conference
- [ ] Design poster

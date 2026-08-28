# The cross-model claim in Prakash et al. — findings and experiments

Everything here is verified against `reference/prakash_2026_lookbacks.pdf` (arXiv 2505.14685v3,
ICLR 2026), downloaded and read this session. Line numbers refer to
`reference/prakash_2026_lookbacks.txt`.

Sorted by which paper each item belongs to.

---

## Part 1 — The side paper (cross-system reference)

### 1.1 The finding

The cross-model appendix states its own limitation and then draws a conclusion the
limitation excludes:

> "Due to computational constraints, **subspace interchange intervention experiments were
> not conducted.** As illustrated in Figures 26–42, the results indicate that
> Llama-3.1-405B-Instruct and Qwen2.5-14B-Instruct **employ the same underlying mechanism**
> as Llama-3-70B-Instruct" — line 2906

Full-residual interchange interventions only, 80 samples per experiment. The subspace
experiments — the ones that individuate address, pointer and payload as *separable
components* — ran on Llama-3-70B alone.

Everything that makes the lookback a *mechanism* rather than a *localization* comes from
the subspace results. Transporting the term without them imports structure that was never
measured in the destination model.

### 1.2 Why full-residual patching cannot carry the claim

Replacing the entire residual stream at a token substitutes every representation at that
position simultaneously. It establishes **where** and **when** information is available.
It cannot distinguish an address from a payload, because both are replaced together along
with everything else.

The layer-wise curve is not vacuous — patching at the wrong layer or token does nothing,
which is why the curve has shape. It is genuine evidence about **position and depth**.
It is not evidence about **structure within** the representation at those positions.

This is the inferential-reach ordering from Mechanistic Reference doing real work: no
quantity of activation-level evidence sums to structural evidence.

### 1.3 The failure mode, by name

`mechanistic-reference/paper/main_v9.tex:313` —

> "**Claim Laundering.** Evidence admissible under one view is used to support claims
> characteristic of a higher-commitment view."

And line 414, which describes this case before having read it:

> "any paper claiming structural-level mechanism identity on the basis of activation
> evidence alone is committing an evidence misfire, **diagnosable from the methods section
> without examining the results.**"

The methods sentence and the conclusion sentence are in the same paragraph. No figure needs
to be inspected. That is a framework making a checkable prediction and the paper supplying
the instance — a stronger argument for the framework than anything currently in it.

### 1.4 Three further gaps in the same appendix

**The transported claim is narrower than stated.** Only the no-visibility condition was
tested cross-model: "All three models succeeded in the no-visibility condition, but none
consistently solved the visibility condition" (line 3085). Visibility lookback — one of the
three sub-mechanisms — is absent from the transport entirely, so "the same mechanism" covers
two of three.

**No layer-correspondence rule.** Llama-3-70B has 80 layers, Llama-3.1-405B has 126,
Qwen2.5-14B has 48. What makes a layer in one "the same layer" as a layer in another is
never stated — proportional depth, absolute index, or visual alignment of curves.

**The within-model type claim is also uncriterioned.** The conclusion calls the lookback "a
single recurring computational pattern" (line 982) across binding, answer and visibility
lookbacks. What licenses treating three instances as one type is a separate identity
question from the cross-model one, and gets no more argument.

### 1.5 Relationship to the IOI worked example

Mechanistic Reference already runs this analysis on "the IOI circuit": object-level transport
fails across seeds, role-level partially succeeds, structural-level fails across families at
0.13 congruence. The lookback case is the same shape with two differences worth having —
the authors state the missing evidence themselves rather than it being reconstructed, and
the claim is *process*-shaped rather than circuit-shaped, which the IOI case does not cover.

### 1.6 Experiments

All on Qwen2.5-14B-Instruct, the model where the gap is stated. Their pipeline is vendored
at `reference/belief_tracking`.

**X1 — Subspace interchange interventions on Qwen, binding lookback.** The experiment they
did not run. Fit DAS subspaces for address, pointer and payload at the corresponding
positions; report IIA against rank.

| outcome | reading |
|---|---|
| subspaces found, IIA comparable to Llama | the structural claim holds; they were right and untested. Transport succeeds at the subspace level |
| nothing above baseline | "the same underlying mechanism" is false for Qwen |
| found, but different rank or geometry | partial transport — the case a hierarchy predicts and a binary claim cannot express |

**X2 — Paired versus single patching on Qwen.** Their own best design (lines 921–938),
applied cross-model. Patch the recalled token alone, the lookback token alone, then both.
The mismatch signature — singles fail, pair succeeds — is evidence of a dependency relation
and needs no DAS. Their visibility version is unavailable here since Qwen fails that
condition, so run it on binding lookback.

**X3 — Random-subspace baseline on Qwen, matched rank.** Makes X1 interpretable. This is E1
of the existing audit, on the other model. Hofmann's 100%-IIA-on-random-models result is why
this is not optional.

**X4 — Layer correspondence.** Once X1 has a result: does the mechanism appear at the same
proportional depth, the same absolute layer, or neither? Answers the question §1.4 says the
paper leaves open, and generalizes past this case.

### 1.7 What must not be overclaimed

The within-Llama analysis is well-evidenced and this is not a refutation of the paper. The
mismatch experiment at lines 921–938 is genuine dependency evidence: they predicted that
patching either token alone would fail through address/pointer mismatch and that patching
both would succeed, and it did. Single-subspace interventions cannot generate that
prediction. The layer ordering is established rather than assumed. Attention knockout appears
in Appendix K.

The finding is narrow and about one appendix. Stated wholesale it is wrong.

The honest sentence for their appendix costs them nothing:

> *Qwen2.5-14B and Llama-405B show the same information flow at corresponding positions and
> depths; whether the same address/pointer/payload decomposition underlies it was not tested.*

---

## Part 2 — Views-specific

Thin, and thinner than it looked before reading the paper. Two items.

### 2.1 The claim is ambiguous across views, and the evidence settles one reading

"The same underlying mechanism" can mean the same components (Structural), the same
operations in the same order (Process), or the same functional organisation (Role). These
come apart here.

Full-residual layer curves are **good** process-level evidence — they establish that
information becomes available in a particular order at particular depths, which is what a
process claim asserts. They are **no** evidence at the structural level.

So the paper's evidence supports one reading of its own conclusion and not another, and
nothing in the text says which was meant. Declaring the view resolves it, which is the
boundary test for a Views violation rather than a validity one. This is open problem C.1.7
(cross-model identity) with a live instance.

Mechanistic Reference handles the same case with more machinery and should own it. The Views
contribution is only the observation that the claim is view-ambiguous before it is
evidence-deficient.

### 2.2 E12, question-framing dissociation

The one result in the audit that stays Views-shaped: the same three subspaces are
indistinguishable under the belief framing (1.000 / 1.000 / 1.000) and separate completely
under action recall (0.886 / 0.400 / 0.057), with reality-state and fill-completion at zero
throughout. Whether two subspaces count as the same causal variable depended on which
question was asked.

Held back at n=35 with the belief row at ceiling and the code flagging its CIs
`"interpretable": false`. The rerun at n-eval 500 addresses this.

---

## Part 3 — Neither (Mechanistic Validity)

Recorded so they are not miscategorised later. E7 necessity (sufficient, zero ablation drop),
E6 cross-stage mediation at 0.34, E11 cross-task transfer, E14/E15 overturned predictions,
E1/E2 controls, confounding sensitivity. All are validity checks that ran and returned
answers. None is fixed by a declaration, so none is a Views case.

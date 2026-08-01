# Lookback Validity Audit

A mechanistic validity audit of Prakash et al. (ICLR 2026), "Language models use lookbacks to track beliefs."

## What this is

The Lookback paper claims language models implement belief tracking via "pointer dereference through attention" — three sub-mechanisms at non-overlapping layer ranges. We audit this claim against 34 validity criteria (v11) from the Mechanistic Validity framework and run 15 pre-registered experiments to test the structural gaps.

## Structure

- `paper/` — LaTeX source for the audit paper
- `experiments/` — Pre-registered experiment scripts (Llama-3.1-70B-Instruct via NDIF/nnsight)
- `results/` — Experiment outputs (per-experiment subdirectories)
- `AMENDMENTS.md` — Full amendment history with pre-committed interpretations

## Experiments

1. **Random subspace baseline (M2)** — Do random subspaces of matched rank achieve comparable IIA?
2. **Shuffled-label DCM control (M2)** — Does a DCM trained on shuffled belief labels match the real DCM?
3. **Non-ToM retrieval control (C4)** — Does the lookback pattern appear on non-belief tasks?
4. **Failure case analysis (I1)** — Is the lookback absent when the model answers incorrectly?
5. **Cross-stage mediation (I6)** — Does the binding representation mediate the answer pointer?
6. **Compositional generalization (C3, C5)** — Does the subspace transfer to novel stories?
7. **Necessity (I1)** — Is the subspace necessary for correct behavior?
8. **Layer spread (M1)** — Is the effect localized to the claimed layers?
9. **Rank decomposition (M4)** — Does each SVD direction contribute, or is one dominant?
10. **Surface heuristic (E1)** — Does the subspace transfer across templates and entity names?
11. **Cross-task contamination (E2)** — Does the subspace steer unrelated tasks?
12. **Question framing (C1)** — Is the subspace sensitive to belief vs non-belief question framing?
13. **Distractor insertion (M3)** — Is the subspace robust to positional perturbation?
14. **False belief (C2)** — Does the subspace handle genuine false beliefs?
15. **Observability dissociation (C2)** — Does the subspace encode observability-mediated knowledge?

## Pre-registration

Experiments are pre-registered in `AMENDMENTS.md` before running. Tags (`prereg-v1`, `prereg-amendment-1` through `prereg-amendment-5`) mark each registration point. See Amendment 10 for the v10→v11 framework version change and provenance notes.

## License

MIT

# Lookback Validity Audit

A mechanistic validity audit of Prakash et al. (ICLR 2026), "Language models use lookbacks to track beliefs."

## What this is

The Lookback paper claims language models implement belief tracking via "pointer dereference through attention" — three sub-mechanisms at non-overlapping layer ranges. We audit this claim against 31 validity criteria from the [Mechanistic Validity framework](https://doi.org/10.5281/zenodo.20478480) and run four pre-registered experiments to test the structural gaps.

## Structure

- `paper/` — LaTeX source for the audit paper
- `experiments/` — Pre-registered experiment scripts (Llama-3-70B-Instruct via NNsight)
- `results/` — Experiment outputs
- `audit/` — Full criterion-by-criterion assessment

## Experiments

1. **Random subspace baseline (M2)** — Do random subspaces of matched dimensionality achieve comparable IIA? (floor control)
2. **Shuffled-label DCM control (M2)** — Does a DCM trained on shuffled belief labels match the real DCM? (the key experiment)
3. **Non-ToM retrieval control (C4)** — Does the lookback pattern appear on non-belief-tracking tasks?
4. **Failure case analysis (I1)** — Is the lookback absent when the model answers incorrectly?
5. **Unity coupling test (I6)** — Do the three sub-mechanisms interact non-additively?

## Pre-registration

Experiments are pre-registered before running. Each commit that adds a pre-registration document is signed and timestamped.

## License

MIT

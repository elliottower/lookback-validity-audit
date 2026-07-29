# Compute Estimate

All experiments use Llama-3-70B-Instruct on A100-80GB GPUs.
Assumes ~0.5s per forward pass (single example, bf16, NNsight hooks).

## Experiment 1: Held-out basis validation

- SVD recomputation: 320 stories × 80 layers × 1 forward pass = 25,600 passes (~3.6 GPU-hours)
- Mask training: 6 layer×concept combos × 80 training stories × ~50 Adam steps = 24,000 passes (~3.3 GPU-hours)
- IIA evaluation: 6 combos × 80 held-out stories = 480 passes (~negligible)
- **Subtotal: ~7 GPU-hours**

## Experiment 2: Matched-capacity controls

### Random subspace baseline
- 200 random subspaces × 6 layer×concept combos × 80 held-out stories = 96,000 passes
- **Subtotal: ~13 GPU-hours**

### Shuffled-label control
Two primary layer-concept combinations at 100 permutations each; four
exploratory combinations at 20 permutations each. Total mask trainings:
(2 × 100) + (4 × 20) = 280.

- 280 mask trainings × 80 training stories × ~50 Adam steps = 1,120,000 passes (mask training)
- 280 mask trainings × 80 held-out stories = 22,400 passes (evaluation)
- **Conservative subtotal: ~156 GPU-hours** (treating every pass as a full forward)

Mask training is single-layer intervention, not full model inference —
actual cost is 10-20× lower than the naive forward-pass estimate.
**Realistic subtotal: ~8-16 GPU-hours.**

## Experiment 3: Task-specificity controls

- 4 conditions × 120 stories × 6 layer×concept combos = 2,880 passes
- **Subtotal: ~0.4 GPU-hours**

## Experiment 4: Failure case analysis

- 500 story generations (model evaluation): 500 passes
- IIA on correct vs incorrect: ~500 × 6 combos = 3,000 passes
- 10,000 permutation test iterations: statistical only (no forward passes)
- **Subtotal: ~0.5 GPU-hours**

## Experiment 5: Cross-stage mediation

- 4 conditions × 80 stories × ~10 layer combinations = 3,200 passes
- Plus mediation analysis: ~3,200 additional passes
- **Subtotal: ~1 GPU-hour**

## Total

| Experiment | GPU-hours (conservative) | GPU-hours (realistic) |
|-----------|------------------------:|---------------------:|
| 1. Held-out basis | 7 | 7 |
| 2a. Random baseline | 13 | 13 |
| 2b. Shuffled-label | 156 | 8-16 |
| 3. Task-specificity | 0.4 | 0.4 |
| 4. Failure cases | 0.5 | 0.5 |
| 5. Mediation | 1 | 1 |
| **Total** | **~178** | **~30-38** |

The realistic estimate accounts for mask training being single-layer
operations, not full model inference. The dominant cost remains
Experiment 2b (shuffled-label retraining), though the 2-primary +
4-exploratory design reduces it by more than half vs. the original
6 × 100 plan.

## Hardware requirements

- Llama-3-70B-Instruct in bf16: ~140GB VRAM → 2× A100-80GB or 1× H100
- NNsight hooks add ~10% memory overhead
- All experiments are embarrassingly parallel across layer×concept combos

## NDIF

All experiments use NNsight, which is NDIF's native interface — scripts
run locally and execute remotely against NDIF-hosted Llama-3-70B-Instruct.
No local GPU rental needed. Estimated ~30-38 realistic GPU-hours across
all experiments; throughput depends on NDIF queue load.

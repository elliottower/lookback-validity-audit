#!/bin/bash
# Relaunch experiments that died from NDIF outage.
# Each checkpoints per-subspace, so completed work is preserved.
# Run: bash experiments/relaunch_stuck.sh

set -e
cd "$(dirname "$0")/.."
TS=$(date +%Y%m%d_%H%M%S)

echo "=== Testing NDIF connectivity ==="
if ! timeout 60 uv run python -c "
from experiments.ndif_utils import setup_nnsight
lm = setup_nnsight()
with lm.trace('The capital of France is', remote=True):
    pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()
print(f'NDIF OK: {lm.tokenizer.decode([pred.item()]).strip()!r}')
" 2>&1; then
    echo "NDIF still down. Aborting."
    exit 1
fi

echo ""
echo "=== Launching experiments ==="

# Necessity test (E7) — all checkpoints were empty, cleared
nohup uv run python experiments/necessity_test.py \
    2>&1 | tee -a "results/necessity_run_log_${TS}.txt" &
echo "  necessity_test.py launched (PID $!)"

# Layer spread (E8) — summary was empty, cleared
nohup uv run python experiments/layer_spread_test.py \
    2>&1 | tee -a "results/layer_spread_run_log_${TS}.txt" &
echo "  layer_spread_test.py launched (PID $!)"

# Rank decomposition (E9) — has cached filtered pairs, will resume
nohup uv run python experiments/rank_decomposition_test.py \
    2>&1 | tee -a "results/rank_decomposition_run_log_${TS}.txt" &
echo "  rank_decomposition_test.py launched (PID $!)"

# Adversarial heuristic (E10) — has template_0 with n=23, needs rerun at n>=80
# Delete old checkpoint so it reruns
rm -f results/adversarial_heuristic/template_0.json
nohup uv run python experiments/adversarial_heuristic_test.py --n-eval 80 \
    2>&1 | tee -a "results/adversarial_heuristic_run_log_${TS}.txt" &
echo "  adversarial_heuristic_test.py launched with --n-eval 80 (PID $!)"

# Compositional generalization (E6) — fixed prompt format, stale results moved
nohup uv run python experiments/compositional_generalization.py \
    --output results/compositional/ \
    2>&1 | tee -a "results/compositional_run_log_${TS}.txt" &
echo "  compositional_generalization.py launched (PID $!)"

echo ""
echo "All 5 experiments launched. Monitor with:"
echo "  tail -f results/*_run_log_${TS}.txt"

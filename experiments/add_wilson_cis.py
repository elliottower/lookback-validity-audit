"""Retrospectively add Wilson CIs to all existing result JSON files.

Scans results/ for JSON files containing IIA values with known n,
adds ci_lower, ci_upper, ci_width, and interpretable fields.

Usage:
    uv run python experiments/add_wilson_cis.py --dry-run
    uv run python experiments/add_wilson_cis.py
"""

import json
from pathlib import Path

from ndif_utils import MIN_N_FOR_INTERPRETATION, wilson_ci

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def add_ci_to_dict(d, iia_key="iia", n_key="n_pairs"):
    """Add Wilson CI fields to a dict that has iia and n_pairs."""
    iia = d.get(iia_key)
    n = d.get(n_key)
    if iia is not None and n is not None and n > 0:
        k = round(iia * n)
        lo, hi = wilson_ci(k, n)
        d["ci_lower"] = lo
        d["ci_upper"] = hi
        d["ci_width"] = hi - lo
        d["interpretable"] = n >= MIN_N_FOR_INTERPRETATION
        return True
    return False


def process_file(path, dry_run=False):
    """Process a single result JSON file."""
    with open(path) as f:
        data = json.load(f)

    modified = False

    if "n_pairs" in data and "iia" in data:
        if add_ci_to_dict(data):
            modified = True

    if "subspace_iias" in data and "n_pairs" in data:
        n = data["n_pairs"]
        for sub_name, iia in data["subspace_iias"].items():
            if isinstance(iia, (int, float)):
                k = round(iia * n)
                lo, hi = wilson_ci(k, n)
                if "subspace_cis" not in data:
                    data["subspace_cis"] = {}
                data["subspace_cis"][sub_name] = {
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "ci_width": hi - lo,
                    "interpretable": n >= MIN_N_FOR_INTERPRETATION,
                }
                modified = True

    for key in ["subspaces", "conditions", "templates"]:
        if key in data and isinstance(data[key], dict):
            for name, entry in data[key].items():
                if isinstance(entry, dict):
                    for iia_k, n_k in [("iia", "n_pairs"), ("iia", "n_samples")]:
                        if add_ci_to_dict(entry, iia_k, n_k):
                            modified = True

    if modified:
        if dry_run:
            print(f"  WOULD update: {path.relative_to(RESULTS_DIR)}")
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Updated: {path.relative_to(RESULTS_DIR)}")
    return modified


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    updated = 0
    for path in sorted(RESULTS_DIR.rglob("*.json")):
        if "stale" in str(path):
            continue
        try:
            if process_file(path, args.dry_run):
                updated += 1
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Skip {path.name}: {e}")

    print(f"\n{'Would update' if args.dry_run else 'Updated'} {updated} files")


if __name__ == "__main__":
    main()

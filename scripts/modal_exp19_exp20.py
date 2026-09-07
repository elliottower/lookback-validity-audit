"""
EXP19 (full-sequence gate) and EXP20 (output identity in the both-condition).

Registered in AMENDMENTS.md Amendment 13 (tag prereg-amendment-9), corrected by
Amendment 14 (tag prereg-amendment-10), and extended by Amendment 15 (tag
prereg-amendment-11), all frozen before either experiment was run in any mode.
Amendment 14 is analysis-side; this script stores raw readouts and classifies
nothing, so it is unaffected by those corrections.

EXP19 is a harness gate. Replacing the entire clean residual sequence with the
counterfactual sequence at a layer must recover the counterfactual output. If it
does not, the source-target alignment or the continuation setup is wrong and no
interchange result in this series is interpretable.

EXP20 re-runs the binding mismatch test recording what the model actually says.
The stored run recorded only equality with the counterfactual target, so an IIA of
0.000 cannot distinguish the two patches cancelling from the two patches
interfering.

Design decisions a reader should not have to infer:

* Classification happens OFFLINE. This script writes token ids, decoded strings,
  logits and span metadata. It assigns no category and runs no test. Storing a
  verdict rather than the evidence is the defect that made this rerun necessary.

* EXP20 runs BOTH resolutions as patches (Amendment 15). The legacy arm reproduces
  the stored intervention and carries every registered endpoint and void condition.
  The offset arm applies Amendment 13's resolution and is exploratory. Agreement
  between the two is recorded per observation and summarised per layer. This
  replaces what would otherwise have been an amendment granting the legacy resolver
  an exception, and it measures the resolver question rather than assuming it.

* Every write is all-or-nothing per (pair, layer). Partial observations are the
  failure mode that makes an incomplete run look finished.

Usage:
    modal run --detach scripts/modal_exp19_exp20.py
    modal run --detach scripts/modal_exp19_exp20.py --dry-run --limit 5
    modal volume get lookback-exp19-exp20 results/ results/exp19_exp20/
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "nnsight==0.7.0",
        "torch==2.5.1",
        "transformers==4.48.3",
        "accelerate==1.3.0",
        "numpy==1.26.4",
        "tqdm==4.67.1",
        "dataclasses-json==0.6.7",
        "huggingface-hub==0.28.1",
    )
    .add_local_dir(
        "/Users/elliottower/Documents/GitHub/lookback-validity-audit/reference/belief_tracking",
        remote_path="/root/belief_tracking",
    )
)

vol = modal.Volume.from_name("lookback-exp19-exp20", create_if_missing=True)
app = modal.App("lookback-exp19-exp20")

MODEL = "Qwen/Qwen2.5-14B-Instruct"

# Pinned literally, before any data was collected, so the registration's claim that the
# model is pinned is true of the frozen code rather than of whatever "main" resolved to on
# the first run. Resolved 2026-09-07; the model was last modified 2024-09-25.
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

# Digest over the four vendored files that generate stimuli. A whole-tree hash would churn
# on .git, .DS_Store and __pycache__ and so could never be asserted.
STIMULUS_FILES = (
    "notebooks/causalToM_novis/utils.py",
    "data/synthetic_entities/characters.json",
    "data/synthetic_entities/bottles.json",
    "data/synthetic_entities/drinks.json",
)
EXPECTED_STIMULUS_SHA = "400c8a5db9a92f59c0701915a0bbc02e8206ce3503cf79f0f85b6051e5b37297"

SEED = 42

LAYERS = [26, 30]
N_EXP19 = 30
N_EXP20 = 200
N_STORIES = 600
TOP_K = 10
COMMIT_EVERY = 10

# Two resolver arms. The legacy arm reproduces the stored cross-model intervention,
# which EXP20's void condition requires. The offset arm applies the resolution
# registered in Amendment 13. Neither is privileged; the comparison is the point, and
# it replaces what would otherwise have been an amendment granting an exception.
LEGACY_CONDITIONS = ("recalled_only", "lookback_only", "both")
OFFSET_CONDITIONS = ("recalled_only_offset", "lookback_only_offset", "both_offset")
EXP20_CONDITIONS = ("clean",) + LEGACY_CONDITIONS + OFFSET_CONDITIONS


# ── Position resolvers ──
# Both are patch sources under Amendment 15. The legacy pair is copied verbatim from
# modal_qwen_mismatch_test.py, defects included, because EXP20's void condition is
# reproducing the stored result and repairing them would defeat it. The offset pair
# below applies the resolution registered in Amendment 13.


def _find_state_positions(tokenizer, prompt, states):
    """LEGACY. Repeated state words pair by occurrence order, not semantic role."""
    full_ids = tokenizer.encode(prompt)
    q_idx = prompt.find("Question:")
    story_prefix = prompt[:q_idx] if q_idx > -1 else prompt
    story_end = len(tokenizer.encode(story_prefix))

    positions = []
    for state in states:
        target_ids = tokenizer.encode(f" {state}", add_special_tokens=False)
        for i in range(story_end - len(target_ids) + 1):
            if full_ids[i:i + len(target_ids)] == target_ids:
                positions.extend(range(i, i + len(target_ids)))
    return sorted(set(positions))


def _find_question_answer_positions(tokenizer, prompt):
    """LEGACY. Tokenizes the prefix separately; BPE does not guarantee that boundary."""
    full_ids = tokenizer.encode(prompt)
    q_idx = prompt.find("Question:")
    if q_idx == -1:
        return list(range(len(full_ids) - 5, len(full_ids)))
    prefix_ids = tokenizer.encode(prompt[:q_idx])
    return list(range(len(prefix_ids), len(full_ids)))


def _offset_question_answer_positions(tokenizer, prompt):
    """DIAGNOSTIC. Character offsets to token indices; never used for patching.

    Returns None where offsets are unavailable, so a tokenizer without them
    disables the diagnostic instead of failing the run.
    """
    q_idx = prompt.find("Question:")
    if q_idx == -1:
        return None
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc.get("offset_mapping")
    if not offsets:
        return None
    return [i for i, (start, end) in enumerate(offsets) if end > q_idx and end > start]


def _offset_state_positions(tokenizer, prompt, states):
    """Offset-mapping resolution of state positions, paired by narrative order.

    Registered in Amendment 13. Two differences from the legacy resolver: token indices
    come from character offsets rather than from separately tokenizing a prefix, and
    matching is word-boundary aware, so a state name cannot match inside a longer word.

    Occurrences are keyed by **narrative order** -- the order in which state strings first
    appear in the story -- not by index in the `states` list. That list is a generator
    slot, not a semantic role: `get_reversed_sentence_counterfacts` builds the
    counterfactual with `list(reversed(states))`, so slot 0 names a different drink in the
    two prompts. Slot order happens to coincide with narrative order for this generator
    because it reverses characters, objects and states together, but relying on that
    coincidence would be the same class of error as the legacy `zip`.

    Returns (positions, roles) or (None, None) where offsets are unavailable.
    """
    import re as _re
    q_idx = prompt.find("Question:")
    story_end_char = q_idx if q_idx > -1 else len(prompt)
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc.get("offset_mapping")
    if not offsets:
        return None, None

    hits = []
    for slot, state in enumerate(states):
        for m in _re.finditer(r"\b" + _re.escape(state) + r"\b", prompt[:story_end_char]):
            hits.append({"state": state, "generator_slot": slot, "char_start": m.start()})
    hits.sort(key=lambda h: h["char_start"])

    positions, roles = [], []
    for narrative_index, h in enumerate(hits):
        c, state = h["char_start"], h["state"]
        span = [i for i, (a, b) in enumerate(offsets)
                if b > c and a < c + len(state) and b > a]
        for k, tok_i in enumerate(span):
            positions.append(tok_i)
            roles.append({"narrative_index": narrative_index, "token_in_span": k,
                          "state": state, "generator_slot": h["generator_slot"],
                          "char_start": c, "token_index": tok_i})
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    return [positions[i] for i in order], [roles[i] for i in order]


def _offset_pair_by_role(src_roles, dst_roles):
    """Align counterfactual to clean state positions by narrative role.

    Role identity is (narrative_index, token_in_span). Returns
    (src_positions, dst_positions, pairing) or None when the two prompts do not share the
    same role set or a role spans a different number of tokens. Returning None is the
    point: silently truncating a mismatch is exactly the legacy defect.
    """
    if src_roles is None or dst_roles is None:
        return None
    def index(roles):
        out = {}
        for r in roles:
            out[(r["narrative_index"], r["token_in_span"])] = r
        return out
    src, dst = index(src_roles), index(dst_roles)
    if set(src) != set(dst) or not src:
        return None
    keys = sorted(src)
    src_pos = [src[k]["token_index"] for k in keys]
    dst_pos = [dst[k]["token_index"] for k in keys]
    pairing = [{"narrative_index": k[0], "token_in_span": k[1],
                "cf_state": src[k]["state"], "clean_state": dst[k]["state"],
                "cf_pos": src[k]["token_index"], "clean_pos": dst[k]["token_index"]}
               for k in keys]
    return src_pos, dst_pos, pairing


# ── Stimuli ──


def _generate_binding_pairs(n_samples, seed):
    import json
    import random
    import sys

    sys.path.insert(0, "/root/belief_tracking")
    from notebooks.causalToM_novis.utils import get_reversed_sentence_counterfacts

    def load(name):
        with open(f"/root/belief_tracking/data/synthetic_entities/{name}.json") as f:
            return json.load(f)

    random.seed(seed)
    return get_reversed_sentence_counterfacts(
        load("characters"), load("bottles"), load("drinks"), n_samples)


def _canonical_drinks():
    import json
    with open("/root/belief_tracking/data/synthetic_entities/drinks.json") as f:
        return json.load(f)


def _pairs_digest(pairs):
    import hashlib
    import json
    payload = json.dumps(
        [[p["clean_prompt"], p["counterfactual_prompt"], p["clean_ans"],
          p["counterfactual_ans"], p["clean_states"], p["counterfactual_states"]]
         for p in pairs],
        sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _stimulus_digest(root):
    """Content hash of the vendored files that generate stimuli."""
    import hashlib
    import os
    h = hashlib.sha256()
    for rel in STIMULUS_FILES:
        path = os.path.join(root, rel)
        h.update(rel.encode())
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def _filter_on_model(lm, pairs, max_size):
    import torch
    from tqdm import tqdm

    filtered = []
    for sample in tqdm(pairs, desc="Filtering"):
        clean_target = sample["clean_ans"].lower().strip()
        cf_target = sample["counterfactual_ans"].lower().strip()
        if clean_target == cf_target:
            continue
        with torch.no_grad():
            with lm.trace(sample["clean_prompt"]):
                clean_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()
            with lm.trace(sample["counterfactual_prompt"]):
                cf_pred = lm.lm_head.output[0, -1].argmax(dim=-1).save()
        if (lm.tokenizer.decode([clean_pred.item()]).lower().strip() == clean_target
                and lm.tokenizer.decode([cf_pred.item()]).lower().strip() == cf_target):
            filtered.append(sample)
            if len(filtered) >= max_size:
                break
    return filtered


# ── Recording ──


def _provenance(model_sha, pairs_sha, source_sha):
    import nnsight
    import torch
    import transformers
    return {
        "model": MODEL,
        "model_revision_requested": MODEL_REVISION,
        "model_commit_sha": model_sha,
        "pairs_digest": pairs_sha,
        "belief_tracking_digest": source_sha,
        "seed": SEED,
        "nnsight": nnsight.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _readout(tokenizer, logits_saved, drink_token_ids, clean_id, cf_id):
    import torch

    logits = logits_saved.detach().float()
    top = torch.topk(logits, TOP_K)
    top_ids = [int(i) for i in top.indices.tolist()]
    pred_id = int(logits.argmax().item())
    pred_text = tokenizer.decode([pred_id])
    return {
        "pred_id": pred_id,
        "pred_repr": repr(pred_text),
        # Filtering and EXP19 score by normalized decoded string. Two token ids can
        # decode to the same normalized answer, so scoring the void conditions on ids
        # could void a run whose decoded outputs reproduce the registered criteria.
        "pred_text": pred_text,
        "pred_normalized": pred_text.lower().strip(),
        "top_ids": top_ids,
        "top_strings": [repr(tokenizer.decode([i])) for i in top_ids],
        "top_logits": [float(v) for v in top.values.tolist()],
        "clean_answer_logit": float(logits[clean_id].item()) if clean_id is not None else None,
        "cf_answer_logit": float(logits[cf_id].item()) if cf_id is not None else None,
        "drink_logits": {str(t): float(logits[t].item()) for t in drink_token_ids},
    }


def _single_token_id(tokenizer, word):
    ids = tokenizer.encode(f" {word}", add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def _load_done(path):
    """Resume identity includes the condition, and the shard is validated before use.

    A set alone would collapse duplicate keys silently, so keys are counted. A pair-layer
    holding some but not all registered conditions is fatal rather than repaired: guessing
    how to complete a half-written observation is how incomplete data acquires the
    appearance of a finished run.
    """
    import json
    import os
    from collections import Counter
    if not os.path.exists(path):
        return set(), None
    counts, digests, rows = Counter(), set(), 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[(row["exp"], row["pair_index"], row["layer"], row["condition"])] += 1
            rows += 1
            if row.get("pairs_digest"):
                digests.add(row["pairs_digest"])

    dupes = [k for k, c in counts.items() if c > 1]
    if dupes:
        raise RuntimeError(f"{len(dupes)} duplicate observation keys in {path}, "
                           f"first: {dupes[:3]}")
    if len(digests) > 1:
        raise RuntimeError(f"shard mixes {len(digests)} pair sets: {sorted(digests)}")
    if rows and not digests:
        raise RuntimeError(f"{path} holds {rows} rows with no pairs_digest; refusing to "
                           "generate a new pair set beside observations of unknown origin")

    per_pair_layer = {}
    for exp, idx, layer, cond in counts:
        if exp == "EXP20":
            per_pair_layer.setdefault((idx, layer), set()).add(cond)
    partial = {k: sorted(v) for k, v in per_pair_layer.items()
               if v != set(EXP20_CONDITIONS)}
    if partial:
        raise RuntimeError(
            f"{len(partial)} EXP20 pair-layers hold an incomplete condition set, e.g. "
            f"{list(partial.items())[:2]}. Expected all of {sorted(EXP20_CONDITIONS)} or "
            "none. Delete those rows and rerun rather than repairing them in place.")

    return set(counts), (digests.pop() if digests else None)


def _validate_exp20_from_disk(path, n_pairs, layers, dry_run):
    """Completeness and void status, computed from the artifact rather than from memory."""
    import json
    import math
    import os
    from collections import defaultdict

    rows = defaultdict(dict)
    undefined = defaultdict(list)
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["exp"] != "EXP20":
                    continue
                key = (r["pair_index"], r["layer"])
                if r.get("undefined"):
                    undefined[key].append(r["condition"])
                else:
                    rows[key][r["condition"]] = r

    def wilson_upper(k, n, z=1.96):
        if n == 0:
            return 1.0
        ph = k / n
        d = 1 + z * z / n
        c = ph + z * z / (2 * n)
        m = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
        return (c + m) / d

    out = {"dry_run": bool(dry_run), "per_layer": {}, "complete": True}
    for layer in layers:
        keys = [k for k in rows if k[1] == layer]
        # "Every condition has a row" and "every registered measurement succeeded" are
        # different facts. Conflating them lets one wholly undefined pair-layer pass as
        # complete while clean accuracy is reported over the remaining 199.
        full = [k for k in keys
                if set(LEGACY_CONDITIONS) | {"clean"} <= set(rows[k])]
        both_arms = [k for k in full if "both_offset" in rows[k]]
        agree = sum(1 for k in both_arms
                    if rows[k]["both"]["pred_normalized"]
                    == rows[k]["both_offset"]["pred_normalized"])
        und = [k for k in undefined if k[1] == layer]
        def norm(v):
            return (v or "").lower().strip()
        clean_hits = sum(
            1 for k in full
            if rows[k]["clean"]["pred_normalized"] == norm(rows[k]["clean"].get("clean_ans")))
        both_hits = sum(
            1 for k in full
            if rows[k]["both"]["pred_normalized"] == norm(rows[k]["both"].get("cf_ans")))
        n = len(full)
        clean_acc = clean_hits / n if n else 0.0
        both_iia = both_hits / n if n else 0.0
        every_condition = [k for k in set(list(rows) + list(undefined)) if k[1] == layer
                           and set(rows.get(k, {})) | set(undefined.get(k, []))
                           == set(EXP20_CONDITIONS)]
        all_keys_accounted_for = len(every_condition) == n_pairs
        legacy_complete = len(full) == n_pairs
        analysis_ready = all_keys_accounted_for and legacy_complete
        out["per_layer"][str(layer)] = {
            "expected": n_pairs,
            "all_keys_accounted_for": all_keys_accounted_for,
            "legacy_defined": len(full),
            "legacy_complete": legacy_complete,
            "offset_defined": len(both_arms),
            "offset_undefined": len(full) - len(both_arms),
            "analysis_ready": analysis_ready,
            "undefined_observations": len(und),
            "clean_accuracy_n": n, "both_iia_n": n,
            "clean_accuracy": clean_acc if legacy_complete else None,
            "both_iia": both_iia if legacy_complete else None,
            "clean_accuracy_preliminary": None if legacy_complete else clean_acc,
            "both_iia_preliminary": None if legacy_complete else both_iia,
            "both_iia_wilson_upper": wilson_upper(both_hits, n) if legacy_complete else None,
            # Void decisions are withheld rather than guessed when the legacy arm is
            # short: an n = 199 verdict against an n = 200 design is not a verdict.
            "void_clean_accuracy_below_1": (clean_acc < 1.0) if legacy_complete else None,
            "void_both_incompatible_with_stored_zero":
                (wilson_upper(both_hits, n) >= 0.05) if legacy_complete else None,
            "resolver_agreement": (agree / len(both_arms)) if both_arms else None,
            "resolver_disagreements": len(both_arms) - agree,
        }
        out["complete"] = out["complete"] and analysis_ready
    return out


def _gate_from_disk(path, dry_run, required=None):
    """The gate is reconstructed from persisted rows, never from in-memory counters.

    An in-memory counter is reset by a restart, so a resumed run would fail the gate
    on a layer whose 30 successful rows are already on disk.
    """
    import json
    import os
    required = N_EXP19 if required is None else required
    per_layer = {layer: {"hits": set(), "misses": set(), "undefined": set()} for layer in LAYERS}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["exp"] != "EXP19" or row["layer"] not in per_layer:
                    continue
                bucket = per_layer[row["layer"]]
                if row.get("undefined"):
                    bucket["undefined"].add(row["pair_index"])
                elif row.get("hit"):
                    bucket["hits"].add(row["pair_index"])
                else:
                    bucket["misses"].add(row["pair_index"])

    summary, ok = {}, True
    for layer, b in per_layer.items():
        defined = len(b["hits"]) + len(b["misses"])
        layer_ok = (defined == required and not b["undefined"] and len(b["misses"]) == 0)
        summary[str(layer)] = {
            "hits": len(b["hits"]), "misses": len(b["misses"]),
            "undefined": len(b["undefined"]), "defined": defined,
            "required": required, "layer_ok": layer_ok,
        }
        ok = ok and layer_ok
    return {"per_layer": summary, "gate_ok": bool(ok), "dry_run": bool(dry_run)}


@app.function(image=image, gpu="A100-40GB", volumes={"/results": vol},
              timeout=24 * 60 * 60, secrets=[modal.Secret.from_name("huggingface")])
def run(dry_run: bool = False, limit: int = 0):
    import json
    import os
    from datetime import datetime, timezone

    import torch
    from huggingface_hub import HfApi
    from nnsight import LanguageModel
    from tqdm import tqdm

    if limit and not dry_run:
        raise RuntimeError(
            "--limit only applies to a dry run. A registered run uses the sample sizes "
            "in Amendment 13; a short real run would report a smaller n than registered.")
    prefix = "DRYRUN_" if dry_run else ""
    d = "/results"
    shard = os.path.join(d, f"{prefix}observations.jsonl")
    meta_path = os.path.join(d, f"{prefix}run_metadata.json")
    gate_path = os.path.join(d, f"{prefix}exp19_gate.json")

    def ts():
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    print(f"[{ts()}] resolving {MODEL}@{MODEL_REVISION}")
    model_sha = HfApi().model_info(MODEL, revision=MODEL_REVISION).sha
    print(f"[{ts()}] model commit {model_sha}")

    lm = LanguageModel(MODEL, revision=model_sha, torch_dtype=torch.float16,
                       device_map="auto")
    tok = lm.tokenizer
    source_sha = _stimulus_digest("/root/belief_tracking")
    if source_sha != EXPECTED_STIMULUS_SHA:
        raise RuntimeError(
            f"stimulus source digest {source_sha[:12]} does not match the pinned "
            f"{EXPECTED_STIMULUS_SHA[:12]}; the vendored generator or entity lists changed")

    # ── Stimuli: cached immutably, never regenerated over an existing shard ──
    done, shard_digest = _load_done(shard)
    pairs_path = None
    if shard_digest:
        pairs_path = os.path.join(d, f"filtered_pairs_{shard_digest}.json")
        if not os.path.exists(pairs_path):
            raise RuntimeError(
                f"shard references pair set {shard_digest[:12]} but {pairs_path} is missing; "
                "refusing to regenerate, because a regenerated set can differ by one "
                "borderline prediction and silently re-point every pair_index")
        with open(pairs_path) as fh:
            cached = json.load(fh)
        pairs = cached["pairs"]
        actual = _pairs_digest(pairs)
        if actual != shard_digest or cached["digest"] != shard_digest:
            raise RuntimeError(
                f"cached pair file does not match the shard: recomputed {actual[:12]}, "
                f"file records {cached['digest'][:12]}, shard rows carry {shard_digest[:12]}. "
                "A modified pair file would silently re-point every pair_index.")
        print(f"[{ts()}] resumed pair set {shard_digest[:12]}, {len(pairs)} pairs, "
              f"{len(done)} observations on disk")
        if cached["provenance"]["model_commit_sha"] != model_sha:
            raise RuntimeError(
                f"model changed since the shard was started: "
                f"{cached['provenance']['model_commit_sha']} -> {model_sha}")
        if cached["provenance"]["belief_tracking_digest"] != source_sha:
            raise RuntimeError("vendored belief_tracking source changed since the shard started")
    else:
        print(f"[{ts()}] generating and filtering stories")
        pairs = _filter_on_model(lm, _generate_binding_pairs(N_STORIES, SEED), N_EXP20)
        digest = _pairs_digest(pairs)
        pairs_path = os.path.join(d, f"filtered_pairs_{digest}.json")
        with open(pairs_path, "w") as fh:
            json.dump({"digest": digest, "n": len(pairs), "pairs": pairs,
                       "provenance": _provenance(model_sha, digest, source_sha)}, fh)
        shard_digest = digest
        print(f"[{ts()}] {len(pairs)} pairs, digest {digest[:12]}, cached")

    if limit:
        pairs = pairs[:limit]
        print(f"[{ts()}] DRY RUN limited to {len(pairs)} pairs; this run is a structural "
              "check and reports no registered result")
    if not limit and len(pairs) != N_EXP20:
        raise RuntimeError(
            f"filtering yielded {len(pairs)} pairs, registration specifies {N_EXP20}. "
            "Amend the registration or widen N_STORIES; do not proceed on a smaller sample.")

    prov = _provenance(model_sha, shard_digest, source_sha)

    drinks = _canonical_drinks()
    drink_ids = {dr: _single_token_id(tok, dr) for dr in drinks}
    multi = sorted(k for k, v in drink_ids.items() if v is None)
    if multi:
        raise RuntimeError(
            f"{len(multi)} canonical drinks are multi-token: {multi}. The registered "
            "greedy-completion fallback is not implemented in this script, so the "
            "confirmatory single-token endpoint cannot be formed. Implement it or amend.")
    drink_token_ids = sorted(v for v in drink_ids.values() if v is not None)

    manifest = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "registration": "AMENDMENTS.md Amendments 13, 14 and 15; tags "
                            "prereg-amendment-9, prereg-amendment-10, "
                            "prereg-amendment-11",
            "layers": LAYERS, "n_exp19": N_EXP19, "n_exp20": len(pairs),
            "filtered_pairs_file": os.path.basename(pairs_path),
            "single_token_drink_ids": drink_ids,
            "all_drinks_single_token": True,
            "greedy_completion_fallback_triggered": False,
            "dry_run": bool(dry_run), "provenance": prov,
    }
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            existing = json.load(fh)
        if existing["provenance"] != prov:
            raise RuntimeError(
                f"run_metadata.json provenance differs from this run: "
                f"{existing['provenance']} vs {prov}")
        with open(os.path.join(d, f"{prefix}resume_events.jsonl"), "a") as fh:
            fh.write(json.dumps({"resumed": datetime.now(timezone.utc).isoformat(),
                                 "observations_on_disk": len(done)}) + "\n")
    else:
        with open(meta_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
    vol.commit()

    def emit_all(rows):
        """Append a complete observation, or nothing. Partial writes are the failure
        mode that makes an incomplete run look finished."""
        stamp = datetime.now(timezone.utc).isoformat()
        for row in rows:
            row.update({"t": stamp, "dry_run": bool(dry_run),
                        "pairs_digest": shard_digest, "model_commit_sha": model_sha})
        # One write, not one per row: a failure between two writes leaves a partial
        # observation, which is the state the resume validator refuses to repair.
        payload = "".join(json.dumps(r) + "\n" for r in rows)
        with open(shard, "a") as fh:
            fh.write(payload)

    def undefined(exp, idx, layer, conditions, reason):
        return [{"exp": exp, "pair_index": idx, "layer": layer, "condition": c,
                 "undefined": True, "reason": reason} for c in conditions]

    # ── EXP19 ──
    n_gate = min(N_EXP19, len(pairs)) if limit else N_EXP19
    equal_len = []
    for i, s in enumerate(pairs):
        if len(tok.encode(s["clean_prompt"])) == len(tok.encode(s["counterfactual_prompt"])):
            equal_len.append(i)
        if len(equal_len) >= n_gate:
            break
    if len(equal_len) < n_gate:
        raise RuntimeError(f"only {len(equal_len)} equal-length pairs, {n_gate} required")

    for layer in LAYERS:
        for n, idx in enumerate(tqdm(equal_len, desc=f"EXP19 L{layer}")):
            if ("EXP19", idx, layer, "full_sequence") in done:
                continue
            s = pairs[idx]
            try:
                with torch.no_grad():
                    with lm.trace(s["counterfactual_prompt"]):
                        cf_out = lm.model.layers[layer].output[0][0].save()
                    with lm.trace(s["clean_prompt"]):
                        lm.model.layers[layer].output[0][0] = cf_out.detach().clone()
                        logits = lm.lm_head.output[0, -1].save()
                r = _readout(tok, logits, drink_token_ids,
                             _single_token_id(tok, s["clean_ans"].strip()),
                             _single_token_id(tok, s["counterfactual_ans"].strip()))
                hit = (tok.decode([r["pred_id"]]).lower().strip()
                       == s["counterfactual_ans"].lower().strip())
                emit_all([{"exp": "EXP19", "pair_index": idx, "layer": layer,
                           "condition": "full_sequence", "hit": bool(hit),
                           "cf_target": s["counterfactual_ans"],
                           "n_tokens": len(tok.encode(s["clean_prompt"])), **r}])
            except Exception as exc:
                emit_all(undefined("EXP19", idx, layer, ["full_sequence"],
                                   f"{type(exc).__name__}: {exc}"))
            if (n + 1) % COMMIT_EVERY == 0:
                vol.commit()
        vol.commit()

    gate = _gate_from_disk(shard, dry_run, required=n_gate)
    with open(gate_path, "w") as fh:
        json.dump(gate, fh, indent=2)
    vol.commit()
    print(f"[{ts()}] EXP19 gate: {json.dumps(gate['per_layer'])}")

    if not gate["gate_ok"]:
        print(f"[{ts()}] GATE FAILED - EXP20 not run. Source-target alignment or "
              "continuation setup is wrong, or observations are missing.")
        return gate

    # ── EXP20 ──
    for layer in LAYERS:
        for n, idx in enumerate(tqdm(range(len(pairs)), desc=f"EXP20 L{layer}")):
            # Checkpointing sits in a finally so every exit path commits: the
            # assertion-failure and union-failure paths both `continue`, and would
            # otherwise leave a run of invalid observations uncommitted.
            try:
                if all(("EXP20", idx, layer, c) in done for c in EXP20_CONDITIONS):
                    continue
                s = pairs[idx]
                clean_id = _single_token_id(tok, s["clean_ans"].strip())
                cf_id = _single_token_id(tok, s["counterfactual_ans"].strip())

                recalled = _find_state_positions(tok, s["clean_prompt"], s["clean_states"])
                lookback = _find_question_answer_positions(tok, s["clean_prompt"])
                cf_recalled = _find_state_positions(tok, s["counterfactual_prompt"],
                                                    s["counterfactual_states"])
                cf_lookback = _find_question_answer_positions(tok, s["counterfactual_prompt"])
                offset_lookback = _offset_question_answer_positions(tok, s["clean_prompt"])
                cf_offset_lookback = _offset_question_answer_positions(
                    tok, s["counterfactual_prompt"])
                off_recalled, off_roles = _offset_state_positions(
                    tok, s["clean_prompt"], s["clean_states"])
                cf_off_recalled, cf_off_roles = _offset_state_positions(
                    tok, s["counterfactual_prompt"], s["counterfactual_states"])
                off_state_pair = _offset_pair_by_role(cf_off_roles, off_roles)

                clean_ids_full = tok.encode(s["clean_prompt"])
                cf_ids_full = tok.encode(s["counterfactual_prompt"])
                n_clean, n_cf = len(clean_ids_full), len(cf_ids_full)

                checks = {
                    "answers_differ": s["clean_ans"].lower().strip()
                                      != s["counterfactual_ans"].lower().strip(),
                    "recalled_lookback_disjoint": not (set(recalled) & set(lookback)),
                    "recalled_cardinality_matches": len(recalled) == len(cf_recalled),
                    "lookback_cardinality_matches": len(lookback) == len(cf_lookback),
                    "recalled_nonempty": len(recalled) > 0,
                    "target_indices_in_range": all(0 <= p < n_clean for p in recalled + lookback),
                    "source_indices_in_range": all(0 <= p < n_cf for p in cf_recalled + cf_lookback),
                    "single_token_answers": clean_id is not None and cf_id is not None,
                }
                checks["all_ok"] = all(checks.values())
                cf_ids_probe = tok.encode(s["counterfactual_prompt"])
                clean_ids_probe = tok.encode(s["clean_prompt"])
                # The second acknowledged defect is that state occurrences pair by sorted
                # position rather than by semantic role. Recording the actual mapping, with
                # both token ids and their decoded text, makes that auditable after the fact
                # instead of merely asserted.
                legacy_state_mapping = [
                    {"cf_pos": int(cp), "clean_pos": int(tp),
                     "cf_id": int(cf_ids_probe[cp]) if cp < len(cf_ids_probe) else None,
                     "clean_id": int(clean_ids_probe[tp]) if tp < len(clean_ids_probe) else None,
                     "cf_tok": repr(tok.decode([cf_ids_probe[cp]])) if cp < len(cf_ids_probe) else None,
                     "clean_tok": repr(tok.decode([clean_ids_probe[tp]])) if tp < len(clean_ids_probe) else None}
                    for cp, tp in zip(cf_recalled, recalled)]
                diagnostic = {
                    "legacy_lookback": lookback,
                    "offset_lookback": offset_lookback,
                    "legacy_offset_agree": (offset_lookback is not None
                                            and set(lookback) == set(offset_lookback)),
                    "legacy_state_mapping": legacy_state_mapping,
                    "state_mapping_tokens_match": all(
                        m["cf_tok"] == m["clean_tok"] for m in legacy_state_mapping),
                    "offset_recalled": off_recalled,
                    "offset_state_roles": off_roles,
                    "offset_state_pairing": off_state_pair[2] if off_state_pair else None,
                    "legacy_offset_recalled_agree": (off_state_pair is not None
                                                     and sorted(off_state_pair[1]) == recalled),
                }

                # An assertion failure is an undefined observation, never a measured one.
                if not checks["all_ok"]:
                    failed = [k for k, v in checks.items() if not v and k != "all_ok"]
                    rows = undefined("EXP20", idx, layer, EXP20_CONDITIONS,
                                     "assertion failure: " + ", ".join(failed))
                    for row in rows:
                        row["assertions"] = checks
                        row["resolver_diagnostic"] = diagnostic
                    emit_all(rows)
                    continue

                try:
                    with torch.no_grad():
                        with lm.trace(s["clean_prompt"]):
                            cl_out = lm.model.layers[layer].output[0][0].save()
                            clean_logits = lm.lm_head.output[0, -1].save()
                        with lm.trace(s["counterfactual_prompt"]):
                            cf_out = lm.model.layers[layer].output[0][0].save()
                    cl_t = cl_out.detach().clone()
                    cf_t = cf_out.detach().clone()

                    # Indices were validated above, so no conditional bounds check here:
                    # an out-of-range index must crash rather than silently shrink the
                    # intervention into an undocumented partial one.
                    def patched_for(sets):
                        out = cl_t.clone()
                        used = []
                        for src, dst in sets:
                            for cp, tp in zip(src, dst):
                                out[tp] = cf_t[cp]
                                used.append(int(tp))
                        return out, sorted(set(used))

                    spec = {
                        "recalled_only": [(cf_recalled, recalled)],
                        "lookback_only": [(cf_lookback, lookback)],
                        "both": [(cf_recalled, recalled), (cf_lookback, lookback)],
                    }

                    # Offset arm. Runs only where every offset resolution succeeds and
                    # every index is in range; otherwise its three conditions are written
                    # undefined and the legacy arm proceeds regardless.
                    offset_ok = (
                        off_state_pair is not None
                        and offset_lookback is not None
                        and cf_offset_lookback is not None
                        and len(offset_lookback) == len(cf_offset_lookback)
                        and len(off_state_pair[0]) > 0
                        and not (set(off_state_pair[1]) & set(offset_lookback))
                        and not (set(off_state_pair[0]) & set(cf_offset_lookback))
                        and all(0 <= q < n_clean for q in off_state_pair[1] + offset_lookback)
                        and all(0 <= q < n_cf for q in off_state_pair[0] + cf_offset_lookback)
                    )
                    if offset_ok:
                        spec["recalled_only_offset"] = [(off_state_pair[0], off_state_pair[1])]
                        spec["lookback_only_offset"] = [(cf_offset_lookback, offset_lookback)]
                        spec["both_offset"] = [(off_state_pair[0], off_state_pair[1]),
                                               (cf_offset_lookback, offset_lookback)]

                    # Compute every condition before writing any of them.
                    pending = [{
                        "exp": "EXP20", "pair_index": idx, "layer": layer, "condition": "clean",
                        "patched_indices": [], "n_patched": 0,
                        **_readout(tok, clean_logits, drink_token_ids, clean_id, cf_id),
                    }]
                    used_by_condition = {}
                    for name, sets in spec.items():
                        patched, used = patched_for(sets)
                        used_by_condition[name] = used
                        with torch.no_grad():
                            with lm.trace(s["clean_prompt"]):
                                lm.model.layers[layer].output[0][0] = patched
                                logits = lm.lm_head.output[0, -1].save()
                        pending.append({
                            "exp": "EXP20", "pair_index": idx, "layer": layer, "condition": name,
                            "patched_indices": used, "n_patched": len(used),
                            **_readout(tok, logits, drink_token_ids, clean_id, cf_id),
                        })

                    if not offset_ok:
                        pending.extend(undefined(
                            "EXP20", idx, layer, OFFSET_CONDITIONS,
                            "offset resolution unavailable or out of range for this pair"))

                    union_ok = set(used_by_condition["both"]) == (
                        set(used_by_condition["recalled_only"])
                        | set(used_by_condition["lookback_only"]))
                    if offset_ok:
                        union_ok = union_ok and set(used_by_condition["both_offset"]) == (
                            set(used_by_condition["recalled_only_offset"])
                            | set(used_by_condition["lookback_only_offset"]))
                    if not union_ok:
                        emit_all(undefined("EXP20", idx, layer, EXP20_CONDITIONS,
                                           "both-condition is not the union of the singles"))
                        continue

                    by_cond = {r["condition"]: r for r in pending if not r.get("undefined")}
                    resolvers_agree = (
                        by_cond["both"]["pred_normalized"]
                        == by_cond["both_offset"]["pred_normalized"]
                        if "both_offset" in by_cond else None)

                    shared = {
                        "both_resolvers_agree": resolvers_agree,
                        "clean_ans": s["clean_ans"], "cf_ans": s["counterfactual_ans"],
                        "clean_answer_id": clean_id, "cf_answer_id": cf_id,
                        "recalled_indices": recalled, "lookback_indices": lookback,
                        "cf_recalled_indices": cf_recalled, "cf_lookback_indices": cf_lookback,
                        "recalled_decoded": repr(tok.decode([clean_ids_full[i] for i in recalled])),
                        "lookback_decoded": repr(tok.decode([clean_ids_full[i] for i in lookback])),
                        "cf_recalled_decoded": repr(tok.decode([cf_ids_full[i] for i in cf_recalled])),
                        "cf_lookback_decoded": repr(tok.decode([cf_ids_full[i] for i in cf_lookback])),
                        "n_tokens_clean": n_clean, "n_tokens_cf": n_cf,
                        "assertions": checks, "resolver_diagnostic": diagnostic,
                        "both_is_union": True,
                    }
                    for row in pending:
                        row.update(shared)
                    emit_all(pending)

                except Exception as exc:
                    emit_all(undefined("EXP20", idx, layer, EXP20_CONDITIONS,
                                       f"{type(exc).__name__}: {exc}"))
            finally:
                if (n + 1) % COMMIT_EVERY == 0:
                    vol.commit()
        vol.commit()

    completeness = _validate_exp20_from_disk(shard, len(pairs), LAYERS, dry_run)
    completeness["limited_dry_run"] = bool(limit)
    with open(os.path.join(d, f"{prefix}exp20_completeness.json"), "w") as fh:
        json.dump(completeness, fh, indent=2)
    vol.commit()

    print(f"[{ts()}] EXP20 completeness: {json.dumps(completeness['per_layer'])}")
    if not completeness["complete"]:
        print(f"[{ts()}] INCOMPLETE - some pair-layers are neither a full condition set "
              "nor an undefined record. Do not analyze until reconciled.")
    for layer, r in completeness["per_layer"].items():
        if r["resolver_agreement"] is not None:
            print(f"[{ts()}] L{layer} resolver agreement: {r['resolver_agreement']:.4f} "
                  f"over {r['offset_arm_defined']} pairs, "
                  f"{r['resolver_disagreements']} disagreements")
        if r["void_clean_accuracy_below_1"]:
            print(f"[{ts()}] L{layer} VOID: clean accuracy {r['clean_accuracy']:.4f} < 1.0")
        if r["void_both_incompatible_with_stored_zero"]:
            print(f"[{ts()}] L{layer} VOID: both-condition Wilson upper "
                  f"{r['both_iia_wilson_upper']:.4f} >= 0.05, not compatible with the "
                  "stored near-zero result")

    print(f"[{ts()}] done. observations in {shard}")
    return {"gate": gate, "completeness": completeness}


@app.local_entrypoint()
def main(dry_run: bool = False, limit: int = 0):
    run.remote(dry_run=dry_run, limit=limit)

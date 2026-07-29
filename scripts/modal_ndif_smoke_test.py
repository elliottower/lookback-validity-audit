"""Smoke test: minimal interchange intervention on NDIF (Llama 3.1 70B).

Swaps layer-34 hidden state between two prompts to verify causal intervention works.

Usage:
    modal run scripts/modal_ndif_smoke_test.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "nnsight==0.7.0",
        "torch==2.5.1",
        "transformers==4.48.3",
        "accelerate==1.3.0",
        "matplotlib==3.10.1",
    )
)

app = modal.App("lookback-ndif-smoke-test")


@app.function(
    image=image,
    secrets=[modal.Secret.from_dict({
        "NDIF_KEY": "801a2649-a054-4703-ad6d-9e93c8cbcb9a",
        "HF_TOKEN": modal.Secret.from_name("huggingface-token"),
    })],
    timeout=86400,
    cpu=2,
)
def smoke_test():
    import json
    import os
    from datetime import datetime, timezone

    ts = lambda: datetime.now(timezone.utc).isoformat()

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    from nnsight import LanguageModel

    try:
        from nnsight import CONFIG
        CONFIG.APP.REMOTE_LOGGING = False
        CONFIG.set_default_api_key(os.environ["NDIF_KEY"])
    except Exception:
        from nnsight import ndif as ndif_mod
        ndif_mod.api_key = os.environ["NDIF_KEY"]

    lm = LanguageModel("meta-llama/Meta-Llama-3.1-70B-Instruct")
    print(f"[{ts()}] Model loaded (meta tensors, no weights)")

    prompt_a = "The capital of France is"
    prompt_b = "The capital of Germany is"
    layer = 34

    # Clean outputs (separate traces, both remote)
    print(f"[{ts()}] Clean forward pass A...")
    with lm.trace(prompt_a, remote=True):
        logits_a = lm.output.logits.save()

    print(f"[{ts()}] Clean forward pass B...")
    with lm.trace(prompt_b, remote=True):
        logits_b = lm.output.logits.save()

    token_a = lm.tokenizer.decode(logits_a[0, -1].argmax().item())
    token_b = lm.tokenizer.decode(logits_b[0, -1].argmax().item())
    print(f"[{ts()}] Clean A: '{prompt_a}' -> '{token_a}'")
    print(f"[{ts()}] Clean B: '{prompt_b}' -> '{token_b}'")

    # Interchange intervention: swap layer hidden state from B into A
    print(f"[{ts()}] Intervention: patching L{layer} from B into A (remote)...")
    with lm.trace(remote=True) as tracer:
        barrier = tracer.barrier(2)

        with tracer.invoke(prompt_b):
            hidden_b = lm.model.layers[layer].output[0]
            barrier()

        with tracer.invoke(prompt_a):
            barrier()
            lm.model.layers[layer].output[0][:] = hidden_b
            logits_patched = lm.output.logits.save()

    token_patched = lm.tokenizer.decode(logits_patched[0, -1].argmax().item())
    print(f"[{ts()}] Patched: '{prompt_a}' + L{layer} from B -> '{token_patched}'")

    changed = token_patched != token_a
    toward_b = token_patched == token_b

    result = {
        "status": "ok",
        "timestamp": ts(),
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "layer": layer,
        "clean_a": token_a,
        "clean_b": token_b,
        "patched": token_patched,
        "changed": changed,
        "toward_b": toward_b,
    }

    print(f"\n{'='*60}")
    if toward_b:
        print(f"Intervention moved output toward B — causal patching works!")
    elif changed:
        print(f"Intervention changed output (to '{token_patched}') but not toward B")
    else:
        print(f"Intervention had no effect")
    print(f"{'='*60}")
    print(f"\nRESULT: {json.dumps(result, indent=2)}")
    return result

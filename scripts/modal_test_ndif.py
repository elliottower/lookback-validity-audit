"""Modal wrapper: test NDIF connection from a Linux container (CPU only, model runs on NDIF).

NNsight 0.6+ requires torch>=2.4.0 which has no Intel Mac wheels.
This runs nnsight on Modal (Linux, CPU-only) to talk to NDIF's remote GPUs.

Usage:
    modal run scripts/modal_test_ndif.py --detach
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "nnsight==0.7.0",
        "torch==2.5.1",
        "transformers==4.48.3",
        "python-dotenv==1.1.0",
        "matplotlib==3.10.1",
        "accelerate==1.3.0",
    )
)

app = modal.App("lookback-ndif-connection-test")


@app.function(
    image=image,
    secrets=[modal.Secret.from_dict({
        "NDIF_KEY": "801a2649-a054-4703-ad6d-9e93c8cbcb9a",
        "HF_TOKEN": modal.Secret.from_name("huggingface-token"),
    })],
    timeout=86400,
    cpu=2,
)
def test_ndif_connection():
    import json
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    ts = lambda: datetime.now(timezone.utc).isoformat()

    print(f"[{ts()}] Starting NDIF connection test")

    ndif_key = os.environ.get("NDIF_KEY")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not ndif_key:
        print("ERROR: NDIF_KEY not found in environment")
        return {"status": "error", "message": "no NDIF_KEY"}

    print(f"[{ts()}] NDIF_KEY present ({len(ndif_key)} chars)")

    os.environ["HF_TOKEN"] = hf_token

    from nnsight import LanguageModel

    try:
        from nnsight import CONFIG
        CONFIG.APP.REMOTE_LOGGING = False
        CONFIG.set_default_api_key(ndif_key)
    except Exception as e:
        print(f"[{ts()}] CONFIG setup note: {e}")
        from nnsight import ndif as ndif_mod
        ndif_mod.api_key = ndif_key

    print(f"[{ts()}] Checking NDIF status...")
    try:
        from nnsight import ndif as ndif_mod
        status = ndif_mod.status()
        print(f"[{ts()}] NDIF status:\n{status}")
    except Exception as e:
        print(f"[{ts()}] Could not check NDIF status: {e}")

    # dispatch=False (DEFAULT in 0.6+): meta tensors only, NO weight download
    print(f"[{ts()}] Loading model handle (meta tensors, no weight download)...")
    # Try multiple model names — NDIF only hosts "pinned" models
    model_candidates = [
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ]

    lm = None
    model_name = None
    for candidate in model_candidates:
        try:
            print(f"[{ts()}] Trying model: {candidate}")
            lm = LanguageModel(candidate)
            model_name = candidate
            print(f"[{ts()}] Model handle loaded: {candidate}")
            break
        except Exception as e:
            print(f"[{ts()}] Failed to load {candidate}: {e}")

    if lm is None:
        print(f"[{ts()}] ERROR: No model loaded")
        return {"status": "error", "message": "no model loaded"}
    print(f"[{ts()}] Model handle loaded")

    prompt = "The capital of France is"
    print(f"[{ts()}] Sending remote trace: '{prompt}'")

    with lm.trace(prompt, remote=True):
        hidden = lm.model.layers[-1].output[0].save()
        logits = lm.output.logits.save()

    pred_id = logits[0, -1].argmax().item()
    pred_token = lm.tokenizer.decode(pred_id)

    result = {
        "status": "ok",
        "timestamp": ts(),
        "hidden_shape": list(hidden.shape),
        "predicted_token": pred_token,
        "model": model_name,
    }

    print(f"[{ts()}] Last-layer hidden shape: {hidden.shape}")
    print(f"[{ts()}] Predicted next token: '{pred_token}'")
    print(f"[{ts()}] NDIF connection OK.")
    print(f"\nRESULT: {json.dumps(result)}")
    return result

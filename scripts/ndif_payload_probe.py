"""Does an NDIF trace carrying the 268 MB projection matrix succeed, and how slow is it?

The random-subspace baseline fails with WriteTimeout on every pair. Three explanations
were live: NDIF down, the model not deployed, or the payload. The first two are ruled
out -- api.ndif.us/ping answers in 0.09s and Llama-3.1-70B-Instruct has three
deployments. This measures the third directly instead of arguing about it.

    uv run --python .venv/bin/python scripts/ndif_payload_probe.py
"""

import os
import sys
import time
from pathlib import Path

import torch
from nnsight import CONFIG, LanguageModel

REPO = Path(__file__).resolve().parents[1]
MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
LAYER = 38
PROMPT = "Question: What is the capital of France? Answer:"


def main() -> None:
    key = os.environ.get("NDIF_KEY") or os.environ.get("NDIF_API_KEY")
    if not key:
        raise RuntimeError("set NDIF_KEY or NDIF_API_KEY")
    CONFIG.APP.REMOTE_LOGGING = False
    CONFIG.set_default_api_key(key)
    print(f"api host: {CONFIG.API.HOST}")

    lm = LanguageModel(MODEL)

    t0 = time.time()
    with lm.trace(PROMPT, remote=True):
        bare = lm.lm_head.output[0, -1].argmax(dim=-1).save()
    print(f"bare trace, nothing sent      {time.time() - t0:6.1f}s  "
          f"-> {lm.tokenizer.decode([bare.item()])!r}")

    basis = torch.load(
        REPO / "results/svd/CausalToM/last_token/singular_vecs" / f"{LAYER}.pt",
        weights_only=True).float()
    V = basis[:3]
    P = V.T @ V
    print(f"projection {tuple(P.shape)}  {P.numel() * 4 / 1e6:.0f} MB")

    t0 = time.time()
    try:
        with lm.trace(PROMPT, remote=True):
            x = lm.model.layers[LAYER].output[0][-1].clone()
            lm.model.layers[LAYER].output[0][-1] = x - (x @ P)
            big = lm.lm_head.output[0, -1].argmax(dim=-1).save()
        print(f"trace carrying 268 MB         {time.time() - t0:6.1f}s  "
              f"-> {lm.tokenizer.decode([big.item()])!r}")
    except Exception as exc:  # noqa: BLE001 - the identity of the failure is the result
        print(f"trace carrying 268 MB         {time.time() - t0:6.1f}s  "
              f"-> {type(exc).__name__}: {str(exc)[:110]}")
        import traceback; print(str(exc)[:1400])

    # Same intervention, factored. x @ (V.T @ V) == (x @ V.T) @ V, so this is the same
    # arithmetic with a 98 KB payload instead of 268 MB.
    print(f"factored basis {tuple(V.shape)}  {V.numel() * 4 / 1e3:.0f} KB")
    t0 = time.time()
    try:
        # Pass both orientations as contiguous tensors. A transpose is a view, and a
        # view may not be serialised the way a plain tensor is.
        Vd = V.to(torch.float16).contiguous()
        VdT = V.T.to(torch.float16).contiguous()
        with lm.trace(PROMPT, remote=True):
            x = lm.model.layers[LAYER].output[0][-1].clone()
            lm.model.layers[LAYER].output[0][-1] = x - (x @ VdT) @ Vd
            small = lm.lm_head.output[0, -1].argmax(dim=-1).save()
        print(f"trace carrying 98 KB          {time.time() - t0:6.1f}s  "
              f"-> {lm.tokenizer.decode([small.item()])!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"trace carrying 98 KB          {time.time() - t0:6.1f}s  "
              f"-> {type(exc).__name__}: {str(exc)[:110]}")
        import traceback; print(str(exc)[:1400])


if __name__ == "__main__":
    main()


def device_probe() -> None:
    """Is the device mismatch general to any CPU tensor, or specific to the factored form?

    If adding a plain 32 KB CPU vector fails the same way, then no script in this
    repository that mixes a local tensor into a remote trace can work under this nnsight
    version -- which would mean the stored results predate it.
    """
    key = os.environ.get("NDIF_KEY") or os.environ.get("NDIF_API_KEY")
    CONFIG.APP.REMOTE_LOGGING = False
    CONFIG.set_default_api_key(key)
    lm = LanguageModel(MODEL)
    import nnsight
    print(f"\nnnsight {nnsight.__version__}")

    small = torch.zeros(8192, dtype=torch.float16)

    for label, build in [
        ("plain CPU tensor", lambda x: small),
        (".to(x.device)", lambda x: small.to(x.device)),
        (".to(x)", lambda x: small.to(x)),
    ]:
        t0 = time.time()
        try:
            with lm.trace(PROMPT, remote=True):
                x = lm.model.layers[LAYER].output[0][-1].clone()
                lm.model.layers[LAYER].output[0][-1] = x + build(x)
                r = lm.lm_head.output[0, -1].argmax(dim=-1).save()
            print(f"  {label:<22} {time.time() - t0:6.1f}s  "
                  f"-> {lm.tokenizer.decode([r.item()])!r}")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).strip().splitlines()[-1][:100]
            print(f"  {label:<22} {time.time() - t0:6.1f}s  -> {type(exc).__name__}: {msg}")


if __name__ == "__main__" and "--device" in sys.argv:
    device_probe()

"""Minimal NDIF connection test: load Llama-3-70B via NNsight, run one remote forward pass."""

import os
from dotenv import load_dotenv

load_dotenv()

from nnsight import CONFIG, LanguageModel

CONFIG.APP.REMOTE_LOGGING = False
CONFIG.set_default_api_key(os.environ["NDIF_KEY"])
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

print("Loading model handle (no local weights)...")
lm = LanguageModel("meta-llama/Meta-Llama-3-70B-Instruct", dispatch=True)

prompt = "The capital of France is"

print(f"Sending remote trace: '{prompt}'")
with lm.trace(prompt, remote=True):
    hidden = lm.model.layers[-1].output[0].save()
    logits = lm.output.logits.save()

pred_id = logits.value[0, -1].argmax().item()
pred_token = lm.tokenizer.decode(pred_id)

print(f"Last-layer hidden shape: {hidden.value.shape}")
print(f"Predicted next token: '{pred_token}'")
print("NDIF connection OK.")

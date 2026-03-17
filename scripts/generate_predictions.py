# scripts/generate_predictions.py
# Runs all 3 ADR-Detect model variants on the test set and saves predictions.
# Run from quantscope/ root: python scripts/generate_predictions.py

import json
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Load config ───────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

RESULTS_DIR = Path(config["results"]["output_dir"])
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load test set ─────────────────────────────────────────────────────────
print("Loading test set...")
dataset  = load_from_disk(config["dataset"]["path"])
split    = config["dataset"]["split"]
test_set = dataset[split]
texts    = list(test_set[config["dataset"]["text_column"]])
labels   = list(test_set[config["dataset"]["label_column"]])
print(f"Test set: {len(texts)} samples")

# ── Inference helpers ─────────────────────────────────────────────────────
def run_pytorch(texts, path):
    path = Path(path).resolve()
    print(f"\n[pytorch] Loading from {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model     = AutoModelForSequenceClassification.from_pretrained(
        path, dtype=torch.float32
    )
    model.eval()

    predictions, confidences, latencies = [], [], []
    for i, text in enumerate(texts):
        if i % 500 == 0:
            print(f"  {i}/{len(texts)}")
        inputs = tokenizer(
            text, return_tensors="pt",
            padding=True, truncation=True, max_length=256
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

        probs = F.softmax(outputs.logits, dim=-1)[0]
        pred  = int(torch.argmax(probs).item())
        conf  = float(probs[pred].item())
        predictions.append(pred)
        confidences.append(conf)

    return predictions, confidences, latencies


def run_onnx(texts, path, file_name, tokenizer_path):
    path           = Path(path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    model_path     = path / file_name

    print(f"\n[onnx] Loading {file_name} from {path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    session   = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"]
    )

    # Check what input names the model expects
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"  Model inputs: {input_names}")

    predictions, confidences, latencies = [], [], []
    for i, text in enumerate(texts):
        if i % 500 == 0:
            print(f"  {i}/{len(texts)}")

        inputs = tokenizer(
            text, return_tensors="np",
            padding="max_length", truncation=True, max_length=256
        )

        # Build ort_inputs from only what the model expects
        ort_inputs = {
            name: inputs[name]
            for name in input_names
            if name in inputs
        }

        t0      = time.perf_counter()
        outputs = session.run(None, ort_inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

        logits = outputs[0][0]
        probs  = np.exp(logits) / np.exp(logits).sum()
        pred   = int(np.argmax(probs))
        conf   = float(probs[pred])
        predictions.append(pred)
        confidences.append(conf)

    return predictions, confidences, latencies


# ── Run all variants ──────────────────────────────────────────────────────
results = {
    "texts" : texts,
    "labels": labels,
}

for variant_name, variant_cfg in config["models"].items():
    if variant_cfg["type"] == "pytorch":
        preds, confs, lats = run_pytorch(
            texts, variant_cfg["path"]
        )
    else:
        preds, confs, lats = run_onnx(
            texts,
            variant_cfg["path"],
            variant_cfg["file"],
            variant_cfg["tokenizer_path"],
        )

    results[variant_name] = {
        "predictions" : preds,
        "confidences" : confs,
        "latencies_ms": lats,
    }
    print(f"  ✅ {variant_name} done")

# ── Save ──────────────────────────────────────────────────────────────────
out_path = RESULTS_DIR / "all_predictions.json"
with open(out_path, "w") as f:
    json.dump(results, f)
print(f"\n✅ Saved to {out_path}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n📊 Summary:")
true = np.array(labels)
for variant_name in config["models"]:
    preds = np.array(results[variant_name]["predictions"])
    acc   = (preds == true).mean()
    lats  = results[variant_name]["latencies_ms"]
    print(f"\n  {variant_name}:")
    print(f"    Accuracy    : {acc:.4f}")
    print(f"    Avg latency : {np.mean(lats):.1f} ms")
    print(f"    p50 latency : {np.median(lats):.1f} ms")
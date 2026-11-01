"""Measure digital zeros, low-level noise and quiet speech without introducing a filter.

All arrays are float32 at 16 kHz; there is no PCM16 quantization or resampling here.
The report contains only signal statistics, transcript hashes/counts, WER and timing.
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from app.viegem_loader import load_viegem  # noqa: E402
from scripts.fetch_fixtures import MANIFEST, sha256, SMALL_DIR, SR, verify  # noqa: E402
from scripts.scoring import text_sha256, wer  # noqa: E402
from scripts.smoke_inference import (  # noqa: E402
    block_network,
    decode_batch,
    environment,
    transcribe,
)
from scripts.stage_models import check as check_assets  # noqa: E402
from scripts.stage_models import MANIFEST as ASSETS_MANIFEST  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()
    block_network()
    manifest = json.loads(MANIFEST.read_text())
    assets = json.loads(ASSETS_MANIFEST.read_text())
    errors = verify(manifest, include_large=False)
    errors += [
        error
        for name, entry in assets["models"].items()
        for error in check_assets(name, entry, args.models_dir / entry["dir"], update=False)
    ]
    if errors:
        raise RuntimeError("; ".join(errors))
    model, tokenizer, units = load_viegem(args.models_dir, "cuda")
    run = partial(decode_batch, model, tokenizer, units, torch.device("cuda"), "bf16")
    results = []

    def measure(case, wav, reference=None):
        text = transcribe(wav, run, False)
        result = {
            "case": case,
            "samples": len(wav),
            "rms": float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))),
            "peak": float(np.max(np.abs(wav))),
            "text_sha256": text_sha256(text),
            "output_words": len(text.split()),
            "nonempty": bool(text),
        }
        if reference is not None:
            result["reference_words"] = len(reference.split())
            result["word_errors"] = round(wer(reference, text) * result["reference_words"])
            result["wer"] = wer(reference, text)
        if not np.any(wav):
            assert not text, case
        results.append(result)

    rng = np.random.default_rng(42)
    for seconds in (0.1, 1601 / SR, 0.5, 1, 2, 10, 30):
        measure(f"zeros/{seconds:g}s", np.zeros(round(seconds * SR), np.float32))
    for seconds in (2, 30):
        for amplitude in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
            measure(
                f"constant/{seconds}s/{amplitude:g}", np.full(seconds * SR, amplitude, np.float32)
            )
            measure(
                f"gaussian/{seconds}s/{amplitude:g}",
                rng.normal(0, amplitude, seconds * SR).astype(np.float32),
            )
    for name in ("vi_a.wav", "vi_b.wav"):
        wav, sr = sf.read(SMALL_DIR / name, dtype="float32")
        assert sr == SR
        for gain in (1, 0.1, 0.01, 0.001, 0.0001):
            measure(f"{name}/gain={gain:g}", wav * gain, manifest["files"][name]["reference"])
    report = {
        "environment": environment(torch),
        "precision": "bf16",
        "rng_seed": 42,
        "models": {k: v["revision"] for k, v in assets["models"].items()},
        "code_sha256": {
            p: sha256(ROOT / p) for p in ("scripts/probe_silence.py", "scripts/smoke_inference.py")
        },
        "method": "16kHz mono float32; no PCM16 quantization; no threshold; gateway digital-zero and <=100ms policies",
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(results), "nonempty": sum(r["nonempty"] for r in results)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

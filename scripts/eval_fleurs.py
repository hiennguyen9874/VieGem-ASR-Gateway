"""Offline, metadata-only greedy evaluation of the pinned FLEURS vi_vn/dev split.

Download the pinned dev.tsv and audio/dev.tar.gz with hf (docs/agents/commands.md).
No serving default or fixture gate is changed. Existing fixture source recordings are
also scored separately from the remaining recordings, so the extra evidence is explicit.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import tarfile
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from app.viegem_loader import load_viegem  # noqa: E402
from scripts.fetch_fixtures import DATASET, MANIFEST, REVISION, sha256, SR  # noqa: E402
from scripts.scoring import text_sha256, wer  # noqa: E402
from scripts.smoke_inference import (  # noqa: E402
    block_network,
    decode_batch,
    environment,
    gpu_mem,
    transcribe,
)
from scripts.stage_models import check as check_assets  # noqa: E402
from scripts.stage_models import MANIFEST as ASSETS_MANIFEST  # noqa: E402


def totals(results):
    words = sum(r["reference_words"] for r in results)
    errors = sum(r["word_errors"] for r in results)
    seconds = sum(r["inference_s"] for r in results)
    audio = sum(r["samples"] for r in results) / SR
    return {
        "recordings": len(results),
        "reference_words": words,
        "word_errors": errors,
        "wer": errors / words if words else None,
        "audio_s": audio,
        "inference_s": seconds,
        "rtf": seconds / audio if audio else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory containing dev.tsv and audio/dev.tar.gz",
    )
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    block_network()
    assets = json.loads(ASSETS_MANIFEST.read_text())
    failures = [
        error
        for name, entry in assets["models"].items()
        for error in check_assets(name, entry, args.models_dir / entry["dir"], update=False)
    ]
    if failures:
        raise RuntimeError("; ".join(failures))
    fixtures = json.loads(MANIFEST.read_text())["files"]
    excluded = {
        name
        for entry in fixtures.values()
        if isinstance(entry["source"], dict)
        for name in entry["source"]["files"]
    }
    rows = {
        r[1]: r
        for r in csv.reader(
            (args.data_dir / "dev.tsv").open(encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )
    }
    archive = args.data_dir / "audio/dev.tar.gz"
    report = {
        "environment": environment(torch),
        "precision": args.precision,
        "model": "viegem-asr",
        "models": {k: v["revision"] for k, v in assets["models"].items()},
        "dataset": {
            "repo": DATASET,
            "revision": REVISION,
            "split": "vi_vn/dev",
            "tsv_sha256": sha256(args.data_dir / "dev.tsv"),
            "archive_sha256": sha256(archive),
        },
        "method": "greedy, full context, gateway split <=30s; whitespace WER on normalized FLEURS column 3; no text normalization",
        "code_sha256": {
            p: sha256(ROOT / p) for p in ("scripts/eval_fleurs.py", "scripts/smoke_inference.py")
        },
        "fixture_source_files": sorted(excluded),
    }
    model, tokenizer, units = load_viegem(args.models_dir, "cuda")
    run = partial(decode_batch, model, tokenizer, units, torch.device("cuda"), args.precision)
    run([np.zeros(2 * SR, np.float32)])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    results = []
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or name not in rows:
                continue
            raw = tar.extractfile(member).read()
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
            assert sr == SR and wav.ndim == 1 and np.isfinite(wav).all()
            ref = rows[name][3]
            assert ref.split(), name
            torch.cuda.synchronize()
            start = time.perf_counter()
            text = transcribe(wav, run, False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            words = len(ref.split())
            results.append(
                {
                    "file": name,
                    "sentence_id": rows[name][0],
                    "samples": len(wav),
                    "audio_sha256": hashlib.sha256(raw).hexdigest(),
                    "text_sha256": text_sha256(text),
                    "reference_words": words,
                    "word_errors": round(wer(ref, text) * words),
                    "inference_s": seconds,
                }
            )
    assert len(results) == len(rows) and {r["file"] for r in results} == set(rows)
    report["all"] = totals(results)
    report["non_fixture"] = totals([r for r in results if r["file"] not in excluded])
    report["gpu"] = gpu_mem(torch)
    report["results"] = results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("precision", "all", "non_fixture", "gpu")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

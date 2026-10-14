"""R1 container gate (docs/m3-design.vi.md §7): build, non-root, no secrets or extra files in the image, offline
start to /ready with the models mounted read-only, smoke against the baseline, graceful SIGTERM, restart.

Needs Docker; the runtime checks also need the NVIDIA Container Toolkit, a free GPU and staged models:
    QUAN_GPU=<free gpu id> uv run pytest tests/container
The gateway runs with `--network none`; requests come from a sidecar container sharing its network namespace, with
this repository mounted read-only (scripts/smoke_server.py + fixtures). The smoke report is written to
docs/baseline/server-container.json.
"""

import json
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("QUAN_MODELS_DIR", ROOT / "models")).resolve()
IMAGE = "quan-transcribe:test"
KEY = "ctr-" + secrets.token_hex(16)
pytestmark = [pytest.mark.container,
              pytest.mark.skipif(shutil.which("docker") is None, reason="needs docker")]


def docker(*args: str, check: bool = True, timeout: float = 120, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True, timeout=timeout, **kw)


@pytest.fixture(scope="module")
def image() -> str:
    docker("build", "-t", IMAGE, str(ROOT), timeout=3600)
    return IMAGE


def run_in_image(image: str, script: str) -> str:
    return docker("run", "--rm", "--network", "none", "--entrypoint", "sh", image, "-c", script).stdout


def test_runs_as_non_root(image):
    assert run_in_image(image, "id -u").strip() == "10001"


def test_image_contains_only_the_app(image):
    listing = run_in_image(image, "ls -A /srv; ls -A /srv/assets").split()
    assert sorted(listing) == sorted([".venv", "app", "assets", "manifest.json"])
    for path in ("/srv/references", "/srv/models", "/srv/tests", "/srv/.git", "/srv/fixtures-large", "/models"):
        assert run_in_image(image, f"test -e {path} && echo present || echo absent").strip() == "absent", path
    versions = run_in_image(image, "python --version; ffmpeg -version | head -1")
    assert "Python 3.12.14" in versions and "ffmpeg version 6.1.1" in versions


def test_no_secrets_in_image_metadata(image):
    meta = docker("inspect", image).stdout + docker("history", "--no-trunc", image).stdout
    config_env = json.loads(docker("inspect", image).stdout)[0]["Config"]["Env"]
    assert not any(e.startswith(("HF_TOKEN=", "QUAN_API_KEYS=", "QUAN_API_KEYS_FILE=")) for e in config_env)
    assert "HF_TOKEN" not in meta and "QUAN_API_KEYS" not in meta
    hf_token = os.environ.get("HF_TOKEN")
    assert not hf_token or hf_token not in meta


def gpu() -> str:
    gpu_id = os.environ.get("QUAN_GPU")
    if gpu_id is None:
        pytest.skip("set QUAN_GPU to a free GPU id for the runtime checks")
    if not (MODELS / "embeddinggemma-2").is_dir():
        pytest.skip("models not staged")
    return gpu_id


def wait_ready(cid: str, timeout: float = 600) -> None:
    probe = "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)"
    deadline = time.monotonic() + timeout
    while docker("exec", cid, "python", "-c", probe, check=False).returncode != 0:
        state = json.loads(docker("inspect", cid).stdout)[0]["State"]
        assert state["Running"], docker("logs", cid, check=False).stderr[-4000:]
        assert time.monotonic() < deadline, "container did not become ready"
        time.sleep(2)


def sidecar(cid: str, image: str, *argv: str, timeout: float = 900, detach: bool = False):
    cmd = ["docker", "run", "--rm", "--network", f"container:{cid}", "-v", f"{ROOT}:/w:ro",
           "-e", f"OPENAI_API_KEY={KEY}", "--entrypoint", "python", image, *argv]
    if detach:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# Streams 6 x vi_long.wav (~640 s, ~22 chunks) and prints the event types.
STREAM_LONG = """
import io, json, os, urllib.request, uuid, wave
src = wave.open('/w/fixtures-large/vi_long.wav'); frames = src.readframes(src.getnframes())
buf = io.BytesIO(); out = wave.open(buf, 'wb'); out.setparams(src.getparams()); out.writeframes(frames * 6); out.close()
b = uuid.uuid4().hex
body = (f'--{b}\\r\\nContent-Disposition: form-data; name="model"\\r\\n\\r\\nquan-transcribe-1.0\\r\\n'
        f'--{b}\\r\\nContent-Disposition: form-data; name="stream"\\r\\n\\r\\ntrue\\r\\n'
        f'--{b}\\r\\nContent-Disposition: form-data; name="file"; filename="a.wav"\\r\\n\\r\\n').encode()
body += buf.getvalue() + f'\\r\\n--{b}--\\r\\n'.encode()
req = urllib.request.Request('http://127.0.0.1:8000/v1/audio/transcriptions', data=body, method='POST', headers={
    'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'], 'Content-Type': f'multipart/form-data; boundary={b}'})
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        if line.strip():
            e = json.loads(line.decode().removeprefix('data: '))
            print(e.get('type', 'error'), flush=True)
"""


def test_offline_ready_smoke_drain_and_restart(image):
    gpu_id = gpu()
    cid = docker("run", "-d", "--network", "none", "--gpus", f"device={gpu_id}", "--read-only",
                 "--tmpfs", "/tmp:size=512m,mode=1777", "--mount", f"type=bind,src={MODELS},dst=/models,readonly",
                 "-e", f"QUAN_API_KEYS={KEY}", image).stdout.strip()
    try:
        wait_ready(cid)
        r = sidecar(cid, image, "/w/scripts/smoke_server.py", "--base-url", "http://127.0.0.1:8000")
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        report = json.loads(r.stdout)
        report["image"] = json.loads(docker("inspect", image).stdout)[0]["Id"]
        (ROOT / "docs/baseline/server-container.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)
                                                                  + "\n")

        if (ROOT / "fixtures-large/vi_long.wav").exists():
            stream = sidecar(cid, image, "-c", STREAM_LONG, detach=True)
            assert stream.stdout.readline().strip() == "transcript.text.delta"  # stream is running
            started = time.monotonic()
            docker("stop", "-t", "70", cid, timeout=120)
            out, err = stream.communicate(timeout=120)
            assert stream.returncode == 0, err[-4000:]
            assert out.strip().splitlines()[-1] == "transcript.text.done"  # finished while draining
            assert time.monotonic() - started < 70
            state = json.loads(docker("inspect", cid).stdout)[0]["State"]
            assert state["ExitCode"] == 0
            logs = docker("logs", cid).stderr
            stopped = [json.loads(line) for line in logs.splitlines() if '"server.stopped"' in line]
            assert stopped and stopped[-1]["drained"] is True
            assert KEY not in logs

            docker("start", cid)
            wait_ready(cid)
    finally:
        docker("rm", "-f", cid, check=False)

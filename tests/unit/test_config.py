"""Startup configuration: required keys, finite limits, consistency, no silent typos."""

from pathlib import Path

import pytest

from app.config import Config, ConfigError

KEY = "k" * 20


def test_defaults_with_one_key():
    cfg = Config.from_env({"VIEGEM_API_KEYS": KEY})
    assert cfg.api_keys == (KEY,) and cfg.whisper_alias is False
    assert cfg.queue_max_buffered_samples == int(cfg.queue_max_buffered_seconds * 16000)
    assert KEY not in repr(cfg)


def test_parses_typed_values():
    cfg = Config.from_env({
        "VIEGEM_API_KEYS": f" {KEY} , {'j' * 20} ", "VIEGEM_WHISPER_ALIAS": "true", "VIEGEM_QUEUE_MAX_JOBS": "8",
        "VIEGEM_MAX_CHUNK_SECONDS": "20.5", "VIEGEM_MODELS_DIR": "/srv/models", "VIEGEM_LOG_LEVEL": "debug",
        "VIEGEM_SPOOL_DIR": "/var/spool/viegem",
    })
    assert cfg.api_keys == (KEY, "j" * 20) and cfg.whisper_alias is True and cfg.queue_max_jobs == 8
    assert cfg.max_chunk_seconds == 20.5 and cfg.models_dir == Path("/srv/models") and cfg.log_level == "DEBUG"
    assert cfg.max_chunk_samples == 328000 and cfg.spool_dir == Path("/var/spool/viegem")


@pytest.mark.parametrize("legacy", ["QUAN_API_KEYS", "QUAN_MODELS_DIR", "QUAN_UNKNOWN", "QUAN_GPU"])
def test_legacy_server_environment_is_rejected(legacy):
    with pytest.raises(ConfigError) as exc:
        Config.from_env({"VIEGEM_API_KEYS": KEY, legacy: "private-value"})
    assert "unknown configuration variables" in str(exc.value)
    assert legacy in str(exc.value) and "private-value" not in str(exc.value)


def test_api_keys_file(tmp_path):
    f = tmp_path / "keys"
    f.write_text(f"# comment\n{KEY}\n\n{'j' * 20}\n")
    assert Config.from_env({"VIEGEM_API_KEYS_FILE": str(f)}).api_keys == (KEY, "j" * 20)
    with pytest.raises(ConfigError):
        Config.from_env({"VIEGEM_API_KEYS_FILE": str(tmp_path / "missing")})


@pytest.mark.parametrize("env", [
    {},  # no public default key
    {"VIEGEM_API_KEYS": " , "},
    {"VIEGEM_API_KEYS": "short"},
    {"VIEGEM_API_KEYS": "é" * 20},
    {"VIEGEM_API_KEYS": "界" * 20},
    {"VIEGEM_API_KEYS": "a" * 19 + "\u007f"},
    {"VIEGEM_API_KEYS": f"{KEY},{KEY}"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_API_KEYS_FILE": "/x"},
    {"VIEGEM_API_KEYS": KEY, "HF_TOKEN": KEY},  # gateway key must not be the HF token
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_QUEUE_MAX_JOB": "4"},  # typo
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_QUEUE_MAX_JOBS": "many"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_QUEUE_MAX_JOBS": "0"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_MAX_AUDIO_SECONDS": "inf"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_MAX_AUDIO_SECONDS": "nan"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_DECODE_TIMEOUT_SECONDS": "-1"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_WHISPER_ALIAS": "maybe"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_LOG_LEVEL": "TRACE"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_MAX_CHUNK_SECONDS": "60", "VIEGEM_QUEUE_MAX_BUFFERED_SECONDS": "30"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_MAX_TURN_SECONDS": "1000", "VIEGEM_QUEUE_MAX_BUFFERED_SECONDS": "900"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_QUEUE_MAX_JOBS_PER_LANE": "10", "VIEGEM_QUEUE_MAX_JOBS": "5"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_MAX_SESSIONS": "40"},
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_MAX_CHUNK_SECONDS": "0.5"},  # splitter needs room above the short-clip floor
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_MAX_DECODED_SECONDS": "100"},  # a max-length file could never be held
    {"VIEGEM_API_KEYS": KEY, "VIEGEM_MAX_UPLOAD_BYTES": "1000", "VIEGEM_MAX_SPOOL_BYTES": "999"},
])
def test_invalid_config_fails_startup(env):
    with pytest.raises(ConfigError) as e:
        Config.from_env(env)
    assert KEY not in str(e.value)

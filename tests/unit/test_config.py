"""Startup configuration: required keys, finite limits, consistency, no silent typos."""

from pathlib import Path

import pytest

from app.config import Config, ConfigError

KEY = "k" * 20


def test_defaults_with_one_key():
    cfg = Config.from_env({"QUAN_API_KEYS": KEY})
    assert cfg.api_keys == (KEY,) and cfg.whisper_alias is False
    assert cfg.queue_max_buffered_samples == int(cfg.queue_max_buffered_seconds * 16000)
    assert KEY not in repr(cfg)


def test_parses_typed_values():
    cfg = Config.from_env({
        "QUAN_API_KEYS": f" {KEY} , {'j' * 20} ", "QUAN_WHISPER_ALIAS": "true", "QUAN_QUEUE_MAX_JOBS": "8",
        "QUAN_MAX_CHUNK_SECONDS": "20.5", "QUAN_MODELS_DIR": "/srv/models", "QUAN_LOG_LEVEL": "debug",
    })
    assert cfg.api_keys == (KEY, "j" * 20) and cfg.whisper_alias is True and cfg.queue_max_jobs == 8
    assert cfg.max_chunk_seconds == 20.5 and cfg.models_dir == Path("/srv/models") and cfg.log_level == "DEBUG"


def test_api_keys_file(tmp_path):
    f = tmp_path / "keys"
    f.write_text(f"# comment\n{KEY}\n\n{'j' * 20}\n")
    assert Config.from_env({"QUAN_API_KEYS_FILE": str(f)}).api_keys == (KEY, "j" * 20)
    with pytest.raises(ConfigError):
        Config.from_env({"QUAN_API_KEYS_FILE": str(tmp_path / "missing")})


@pytest.mark.parametrize("env", [
    {},  # no public default key
    {"QUAN_API_KEYS": " , "},
    {"QUAN_API_KEYS": "short"},
    {"QUAN_API_KEYS": "é" * 20},
    {"QUAN_API_KEYS": "界" * 20},
    {"QUAN_API_KEYS": "a" * 19 + "\u007f"},
    {"QUAN_API_KEYS": f"{KEY},{KEY}"},
    {"QUAN_API_KEYS": KEY, "QUAN_API_KEYS_FILE": "/x"},
    {"QUAN_API_KEYS": KEY, "HF_TOKEN": KEY},  # gateway key must not be the HF token
    {"QUAN_API_KEYS": KEY, "QUAN_QUEUE_MAX_JOB": "4"},  # typo
    {"QUAN_API_KEYS": KEY, "QUAN_QUEUE_MAX_JOBS": "many"},
    {"QUAN_API_KEYS": KEY, "QUAN_QUEUE_MAX_JOBS": "0"},
    {"QUAN_API_KEYS": KEY, "QUAN_MAX_AUDIO_SECONDS": "inf"},
    {"QUAN_API_KEYS": KEY, "QUAN_MAX_AUDIO_SECONDS": "nan"},
    {"QUAN_API_KEYS": KEY, "QUAN_DECODE_TIMEOUT_SECONDS": "-1"},
    {"QUAN_API_KEYS": KEY, "QUAN_WHISPER_ALIAS": "maybe"},
    {"QUAN_API_KEYS": KEY, "QUAN_LOG_LEVEL": "TRACE"},
    {"QUAN_API_KEYS": KEY, "QUAN_MAX_CHUNK_SECONDS": "60", "QUAN_QUEUE_MAX_BUFFERED_SECONDS": "30"},
    {"QUAN_API_KEYS": KEY, "QUAN_MAX_TURN_SECONDS": "1000", "QUAN_QUEUE_MAX_BUFFERED_SECONDS": "900"},
    {"QUAN_API_KEYS": KEY, "QUAN_QUEUE_MAX_JOBS_PER_LANE": "10", "QUAN_QUEUE_MAX_JOBS": "5"},
    {"QUAN_API_KEYS": KEY, "QUAN_MAX_SESSIONS": "40"},
])
def test_invalid_config_fails_startup(env):
    with pytest.raises(ConfigError) as e:
        Config.from_env(env)
    assert KEY not in str(e.value)

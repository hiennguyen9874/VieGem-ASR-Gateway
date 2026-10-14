"""Strict shared subtitle parsing for smoke and server gates."""

import pytest

from scripts.subtitles import parse_cues


@pytest.mark.parametrize("fmt, header, sep", [("srt", "1\n", ","), ("vtt", "WEBVTT\n\n", ".")])
def test_parse_cues(fmt, header, sep):
    body = f"{header}00:00:00{sep}040 --> 00:00:01{sep}200\nhello\n\n"
    assert parse_cues(body, fmt) == [(0.04, 1.2, "hello")]
    assert parse_cues(body.replace(" --> ", " "), fmt) is None
    assert parse_cues(body.replace(sep, "." if sep == "," else ","), fmt) is None
    assert parse_cues(body.replace("00:00:01", "00:60:01"), fmt) is None
    assert parse_cues(body.rstrip(), fmt) is None
    assert parse_cues(body.replace("hello", ""), fmt) is None


def test_empty_cues_and_bad_headers():
    assert parse_cues("", "srt") == []
    assert parse_cues("WEBVTT\n\n", "vtt") == []
    assert parse_cues("", "vtt") is None
    assert parse_cues("WEBVTT\n", "vtt") is None
    assert parse_cues("", "unknown") is None
    assert parse_cues("2\n00:00:00,040 --> 00:00:01,200\nhello\n\n", "srt") is None

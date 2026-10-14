"""Dependency-free parser for gateway SRT/VTT, shared by smoke and server tests."""

import re

_CLOCK = r"(\d{{2,}}):([0-5]\d):([0-5]\d){sep}(\d{{3}})"
_TIMING = {fmt: re.compile(_CLOCK.format(sep=sep) + " --> " + _CLOCK.format(sep=sep))
           for fmt, sep in (("srt", ","), ("vtt", r"\."))}


def parse_cues(body: str, fmt: str) -> list[tuple[float, float, str]] | None:
    """Return (start s, end s, text) cues, or None for malformed gateway output."""
    if fmt not in _TIMING:
        return None
    if fmt == "vtt":
        if not body.startswith("WEBVTT\n\n"):
            return None
        body = body[len("WEBVTT\n\n"):]
    if body and not body.endswith("\n\n"):
        return None
    cues = []
    for i, block in enumerate(body.split("\n\n")[:-1], 1):
        lines = block.split("\n")
        if fmt == "srt":
            if lines[0] != str(i):
                return None
            lines = lines[1:]
        if len(lines) != 2 or not lines[1]:
            return None
        m = _TIMING[fmt].fullmatch(lines[0])
        if m is None:
            return None
        h, minute, s, ms, eh, em, es, ems = map(int, m.groups())
        cues.append((h * 3600 + minute * 60 + s + ms / 1000,
                     eh * 3600 + em * 60 + es + ems / 1000, lines[1]))
    return cues

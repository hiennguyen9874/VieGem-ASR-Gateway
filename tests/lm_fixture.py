"""Tiny hand-made ARPA language model for CPU tests (KenLM reads ARPA text under any file name).

Words: "ab" and "d" are likely ("<s> ab d </s>"), "e" and "aa" exist, "ac" is outside the vocabulary.
"""

from pathlib import Path

from app.lm import LM_DIR, LM_FILE, UNIGRAMS_FILE

WORDS = ["ab", "d", "e", "aa"]

# Hand-made bigram model: "ab d" is likely, "ac" is not in the vocabulary.
ARPA = """
\\data\\
ngram 1=7
ngram 2=4

\\1-grams:
-1.0\t<unk>\t0
-99\t<s>\t-0.3
-1.0\t</s>\t0
-0.4\tab\t-0.3
-0.8\td\t-0.2
-1.2\te\t-0.2
-1.5\taa\t-0.2

\\2-grams:
-0.1\t<s> ab
-0.1\tab d
-0.2\td </s>
-0.5\te </s>

\\end\\
"""


def write_arpa(path: Path) -> Path:
    path.write_text(ARPA.lstrip())
    return path


def stage_tiny_lm(models_dir: Path) -> Path:
    """Stage the tiny LM where `LMPool` looks for the real one; returns `models_dir`."""
    lm_dir = models_dir / LM_DIR
    lm_dir.mkdir(parents=True, exist_ok=True)
    write_arpa(lm_dir / LM_FILE)
    (lm_dir / UNIGRAMS_FILE).write_text("\n".join(WORDS) + "\n")
    return models_dir

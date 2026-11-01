# VieGem-ASR Gateway

A self-hosted gateway that serves one Vietnamese speech-recognition model through a scoped OpenAI-compatible API.

## Language

**Model**:
VieGem-ASR, the one speech-recognition model the gateway serves; clients address it as `viegem-asr`.
_Avoid_: QUAN, QUAN-Transcribe (the previous model, no longer served), engine (that is the code that runs the Model)

**Base encoder**:
EmbeddingGemma 2, the pretrained audio encoder the Model is built on and that is staged next to it.
_Avoid_: base model, Gemma

**Language model (LM)**:
The n-gram word model shipped with the Model that rescores its hypotheses during decoding.
_Avoid_: KenLM (the file format/library), n-gram

---
status: accepted
---

# LM decoding on a CPU pool, timestamps by forced alignment

The VieGem-ASR language model (KenLM, Python prefix beam search) is CPU-bound and slow next to the GPU forward pass, so it runs in a separate, bounded CPU worker pool; the engine thread only runs the model and hands back logits pruned to the beam-search candidates. Because the LM changes the text, timed formats (verbose_json/srt/vtt) get their times by CTC forced alignment of the LM's units on the same logits, which keeps the rule that times come from the same inference as the text. The LM is a server setting that stays off until its gate (WER, decode latency) passes.

## Considered Options

- Beam search on the engine thread: simpler, but blocks the single GPU worker for every chunk.
- Greedy text and times for timed formats: plain `json` and `srt` of the same audio would disagree.
- Tracking frames inside each beam hypothesis: invasive change to the vendored decoder.

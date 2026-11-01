# Audio and timestamps

## Ingress and chunking

- Decode client audio through `AudioIngress.decode` and split through `chunking.split`. FFmpeg uses fixed argv, the `file` protocol and demuxer/decoder allowlists; client filenames, paths and URLs must not reach its arguments, disk paths or logs.
- Preserve the body limit before multipart parsing and the spool and decoded-audio RAM budgets (`hold`, Realtime `reserve`). Resource exhaustion must use the appropriate OpenAI error envelope; see [API errors](../api/errors.md).
- The frontend crashes below 161 samples at 16 kHz. Audio ≤ 100 ms returns empty text without a model call. After inference an all-zero chunk returns empty text/path while retaining its measured frame count. Keep nonzero audio on the decode path: an amplitude threshold can discard quiet speech ([baseline](../baseline.md#giới-hạn-frontend-và-im-lặng)).

## Timestamps

Use the CTC path from the same inference as the text, plus the chunk offset: greedy path normally, forced alignment of LM units on the same lattice with LM enabled. A mismatch raises `AlignmentError` and returns 500 `alignment_failed`; never fabricate timing. Word timestamps remain behind `WORD_TIMESTAMPS` until the [R2b gate](../roadmap.md) passes; flip the switch in the commit containing the gate report.

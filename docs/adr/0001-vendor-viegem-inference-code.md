---
status: accepted
---

# Vendor VieGem-ASR inference code, isolated under its own licence

VieGem-ASR's model code ships only inside its Hugging Face repo (`5CD-AI/VieGem-ASR`, `asr/`), not as a package, and is licensed CC-BY-NC-SA-4.0. We keep a trimmed, inference-only copy in `app/viegem/` with its own `LICENSE` and a `NOTICE` naming the source revision and our changes, so share-alike stays confined to that directory and the image never executes code from the models mount. Parity with the upstream code is proven by an exact-logits test against the read-only submodule `references/VieGem-ASR` pinned at the same revision as the staged weights.

## Considered Options

- Import `asr/` from the staged, SHA-pinned model directory at runtime: no fork to maintain, but code would run from a data mount and every upstream file would become part of the trusted runtime.
- Declare the whole repository CC-BY-NC-SA-4.0: simpler, but forces the licence onto gateway code that does not derive from VieGem.

## Consequences

- The deployment is non-commercial only (CC-BY-NC-SA-4.0 plus the Gemma Terms of Use).
- Upstream fixes are pulled by hand: bump the submodule and the manifest pin together, then rerun the parity test.

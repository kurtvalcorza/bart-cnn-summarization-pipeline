# Weight provenance and DIMER hosting

- Upstream: `facebook/bart-large-cnn`
- Immutable revision: `37f520fa929c961707657b28798b30c003dd100b`
- Weight format: SafeTensors (`model.safetensors`, 1625222120 bytes, SHA-256 `40041830399afb5348525ef8354b007ecec4286fdf3524f7e6b54377e17096cb`; 512 float32 tensors, 406 290 432 parameters)
- Upstream weight license: MIT (`license: mit` in the snapshot `README.md` front matter; the snapshot carries no separate `LICENSE` file)
- Local snapshot: `weights/bart-large-cnn/` with `dimer-base-manifest.json` (8 files, per-file bytes + SHA-256, `totalBytes` 1627941444); the Git repository does not vendor the checkpoint.
- Generation defaults: `generation_config_for_summarization.json` (SHA-256 `4897361917410254e3132e8fe7786d37f3ef7cff54a650845c1147c2450a790f`, byte-identical to `generation_config.json`) — `num_beams` 4, `length_penalty` 2.0, `no_repeat_ngram_size` 3, `early_stopping` true, `max_length` 142, `min_length` 56; exposed by the package as `DEFAULT_*` constants (length bounds converted to new-token counts).
- Load-time check: `stage_missing_files()` then `verify_snapshot()` in `src/bart_summarization_pipeline/pipeline.py` — the first refuses a manifest naming another model or revision and fetches only missing manifest entries when `allow_download=True`; the second re-hashes every manifest entry and refuses on any mismatch. Both run before `torch`/`transformers` are imported.
- DIMER hosting: MIT permits use, modification, distribution and commercial use provided the licence and copyright notice accompany copies; DIMER may mirror the pinned checkpoint in its model store under the upstream license with that notice.
- Loader trust boundary: `transformers==4.57.6` built-in `BartForConditionalGeneration` + `BartTokenizerFast` (named explicitly because the snapshot has no `tokenizer_config.json`), `trust_remote_code=False`, `local_files_only=True` from the snapshot directory. Hub download is opt-in and pinned to the revision above.

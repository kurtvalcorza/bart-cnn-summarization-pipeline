# BART-CNN Summarization Pipeline

DIMER inference and fine-tuning wrapper for **`facebook/bart-large-cnn`** — abstractive summarization of one English document by deterministic beam search — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot, with a bounded adaptation contract: a digest-pinned real referenced corpus (SciTLDR-A, one-sentence TL;DRs of paper abstracts), corpus ROUGE-1/2/L with Lead-N baselines, supervised fine-tuning of the last decoder blocks with validation-ROUGE-L epoch selection, and a safetensors adapter that reloads onto the digest-verified base.

## Upstream alignment

- Model: `facebook/bart-large-cnn` (BART-large fine-tuned on CNN/DailyMail; 12+12 layers, `d_model` 1024, 406 M parameters, English)
- Revision: `37f520fa929c961707657b28798b30c003dd100b`
- Upstream weight license: MIT
- Upstream task: news summarization; the generation defaults are the snapshot's `generation_config_for_summarization.json` (`num_beams` 4, `length_penalty` 2.0, `no_repeat_ngram_size` 3, `early_stopping`, `max_length` 142 / `min_length` 56 → `max_new_tokens` 141 / `min_new_tokens` 55)
- Repository adaptation: bounded supervised fine-tuning of the last *k* decoder blocks (`adapt`, default 2 of 12 = 33,593,344 of 406,290,432 parameters) on caller-supplied or pinned SciTLDR-A records; the encoder, embeddings and tied output projection are never modified; the adapter carries only the trained tensors and is bound to the base `model.safetensors` SHA-256

## Quick start

```python
from bart_summarization_pipeline import BARTSummarizationPipeline

pipe = BARTSummarizationPipeline.from_pretrained()          # cuda:0 if available, else cpu
result = pipe.summarize(document_text)                       # pinned generation defaults
print(result["summary"], result["generated_tokens"], result["truncated"])

short = pipe.summarize(document_text, max_new_tokens=60, min_new_tokens=20, num_beams=2)
```

`summarize` returns the summary text plus `generated_tokens`, `input_tokens`, `truncated` (the decoder hit `max_new_tokens` before end-of-sequence), `stopped_by` and the full generation settings. Decoding is deterministic beam search (`do_sample=False`), so the same input on the same device gives the same summary; **no score, probability or quality metric is emitted**. Ceilings: `MAX_TEXT_CHARS = 40000`, `MAX_INPUT_TOKENS = 1024` (rejected, not truncated), `MAX_NEW_TOKENS = 512`, `MAX_NUM_BEAMS = 8`, `LENGTH_PENALTY_RANGE = (-5.0, 5.0)`, `MAX_NO_REPEAT_NGRAM_SIZE = 10`. `evaluate(records, **generation)` summarises a validated `{id, source, targets}` dataset under explicit settings and reports corpus ROUGE-1/2/L with the mean summary and reference lengths (`measured` / `measured-small-sample`); `evaluation_report(result, references)` scores one document as `sample-sanity` and still returns `not-measurable` without references; `adapt(train, val, *, epochs=2, lr=3e-5, batch_size=8, trainable_decoder_layers=2, seed=0, eval_generation=...)` fine-tunes the last decoder blocks and keeps the best-validation-ROUGE-L epoch; `save_artifact` / `from_artifact` export and reload the trained tensors as safetensors with a manifest bound to the base weight digest. Dataset helpers (`fetch_corpus`, `read_corpus`, `build_sample_dataset`, `validate_dataset`, `split_dataset`, `check_split_disjoint`, `load_byod_dataset`, `write_dataset_csv`) live in `samples.py`; ROUGE and `lead_baseline` in `metrics.py`; records are 8–20,000 mappings with one or more reference summaries each, and every inference ceiling is a refusal, never a silent cut (training truncates sources to 512 and targets to 64 tokens).

Adaptation on the pinned SciTLDR-A sample (CPU, about ten minutes including the validation passes):

```python
from bart_summarization_pipeline import BARTSummarizationPipeline, fetch_sample_dataset, check_split_disjoint, lead_baseline

TLDR = {'max_new_tokens': 48, 'min_new_tokens': 0, 'num_beams': 4}   # the news defaults force >= 55 tokens; score both models at TL;DR length
splits = fetch_sample_dataset()            # three pinned JSON-Lines files (5.5 MB), digest-verified, cached under weights/scitldr/
check_split_disjoint(splits)               # 300 / 50 / 100 records from SciTLDR-A's own paper-disjoint members
pipe = BARTSummarizationPipeline.from_pretrained()
print(lead_baseline(splits['test'])['rougeL'], pipe.evaluate(splits['test'], **TLDR)['rougeL'])   # 23.61, 25.52 in the recorded run
pipe.adapt(splits['train'], splits['validation'], eval_generation=TLDR)                          # last 2 decoder blocks, 2 epochs, best validation ROUGE-L kept
print(pipe.evaluate(splits['test'], **TLDR)['rougeL'])                                          # 34.28 in the recorded run
artifact = pipe.save_artifact('outputs/adapter')                                                # adapter.safetensors (134 MB) + manifest.json
again = BARTSummarizationPipeline.from_artifact(artifact)                                       # verifies base digest + artifact digest before applying
```

## Weights layout

```
weights/bart-large-cnn/
  dimer-base-manifest.json                  # modelId, revision, per-file bytes + sha256 (verified on every load)
  config.json                               # BartForConditionalGeneration architecture
  generation_config.json                    # beam-search defaults (identical to the file below)
  generation_config_for_summarization.json  # the defaults this package exposes
  tokenizer.json, vocab.json, merges.txt    # byte-level BPE (no tokenizer_config.json upstream; BartTokenizerFast is named explicitly)
  model.safetensors                         # 1625222120 bytes, git-ignored
  README.md                                 # upstream card, listed in the manifest; not used by the loader
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` and refuses to load if any manifest file is missing or its SHA-256 differs. On a fresh clone (manifest committed, weights git-ignored) `from_pretrained(allow_download=True)` fetches only the missing files at the pinned revision; the default is to refuse. Without any manifest, `allow_download=True` loads from the Hub with `revision=37f520fa929c961707657b28798b30c003dd100b`.

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed (34 tests + 5 notebook-parity tests; 2 model-backed tests run only with the staged snapshot)
```

Smoke (loads the verified snapshot on CPU; measured numbers are in `MODEL_CARD.md` → Runtime):

```python
from bart_summarization_pipeline import BARTSummarizationPipeline

pipe = BARTSummarizationPipeline.from_pretrained(device="cpu")
print(pipe.summarize("<a few paragraphs of English prose>")["summary"])
```

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/tutorials/bart_summarization_colab.ipynb)

`tutorials/bart_summarization_colab.ipynb` is declared `E2E` (mode `GUIDED`) under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`; see `tutorials/README.md`). Its default path fetches the three pinned SciTLDR-A files (5.5 MB, Apache-2.0, each refused on any digest mismatch) and draws 300 / 50 / 100 source-disjoint records with four refusal probes, stages the git-ignored `model.safetensors` with `stage_missing_files(..., allow_download=True)` and digest-verifies the snapshot, exercises the inference contract on a synthetic passage under the pinned news defaults and then under TL;DR-length settings, scores the Lead-1 and Lead-3 baselines and the frozen model on the test split (ROUGE-L 23.61 / 19.86 / 25.52 in the recorded run, the frozen summaries about twice the reference length), fine-tunes the last two decoder blocks for two epochs with validation-ROUGE-L epoch selection (362.1 s on CPU), re-scores the test split (ROUGE-L 34.28, summaries at reference length), summarises four unseen abstracts with a `measured-small-sample` verdict, exports a 134 MB safetensors adapter and reloads it with 4/4 identical summaries. Every number is one seeded split with no dispersion estimate. BYOD (`{id, source, target}` / `{id, source, targets}` as CSV, JSON or JSONL) is optional and gated off by default. See `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — the `E2E` notebook blob `e0501a20` (committed at `833efec`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 518.6 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.1 card
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance and hosting
- [`STATUS.md`](STATUS.md) — release status

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights are MIT; see `docs/WEIGHTS.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.

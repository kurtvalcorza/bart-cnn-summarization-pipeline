# Release verification

`tutorials/bart_summarization_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `BARTSummarizationPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `check_split_disjoint`, `write_dataset_csv`, `validate_inputs` with the `num_beams` refusal probe, `pipe.summarize`
  under the pinned defaults with the sanity checks and again under the TL;DR settings, `lead_baseline` for one and
  three sentences, `pipe.evaluate` on the frozen model and on the validation and test splits after adaptation with
  the ROUGE-L assertion, `pipe.adapt` with its explicit hyperparameters and `eval_generation=TLDR`, `pipe.evaluate`
  on the unseen abstracts plus the single-document `evaluation_report`, `pipe.save_artifact`,
  `BARTSummarizationPipeline.from_artifact` and the reload-parity assertion, and the provenance fields
  `weight_format`, `weight_sha256` and the `corpus` block), the six expected `outputs/` paths, the learner-facing
  statements (no score emitted, deterministic beam search, adaptation with references, out of domain, the Lead
  baselines, the length note, rouge-score-style not rouge-score-identical, no dispersion estimate, inputs above the
  token ceiling rejected not truncated or chunked, named exclusions, the Apache-2.0 corpus licence) and the gated-off
  BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the
  primary path, a mutable `revision='main'`, direct `from transformers import` / `BartTokenizerFast` /
  `BartForConditionalGeneration` / `.generate(` / `from huggingface_hub import` / `urllib.request` / `safetensors` /
  `torch.optim` / `.backward(` / `pipe._model` use **outside the carried module cells**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `tokenizers`, `huggingface-hub`, `safetensors`
and `numpy`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`; injected runner, token counter and corpus fetcher, temporary manifests, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/bart-large-cnn/` or the corpus cache `weights/scitldr/` (the standalone path writes the manifest
   itself, stages the missing files from the Hub, and fetches the three pinned SciTLDR-A files from the project
   repository, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `TLDR_MAX_NEW_TOKENS = 48`, `TLDR_MIN_NEW_TOKENS = 0`, `NUM_BEAMS = 4`,
   `EPOCHS = 2`, `LEARNING_RATE = 3e-5`, `BATCH_SIZE = 8`, `TRAINABLE_DECODER_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3` (an interpreter restart after the install is expected where the runtime's
   preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `BARTSummarizationPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `filter_records`, `validate_dataset`, `check_split_disjoint`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv`, `rouge_n`, `rouge_l`, `summary_metrics`, `lead_baseline` and the
     ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `facebook/bart-large-cnn` at the immutable revision, and `verify_snapshot` returning its dict (8 files);
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source` `local-snapshot`;
   - Section 4: `fetch_corpus` fetching the three pinned files (3,155,015 / 1,124,865 / 1,204,107 bytes) from
     `raw.githubusercontent.com` into `weights/scitldr/`, 1,992 + 619 + 618 raw papers read, and the seeded draw
     of 300 / 50 / 100 records with `check_split_disjoint` reporting no shared source and the three dataset
     digests `5fc73d14…` / `4693cd21…` / `55db9c45…`; `outputs/…_train.csv` written; the four dataset refusal
     probes each raising `ValueError`;
   - Section 5: the ceilings (`MAX_TEXT_CHARS` 40000, `MAX_INPUT_TOKENS` 1024, `MAX_NEW_TOKENS` 512, `MAX_NUM_BEAMS`
     8, `LENGTH_PENALTY_RANGE`, `MAX_NO_REPEAT_NGRAM_SIZE` 10), the pinned defaults and `DECISION_RULE` surfaced;
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding
     from the `num_beams` ceiling probe); `pipe.summarize` on the synthetic passage under the pinned defaults with
     every sanity check `True` (the card-pass smoke returned 81 tokens, `stopped_by: eos`), then under the TL;DR
     settings with a shorter summary;
   - Section 6: the Lead-1 and Lead-3 baselines (ROUGE-L ≈ 23.6 / 19.9 on the sample) and the frozen model's test
     score (ROUGE-1/2/L ≈ 33.5 / 14.7 / 25.5 at ≈ 35 words against ≈ 19-word references, on CPU float32), with the
     cell's assertion that the frozen ROUGE-1 is positive;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 33,593,344 trainable of 406,290,432 parameters,
     300 training abstracts, and a two-epoch history with validation ROUGE-L rising and `mean_summary_words`
     falling towards the reference length (`best_epoch` 2 in the recorded run);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison and
     `outputs/…_evaluation_report.json` written (the cell asserts the adapted test ROUGE-L exceeds the frozen one —
     on the sample ≈ 34.28 versus ≈ 25.52, with summaries at ≈ 19.8 words);
   - Section 9: four unseen dev abstracts summarised with `pipe.evaluate` returning `measured-small-sample`, the
     single-document `evaluation_report` verdict `sample-sanity`, `outputs/…_summaries.csv` written;
     `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (52 tensors, about
     134 MB) and `BARTSummarizationPipeline.from_artifact` reloading it with 4/4 identical summaries (the cell
     asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the
     snapshot block (`weight_format`, `weight_sha256`), the `corpus` block, the inference-contract items, the
     comparison, the artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `bart_summarization_colab.ipynb` (`E2E`) | `52c3522` / `afa4e964` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/bart_summarization_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/bart_summarization_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates. No hosted run of the earlier `TASK-INFERENCE` notebook was ever recorded.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `52c3522` / `afa4e964` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 8 entries because the snapshot was pre-staged → `verify_snapshot` 8 files → `from_pretrained` on CPU → `fetch_corpus` served from the pre-staged cache after its digest checks → 1,992 + 619 + 618 papers read, 300 / 50 / 100 drawn with `check_split_disjoint` clean and digests `5fc73d14…` / `4693cd21…` / `55db9c45…` → four dataset refusals → input manifest + `num_beams` refusal probe → `summarize` under the pinned defaults (81 tokens, `eos`, 3.5 s) and TL;DR settings (46 tokens, 1.8 s) with every sanity check `True` → Lead-1/Lead-3 → frozen evaluation → `adapt` → validation + test evaluation → four unseen abstracts → adapter export → reload parity) | 746.4 s | **PASSED** — 11/11 code cells; Lead-1 ROUGE-1/2/L 29.45 / 11.23 / 23.61, Lead-3 26.76 / 10.38 / 19.86; frozen test 33.53 / 14.67 / 25.52 at 35.0 words against 17.7-word references (178.3 s, 0 at the token ceiling); `adapt` 33,593,344 of 406,290,432 params, 300 abstracts, 2 epochs, 353.1 s, validation ROUGE-L 24.19 → 29.49 → 32.28 (`best_epoch` 2, train loss 2.608 → 2.294); **adapted test 39.17 / 21.34 / 34.28 at 19.8 words (Δ +5.64 / +6.66 / +8.76)**; four unseen abstracts 56.18 / 36.20 / 47.64 `measured-small-sample`, single-document report `sample-sanity`; adapter 134,379,512 B / 52 tensors, SHA-256 `17fc98c3…`; reload parity 4/4; six exports written. Pre-flight; hosted clean-runtime run still required |

## Current status

The `E2E` notebook source is complete and passes all static checks, including the generator parity checks
(`--check` OK). A local pre-flight execution of the committed blob completed the whole default path on CPU — corpus
read from the cache, validation and split, the inference contract under two settings, both Lead baselines, two epochs
of decoder fine-tuning, held-out evaluation, unseen-abstract summarization, adapter export and reload parity — which
catches defects but is **not** a supported runtime under REL1/REL10, and it ran with the snapshot and the three
SciTLDR files pre-staged, so neither the 1.6 GB Hub fetch nor the corpus download has been exercised by this notebook
end to end; the hosted run must cover both. The repository stays at **Candidate** until a Colab or fresh-container run
of the exact `E2E` release revision is recorded above.

# BART-large CNN Summarization E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/bart-cnn-summarization-pipeline`  
**Notebook:** `tutorials/bart_summarization_colab.ipynb`  
**Reviewed commit:** `e038514f47044ee293d2bbe6aaa35d66fafbeeaf` (`main`, confirmed with `gh api repos/kurtvalcorza/bart-cnn-summarization-pipeline/commits/main`)  
**Notebook Git blob:** `e0501a20fb54ee3bd0bfe1b7c654b2338f09b3d2`, the same blob as at `833efec` (the only later commit touching `tutorials/`, `src/` or `tools/` is `caab7b8`, a test-only change)  
**Finding prefix:** `BART`

## Executive assessment

As a reference pipeline the notebook is disciplined. It carries its three package modules byte for byte (generator `--check` exits 0), pins `facebook/bart-large-cnn` to an immutable revision and digest-verifies all 8 snapshot files before loading, fetches a real, digest-pinned and licensed out-of-domain corpus (SciTLDR-A), keeps the release's paper-disjoint partition and re-checks it, scores Lead-1, Lead-3, the frozen model and the adapted model on the same 100 test abstracts, selects the epoch on validation only, and reloads a safetensors adapter from files with a parity assertion. Its prose about ROUGE's length sensitivity, faithfulness and what the result does not establish is careful.

Five problems stand in the way of the Release-grade label and the `GUIDED` promise:

1. The one clean-runtime run of this blob (Kaggle T4) stopped at the install cell and needed a manual restart, and the repository records it as PASSED / Release-grade (BART-M1).
2. The inference contract's `truncated` / `stopped_by` flag, and therefore `hit_token_ceiling`, is wrong whenever the token cap is hit, because the snapshot forces an EOS token at the cap. The notebook tells learners to read `hit_token_ceiling` before trusting any score; it reports 0 while the frozen model's summaries are visibly cut mid-sentence at exactly the cap (BART-M2).
3. Every rerun the notebook prescribes (BYOD "re-run from that cell", the optional experiments) reuses the already-adapted pipeline. "Frozen" sections then score the adapted model, fine-tuning stacks on the previous run under a "frozen model" epoch-0 label, and the suggested `TRAINABLE_DECODER_LAYERS = 1` experiment exports an artifact that omits a layer the evaluated model still carries (BART-M3).
4. The documented BYOD minimum (8 records) is not the workflow's real minimum (50), and an over-long BYOD source passes Section 4 validation and fails later inside model execution without naming the record (BART-M4).
5. The notebook declares `GUIDED`, but most of the guided layer is missing and the learning objectives are mostly procedural (BART-M5).

The default-path adaptation and evaluation design is not shown to be wrong. The problems are the `Run all` contract, the correctness of one output-contract field the lesson leans on, the validity of every non-default run, and the BYOD and learning promises.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, and the opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`, `tutorials/README.md`, References) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main`. GDL1–GDL15 are `SHOULD`s; per spec §33 a 2.0/2.1 notebook does not become non-conformant merely for lacking them |
| Intended audience | Stated knowledge prerequisite: basic Python, seq2seq, beam search, hallucination, ROUGE. Notebook familiarity level is not stated |
| Supported runtime | "Google Colab or Jupyter, Python 3.12", CPU float32 by default, CUDA used when present |
| Promised outcomes | Pinned install; carried modules; staged, digest-verified snapshot; SciTLDR-A fetch, validation, four refusal probes, 300/50/100 disjoint split; inference contract with rejection probe and `truncated`/`stopped_by` semantics; Lead-1/Lead-3 and frozen ROUGE-1/2/L under TL;DR settings; bounded fine-tuning of the last two decoder blocks with validation-ROUGE-L selection; held-out four-way comparison; summaries of unseen abstracts; safetensors adapter export and reload parity; BYOD through every stage |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; carried modules from `src/bart_summarization_pipeline/` @ `386680b6` |

### Evidence actually obtained

- **Source inspection:** all 25 cells (11 code), the three carried modules (`pipeline.py` `from_pretrained` runner, `summarize`, `evaluate`, `adapt`, `save_artifact`, `load_artifact` read in full; `samples.py` and `metrics.py` read in full), the generator and template, `tutorials/README.md`, `docs/release-verification.md`, `STATUS.md`, the snapshot's `config.json` and `generation_config.json`.
- **Documented execution evidence:** Kaggle T4, 2026-09-19, commit `833efec` / blob `e0501a20`, **the same blob as the reviewed revision**. Outcome: 11/11 cells after "1 restart after install cell". The archived `run_summary.json` and `executed-pass1.ipynb` (workspace `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-bart-summarization/v2/evidence/`) record the pass-1 failure: `Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` The executed notebook's outputs (summaries, histories, comparison) were read and are cited below. A local Windows pre-flight (blob `afa4e964`, pins pre-installed) is labelled not-promotion evidence and is used only for the timing comparison in BART-m3.
- **Direct execution (this review):** `run_probes.py`, Windows, Python 3.12.14, torch 2.13.0+cpu, transformers 4.57.6, CPU. **No BART-large weights.** Static probes: notebook JSON parse and compile of all 11 code cells, blob id, generator `--check` (exit 0). Pure-Python probes of the carried `split_dataset` / `validate_dataset` exactly as cell 13 calls them. Model probes use a **tiny randomly initialised `BartForConditionalGeneration`** built from the snapshot's own `config.json` and `generation_config.json` (12 decoder layers kept, widths shrunk) and the snapshot's real tokenizer, staged with a manifest so the carried `from_pretrained`, `summarize`, `evaluate`, `adapt`, `save_artifact` and `from_artifact` run unmodified. The stand-in exercises the carried control flow and transformers' generation logic, not BART-large-CNN's numerics. The real tokenizer was also used to count the tokens in the Kaggle-recorded summaries.
- **Not verified:** any Colab run of this notebook; whether a fresh Colab kernel triggers the restart; the CPU default runtime on any hosted platform; any real BART-large forward pass in this review; BYOD end to end (upload dialog, real model); the optional experiments on the real model; learner understanding.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | Strong on the default path: immutable revision, per-file digests, no remote code, validation before model work, transactional `adapt`, digest-checked artifact loading with an exact tensor set. Defects: the install guard turns a fresh-runtime `Run all` into a two-pass run (BART-M1); `truncated` / `stopped_by` / `hit_token_ceiling` misreport capped generations (BART-M2); `adapt` and the "frozen" sections are not idempotent across prescribed reruns, and a changed `TRAINABLE_DECODER_LAYERS` exports an artifact that is not the evaluated model (BART-M3); the token ceiling is not checked at Section 4 validation (BART-M4). |
| Scientific / experimental validity | Good default design: source-disjoint release partition, validation-only selection, test used once per model, two meaningful baselines, `mean_summary_words` beside every score, explicit no-dispersion caveat. Weaknesses: the frozen model is scored under a 48-token cap that cuts its summaries mid-sentence while the notebook reports none were cut, so the frozen-versus-adapted gap partly measures the cap (BART-M2); every non-default run silently replaces the frozen reference (BART-M3); the Lead baselines are described as produced "under the TL;DR-length settings" although no cap applies to them (BART-m4); pretraining overlap of SciTLDR is not addressed (BART-m2). |
| Promise fulfilment | Every listed default stage ran in the documented Kaggle run. Not met: the `Run all` promise (BART-M1); the inference-contract promise that `truncated` reports a summary that hit `max_new_tokens` (BART-M2); the BYOD promise at the stated minimum and its "same cells" claim (BART-M4, BART-m6); "a few minutes on CPU" and the per-epoch timings, which no recorded run supports as stated (BART-m3). |
| Learner experience | Good "Look for" / "Expect" notes before most stages and strong interpretation prose. Missing: how-to-use guidance, roadmap, glossary, predictions, checkpoints with worked answers, a structured experiment with rerun instructions, troubleshooting, a conclusion scaffold; 56k characters of carried code are not labelled as infrastructure (BART-M5). Schema text shows doubled braces, including in the id regex (BART-m7). |
| Spec conformance | Unmet applicable `MUST`s: RUN1/RUN10/ENV6/REL2/REL11 and §27 release-grade marking (BART-M1); INF3/VAL7 (BART-M2); DAT13/DAT14/UX7 and VER5 for the prescribed experiment (BART-M3); DAT12/DAT19/VAL1/VAL6 and REL12 (BART-M4); UX1 (BART-M5); DAT9 (BART-m2); UX12 (BART-m3). `SHOULD` gaps: GDL1–GDL15, UX8, UX9, EXE1, EXE2, RUN9 (via result asserts). |

## 3. Promise and objective tracing

| Claim (opening / section text) | Implementation | Observable result | Learner interpretation | Holds? |
|---|---|---|---|---|
| "Run all … needs no … configuration edit" in a fresh runtime | Cell 3 install guard | Kaggle pass 1: `RuntimeError` at cell 3; pass 2 after restart: 11/11 | Learner must restart and rerun | **No** (BART-M1) |
| Snapshot staged and digest-verified, no fallback | Cell 11, `stage_missing_files`, `verify_snapshot`, `from_pretrained(local_files_only=True)` | Kaggle: 8 files fetched, `verified_files: 8`, `source: local-snapshot` | Clear | Yes |
| Corpus digest-pinned, split without leakage | Cell 13, `fetch_corpus`, `build_sample_dataset`, `check_split_disjoint` | Kaggle: 1,992/619/618 papers, 300/50/100, four refusals | Clear | Yes |
| `truncated` = the summary hit `max_new_tokens` | `runner` (`pipeline.py` 378), `summarize` (406) | Kaggle Section 5 TL;DR: 46 content tokens (= cap 48 − forced BOS − forced EOS), ends "will cost an estimated", reported `stopped_by: eos` | "not truncated" | **No** (BART-M2) |
| Read `hit_token_ceiling` before trusting any score | `evaluate` (440) | Kaggle frozen test: `hit_token_ceiling: 0`, `mean_summary_words: 35.0`; both printed frozen examples are exactly 46 tokens, one cut mid-sentence | "no summary was cut" | **No** (BART-M2) |
| Lead baselines and frozen/adapted models under the same TL;DR settings | Cells 17, 21 | Lead-3 at 68.3 words; Lead baselines take no generation settings | "all under the TL;DR-length settings" | Partly (BART-m4) |
| Bounded fine-tuning with validation selection | Cell 19, `adapt` | Kaggle: 33,593,344 trainable params, val ROUGE-L 24.19 → 31.03 → 31.93, `best_epoch` 2 | Clear | Yes (default path) |
| Held-out four-way comparison | Cell 21 | Kaggle: ROUGE-L 23.61 / 19.86 / 25.52 / 33.62 | Clear, with no-dispersion caveat | Yes (default path; frozen number affected by BART-M2) |
| Adapter reloads with parity | Cell 23, `save_artifact`, `from_artifact` | Kaggle: 52 tensors, 4/4 identical | Clear | Yes (default path); **No** after the suggested experiment (BART-M3) |
| BYOD passes "through the same … cells" | Cells 13, 23 | Probe P2: 8–49 records fail cell 13; P5: over-ceiling source fails in Section 6; Section 9 uses test records | — | **No** (BART-M4, BART-m6) |

| Learning objective (opening cell) | Learner activity | Evidence the objective is exercised |
|---|---|---|
| Install the pinned runtime; stage and digest-verify the snapshot | Run cells 3 and 11 | Printed versions and verified-file count; nothing to decide or interpret |
| Read what the carried modules guarantee | Scroll past three long module cells | No prompt, checkpoint or summary of the guarantees |
| Fetch, validate and split without leakage | Run cell 13; read four refusals | Refusal messages printed; no question asks the learner to explain them |
| Read `generated_tokens`, `truncated`, `stopped_by` correctly | Run cell 15 | The fields printed are wrong at the cap (BART-M2), so the objective teaches the wrong reading |
| Score frozen vs Lead and read why length drives ROUGE | Run cell 17; read `mean_summary_words` | Partly exercised by observation; no prediction or explanation asked |
| Run bounded fine-tuning; evaluate on an independent split | Run cells 19, 21 | Exercised on the default path |
| Summarise new abstracts; export and reload | Run cell 23 | Exercised on the default path |

## 4. Prioritized findings

### BART-M1 — Major: fresh-runtime `Run all` stops at the install cell and needs a manual restart, yet the notebook is registered Release-grade

**Cell/section:** Section 1, install cell (cell 3). Generated from `_INSTALL_GUARD` in `tools/build_notebook.py` (lines 47–69).

**Observed issue:** The guard records the distributions already imported, runs `pip install` of the pins (`numpy==2.5.3`, `torch==2.14.0`, …) and raises `RuntimeError(... 'Restart the runtime, then rerun from the top.')` if a loaded distribution changed. On the documented fresh Kaggle T4 image, NumPy 2.0.2 and cuda-bindings 12.9.4 were already loaded, so pass 1 failed at this cell after 205 s and the executor restarted. `docs/release-verification.md` step 4 accepts this ("an interpreter restart after the install is expected where the runtime's preinstalled torch or numpy differ from the pins"). `STATUS.md`, `tutorials/README.md` and the release record nevertheless mark the blob **Release-grade** on that run. No Colab run of this notebook exists, though `docs/release-verification.md` names Colab as "the runtime the tutorial is written for".

**Consequence:** A learner on a fresh hosted runtime that preloads NumPy gets an error on the first `Run all` and must restart and rerun by hand. Spec §5 says such a notebook "is not `Run all` conformant"; the release label overstates the evidence.

**Evidence:** Documented execution: `run_summary.json` passes[0] (`ok: false`, error quoted above, `restarted_after_install_cell: true`); "1 restart after install cell" in `docs/release-verification.md` (manual-evidence table, recorded-executions table, Current status) and `STATUS.md` line 3. Source inspection of cell 3. Colab behaviour: **not verified**.

**Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in the repository's notebook generator, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.

**Acceptance check:** A fresh Colab runtime and a fresh Kaggle image each run the regenerated notebook with **Run all** once, with no restart and no `RuntimeError` from cell 3, recorded with blob, runtime and outcome. Until then `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md` do not say Release-grade.

**Spec:** RUN1, RUN10, ENV6, REL2, REL11, §27 (MUST).

### BART-M2 — Major: `truncated`, `stopped_by` and `hit_token_ceiling` report "not cut" for every summary that hits the token cap

**Cell/section:** Section 5 (cell 15) inference contract and sanity checks; Section 6 (cell 17) `hit_token_ceiling`; every `evaluate` call. Carried `runner` in `BARTSummarizationPipeline.from_pretrained` (`src/bart_summarization_pipeline/pipeline.py` line 378) and `summarize` (line 406).

**Observed issue:** The runner sets `stopped_by = "eos" if tokenizer.eos_token_id in ids[1:] else "max_new_tokens"`, and `truncated = stopped_by != "eos"`. The pinned snapshot sets `forced_eos_token_id: 2` (`config.json` line 28, `generation_config.json`), so transformers forces EOS as the last token whenever generation reaches the cap. A capped summary therefore always contains EOS and is reported `stopped_by: eos`, `truncated: False`; `hit_token_ceiling` counts zero. The cell-15 sanity check `truncated_matches_stopped_by` cannot catch this: it compares two fields derived from the same test.

The lesson leans on this field. Section 6 tells the learner to "read `mean_summary_words` and `hit_token_ceiling` (summaries cut at `max_new_tokens`) before trusting any score". The `evaluation_report` "needs" text tells deployers to exclude or re-run outputs "whose truncated flag is true".

**Consequence:** Under the 48-token TL;DR cap the frozen model's summaries are cut, often mid-sentence, while the notebook reports that none were. The learner is steered to the wrong reading of the central comparison: part of the frozen model's gap to the adapted model is the cap cutting news-length summaries, not only register. The output contract promised in the opening and in `tutorials/README.md` ("`truncated`/`stopped_by` semantics") is wrong at the boundary it exists to report.

**Evidence:** Documented execution (Kaggle `executed.ipynb`): Section 5 TL;DR summary ends "…will cost an estimated", `generated_tokens: 46`, `stopped_by: eos`; Section 6 frozen test `hit_token_ceiling: 0`, `mean_summary_words: 35.0`; first printed frozen example ends "…this method improves the". Direct execution (probe P3): the snapshot's real tokenizer counts all three recorded texts at exactly **46 content tokens = 48 − forced BOS − forced EOS**, i.e. the cap. With the tiny stand-in model built from the snapshot's config, `summarize` at `max_new_tokens` 6 / 10 / 16 returns `generated_tokens` 4 / 8 / 14 with `stopped_by: eos`, `truncated: False` every time. How many of the 100 frozen test summaries hit the cap on the real model is **not verified**; the 35.0-word mean and the two printed examples suggest most did.

**Recommended correction:** Decide truncation from the length, not the presence of EOS: e.g. `truncated = n_generated_including_forced_tokens >= max_new_tokens` (or count content tokens against `max_new_tokens - 2` when forced BOS/EOS are configured), or disable `forced_eos_token_id` in the runner and keep the current test. Replace the tautological `truncated_matches_stopped_by` check with one that compares `generated_tokens` against the cap. Add a model-backed or stand-in unit test with `forced_eos_token_id` set. Then re-read the Section 6/8 prose: report how many frozen summaries were cut and say what that means for the comparison.

**Acceptance check:** With the snapshot config, a `summarize` call whose generation reaches `max_new_tokens` returns `truncated: True`, `stopped_by: "max_new_tokens"`; the probe-P3 stand-in reproduces this at caps 6/10/16; a regenerated run's Section 5 TL;DR result and Section 6 `hit_token_ceiling` agree with a token count of the printed summaries.

**Spec:** INF3, VAL7 (MUST); UNC1 (output semantics as actually provided).

### BART-M3 — Major: prescribed reruns score the adapted model as "frozen", stack a second fine-tuning, and the suggested experiment exports an artifact that is not the evaluated model

**Cell/section:** Opening BYOD instruction ("set `USE_BYOD = True` in Section 4 and re-run from that cell", template line 38); "Optional experiments" in Interpretation (template line 471: `TRAINABLE_DECODER_LAYERS = 1`, raise `TLDR_MAX_NEW_TOKENS`, `NUM_BEAMS = 1`, BYOD); Sections 6–9 (cells 17, 19, 21, 23); `adapt` and `save_artifact` (`pipeline.py` 462–611, 613–651).

**Observed issue:** `pipe` is built once, in Section 3. `adapt()` trains from the model's **current** weights; `initial_state` is used only to roll back on an exception, and nothing resets the decoder blocks at entry. It still writes `"note": "frozen model"` on epoch 0. After the default run, rerunning as instructed gives:
- Section 6 "the frozen model's score on the test split" scores the adapted model (only its `adapted: True` field says so), so Section 8's comparison and `evaluation_report.json` call adapted-versus-re-adapted "frozen versus adapted";
- Section 7 fine-tunes on top of the previous adaptation under the "frozen model" epoch-0 label;
- with `TRAINABLE_DECODER_LAYERS = 1`, layer 10 keeps the first run's trained weights in `pipe`, but `save_artifact` writes only the tensors of the latest `trainable_names` (layer 11). The exported artifact, reloaded onto a fresh base, lacks the adapted layer 10 that produced the evaluated scores. The 4-summary parity assertion may or may not notice, so "compare the artifact size and the test scores" compares a size and scores that belong to different models.

**Consequence:** A learner who follows the BYOD instruction or any suggested experiment gets "frozen" and "adapted" numbers that do not mean what the headings and the exported report say, and may get an artifact that does not reproduce the evaluated model. The conclusion the notebook teaches ("what a bounded adaptation adds over the frozen model") becomes invalid for every non-default run, BYOD included.

**Evidence:** Source inspection of `adapt` (no reset; `initial_state` appears only in the `except` path) and `save_artifact` (`names = set(self.adapter["trainable_names"])`). Direct execution with the tiny stand-in (probe P4, carried code unmodified): the default Section 6 evaluation reports `adapted: False`, the rerun reports `adapted: True`; the second `adapt` again labels epoch 0 "frozen model"; after the default 2-layer run followed by a 1-layer run, `pipe`'s layer 10 equals the first run's trained weights, the artifact holds only layer `11`, the reloaded model's layer 10 equals the base, and the logits differ (max abs 0.0057), yet **4/4 summaries were identical** on the stand-in, so the parity assertion passed while the models differed. Real-model behaviour is inferred from the same control flow, **not executed**. (The stand-in runs used no validation split, because a random model scores ROUGE 0 and selection would keep epoch 0; the state mechanics do not depend on selection.)

**Recommended correction:** Make each pass start from the frozen base: have Section 4 (or the start of Section 6) rebuild `pipe` with `BARTSummarizationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` from the already-verified files, or add a `reset_adapter()` that restores the base tensors and clears `self.adapter`, and call it before Sections 6 and 7. Alternatively, have `adapt()` refuse to run when `self.adapter is not None`. Make `save_artifact` refuse, or include every tensor that differs from the base, when the model's adapted set and `trainable_names` disagree. Tell learners which section to rerun from for each experiment. Generator locations: the BYOD and experiments text in `tools/notebook_template.py` lines 38 and 471; the cells in the same file.

**Acceptance check:** On a stand-in or the real model, running the default path and then rerunning from Section 4 as instructed gives Section 6 `adapted: False` and a frozen test score equal to the first pass; the second `adapt` starts from base weights; after a `TRAINABLE_DECODER_LAYERS = 1` rerun, the reloaded artifact's state dict equals `pipe`'s on every decoder tensor.

**Spec:** DAT13, DAT14, UX7, VER5 (MUST); GDL10, VER4 (SHOULD).

### BART-M4 — Major: the documented BYOD minimum fails Section 4, and an over-long BYOD source passes validation and fails inside model execution

**Cell/section:** Prerequisites "a dataset needs 8..20,000 records" (template line 122) and Section 4 (cell 13); `split_dataset` (`samples.py`), `validate_dataset`; `summarize` token check (`pipeline.py` `_check_input_tokens`).

**Observed issue:** Cell 13 calls `split_dataset` (test 20 %, validation 15 %, train ≥ 8) and then `validate_dataset(part)` on **each split**, which requires at least 8 records per split. Any BYOD file of 8–49 records passes the stated contract and `split_dataset`, then fails in cell 13 with a message such as `7 records; 8..20000 are required` that does not say which split or that the whole dataset needs at least 50 records. Separately, Section 4 validation checks the 40,000-character ceiling but not the 1,024-token ceiling. A BYOD source between about 4,000 and 40,000 characters passes Section 4; in the training split it is silently truncated to 512 tokens (documented), but in the validation or test split it fails inside `pipe.evaluate` during Section 6 or Section 7 epoch 0, after the Lead baselines and possibly minutes of generation, with `input is 1325 tokens; ceiling is MAX_INPUT_TOKENS=1024` and no record id.

**Consequence:** A learner who follows the stated contract with a small dataset is rejected with a misleading message; one with realistic reports fails mid-run without being told which record to fix. The BYOD promise ("the same validation … cells", "the expected schema and the ceilings are stated") is not met at its own boundaries.

**Evidence:** Direct execution (probe P2, carried code): N = 8, 20, 37, 38, 49 all pass `validate_dataset(records)` and `split_dataset`, then fail cell 13's per-split validation (e.g. N = 49 → splits 10/7/32, `7 records; 8..20000 are required`); N = 50 passes. Probe P5 (stand-in model, real tokenizer): a 6,719-character source of 1,325 tokens passes Section 4 validation and the split, lands in the test split, and `pipe.evaluate` on that split raises the error above; `error_names_record_id: false`. BYOD through the upload widget and the real model: **not verified**.

**Recommended correction:** State the real minimum (50 at the default fractions) or validate splits with `min_records=1` for validation/test while keeping the training minimum, and name the split in the message. Count tokens for every BYOD source in Section 4 (the tokenizer is already loaded in Section 3) and reject over-ceiling validation/test sources there, naming the id, or report and drop them (VAL7). Generator: cell 13 in `tools/notebook_template.py` (around line 153) and the Prerequisites text (line 122).

**Acceptance check:** A 20-record BYOD file either runs through Section 4 or is rejected in Section 4 with a message naming the minimum and the split; a BYOD file containing one 1,300-token record in the test split is rejected in Section 4 with its id, before any model call.

**Spec:** DAT12, DAT19, VAL1, VAL6 (MUST); REL12.

### BART-M5 — Major: declared `GUIDED`, but most of the guided layer is absent and the objectives are procedural

**Cell/section:** Whole notebook; opening "Learning objectives"; Interpretation's one-line "Optional experiments".

**Observed issue:** The opening's objectives are mostly infrastructure steps ("install the pinned runtime; read what the carried … modules guarantee; stage and digest-verify…") rather than observable learning outcomes. There is no **How to use this notebook**, roadmap, Input → Model → Output contract, glossary (beam search, length penalty, `no_repeat_ngram_size`, teacher forcing and longest common subsequence are used without being explained in the notebook; the Prerequisites assume them), prediction before a principal result, checkpoint with a worked answer, troubleshooting section, conclusion scaffold, or section-end synthesis. The optional experiments are a single sentence with no prediction, rerun instructions or interpretation guidance, and following them gives invalid results (BART-M3). Three module cells (56k characters) appear between Sections 1 and 3 with no **Infrastructure** label or collapse, so they read as required study.

**Consequence:** A self-paced learner can run the notebook and read its outputs but is not asked to predict, change, explain or conclude anything. Several stated objectives ("read why summary length drives ROUGE", "read `truncated` correctly") are exercised only by passive reading, and one teaches a wrong reading (BART-M2).

**Evidence:** Source inspection of all 14 markdown cells. Strengths noted: "Look for" / "Expect" notes before Sections 1, 4, 6, 7, 8 (GDL8 in part) and a careful limits section.

**Recommended correction:** Add the guided layer following the spec §25.13 reference notebook: observable objectives, how-to-use, roadmap, glossary, a prediction before Sections 6 and 8, collapsible "Check your reasoning" answers, one **Predict → Change one thing → Run → Observe → Explain** activity with exact rerun instructions (after BART-M3 is fixed, e.g. `TLDR_MAX_NEW_TOKENS`), troubleshooting (restart, download, memory, BYOD rejections), and a conclusion template. Title module and install cells `Infrastructure: …` and collapse them (`cellView: form`). Generator: `tools/notebook_template.py`.

**Acceptance check:** The regenerated notebook contains each GDL1–GDL15 element (checkable by a validator rule or reviewer checklist), its objectives use observable verbs and each maps to an executed cell, and at least one activity names the change, the cells to rerun and what to compare.

**Spec:** UX1 (MUST); UX5, UX8, UX9, GDL1–GDL15 (SHOULD).

### BART-m1 — Minor: result-dependent asserts and fixed-result prose turn a legitimate negative result into a crash before export

**Cell/section:** Cell 21 `assert adapted_test['rougeL'] > frozen_test['rougeL']` (template line 368), before the artifact export in cell 23; cell 17 `assert frozen_test['rouge1'] > 0.0`; Sections 7, 8 and Interpretation prose ("Watch validation ROUGE-L rise by several points", "Look for a ROUGE-L gain of several points", "lifts held-out ROUGE-L by several points").

**Observed issue:** On BYOD data or an optional experiment, adaptation may legitimately not beat the frozen model. The assert then stops the notebook before the summaries, artifact, reload and `result.json`, and the prose has already told the learner what to see.

**Consequence:** The learner loses the outputs that would let them diagnose a negative result and is pushed to treat it as a failure.

**Evidence:** Source inspection; probe P6 lists the three asserts. Not executed with a negative result.

**Recommended correction:** Keep the assert only for the pinned default sample (e.g. `if not USE_BYOD`), or replace it with a printed verdict; phrase "Look for" notes as what to compare rather than the expected winner.

**Acceptance check:** With BYOD data where adaptation does not help, the notebook completes through Section 9 and prints that the adapted model did not improve.

**Spec:** RUN9 (MUST for BYOD/optional branches); GDL8, GDL14 (SHOULD).

### BART-m2 — Minor: pretraining overlap of the SciTLDR sample is not addressed

**Cell/section:** Opening cell and Section 4.

**Observed issue:** BART was pretrained on large web, news and book corpora; SciTLDR abstracts are public OpenReview/arXiv text. The notebook calls the corpus "out of domain" but does not say whether the abstracts may have been seen in pretraining.

**Consequence:** Learners may read the frozen and adapted scores as unaffected by memorisation.

**Evidence:** Source inspection (no mention of pretraining overlap in any cell).

**Recommended correction:** One sentence: overlap with BART's pretraining text cannot be ruled out; the comparison is between models with the same exposure.

**Acceptance check:** The opening or Section 4 states the limitation.

**Spec:** DAT9 (MUST).

### BART-m3 — Minor: runtime claims do not name their environment and disagree with the recorded runs

**Cell/section:** Opening ("about twelve minutes of model time on CPU"), Prerequisites ("the build record measured … about 40 s per training epoch"), Section 7 ("about a minute of training plus a validation pass per epoch on CPU"), Interpretation ("in a few minutes on CPU").

**Observed issue:** The numbers come from the local Windows pre-flight (746.4 s total, `adapt` 353.1 s for two epochs plus three validation passes, about 176 s per epoch) but are presented as CPU expectations without naming the environment; "40 s per epoch" and "about a minute" do not match that record. The only hosted run was on a T4 GPU (`adapt` 105.4 s).

**Consequence:** A learner on a hosted CPU runtime cannot judge whether the run is stuck.

**Evidence:** Source inspection; `docs/release-verification.md` recorded-executions table; Kaggle `run_summary.json`.

**Recommended correction:** Label each figure with its environment (e.g. "local Windows CPU pre-flight"), give the T4 figure, and mark untested hosted-CPU times as estimates.

**Acceptance check:** Every runtime figure in the notebook names its environment or is labelled an estimate, and agrees with a recorded run.

**Spec:** UX12 (MUST).

### BART-m4 — Minor: Lead baselines described as produced under the same TL;DR settings

**Cell/section:** Section 6 markdown ("Three numbers frame the adaptation, all under the TL;DR-length settings of Section 5", template line 265); `tutorials/README.md` ("applied identically to the Lead baselines, the frozen model and the adapted model").

**Observed issue:** Lead baselines take no generation settings; Lead-3 averages 68.3 words against a 48-token cap on the models.

**Consequence:** Learners may read the four-way comparison as length-matched when it is not, which is the very confound the section warns about.

**Evidence:** Source inspection of `lead_baseline` (`metrics.py`); Kaggle comparison `mean_summary_words` {lead1 20.18, lead3 68.29, frozen 35, adapted 19.55}.

**Recommended correction:** Say the cap applies to the models only, and point to `mean_summary_words` as the length control for the baselines.

**Acceptance check:** No learner-facing text says the TL;DR settings apply to the Lead baselines.

**Spec:** EVAL10 (context); framework dimension 3.

### BART-m5 — Minor: BYOD intake is upload-only, has no location field, and does not report de-duplication

**Cell/section:** Section 4 (cell 13), BYOD branch.

**Observed issue:** BYOD always imports `google.colab` and opens `files.upload()`, though Jupyter is a stated runtime; there is no path field. An empty upload raises a bare `StopIteration`. `split_dataset` silently drops duplicate-source records without reporting how many.

**Consequence:** BYOD is unusable outside Colab and cannot be driven by an executor; a cancelled upload gives no recovery message; a learner does not learn their dataset shrank.

**Evidence:** Source inspection of cell 13 and `split_dataset`.

**Recommended correction:** Add `BYOD_PATH = ''  # @param` read when set, guard the empty upload with an actionable message, and print the number of de-duplicated records.

**Acceptance check:** With `BYOD_PATH` set, cell 13 reads the file without importing `google.colab`; an empty upload prints a recovery instruction; the cell prints `dropped_duplicate_sources`.

**Spec:** DAT16, VAL7 (MUST for reporting dropped rows); EXE1, EXE2, UX10 (SHOULD).

### BART-m6 — Minor: under BYOD, Section 9's "new" documents are the first four test records

**Cell/section:** Section 9 (cell 23, template line 392) and its markdown (template line 375: "Four abstracts that were in none of the splits").

**Observed issue:** With `USE_BYOD = True`, `new_records = test_records[:4]`; the reload-parity set is also `test_records[:4]`. The heading still says they were in none of the splits.

**Consequence:** A BYOD learner is told the inference demonstration is on unseen data when it reuses scored test data.

**Evidence:** Source inspection.

**Recommended correction:** Hold four BYOD records out of the split for Section 9, or label them honestly as test records.

**Acceptance check:** Under BYOD, Section 9's text and its `sample_kind` agree with where the records came from.

**Spec:** INF2, INF3 (context).

### BART-m7 — Minor: schema text shows doubled braces, including a wrong id regex

**Cell/section:** Opening BYOD paragraph and Prerequisites data contract (template lines 38–40, 122).

**Observed issue:** The markdown shows `{{id, source, targets}}` and the id pattern `[A-Za-z0-9_.:-]{{1,64}}`; the template escapes braces for formatting that markdown cells never receive. The printed regex means something different from the enforced `{1,64}`.

**Consequence:** A learner copying the documented schema or regex gets a wrong pattern.

**Evidence:** Probe P1: markdown cells 0 and 1 contain `{{`; `_ID_RE` in `samples.py` is `^[A-Za-z0-9_.:-]{1,64}$`.

**Recommended correction:** Use single braces in markdown template strings that are not `.format()`ed, and add a validator check for `{{` in markdown cells.

**Acceptance check:** No markdown cell of the regenerated notebook contains `{{` or `}}`.

**Spec:** DAT12 (MUST: schema stated before upload).

### Suggestions

- **BART-S1:** Update the declared `notebook_spec` and the in-notebook "NOTEBOOK_SPEC 2.0" references to 2.2 when the notebook is next regenerated.
- **BART-S2:** Note CPU versus CUDA metric drift: `docs/release-verification.md` step 5 quotes the local CPU numbers (adapted ROUGE-L ≈ 34.28) as "on the sample", while the T4 run gave 33.62. A sentence in Section 8 would stop learners treating a 0.7-point difference as a defect (ENV8 context).
- **BART-S3:** Strengthen reload verification: compare the reloaded decoder tensors (or logits on a fixed input) with `pipe`'s, not only 4 beam-search summaries; probe P4 shows summary parity can pass with different weights.
- **BART-S4:** Print a bootstrap interval over the 100 test documents for the frozen-versus-adapted ROUGE-L delta; it is cheap and makes the no-dispersion caveat concrete.

## 5. Readiness

**Needs revision.** Five Major findings are open. Applicable `MUST`s unmet: RUN1/RUN10/ENV6/REL2/REL11 and §27 (BART-M1), INF3/VAL7 (BART-M2), DAT13/DAT14/UX7/VER5 (BART-M3), DAT12/DAT19/VAL1/VAL6 (BART-M4), UX1 (BART-M5), DAT9 (BART-m2), UX12 (BART-m3), DAT16/VAL7 (BART-m5), DAT12 (BART-m7).

Remaining gates after fixes: a one-pass clean-runtime run on Colab and on a fresh Kaggle image, recorded against the new blob; a BYOD positive and negative run through Section 9 on a hosted runtime (REL12); a rerun of an optional experiment showing a frozen baseline equal to the first pass.

## 6. Verified versus inferred

- **Verified by direct execution (CPU, this review):** notebook parse and compile (11/11 code cells), blob `e0501a20`, generator `--check` exit 0; BYOD minimum behaviour (P2); forced-EOS mislabelling with the snapshot's real config and tokenizer on a tiny stand-in model, and the 46-token count of the three recorded Kaggle summaries (P3); rerun state, artifact tensor set and parity behaviour on the stand-in (P4); over-ceiling BYOD failure location and message (P5); result-dependent asserts (P6).
- **Verified from documented execution:** the Kaggle T4 restart and pass-2 outputs for this exact blob.
- **Inferred, not executed:** real-model behaviour for BART-M2 (how many of the 100 frozen summaries hit the cap), BART-M3 (whether real-model parity would catch the missing layer), and BART-M4 through the upload widget; any Colab behaviour; learner understanding.
- **Finding most likely to be wrong:** BART-M2's practical scale. The mechanism is certain (forced EOS at the cap is reported as `eos`), and all three recorded summaries sit exactly at the cap, but the share of the 100 frozen test summaries that were cut is inferred from a 35.0-word mean and two printed examples, not counted.

Probe ZIP: `bart_summarization_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).

"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone carrier, isolated environment).

Only the task-specific prose and the learner cells live here. The runtime check, the carrier of the package / stage
runner / lock / manifest, the isolated install and the weight staging are produced by the generator from repository
sources, so they cannot drift from the package. Every learner cell runs one stage of ``tools/tutorial_stages.py`` with
``run_stage`` and prints what that stage wrote.

This template configures an E2E summarization workflow: the pinned BART-large CNN snapshot is digest-verified; a
digest-pinned real out-of-domain corpus (SciTLDR-A, one-sentence TL;DRs of paper abstracts) is fetched, validated
against the workflow and split; one synthetic document is summarised through the inference contract; the frozen model
is scored with ROUGE against the Lead-1 and Lead-3 baselines under TL;DR-length settings; a bounded fine-tuning of the
last decoder blocks runs; the held-out split is scored again with the adapter rebuilt from its files; new documents are
summarised; and the adapter is reloaded in a new process with decoder-tensor parity.

Markdown passed through ``str.format`` (prerequisites, setup, cells, guided opening, closing) writes literal braces
doubled; the header strings (run_all, byod, intro, learning_objectives, exclusions) are not formatted and use single
braces (BART-m7).
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "bart-cnn-summarization-pipeline"
COLAB_URL = f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/bart_summarization_colab.ipynb"

BADGES = [
    ("GitHub", "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white", f"https://github.com/kurtvalcorza/{REPO}"),
    ("Open In Colab", "https://colab.research.google.com/assets/colab-badge.svg", COLAB_URL),
    ("Hugging Face", "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fbart--large--cnn-ffcc4d?style=flat", "https://huggingface.co/facebook/bart-large-cnn"),
    ("Upstream", "https://img.shields.io/badge/Upstream-facebookresearch%2Ffairseq-181717?style=flat&logo=github&logoColor=white", "https://github.com/facebookresearch/fairseq/tree/main/examples/bart"),
    ("arXiv", "https://img.shields.io/badge/arXiv-1910.13461-b31b1b.svg", "https://arxiv.org/abs/1910.13461"),
]

TEMPLATE = {
    "package": "bart_summarization_pipeline",
    "repo_name": REPO,
    "stem": "bart_summarization",
    "notebook_name": "bart_summarization_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "weights_key": "bart-large-cnn",
    "carried": {
        "src/bart_summarization_pipeline/__init__.py": "src/bart_summarization_pipeline/__init__.py",
        "src/bart_summarization_pipeline/pipeline.py": "src/bart_summarization_pipeline/pipeline.py",
        "src/bart_summarization_pipeline/samples.py": "src/bart_summarization_pipeline/samples.py",
        "src/bart_summarization_pipeline/metrics.py": "src/bart_summarization_pipeline/metrics.py",
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-colab.lock.txt",
        "weights/bart-large-cnn/dimer-base-manifest.json": "weights/bart-large-cnn/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    },
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "disk_gib": {"weights": 2.0, "environment": 10},
    "runtime_modules": ["torch", "transformers"],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime checks the runtime, writes and hash-verifies the carried files, "
        "builds an isolated hash-locked Python environment (nothing is installed into the notebook kernel, so no restart is "
        "needed), stages and digest-verifies the pinned BART-large CNN snapshot (safetensors, 1.6 GB), fetches the three "
        "digest-pinned SciTLDR-A files from the project repository (5.5 MB, no credential), draws 300 / 50 / 100 training, "
        "validation and test abstracts from the release's own paper-disjoint members and checks them against the workflow, "
        "summarises one synthetic news-style document through the inference contract, scores the Lead-1 and Lead-3 baselines "
        "and the frozen model on the test abstracts with ROUGE-1/2/L, runs a bounded fine-tuning of the last two decoder "
        "blocks with validation-ROUGE-L epoch selection, scores the held-out split again with the adapter rebuilt from its "
        "files, summarises four abstracts that are in none of the splits, reloads the adapter in another fresh process to "
        "verify that every decoder tensor and the summaries are reproduced, and writes a provenance record. Every stage that "
        "runs the model builds it from the verified files in its own process. The default path needs no repository clone, "
        "DIMER worker or service, credential, upload dialog, or configuration edit (NOTEBOOK_SPEC 2.2 §5, RUN1, RUN10)."
    ),
    "byod": (
        "`USE_BYOD = True` in Section 5 replaces the SciTLDR sample with your own document–summary pairs as a CSV (columns "
        "`id`, `source`, `target`), a JSON array, or a JSONL file of `{id, source, target}` or `{id, source, targets: [...]}` "
        "records. Put the file's path in `BYOD_PATH` (any runtime), or leave it empty for the Google Colab upload dialog. "
        "Section 5 de-duplicates sources (and reports how many it dropped), splits by a seeded source-disjoint shuffle, and "
        "refuses before any model runs a dataset this workflow cannot complete: fewer than **12 records with distinct "
        "sources** (the smallest number whose split leaves 8 training records and a validation record), or a validation or "
        "test source over the 1,024-token ceiling, naming the record. Your pairs then pass through the same baselines, "
        "fine-tuning, held-out evaluation, inference, export and reload stages as the sample; Section 10 summarises four of "
        "your test records and says so. Files stay in this runtime."
    ),
    "title": "BART-large CNN — DIMER E2E abstractive summarization fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "abstractive summarization of one English document by deterministic beam search (pinned generation defaults; token counts and a truncation flag reported; no score) and bounded supervised fine-tuning of the last decoder blocks on a referenced summarization corpus, using the pinned `facebook/bart-large-cnn` weights",
    "intro": (
        "At inference the byte-level BPE tokenizer encodes the whole document once, the 12-layer bidirectional encoder of "
        "the 406 M-parameter BART-large reads it, and the 12-layer autoregressive decoder — fine-tuned upstream on "
        "CNN/DailyMail article/highlight pairs — writes a summary token by token under **deterministic beam search**: "
        "`num_beams` 4, `length_penalty` 2.0, `no_repeat_ngram_size` 3, `early_stopping`, and the length bounds of the "
        "snapshot's `generation_config_for_summarization.json` (`max_length` 142 / `min_length` 56, exposed as "
        "`DEFAULT_MAX_NEW_TOKENS` 141 / `DEFAULT_MIN_NEW_TOKENS` 55 because the upstream bounds count the decoder start "
        "token). What the upstream checkpoint supplies is the encoder-decoder, the language-model head, the generation "
        "config and the tokenizer; what the carried pipeline module adds is manifest verification, input and "
        "generation-setting validation against named ceilings (over-long documents are rejected, not truncated or chunked), "
        "a fixed output contract that reports `generated_tokens`, `input_tokens`, `truncated` and `stopped_by`, and the "
        "`validate_inputs` and `evaluation_report` stage helpers. **The pipeline emits no score, probability or quality "
        "metric** — a summary is free text.\n\n"
        "What this notebook adds to inference is **adaptation with references**. The dataset is real and deliberately "
        "**out of domain**: SciTLDR-A (Cachola et al., 2020; Apache-2.0), one-sentence TL;DR summaries of computer-science "
        "paper abstracts — three digest-pinned JSON-Lines files fetched from the project repository at a pinned commit. A "
        "news summariser writes three-sentence, largely extractive highlights here; the fine-tuning question is whether a "
        "bounded adaptation of the last two decoder blocks moves it to the short TL;DR register on held-out papers. Three "
        "metrics are implemented in the carried `metrics.py` (corpus **ROUGE-1/2/L** F1, best over the references, "
        "rouge-score-style, not rouge-score-identical), and two **Lead baselines** — the first sentence and the first three "
        "sentences of the abstract as the summary — show where a system that does no modelling sits. Nothing here is a "
        "quality claim about your documents: it is one seeded split of one corpus.\n\n"
        "**Length note:** the frozen model's pinned news defaults force at least 55 new tokens, three times a TL;DR; every "
        "model ROUGE number in this notebook is therefore produced under explicit TL;DR-length settings (`max_new_tokens` "
        "48, `min_new_tokens` 0, `num_beams` 4) applied identically to the frozen and the adapted model. The Lead baselines "
        "copy sentences and take no generation settings, so no cap applies to them: `mean_summary_words`, printed beside "
        "every score, is the length control for them. A 48-token cap also **cuts** summaries that would have run longer, and "
        "the notebook counts how many were cut (`truncated`, `hit_token_ceiling`) before any score is read."
    ),
    "learning_objectives": (
        "after this notebook you should be able to (1) describe what BART-large CNN takes as input and how deterministic beam "
        "search turns it into a summary, and read the pinned generation defaults (Sections 4 and 6); (2) read "
        "`generated_tokens`, `truncated` and `stopped_by` and explain why a summary that reached `max_new_tokens` is cut even "
        "though it ends with an end token (Section 6); (3) explain, from the Lead-1 and Lead-3 baselines and "
        "`mean_summary_words`, why ROUGE rewards length matching and how many frozen summaries the cap cut (Section 7); "
        "(4) explain why the split must be source-disjoint and why the validation split, not the test split, selects the "
        "epoch (Sections 5 and 8); (5) run a bounded fine-tuning and judge it on the held-out split against two baselines and "
        "the frozen model (Sections 8 and 9); (6) export the adapter, rebuild the model from files in a new process and "
        "verify parity (Section 11); and (7) change one variable — the summary-length cap — and explain how much of the "
        "frozen-versus-adapted gap it accounts for (Section 13)."
    ),
    "exclusions": (
        "extractive summarization with guaranteed source spans, multi-document or long-document (chunked) summarization, "
        "headline generation, sampling-based or diverse decoding, full-model or encoder fine-tuning, classification or "
        "question answering (the `bart-mnli-zero-shot-classification-pipeline` sibling covers zero-shot classification), "
        "non-English text, any faithfulness or factuality score, and any claim that a SciTLDR split stands in for your "
        "documents. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh Linux x86_64 runtime — Google Colab, Kaggle, or a Linux Jupyter server — with about 12 GB of free disk and about 6 GB of RAM. A GPU is used automatically when present (the default runtime type is a T4) but is not required: every stage also runs on CPU, in float32 on both. The stages run in CPython 3.12.12 inside the isolated environment, whatever Python the notebook kernel itself uses (Colab's kernel is 3.13). Windows and macOS kernels are not supported, because the hash-locked environment is built for manylinux x86_64.",
        "- **Time (measured where stated, otherwise an estimate):** building the isolated environment and downloading about 1.6 GB of weights take a few minutes on a hosted runtime — an estimate that depends on the network. On the 2026-09-19 Kaggle T4 run of the previous, in-kernel version of this notebook the whole path took 518.6 s including downloads, of which the fine-tuning took 105.4 s. On a Windows CPU workstation (a local pre-flight of the previous version, not a supported runtime) scoring the 100 test abstracts with the frozen model took 178 s and the two-epoch fine-tuning 353 s; a local pre-flight of this version on the same kind of workstation, shared with other jobs, took several times longer and is not a timing reference. A hosted CPU runtime is usually slower than an unloaded workstation, so expect tens of minutes on CPU (an estimate); a T4 GPU is much faster.",
        "- **Knowledge:** basic Python; what an encoder-decoder (seq2seq) model is; why an abstractive summary can state things the source does not (hallucination). Beam search, the length settings, teacher forcing, ROUGE and the longest common subsequence are explained where they are first used; the glossary collects them.",
        "- **Data contract:** records are `{{id, source, targets}}` — a document and one or more reference summaries (`{{id, source, target}}` with a single string is accepted and normalised); the source 1..40,000 characters, each reference 1..2,000 characters, ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique. The package accepts 8..20,000 records, but **this workflow needs at least 12 records with distinct sources**: sources are de-duplicated case-insensitively before splitting (so the same document never sits in two splits), and the 15 % / 20 % validation / test split must leave 8 training records and one validation record. Every validation and test source must fit the 1,024-token encoder ceiling (inference never truncates — it rejects), and Section 5 checks that before any model runs; during training only, sources are truncated to 512 and targets to 64 BPE tokens. BYOD accepts CSV, JSON or JSONL in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a reference is a faithful summary of its source or that a source is prose — a mislabelled corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal report archive with its executive summaries is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub and PyPI, the default path fetches three pinned objects (`train.jsonl` 3,155,015 bytes, `dev.jsonl` 1,124,865 bytes, `test.jsonl` 1,204,107 bytes; SHA-256 `b222771d…` / `3191fa98…` / `fb42dd6c…`) from `raw.githubusercontent.com` at the pinned `allenai/scitldr` commit over HTTPS, each refused on any mismatch before it is read; SciTLDR is Apache-2.0 (Cachola et al., 2020).",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook and read short Python, and who "
                "want to see how a pretrained summariser is evaluated against references and baselines and adapted to a new "
                "kind of summary. You should know what an encoder-decoder model is; every summarization term after that is "
                "explained where it is first needed, and the glossary below collects them.\n\n"
                "**Running it.** In Colab, choose *Runtime → Run all*. The default path needs no edit, no upload, no account, "
                "no token and no runtime restart. Sections 1–3 build an isolated environment from hash-locked packages and "
                "download about 1.6 GB of verified weights, so they take the longest before any model runs; read ahead while "
                "they finish, or run one cell at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files in the run directory — the verified snapshot, the "
                "corpus cache, the splits, the generation settings, the adapter and JSON records. **Every stage that runs the "
                "model builds it from the verified files**, so Sections 6 and 7 always show the frozen model and Section 8 "
                "always starts from the pinned base weights, however often you re-run them.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–13) are the summarization workflow; each runs one stage and "
                "prints compact dictionaries for you to read. *Infrastructure cells* (Sections 1–3: the runtime check, the "
                "carried code, the isolated install and the pinned-weight staging) are collapsed and titled **Infrastructure**. "
                "You may run them without studying their implementation: they exist for reproducibility and provenance, not as "
                "prerequisite machine-learning knowledge.\n\n"
                "**Form controls.** Some learner cells start with fields that Colab renders as a form: `USE_BYOD`, `BYOD_PATH` "
                "and `SPLIT_SEED` (Section 5); `TLDR_MAX_NEW_TOKENS`, `TLDR_MIN_NEW_TOKENS` and `NUM_BEAMS` (Section 6); "
                "`EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE` and `TRAINABLE_DECODER_LAYERS` (Section 8); and `RUN_ACTIVITY` and "
                "`ACTIVITY_MAX_NEW_TOKENS` in the optional activity (Section 13). Leave them at their defaults for the first "
                "run: the notes and sample answers describe the default path.\n\n"
                "**Changing a setting.** After changing a Section 5 field (data), re-run from Section 5 to the end. After "
                "changing a Section 6 field (generation settings), re-run from Section 6 to the end. After changing a Section 8 "
                "field (training), re-run from Section 8 to the end. A re-run section clears every result computed from its old "
                "inputs, and a later section that needs a cleared result says which section to run. Nothing else needs to be "
                "repeated, and the frozen model is never replaced by an earlier adaptation.\n\n"
                "**Section tags.** Each numbered heading carries one tag. **[Concept]** — what the model does and why. "
                "**[Evaluation practice]** — how the evidence is produced and how to read it. **[Engineering]** — "
                "reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit "
                "to an expectation; after it, **What to notice** describes normal output, and a collapsed **Check your "
                "reasoning** answer follows each checkpoint. Write your own answer first, then open it. Exact numbers can vary "
                "between CPU and GPU and between library builds (a few tenths of a ROUGE point, and occasionally a different "
                "word in a summary), so the notes describe the shape of a normal result rather than fixed values."
            ),
            (
                "## The task: Input → Model/System → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Summarise** | one English document (at most 1,024 BPE tokens) and generation settings | BART-large CNN: bidirectional encoder → autoregressive decoder under deterministic beam search | a summary with `generated_tokens`, `input_tokens`, `truncated` and `stopped_by`; no score |\n"
                "| **Baselines and evaluation** | the summaries and the reference TL;DRs of the test abstracts | Lead-1 and Lead-3 (copy the first sentences); corpus ROUGE-1/2/L F1, best over the references | a table of scores with summary lengths and the number of summaries cut at the cap |\n"
                "| **Adaptation** | 300 training abstracts with one reference each; 50 validation abstracts | the last two decoder blocks trained with teacher-forced cross-entropy; the epoch with the best validation ROUGE-L kept | a 134 MB adapter file holding only those blocks |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux and disk checked; a fresh run directory | the GPU (or CPU) |\n"
                "| 2. Carry the code, install the runtime | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage and verify | [Engineering] | snapshot downloaded and digest-checked | the verified files |\n"
                "| 4. Confirm the runtime | [Engineering] | versions checked against the lock | device, ceilings, pinned defaults |\n"
                "| 5. Corpus, validation and split | [Evaluation practice] | three pinned files fetched, validated, split, checked against the workflow | splits, refusals |\n"
                "| 6. The inference contract | [Concept] | one document summarised under the news defaults and the TL;DR settings | `truncated`, `stopped_by` |\n"
                "| 7. Baselines and the frozen model | [Evaluation practice] | Lead-1, Lead-3 and the frozen model scored on the test split | the score to beat; summaries cut |\n"
                "| 8. Bounded fine-tuning | [Concept] | two epochs on the last two decoder blocks; adapter exported | the epoch history |\n"
                "| 9. Held-out evaluation | [Evaluation practice] | the adapter rebuilt from files and scored on the test split | the principal result |\n"
                "| 10. New documents | [Concept] | four documents summarised by the adapted model | the summaries |\n"
                "| 11. Fresh reload and parity | [Engineering] | another fresh process reproduces the trained model | `PASSED` |\n"
                "| 12. Provenance record | [Engineering] | one JSON linking every output | the file list |\n"
                "| 13. Optional activity | [Concept] | change the summary-length cap (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common failures and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Short on time? Run all, then read Sections 6, 7, 9 and 11 and the conclusion: they carry the "
                "principal results. The canonical path ends with Section 12; Section 13 changes nothing unless you switch it on."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Abstractive / extractive summary** | An extractive summary copies sentences from the source; an abstractive one writes new text, which can paraphrase — or state something the source does not. |\n"
                "| **Encoder-decoder (seq2seq)** | The encoder reads the whole document at once; the decoder writes the summary one token at a time, looking at the encoder's output and at what it has written so far. |\n"
                "| **Token / BPE** | The unit the model reads and writes: byte-level byte-pair-encoding pieces, roughly ¾ of an English word on average. |\n"
                "| **Beam search** | Keeps the `num_beams` best partial summaries at every step and returns the best complete one; with no sampling it is deterministic. |\n"
                "| **Length penalty** | A term in the beam score that favours longer (above 1) or shorter (below 1) beams. |\n"
                "| **`no_repeat_ngram_size`** | Forbids repeating any n-word sequence (3 here) inside one summary. |\n"
                "| **`max_new_tokens` / the cap** | The most tokens the decoder may write. The pinned config forces a start token first and an end token at the cap, so a capped summary holds `max_new_tokens − 2` content tokens and ends where it was cut. |\n"
                "| **`truncated` / `stopped_by`** | Whether the summary reached the cap (`stopped_by: max_new_tokens`) or ended on its own (`stopped_by: eos`). `hit_token_ceiling` counts the truncated summaries in a scored set. |\n"
                "| **Reference summary** | A human-written summary the output is compared with; SciTLDR has one per training paper and up to several per test paper. |\n"
                "| **ROUGE-1 / ROUGE-2** | F1 overlap of single words / word pairs between a summary and its best-matching reference, on 0–100. |\n"
                "| **ROUGE-L / longest common subsequence** | F1 of the longest sequence of words that appears in both texts in the same order (not necessarily adjacent). |\n"
                "| **Lead-k baseline** | The first k sentences of the source used as the summary: no model at all. |\n"
                "| **Frozen model** | The pinned pretrained model with no adaptation. |\n"
                "| **Decoder block** | One of the decoder's 12 identical layers (attention plus a feed-forward network); this notebook trains only the last two. |\n"
                "| **Teacher forcing / cross-entropy** | Training feeds the reference summary to the decoder and penalises the log-probability it gave each correct next token. |\n"
                "| **Epoch / validation / test** | One pass over the training split; the validation split picks the epoch; the test split is used once, for the final score. |\n"
                "| **Source-disjoint split** | No document appears in two splits, so the test measures summarising new documents, not remembered ones. |\n"
                "| **Adapter** | The exported file holding only the trained decoder blocks, with a manifest naming the base model it fits. |\n"
                "| **Safetensors** | A tensor file format that stores only numbers, so loading it cannot run code. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes; a single changed byte changes it. |\n"
                "| **Hash-locked environment** | A separate Python environment built from a requirements file that pins every package to one version and one set of SHA-256 digests; the installer refuses anything else. |\n"
                "| **Stage** | One step of the workflow run as its own process by `run_stage`; it reads the files earlier stages wrote and writes its own. |\n"
                "| **BYOD** | Bring Your Own Data: an optional switch to run the same workflow on your own document–summary pairs. |\n\n"
                "</details>"
            ),
        ],
    },
    "setup": [
        {
            "cell": "check",
            "md": (
                "## 1. Check the runtime · [Engineering]\n\n"
                "> **Infrastructure.** The code cells in Sections 1–3 are collapsed. You may run them without studying their "
                "implementation; they exist for reproducibility and provenance. The learning activities start in Section 4.\n\n"
                "**Input:** a fresh hosted runtime. **System:** checks that it is Linux x86_64 with enough free disk, reports the "
                "GPU if there is one, and creates a new run directory. **Output:** the directories this run will use. Each run "
                "writes to a new directory under `outputs/{stem}/`, so an earlier export cannot be mistaken for a current result. "
                "The verified snapshot and the corpus cache are kept in `weights/` and reused by a later run."
            ),
            "after": (
                "**Expected result:** one dictionary naming the GPU (for example `Tesla T4, 15360 MiB`) or `none (the stages run "
                "on CPU)`, the kernel's Python version, the run directory, the weights directory, the isolated environment's "
                "directory and the free disk. If the cell stops with a platform or disk message, see **Troubleshooting**."
            ),
        },
        {
            "cell": "carrier",
            "md": (
                "## 2. Carry the code and install the locked runtime · [Engineering]\n\n"
                "> **Infrastructure.** The next two code cells are collapsed. The first **is** the repository's code, carried so "
                "that this notebook works on its own; the second builds the environment every stage runs in.\n\n"
                "The first cell holds, as text, the files the workflow needs: the package's four modules under "
                "`src/bart_summarization_pipeline/` (identity constants, snapshot verification and staging, the named ceilings, "
                "validation, the pipeline class with its inference and adaptation contracts, the pinned-corpus fetcher, the "
                "dataset contract and the ROUGE metrics), the stage runner `tutorial_stages.py`, the hash-locked "
                "`requirements.txt` ({n_locked} packages), the snapshot manifest and the licence. It writes each file into the "
                "run directory and checks its SHA-256 against `CARRIED_HASHES`, stopping on any mismatch. The text is the "
                "repository's files byte for byte; the repository's parity test (`tests/test_notebook_parity.py`) fails whenever "
                "the two diverge, so what runs here is what the repository tests. Nothing in this cell runs a model."
            ),
            "after": (
                "**Expected result:** `carried_files`, `verified: True`, and the repository revision the notebook was generated "
                "from.\n\n"
                "The next cell installs nothing into this notebook's kernel. It downloads one pinned file — the `uv` installer "
                "wheel, refused unless its size and SHA-256 match — creates a separate virtual environment with its own CPython "
                "3.12.12, and installs `requirements.txt` into it with `--require-hashes --only-binary :all:`: every package must "
                "be the locked version, a prebuilt wheel, and match a locked digest. The hosted runtime's own packages are never "
                "replaced, which is why no restart is needed. The cell also defines `run_stage`, `load_record` and "
                "`obtain_upload`, the helpers the learner cells use."
            ),
        },
        {
            "cell": "install",
            "md": (
                "**Infrastructure: the isolated environment.** Installation messages from `uv` are normal and can take a few "
                "minutes (the CUDA build of `torch` is the largest download). A failed download or a hash mismatch stops the "
                "cell; never remove a pin or a hash to get past one."
            ),
            "after": (
                "**Expected result:** one dictionary with the generating revision, the isolated environment's Python (3.12.12), "
                "the `torch` and `transformers` versions, `cuda` (`True` on a GPU runtime, `False` on CPU — both are supported), "
                "the number of locked packages and the setup time."
            ),
        },
        {
            "cell": "weights",
            "md": (
                "## 3. Pin, stage and verify the model · [Engineering]\n\n"
                "> **Infrastructure.** The next code cell is collapsed. It downloads about 1.6 GB of pinned upstream files and "
                "checks every file's size and SHA-256; you may run it without studying its implementation.\n\n"
                "The model identity is carried twice — `MODEL_ID`/`MODEL_REVISION` in the carried `pipeline.py` and the 8-file "
                "snapshot manifest (paths, byte sizes, SHA-256) — and the `weights` stage first checks that they agree. It "
                "installs the carried manifest into `weights/`, fetches exactly the files that are absent from the Hugging Face "
                "Hub **at the pinned revision** of `{MODEL_ID}` (never `main`), and re-hashes every file, raising on the first "
                "size or digest mismatch. The weights are safetensors, which hold only numbers; no remote model code is executed, "
                "and every later stage loads the model from these verified files with `local_files_only=True`."
            ),
            "after": (
                "**What to notice:** the model id, revision, licence and file count; a `fetched` list (empty on a rerun, because "
                "staging only fetches absent files); and `verified_files: 8`. A mismatch stops the cell with an error naming the "
                "file — see **Troubleshooting**, and never edit a manifest to get past one."
            ),
        },
    ],
    "cells": [
        # ------------------------------------------------------------------ 4. runtime
        {
            "md": (
                "## 4. Confirm the isolated runtime · [Engineering]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`. This cell runs the "
                "`runtime` stage. It compares the versions installed in the isolated environment (`torch`, `transformers`, "
                "`tokenizers`, `safetensors`, `huggingface-hub`, `numpy`) with the versions in the carried lock and **stops if "
                "any differs**; it then prints the device the stages will use, the input and generation ceilings the pipeline "
                "enforces, and the pinned generation defaults read from the snapshot's `generation_config_for_summarization.json`.\n\n"
                "**Expected result:** `versions_match_lock: True`, the device (`cuda:0` on a GPU runtime, `cpu` otherwise), the "
                "ceilings (`MAX_TEXT_CHARS` 40,000, `MAX_INPUT_TOKENS` 1,024, `MAX_NEW_TOKENS` 512, `MAX_NUM_BEAMS` 8), the "
                "pinned defaults (`num_beams` 4, `length_penalty` 2.0, `no_repeat_ngram_size` 3, 55..141 new tokens) and the "
                "decision rule: deterministic beam search, no sampling, no seed."
            ),
            "code": "run_stage('runtime')",
        },
        # ------------------------------------------------------------------ 5. data
        {
            "md": (
                "## 5. Referenced corpus, validation and split · [Evaluation practice]\n\n"
                "The `data` stage downloads the three pinned SciTLDR-A files (or reads them from the cache), refuses a byte-size "
                "or SHA-256 mismatch per file before it is parsed, and flattens each JSON-Lines member into records whose "
                "`source` is the abstract's sentences joined by a space and whose `targets` are its reference TL;DRs (one "
                "author-written TL;DR for training papers; author plus peer-review-derived TL;DRs for dev and test papers). It "
                "keeps abstracts of 200..2,400 characters with at least one reference, drops repeated sources, and draws 300 "
                "training records from the `train` member, 50 validation records from `dev` and 100 test records from `test` by "
                "a seeded shuffle — the release's own paper-disjoint partition. It also sets aside four `dev` abstracts that are "
                "in none of the splits for Section 10. `validate_dataset` checks every record against the contract, "
                "`check_split_disjoint` asserts that no source appears in two splits, and the training split is written to "
                "`outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "**Checked against the workflow, here.** A dataset that passes the record contract can still fail later, so "
                "this stage also counts the BPE tokens of every validation, test and Section 10 source with the snapshot's "
                "tokenizer: one over the 1,024-token ceiling would fail inside the model in Section 7 or 9, so it is refused "
                "**here**, naming its split and id. Training sources over 512 tokens are only counted (they are truncated during "
                "training). Nothing in this stage runs the model.\n\n"
                "**Pretraining overlap.** BART was pretrained on about 160 GB of news, books, stories and web text (Lewis et al., "
                "2019) and the checkpoint was then fine-tuned on CNN/DailyMail. SciTLDR's abstracts are public OpenReview papers, "
                "so overlap between them and BART's pretraining text **cannot be ruled out**. The comparison in this notebook is "
                "between the frozen and the adapted versions of the same model, which share whatever exposure there was; the "
                "splits are independent of *each other*, which is what the train / validation / test reading below relies on.\n\n"
                "**BYOD:** set `USE_BYOD = True` and put a CSV, JSON or JSONL path in `BYOD_PATH` (any runtime), or leave the "
                "path empty for the Colab upload dialog. The training CSV written below is a valid example. Your file is split "
                "15 % validation / 20 % test / the rest training after de-duplication; the stage prints how many duplicate "
                "sources it dropped and refuses a dataset with fewer than 12 distinct sources, naming the split it would leave "
                "too small. Section 10 then summarises four of your *test* records and labels them as such.\n\n"
                "**Expected result:** 1,992 + 619 + 618 raw papers, three file digests, splits 300 / 50 / 100 and 4 new "
                "documents, one reference per training record and up to four per test record, the largest token count per split "
                "(all far below 1,024), and five refusal probes — a duplicate id, an empty reference list, a missing field, a "
                "dataset too small for this workflow and an over-long test source — each rejected before any model runs."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "data_options = ['--split-seed', SPLIT_SEED]\n"
                "if USE_BYOD:\n"
                "    data_options += ['--byod', obtain_upload(BYOD_PATH, ('.csv', '.json', '.jsonl'), 'BYOD_PATH')]\n"
                "run_stage('data', *data_options)"
            ),
        },
        {
            "md": (
                "**Checkpoint:** SciTLDR already ships separate train, dev and test files. Why does the stage still check that no "
                "source appears in two splits, and why does it de-duplicate before splitting?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Because the test score is only meaningful if the model has never been trained on the documents it is tested on. "
                "If the same abstract (or a copy differing only in case) sat in training and test, the adapted model could score "
                "well by remembering its reference, and the test would measure memory rather than summarization. The release's "
                "partition is paper-disjoint, but a check costs nothing and catches a corrupted or re-used file; with your own "
                "data, where duplicates are common, de-duplication before the split is what keeps the splits apart. When pairs "
                "come from one document collection or author, split by that grouping as well.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 6. inference
        {
            "md": (
                "## 6. Summarise through the inference contract · [Concept]\n\n"
                "Before any adaptation, the inference contract is exercised on a synthetic three-paragraph news-style document "
                "carried in the stage runner (the model card's smoke passage). `validate_inputs` applies exactly the checks "
                "`summarize` applies — text type and character ceiling, and the generation settings against their ceilings — and "
                "returns an input manifest; an out-of-range `num_beams` is validated too and its rejection recorded as a finding. "
                "`summarize` counts the encoder tokens with the real tokenizer and **rejects with a `ValueError` naming the "
                "count, never truncates or chunks**, an input over `MAX_INPUT_TOKENS`. It returns the summary with "
                "`generated_tokens` (content tokens), `input_tokens`, `truncated` and `stopped_by`, and echoes the settings. "
                "**Score semantics:** the pipeline emits **no probability, confidence or score of any kind**; `truncated` is a "
                "length flag.\n\n"
                "**How `truncated` is decided.** The pinned config forces a start token as the first new token and an end token "
                "when generation reaches `max_new_tokens`, so every summary ends with an end token — even one that was cut "
                "mid-sentence. The contract therefore decides from the length: a summary that used every allowed step is "
                "`truncated: True`, `stopped_by: max_new_tokens`, and holds `max_new_tokens − 2` content tokens. The stage checks "
                "that independently of the flag's own logic: `full_length_summary_flagged_truncated` fails if a summary holding "
                "every content token the cap allows is not flagged as cut. The first call uses the "
                "pinned news defaults (55..141 new tokens); the second uses the TL;DR settings below, which every model score in "
                "Sections 7–10 is produced under.\n\n"
                "**Predict before running:** the news defaults allow up to 141 new tokens and the TL;DR settings 48. Will the "
                "TL;DR summary of a three-paragraph news story end at a sentence boundary, or be cut?"
            ),
            "code": (
                "TLDR_MAX_NEW_TOKENS = 48  # @param {{type:\"integer\"}}\n"
                "TLDR_MIN_NEW_TOKENS = 0  # @param {{type:\"integer\"}}\n"
                "NUM_BEAMS = 4  # @param {{type:\"integer\"}}\n\n"
                "run_stage('inference', '--max-new-tokens', TLDR_MAX_NEW_TOKENS, '--min-new-tokens', TLDR_MIN_NEW_TOKENS, '--num-beams', NUM_BEAMS)"
            ),
        },
        {
            "md": (
                "**What to notice:** for each setting, `generated_tokens`, `content_tokens_at_cap` (the cap minus the two forced "
                "tokens), `truncated` and `stopped_by`, then the summary. Every sanity check is `True`, including "
                "`full_length_summary_flagged_truncated`.\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Under the news defaults the model writes a complete multi-sentence summary and stops on its own well before 141 "
                "tokens (`stopped_by: eos`). Under the TL;DR settings it tries to write the same kind of news highlight, runs into "
                "the 48-token cap and is cut mid-sentence: `generated_tokens` equals `content_tokens_at_cap` (46), `truncated: "
                "True`, `stopped_by: max_new_tokens`. That matters for everything that follows: a news summariser scored under a "
                "TL;DR cap is partly scored on text the cap chopped off, and Section 7 counts how often that happens.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 7. frozen
        {
            "md": (
                "## 7. Baselines and the frozen model on the test split · [Evaluation practice]\n\n"
                "Three numbers frame the adaptation. The **Lead-1 baseline** submits the first sentence of each abstract as its "
                "summary and the **Lead-3 baseline** the first three: what a system that does no modelling gets, and a reminder "
                "that ROUGE rewards length matching — Lead-1 usually beats Lead-3 against one-sentence references. The Lead "
                "baselines copy sentences, so no generation settings or token cap apply to them; compare their "
                "`mean_summary_words` with the references instead. The `frozen` stage builds the frozen model afresh, summarises "
                "the 100 test abstracts under the TL;DR settings of Section 6 and scores them with the same three metrics: corpus "
                "**ROUGE-1**, **ROUGE-2** and **ROUGE-L** F1 (best over the references, rouge-score-style, not "
                "rouge-score-identical). It then prints **how many of the 100 frozen summaries the cap cut**, and two examples "
                "with their `truncated` flag, and writes everything to `outputs/{stem}_frozen_test.json` before any adaptation, so "
                "the reference cannot be overwritten by a later step.\n\n"
                "**Predict before running:** the frozen model writes news highlights of three sentences. Against one-sentence "
                "TL;DRs, will it beat Lead-1? And how many of its 100 summaries do you expect the 48-token cap to cut — a few, or "
                "most?"
            ),
            "code": "run_stage('frozen')",
        },
        {
            "md": (
                "**What to notice:** the frozen model's ROUGE scores beside both baselines, its `mean_summary_words` against "
                "`mean_reference_words`, `adapted: False`, and the `cut_at_the_cap` line.\n\n"
                "**Question tested:** before any adaptation, is a news summariser better than copying the first sentence, and "
                "how much is its score shaped by the length cap rather than by its content?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "On the default sample the frozen model lands just above Lead-1 on ROUGE-L (about 25.5 against 23.6 in the "
                "recorded runs) with summaries about twice the reference length (about 35 words against about 20), and most of "
                "its summaries reach the cap and are cut — 86 of 100 in a local CPU pre-flight of this version; read the count "
                "your run printed. So part of the frozen score is "
                "produced by the cap: without it the summaries would run longer and lose precision; with it some are scored on "
                "a sentence that stops halfway. Either way the frozen number is the honest baseline the adaptation is read "
                "against, and the optional activity in Section 13 measures how much the cap moves it.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 8. adapt
        {
            "md": (
                "## 8. Bounded fine-tuning of the last decoder blocks · [Concept]\n\n"
                "The `adapt` stage builds the frozen model afresh (it prints `adapted: False` to show it starts from the pinned "
                "base) and trains only the last `TRAINABLE_DECODER_LAYERS` decoder blocks — two by default, 33,593,344 of "
                "406,290,432 parameters; the encoder, the shared embeddings, the tied output projection and the earlier decoder "
                "blocks stay frozen — with teacher-forced cross-entropy on the first reference TL;DR, AdamW at a fixed learning "
                "rate, gradient clipping at 1.0, seeded shuffling and no scheduler. Sources are truncated to 512 and targets to 64 "
                "BPE tokens **during training only**. Epoch 0 records the frozen model's validation ROUGE under the TL;DR "
                "settings; every epoch is scored on the validation split the same way, and the epoch with the highest "
                "**validation** ROUGE-L is kept — the test split plays no part. The stage then exports the kept blocks as "
                "`outputs/{stem}_adapter/adapter.safetensors` with a `manifest.json` (format, base model id, revision and weight "
                "digest, tensor names, size and SHA-256, training configuration and epoch history), and records a fingerprint of "
                "every decoder tensor and four test summaries for the parity check in Section 11.\n\n"
                "**Predict before running:** will validation ROUGE-L rise every epoch? What should happen to "
                "`val_mean_summary_words` and to `val_cut_at_cap`?"
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 3e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n\n"
                "run_stage('adapt', '--epochs', EPOCHS, '--learning-rate', LEARNING_RATE, '--batch-size', BATCH_SIZE, '--trainable-decoder-layers', TRAINABLE_DECODER_LAYERS)"
            ),
        },
        {
            "md": (
                "**What to notice:** epoch 0 labelled `frozen model`, then per epoch the training loss, the validation ROUGE "
                "scores, `val_mean_summary_words` and `val_cut_at_cap`; 33,593,344 trainable parameters, the kept epoch, and the "
                "adapter (52 tensors, about 134 MB).\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "On the default sample, validation ROUGE-L rose from about 24 to about 31–32 over two epochs in the recorded runs, "
                "and most of the gain came in the first epoch, as the summaries shrank from news length to roughly the reference "
                "length and almost none reached the cap any more. That is a register shift: the model learns to write one short "
                "sentence. The notebook does not promise a rise on other data or settings — epoch 0 (the frozen model) can win, "
                "and then it is kept. That is why the epoch is chosen on the validation split and never on the test split: "
                "choosing it on the test numbers would make Section 9 a measure of the choice, not of the model.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 9. evaluate
        {
            "md": (
                "## 9. Held-out evaluation · [Evaluation practice]\n\n"
                "The test split was never used for training or epoch selection, and no abstract in it appears in the training or "
                "validation splits. The `evaluate` stage starts a new process, rebuilds the base model from the verified files, "
                "loads the adapter from its files (manifest, base digest and tensor set checked before anything is "
                "deserialised), and scores it exactly as the frozen model was scored in Section 7. The table puts Lead-1, Lead-3, "
                "frozen and adapted side by side for ROUGE-1/2/L, `mean_summary_words` and the number of summaries cut at the "
                "cap, and `outputs/{stem}_evaluation_report.json` records it with the settings, the training configuration and the "
                "epoch history.\n\n"
                "The stage then prints the **outcome** as a fact, not an assertion: whether the adapted ROUGE-L is above the "
                "frozen one. Only on the default sample with the default settings, where the recorded runs show the improvement, "
                "does the stage treat its absence as a failure. With BYOD, other settings or other training choices a negative "
                "result is a legitimate finding: the notebook continues, and Sections 10–12 still export everything.\n\n"
                "**Predict before running:** will the adapted model beat the frozen model on all three ROUGE scores? Will it beat "
                "Lead-1?"
            ),
            "code": "run_stage('evaluate')",
        },
        {
            "md": (
                "**What to notice:** `delta_adapted_vs_frozen`, the adapted `mean_summary_words` against the references, the "
                "`cut_at_cap` row, the two adapted examples beside their references, and the `outcome` line.\n\n"
                "**Question tested:** on papers it has never seen, does adapting two decoder blocks on 300 abstracts move a news "
                "summariser to the TL;DR register, by more than copying the first sentence would?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "On the default sample, yes: in the recorded runs held-out ROUGE-L went from about 25.5 (frozen) to about 33.6 on "
                "a T4 GPU and about 34.3 on CPU, above both Lead baselines, with summaries close to the reference length and "
                "almost none cut. The CPU/GPU difference is floating-point arithmetic, not a defect. One hundred abstracts from "
                "one seeded split of one corpus give no dispersion estimate, so the gain shows that the adaptation contract works "
                "here, not by how much it would help on other papers. The adaptation also changes the model: it now writes one "
                "short sentence for any input, and its long news-highlight behaviour is traded away in the adapter. If your run "
                "did not improve — with BYOD or other settings — read the baselines first: when Lead-1 is already close to the "
                "references, there is little to gain.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 10. new
        {
            "md": (
                "## 10. Summarise new documents · [Concept]\n\n"
                "The `new` stage rebuilds the adapted model from its files and summarises the four documents Section 5 set aside, "
                "through the same `summarize` contract as Section 6, printing each summary beside its reference with its "
                "`truncated` flag. On the default sample these are four `dev` abstracts that are in **none** of the splits. Under "
                "BYOD there are no spare documents, so they are the first four of your **test** records, already scored in "
                "Section 9, and the output says so. The four are scored with `pipe.evaluate` (a `measured-small-sample` verdict, "
                "because four documents carry no dispersion estimate), and the single-document `evaluation_report` helper — the "
                "inference-stage helper, which scores supplied references as `sample-sanity` — is written for the first of them. "
                "The summaries go to `outputs/{stem}_summaries.csv`.\n\n"
                "**Expected result:** four one-sentence summaries, `documents_are` naming where they came from, a "
                "`measured-small-sample` verdict and a `sample-sanity` single-document verdict."
            ),
            "code": "run_stage('new')",
        },
        {
            "md": (
                "**What to notice:** read the summaries against the abstracts' meaning, not only against the references. ROUGE "
                "counts shared words; it cannot tell whether a summary states a result the paper does not. An abstractive "
                "summary can overlap a reference well and still be unfaithful."
            ),
        },
        # ------------------------------------------------------------------ 11. reload
        {
            "md": (
                "## 11. Fresh reload and parity · [Engineering]\n\n"
                "Sections 9 and 10 already loaded the adapter in new processes. The `reload` stage checks that this is faithful: "
                "it re-verifies the base snapshot and the adapter's manifest and digest, rebuilds the pipeline from the files in "
                "another fresh process, and compares it with the trained model of Section 8 in two ways — a SHA-256 fingerprint "
                "of **every decoder tensor** (all twelve blocks, not only the trained ones), and the summaries of four test "
                "documents. A mismatch stops the notebook.\n\n"
                "**Expected result:** `decoder_tensors_identical: True`, `identical_summaries: 4` of 4 and "
                "`reload_verification: PASSED`."
            ),
            "code": "run_stage('reload')",
        },
        {
            "md": (
                "**What to notice:** the tensor fingerprint is the stronger check. Four identical summaries can hide a small "
                "difference in the weights — beam search often returns the same words from slightly different numbers — while "
                "the fingerprint changes if any decoder number differs. Parity shows that the export is complete and loads "
                "safely, not that the summaries are good: a saved model reproduces its errors just as faithfully as its skill."
            ),
        },
        # ------------------------------------------------------------------ 12. bundle
        {
            "md": (
                "## 12. Export the provenance record · [Engineering]\n\n"
                "The `bundle` stage writes `outputs/{stem}_result.json`, which links the run's records — the dataset manifest, "
                "input manifest and inference record, the frozen scores, training history, evaluation report, new-document "
                "summaries and reload parity — with the notebook's source revision, the model id, revision and licence, the "
                "snapshot block (`weight_format`, `weight_sha256`, `remote_code_executed: false`), the `corpus` block (name, "
                "release, base URL, the three files with sizes and digests, licence), the runtime versions, and the SHA-256 of "
                "every output file.\n\n"
                "**Expected result:** a list of the run's output files, including `{stem}_train.csv`, "
                "`{stem}_dataset_manifest.json`, `{stem}_input_manifest.json`, `{stem}_frozen_test.json`, "
                "`{stem}_evaluation_report.json`, `{stem}_summaries.csv`, `{stem}_adapter/adapter.safetensors`, "
                "`{stem}_reload_parity.json` and `{stem}_result.json`."
            ),
            "code": "run_stage('bundle')",
        },
        # ------------------------------------------------------------------ 13. activity
        {
            "md": (
                "## 13. Optional activity: change one thing — the summary-length cap · [Concept]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** This activity is off by default and changes nothing "
                "the canonical path produced: with `RUN_ACTIVITY = False` the next cell only prints how to switch it on. When on, "
                "the `activity` stage re-scores the test split with one change — `ACTIVITY_MAX_NEW_TOKENS` (96 by default, twice "
                "the canonical 48) — for both the frozen model (built afresh) and the adapted model (rebuilt from its files), with "
                "the same beams, minimum length and test documents. It compares ROUGE-L, `mean_summary_words` and the number of "
                "summaries cut with the canonical result, writes only under `outputs/activity/`, and stops if a canonical output "
                "changed. It takes about as long as Sections 7 and 9 together.\n\n"
                "**Change one thing:** the summary-length cap. Set `RUN_ACTIVITY = True` and run the cell.\n\n"
                "**Predict before running:** with twice the room, will the frozen model's ROUGE-L rise (no more cut sentences) or "
                "fall (longer summaries against one-sentence references)? Will the adapted model change much? Will the gap between "
                "them grow or shrink? Write it down."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_MAX_NEW_TOKENS = 96  # @param {{type:\"integer\"}}\n\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', '--max-new-tokens', ACTIVITY_MAX_NEW_TOKENS)\n"
                "else:\n"
                "    print({{'activity': 'skipped (optional)', 'to_run': 'set RUN_ACTIVITY = True, then run this cell'}})"
            ),
        },
        {
            "md": (
                "**Observe:** per model, ROUGE-L, `mean_summary_words` and `cut_at_cap` at the canonical and the activity cap; "
                "then the adapted-minus-frozen ROUGE-L gap at both caps.\n\n"
                "**Explain:** did the result match your prediction? Which model did the cap affect, and why?\n\n"
                "**Question tested:** how much of the frozen-versus-adapted gap in Section 9 is the length cap, and how much is "
                "the register the adaptation learned?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The cap can only matter for summaries that reach it, so start from the `cut_at_cap` counts. The adapted model "
                "usually stops on its own after one sentence, so a higher cap should change it little. The frozen model, given "
                "room to finish, writes its full news highlight: fewer summaries are cut, but they get longer — further from a "
                "one-sentence reference — so two effects pull against each other: completed sentences can add matching words, "
                "while extra length lowers ROUGE precision. Which one wins is what your run measures. If the adapted-minus-frozen "
                "gap *grows* at the larger cap, the 48-token cap was flattering the frozen model, and the adaptation's gain is not "
                "an artefact of cutting the frozen summaries short; if it *shrinks*, part of the Section 9 gap was the cap. "
                "Whatever your run shows is an observation about this run, not a failed activity; because only the cap changed, "
                "any difference is caused by it. To try another variable "
                "instead — `NUM_BEAMS = 1` (greedy decoding) in Section 6, or `TRAINABLE_DECODER_LAYERS = 1` in Section 8 — "
                "change that field and re-run from its section to the end, as **Changing a setting** describes; every stage "
                "starts again from the pinned base, so the frozen numbers stay the frozen model's.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ troubleshooting
        {
            "md": (
                "## Troubleshooting · [Engineering]\n\n"
                "| Observation | Action |\n"
                "|---|---|\n"
                "| Section 1 stops: not Linux x86_64 | Use Google Colab, Kaggle or a Linux Jupyter server; the locked environment is built for manylinux x86_64. |\n"
                "| Section 1 stops: not enough disk | Start a fresh runtime, or delete earlier `outputs/{stem}/` run directories. |\n"
                "| Section 2: download, `uv` or hash failure | Retry once on a stable connection. Never remove a pin or a hash; a hash mismatch means the file is not the locked one. |\n"
                "| Section 3: Hugging Face download fails or a digest mismatches | Retry; delete `weights/bart-large-cnn/` and rerun Section 3 if a partial file remains. Never edit the manifest. |\n"
                "| Section 5: a SciTLDR file fails to download or mismatches | Retry; the files are cached in `weights/scitldr/` once verified. A digest mismatch is refused, never parsed. |\n"
                "| Section 5: `… records, … with distinct sources …; the workflow needs at least 12` | Your BYOD file is too small for this workflow after de-duplication: add records, then re-run from Section 5. |\n"
                "| Section 5: `… record '…' has N tokens; MAX_INPUT_TOKENS is 1024` | Shorten or remove the named record (or split the document), then re-run from Section 5. |\n"
                "| Section 5: id, field, reference or length refusal | Fix the file as the message says: unique ids matching `[A-Za-z0-9_.:-]{{1,64}}`, a non-empty `source`, at least one non-empty reference. |\n"
                "| `No file was uploaded` or `The upload dialog needs Google Colab` | Run the cell again and choose one file, or put the file on the machine and set `BYOD_PATH` to its path. |\n"
                "| Section 6: a generation setting is refused | Keep `TLDR_MAX_NEW_TOKENS` in 1..512, `TLDR_MIN_NEW_TOKENS` ≤ it, `NUM_BEAMS` in 1..8. |\n"
                "| A stage fails | The cell repeats the stage's error message; the full log is in the run directory under `logs/<stage>.log`. Fix the cause and rerun from that section. |\n"
                "| \"… is missing: run the stage that writes it\" | A later cell ran before an earlier one, or a re-run cleared it. Run from the section named, or from Section 5, to the end. |\n"
                "| \"the adapter was trained before the data or the generation settings changed\" | Re-run from Section 8. |\n"
                "| Out of memory | Lower `BATCH_SIZE` in Section 8 and re-run from Section 8; a CPU runtime needs about 6 GB of RAM. |\n"
                "| Section 9 reports that the adapted model does not score above the frozen model | A legitimate result outside the default configuration: read the Lead baselines first. On the default configuration the stage stops, because the recorded runs improved; keep the logs and report it. |\n"
                "| Reload parity fails | Do not use the export. Re-run from Section 8; if it persists, keep the logs and report it. |\n\n"
                "Nothing is installed into the kernel, so no runtime restart is ever needed."
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen news summariser, asked for a TL;DR of a paper abstract, writes something close to the abstract's own "
        "opening sentences at about twice the reference length, is cut by the 48-token cap on most documents, and scores just "
        "above Lead-1. On the default sample, a bounded fine-tuning of the last two decoder blocks on 300 in-domain abstracts "
        "moved it to the one-sentence register and raised held-out ROUGE-L by about eight points in the recorded runs, with a "
        "134 MB adapter that reloads to identical decoder tensors; your run reports its own outcome in Section 9, and with "
        "other data or settings that outcome may be negative. That is the claim: the adaptation contract works end to end on a "
        "real referenced corpus, and the numbers it produces are read against two Lead baselines and the frozen model rather "
        "than in isolation.\n\n"
        "The test split is 100 abstracts from one seeded split of one corpus, the metrics are three n-gram overlap scores "
        "(own implementation, not rouge-score-identical, and none a judgement of faithfulness), and SciTLDR references are "
        "short and formulaic. So a gain here says the contract works, not that the adapted model is better on your documents, "
        "that it handles long or technical text, or that its summaries are faithful — an abstractive summary can state a "
        "result the source does not and still overlap the reference well. Part of the frozen model's score is shaped by the "
        "cap that cut its summaries (Sections 7 and 13). Fine-tuning on a narrow corpus also changes the model elsewhere — "
        "the adapter writes one short sentence for any input — and nothing here measures that. Overlap between SciTLDR and "
        "BART's pretraining text cannot be ruled out (Section 5). CPU and GPU runs agree to a few tenths of a ROUGE point, not "
        "bit for bit.\n\n"
        "Three things to carry to real data. **References first:** the Lead baselines and the frozen model's score on *your* "
        "references, under *your* length settings, are the numbers to read before any adapted one — ROUGE moves with summary "
        "length as much as with content, so read `mean_summary_words` and the count of summaries cut at the cap beside every "
        "score. **Leakage:** de-duplicate sources across splits (the contract does this case-insensitively) and split by "
        "document collection or author when your pairs come from one. **Ceilings:** inputs over `MAX_INPUT_TOKENS` are refused "
        "at inference and truncated to 512 tokens only during training — long-document summarization is out of scope.\n\n"
        "Successful execution proves that the recorded repository revision's package and stage runner, carried in this "
        "standalone notebook, can build a hash-locked environment without touching the kernel, acquire and digest-verify the "
        "pinned model snapshot, fetch and digest-verify a real referenced corpus, validate the demonstrated dataset contract "
        "against the workflow without leakage, execute the inference contract and a bounded fine-tuning, evaluate against two "
        "trivial baselines and the frozen model on an independent split, and emit the shown machine-readable artifacts — "
        "without the repository being reachable. It does **not** establish benchmark superiority, summary quality or "
        "faithfulness on any other domain, a usable acceptance threshold, or production fitness.\n\n"
        "## Conclude with evidence · [Evaluation practice]\n\n"
        "Complete this paragraph with the numbers your run printed:\n\n"
        "> On the [data source] test split of [n] documents, under a [cap]-token TL;DR cap, Lead-1 scored ROUGE-L [value] and "
        "the frozen model [value] at [words] words per summary against [words]-word references, with [n] of [n] frozen "
        "summaries cut at the cap. After fine-tuning the last [n] decoder blocks on [n] training documents (best epoch [value], "
        "chosen on the validation split), the adapted model scored ROUGE-L [value] at [words] words, with [n] summaries cut. "
        "The reloaded adapter reproduced every decoder tensor: [True/False]. With the cap at [value] (optional activity), the "
        "adapted-minus-frozen ROUGE-L gap was [value]. These results show [what they do show] and do not show [one thing they "
        "cannot show].\n\n"
        "**Transfer:** switch on BYOD (Section 5) with document–summary pairs from your own domain — at least 12 distinct "
        "documents, nothing confidential on a hosted runtime — and read the Lead baselines and the count of summaries cut "
        "before any adapted number. Which length setting would you choose for your references, and how would you check that "
        "the summaries are faithful, which ROUGE cannot tell you?\n\n"
        "**AI Assistance Disclosure:** this notebook's code and explanations were developed with generative AI assistance under "
        "maintainer direction. The maintainer remains responsible for reviewing implementation, validating results and making "
        "release decisions.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/fairseq/tree/main/examples/bart\n"
        "- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (Lewis et al., 2019): https://arxiv.org/abs/1910.13461\n"
        "- TLDR: Extreme Summarization of Scientific Documents (Cachola et al., EMNLP Findings 2020; SciTLDR, Apache-2.0): https://arxiv.org/abs/2004.15011\n"
        "- ROUGE: A Package for Automatic Evaluation of Summaries (Lin, 2004): https://aclanthology.org/W04-1013\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}

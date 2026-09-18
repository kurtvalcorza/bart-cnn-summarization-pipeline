"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E summarization workflow: the pinned BART-large CNN snapshot is
digest-verified and loaded, a digest-pinned real out-of-domain corpus (SciTLDR-A, one-sentence TL;DRs of
paper abstracts) is fetched, validated and split, one synthetic document is summarised through the
inference contract with the pinned news defaults, the frozen model is scored with ROUGE against the
Lead-1 and Lead-3 baselines under TL;DR-length settings, a bounded fine-tuning of the last decoder blocks
runs in the kernel, the held-out split is scored again, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "bart_summarization_pipeline",
    "repo_name": "bart-cnn-summarization-pipeline",
    "stem": "bart_summarization",
    "notebook_name": "bart_summarization_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned BART-large CNN snapshot (safetensors, 1.6 GB), fetches the three digest-pinned SciTLDR-A files from the "
        "project repository (5.5 MB, no credential), filters and draws 300 / 50 / 100 training, validation and test "
        "abstracts from the release's own paper-disjoint members, summarises one synthetic news-style document through the "
        "inference contract with the pinned generation defaults and a rejection probe, scores the frozen model on the test "
        "abstracts with ROUGE-1/2/L under TL;DR-length settings beside the Lead-1 and Lead-3 baselines, runs a bounded "
        "fine-tuning of the last two decoder blocks on the training abstracts with validation-ROUGE-L epoch selection, scores "
        "the held-out split again, summarises new abstracts with the adapted model, exports the adapter as safetensors with a "
        "manifest, and reloads that artifact into a fresh pipeline to verify summary parity. The default path needs no "
        "repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about twelve minutes of model time after the downloads; a CUDA "
        "runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "document–summary pairs as a CSV (columns `id`, `source`, `target`), a JSON array or a JSONL file of `{{id, source, "
        "target}}` or `{{id, source, targets: [...]}}` records. They pass through the same validation, seeded source-disjoint "
        "split, baselines, fine-tuning, held-out evaluation, inference, artifact export and reload-parity cells as the SciTLDR "
        "sample. The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay "
        "inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "BARTSummarizationPipeline",
    "weights_key": "bart-large-cnn",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "BART-large CNN — DIMER E2E abstractive summarization fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/bart-cnn-summarization-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/tutorials/bart_summarization_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fbart--large--cnn-ffcc4d?style=flat",
            "https://huggingface.co/facebook/bart-large-cnn",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-facebookresearch%2Ffairseq-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/facebookresearch/fairseq/tree/main/examples/bart",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-1910.13461-b31b1b.svg", "https://arxiv.org/abs/1910.13461"),
    ],
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
        "ROUGE number in this notebook is therefore produced under explicit TL;DR-length settings (`max_new_tokens` 48, "
        "`min_new_tokens` 0, `num_beams` 4) applied identically to the frozen and the adapted model, and the inference "
        "contract in Section 5 shows the pinned defaults on a news-style document first."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, dataset and metrics modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned referenced summarization corpus and validate "
        "and split it without leakage; summarise through the public API with the pinned generation defaults and read "
        "`generated_tokens`, `truncated` and `stopped_by` correctly; score the frozen model against references beside two "
        "Lead baselines and read why summary length drives ROUGE; run a bounded fine-tuning with explicit hyperparameters "
        "and validation-based epoch selection; evaluate on an independent test split; summarise new abstracts; and export "
        "a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "extractive summarization with guaranteed source spans, multi-document or long-document (chunked) summarization, "
        "headline generation, sampling-based or diverse decoding, full-model or encoder fine-tuning, classification or "
        "question answering (the `bart-mnli-zero-shot-classification-pipeline` sibling covers zero-shot classification), "
        "non-English text, any faithfulness or factuality score, and any claim that a SciTLDR split stands in for your "
        "documents. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate but slow for a 406 M-parameter decoder: the build record measured 6 s to load and digest-verify the 1.6 GB snapshot, about 1.8 s per abstract for 4-beam TL;DR-length generation (three minutes for the 100-abstract test split) and about 40 s per training epoch over 300 abstracts plus a 50-abstract validation pass per epoch. The pinned `torch==2.14.0` install and the 1.6 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what an encoder-decoder (seq2seq) model is; what beam search does and why it is deterministic; why an abstractive summary can state things the source does not (hallucination); what ROUGE measures and why it is neither faithfulness nor a human judgement.",
        "- **Data contract:** records are `{{id, source, targets}}` — a document and one or more reference summaries (`{{id, source, target}}` with a single string is accepted and normalised), the source 1..40,000 characters and at most 1,024 BPE tokens at inference, each reference 1..2,000 characters, ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a dataset needs 8..20,000 records; sources are de-duplicated case-insensitively before splitting so the same document never sits in two splits; during training only, sources are truncated to 512 and targets to 64 BPE tokens (inference never truncates — it rejects). BYOD accepts CSV, JSON or JSONL in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a reference is a faithful summary of its source or that a source is prose — a mislabelled corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal report archive with its executive summaries is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches three pinned objects (`train.jsonl` 3,155,015 bytes, `dev.jsonl` 1,124,865 bytes, `test.jsonl` 1,204,107 bytes; SHA-256 `b222771d…` / `3191fa98…` / `fb42dd6c…`) from `raw.githubusercontent.com` at the pinned `allenai/scitldr` commit over HTTPS, each refused on any mismatch before it is read; SciTLDR is Apache-2.0 (Cachola et al., 2020).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and split\n\n"
                "`fetch_corpus` downloads the three pinned SciTLDR-A files (or reads them from the cache), refuses a "
                "byte-size or SHA-256 mismatch per file before it is parsed, and `read_corpus` flattens each JSON-Lines "
                "member into records whose `source` is the abstract's sentences joined by a space and whose `targets` are "
                "its reference TL;DRs (one author-written TL;DR for training papers; author plus peer-review-derived "
                "TL;DRs for dev and test papers). `build_sample_dataset` keeps abstracts of 200..2,400 characters with at "
                "least one reference, drops repeated sources, and draws 300 training records from the `train` member, 50 "
                "validation records from `dev` and 100 test records from `test` by a seeded shuffle — the release's own "
                "paper-disjoint partition. `validate_dataset` then checks every record against the contract, "
                "`check_split_disjoint` asserts no source appears in two splits, and the training split is written to "
                "`outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 1,992 + 619 + 618 raw papers, three digests, splits 300 / 50 / 100, one reference per training "
                "record and up to four per test record, and four refusal probes — a duplicate id, an empty reference "
                "list, a missing field and a dataset too small to split — each rejected before `torch` does anything."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_papers = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/scitldr'))\n"
                "    raw_papers = {{name: len(part) for name, part in corpus.items()}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_papers': raw_papers, 'splits': disjoint, 'file_sha256': {{k: v[2][:12] + '...' for k, v in CORPUS_FILES.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_sources': manifest['unique_sources'], 'references_per_record': manifest['references_per_record'], 'source_chars': manifest['source_chars'], 'target_words': manifest['target_words'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'example': {{'id': train_records[0]['id'], 'source': train_records[0]['source'][:200] + '...', 'targets': train_records[0]['targets']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty reference list': [{{**train_records[0], 'targets': []}}, *train_records[1:8]],\n"
                "    'missing field': [{{'id': r['id'], 'source': r['source']}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Summarise through the inference contract\n\n"
                "Before any adaptation, the inference contract is exercised as it always was, on a synthetic three-paragraph "
                "news-style document authored in this cell (the model card's smoke passage). `validate_inputs` applies "
                "exactly the checks `summarize` applies — text type and character ceiling, and the generation settings "
                "against their ceilings — and returns an input manifest; the encoder-token ceiling `MAX_INPUT_TOKENS` "
                "(1,024) needs the real tokenizer and is enforced inside `summarize`, which **rejects with a `ValueError` "
                "naming the count, never truncates or chunks**. An out-of-range `num_beams` is validated too and its "
                "rejection recorded as a finding. `summarize` returns the summary with `generated_tokens`, `input_tokens`, "
                "`truncated` (the summary hit `max_new_tokens`) and `stopped_by`, and echoes the settings. **Score "
                "semantics:** the pipeline emits **no probability, confidence or score of any kind**; `truncated` is a "
                "length flag. The first call uses the pinned news defaults (at least 55 new tokens); the second uses the "
                "TL;DR-length settings every ROUGE number below is produced under, so the length difference is visible "
                "before any metric is read."
            ),
            "code": (
                "import time\n\n"
                "TLDR_MAX_NEW_TOKENS = 48  # @param {{type:\"integer\"}}\n"
                "TLDR_MIN_NEW_TOKENS = 0  # @param {{type:\"integer\"}}\n"
                "NUM_BEAMS = 4  # @param {{type:\"integer\"}}\n\n"
                "TLDR = {{'max_new_tokens': TLDR_MAX_NEW_TOKENS, 'min_new_tokens': TLDR_MIN_NEW_TOKENS, 'num_beams': NUM_BEAMS}}\n"
                "document = (\n"
                "    'The town council of Millbrook voted on Tuesday evening to approve a three-year plan to replace the '\n"
                "    'aging water mains beneath the historic district. The plan, which had been debated for more than a '\n"
                "    'year, will cost an estimated 4.2 million dollars and is scheduled to begin in the spring. Council '\n"
                "    'members said the decision was driven by a series of pipe failures last winter that left several '\n"
                "    'streets without water for days.\\n\\n'\n"
                "    'Under the approved schedule, crews will work one block at a time so that no more than two streets are '\n"
                "    'closed on any given day. The public works director told residents that most of the disruption would '\n"
                "    'fall in the first eighteen months, with paving and landscaping to follow. Businesses along Main Street '\n"
                "    'will receive advance notice of closures and a dedicated contact for complaints.\\n\\n'\n"
                "    'Funding will come from a combination of a state infrastructure grant and a modest increase in water '\n"
                "    'rates, which the council set at three percent per year for the duration of the project. Two members '\n"
                "    'voted against the rate increase, arguing that the grant alone should have covered the work, but the '\n"
                "    'majority said delaying the project any further would only raise its cost.'\n"
                ")\n"
                "document_id = 'doc'\n"
                "ceilings = {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_INPUT_TOKENS': MAX_INPUT_TOKENS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'MAX_NUM_BEAMS': MAX_NUM_BEAMS, 'LENGTH_PENALTY_RANGE': LENGTH_PENALTY_RANGE, 'MAX_NO_REPEAT_NGRAM_SIZE': MAX_NO_REPEAT_NGRAM_SIZE}}\n"
                "print(ceilings)\n"
                "print({{'defaults_from': GENERATION_CONFIG_FILE, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DEFAULT_MIN_NEW_TOKENS': DEFAULT_MIN_NEW_TOKENS, 'DEFAULT_NUM_BEAMS': DEFAULT_NUM_BEAMS, 'DEFAULT_LENGTH_PENALTY': DEFAULT_LENGTH_PENALTY, 'DEFAULT_NO_REPEAT_NGRAM_SIZE': DEFAULT_NO_REPEAT_NGRAM_SIZE, 'EARLY_STOPPING': EARLY_STOPPING, 'DECISION_RULE': DECISION_RULE}})\n"
                "input_manifest = validate_inputs([document], num_beams=NUM_BEAMS, names=[document_id])\n"
                "try:\n"
                "    validate_inputs([document], num_beams=MAX_NUM_BEAMS + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'num-beams-ceiling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "started = time.perf_counter()\n"
                "result = pipe.summarize(document, num_beams=NUM_BEAMS)\n"
                "summarize_elapsed = time.perf_counter() - started\n"
                "checks = {{\n"
                "    'summary_is_non_empty_text': isinstance(result['summary'], str) and bool(result['summary'].strip()),\n"
                "    'generated_within_bound': 1 <= result['generated_tokens'] <= DEFAULT_MAX_NEW_TOKENS <= MAX_NEW_TOKENS,\n"
                "    'input_within_ceiling': 1 <= result['input_tokens'] <= MAX_INPUT_TOKENS,\n"
                "    'truncated_matches_stopped_by': result['truncated'] == (result['stopped_by'] != 'eos'),\n"
                "    'settings_echoed': result['generation']['max_new_tokens'] == DEFAULT_MAX_NEW_TOKENS and result['generation']['num_beams'] == NUM_BEAMS,\n"
                "    'deterministic_decoding': result['generation']['do_sample'] is False,\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'summarize output failed a sanity check: {{checks}}')\n"
                "print({{key: value for key, value in result.items() if key != 'summary'}})\n"
                "print({{'seconds': round(summarize_elapsed, 3), 'checks': checks, 'no_score': 'the pipeline emits no probability or quality score; truncated is a length flag', 'findings': len(input_manifest['findings'])}})\n"
                "print('summary (pinned news defaults):')\n"
                "print(result['summary'])\n"
                "started = time.perf_counter()\n"
                "tldr_result = pipe.summarize(document, **TLDR)\n"
                "print({{'tldr_settings': TLDR, 'generated_tokens': tldr_result['generated_tokens'], 'stopped_by': tldr_result['stopped_by'], 'seconds': round(time.perf_counter() - started, 3)}})\n"
                "print('summary (TL;DR-length settings):')\n"
                "print(tldr_result['summary'])"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's score on the test split\n\n"
                "Three numbers frame the adaptation, all under the TL;DR-length settings of Section 5. The **Lead-1 "
                "baseline** submits the first sentence of each abstract as its summary and the **Lead-3 baseline** the first "
                "three: what a system that does no modelling gets, and a reminder that ROUGE rewards length matching — "
                "Lead-1 usually beats Lead-3 against one-sentence references. The **frozen model** summarises the 100 test "
                "abstracts and is scored with the same three metrics: corpus **ROUGE-1**, **ROUGE-2** and **ROUGE-L** F1 "
                "(best over the references, rouge-score-style, not rouge-score-identical). Expect the frozen model to land "
                "near the Lead baselines with summaries roughly twice as long as the references — it copies the abstract's "
                "opening sentences in the news register it was trained for — and read `mean_summary_words` and "
                "`hit_token_ceiling` (summaries cut at `max_new_tokens`) before trusting any score. About three minutes on "
                "CPU."
            ),
            "code": (
                "baseline_lead1 = lead_baseline(test_records, n_sentences=1)\n"
                "baseline_lead3 = lead_baseline(test_records, n_sentences=3)\n"
                "for name, baseline in (('lead1_baseline', baseline_lead1), ('lead3_baseline', baseline_lead3)):\n"
                "    print({{name: {{'rouge1': round(baseline['rouge1'], 2), 'rouge2': round(baseline['rouge2'], 2), 'rougeL': round(baseline['rougeL'], 2), 'mean_summary_words': round(baseline['mean_summary_words'], 1), 'n': baseline['n']}}}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, **TLDR)\n"
                "print({{'frozen_model_test': {{'rouge1': round(frozen_test['rouge1'], 2), 'rouge2': round(frozen_test['rouge2'], 2), 'rougeL': round(frozen_test['rougeL'], 2), 'mean_summary_words': round(frozen_test['mean_summary_words'], 1), 'mean_reference_words': round(frozen_test['mean_reference_words'], 1), 'hit_token_ceiling': frozen_test['hit_token_ceiling'], 'n': frozen_test['n'], 'verdict': frozen_test['verdict']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for record in test_records[:2]:\n"
                "    print({{'frozen': pipe.summarize(record['source'], **TLDR)['summary'], 'reference': record['targets'][0]}})\n"
                "assert frozen_test['rouge1'] > 0.0"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last decoder blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_DECODER_LAYERS` decoder blocks — two by default, 33,593,344 of "
                "406,290,432 parameters; the encoder, the shared embeddings, the tied output projection and the earlier "
                "decoder blocks stay frozen — with teacher-forced cross-entropy on the first reference TL;DR, AdamW at a "
                "fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler. Sources are truncated to "
                "512 and targets to 64 BPE tokens **during training only**. Epoch 0 records the frozen model's validation "
                "ROUGE under the TL;DR settings; every epoch is scored on the validation split the same way, and the epoch "
                "with the highest validation ROUGE-L is kept.\n\n"
                "Watch validation ROUGE-L rise by several points and `mean_summary_words` fall towards the reference length "
                "within the first epoch (about a minute of training plus a validation pass per epoch on CPU). The build "
                "record's counter-example: one epoch over 200 abstracts already captured most of the gain."
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 3e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_rouge1'] = round(entry['val']['rouge1'], 2)\n"
                "        row['val_rouge2'] = round(entry['val']['rouge2'], 2)\n"
                "        row['val_rougeL'] = round(entry['val']['rougeL'], 2)\n"
                "        row['val_mean_summary_words'] = round(entry['val']['mean_summary_words'], 1)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, eval_generation=TLDR, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and no abstract in it appears in the "
                "training or validation splits. The adapted model is scored exactly as the frozen model was in Section 6, "
                "and the four numbers are put side by side. Look for a ROUGE-L gain of several points over the frozen model "
                "and over both Lead baselines, and for `mean_summary_words` close to the reference length — the cell asserts "
                "the adapted ROUGE-L is above the frozen ROUGE-L — and for the same two abstracts summarised by the adapted "
                "model. One hundred abstracts from one seeded split of one corpus give no dispersion estimate; the deltas "
                "are sample-sanity evidence that the adaptation contract works, not a benchmark, and a register shift on "
                "paper abstracts says nothing about your documents until you measure it there. Note what the adaptation "
                "also changes: the model now writes one short sentence, so its long news-highlight behaviour is traded "
                "away in the adapter."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, **TLDR)\n"
                "adapted_val = pipe.evaluate(val_records, **TLDR)\n"
                "comparison = {{\n"
                "    metric: {{'lead1': round(baseline_lead1[metric], 2), 'lead3': round(baseline_lead3[metric], 2), 'frozen': round(frozen_test[metric], 2), 'adapted': round(adapted_test[metric], 2)}}\n"
                "    for metric in ('rouge1', 'rouge2', 'rougeL', 'mean_summary_words')\n"
                "}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 2) for metric in ('rouge1', 'rouge2', 'rougeL')}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "for record in test_records[:2]:\n"
                "    print({{'adapted': pipe.summarize(record['source'], **TLDR)['summary'], 'reference': record['targets'][0]}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'generation': frozen_test['generation'],\n"
                "    'baselines': {{'lead1': baseline_lead1, 'lead3': baseline_lead3}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['rougeL'] > frozen_test['rougeL']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Summarise new abstracts, export the adapter and reload it\n\n"
                "Four abstracts that were in none of the splits are summarised by the adapted model through the same "
                "`summarize` contract as Section 5 and scored with `pipe.evaluate` (a `measured-small-sample` verdict, "
                "because four documents carry no dispersion estimate); the single-document `evaluation_report` helper — "
                "the inference-stage helper, which now scores supplied references as `sample-sanity` — is written for the "
                "first of them.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last two decoder blocks, about 134 MB — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "training configuration and the epoch history (OUT8). `BARTSummarizationPipeline.from_artifact` re-verifies "
                "the base snapshot, checks the artifact manifest and digest **before** deserialising, refuses any tensor "
                "that is not an adaptable decoder tensor, and overlays the tensors onto a freshly loaded base — a new object "
                "from files, not the in-memory model (VER2). The cell asserts identical summaries (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "if USE_BYOD:\n"
                "    new_records = [{{**r, 'id': f'new-{{i:02d}}'}} for i, r in enumerate(test_records[:4])]\n"
                "else:\n"
                "    used = {{r['source'].lower() for part in splits.values() for r in part}}\n"
                "    new_records = [{{**r, 'id': f'new-{{i:02d}}'}} for i, r in enumerate([r for r in filter_records(corpus['dev']) if r['source'].lower() not in used][:4])]\n"
                "new_metrics = pipe.evaluate(new_records, **TLDR)\n"
                "new_results = []\n"
                "for record in new_records:\n"
                "    item = pipe.summarize(record['source'], **TLDR)\n"
                "    new_results.append({{'id': record['id'], 'summary': item['summary'], 'reference': record['targets'][0], 'generated_tokens': item['generated_tokens'], 'input_tokens': item['input_tokens'], 'truncated': item['truncated'], 'stopped_by': item['stopped_by']}})\n"
                "    print({{k: new_results[-1][k] for k in ('id', 'summary', 'reference')}})\n"
                "single_report = evaluation_report(pipe.summarize(new_records[0]['source'], **TLDR), new_records[0]['targets'], sample_kind='one unseen SciTLDR abstract' if not USE_BYOD else 'one BYOD test record')\n"
                "print({{'new_abstracts': {{'n': new_metrics['n'], 'rouge1': round(new_metrics['rouge1'], 2), 'rougeL': round(new_metrics['rougeL'], 2), 'verdict': new_metrics['verdict']}}, 'single_document_report_verdict': single_report['verdict']}})\n"
                "with open('outputs/{stem}_summaries.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(new_results[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(new_results)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = BARTSummarizationPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [pipe.summarize(r['source'], **TLDR)['summary'] for r in test_records[:4]]\n"
                "after = [reloaded.summarize(r['source'], **TLDR)['summary'] for r in test_records[:4]]\n"
                "parity = {{'identical_summaries': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_summaries'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'files': {{k: {{'name': v[0], 'bytes': v[1], 'sha256': v[2]}} for k, v in CORPUS_FILES.items()}}, 'license': CORPUS_LICENSE}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'pinned_defaults_result': {{k: v for k, v in result.items() if k != 'summary'}}, 'tldr_settings': TLDR}},\n"
                "    'comparison': comparison,\n"
                "    'new_abstracts': new_metrics,\n"
                "    'single_document_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen news summariser, asked for a TL;DR of a paper abstract, writes something close to the abstract's own "
        "opening sentences at twice the reference length and scores near the Lead baselines; a bounded fine-tuning of the "
        "last two decoder blocks on 300 in-domain abstracts moves it to the one-sentence register and lifts held-out ROUGE-L "
        "by several points in a few minutes on CPU, with a 134 MB adapter that reloads to identical summaries. That is the "
        "claim: the adaptation contract works end to end on a real referenced corpus, and the numbers it produces are read "
        "against two Lead baselines and the frozen model rather than in isolation.\n\n"
        "The test split is 100 abstracts from one seeded split of one corpus, the metrics are three n-gram overlap scores "
        "(own implementation, not rouge-score-identical, and none a judgement of faithfulness), and SciTLDR references are "
        "short and formulaic. So a gain here says the contract works, not that the adapted model is better on your "
        "documents, that it handles long or technical text, or that its summaries are faithful — an abstractive summary can "
        "state a result the source does not and still overlap the reference well. Fine-tuning on a narrow corpus also "
        "changes the model elsewhere — the adapter writes one short sentence for any input — and nothing here measures "
        "that.\n\n"
        "Three things to carry to real data. **References first:** the Lead baselines and the frozen model's score on *your* "
        "references, under *your* length settings, are the numbers to read before any adapted one — ROUGE moves with summary "
        "length as much as with content. **Leakage:** de-duplicate sources across splits (the contract does this "
        "case-insensitively) and split by document collection or author when your pairs come from one. **Ceilings:** inputs "
        "over `MAX_INPUT_TOKENS` are refused at inference and truncated to 512 tokens only during training — long-document "
        "summarization is out of scope.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real referenced corpus, "
        "validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded "
        "fine-tuning, evaluate against two trivial baselines and the frozen model on an independent split, and emit the "
        "shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, summary quality or faithfulness on any other domain, a usable acceptance threshold, or production "
        "fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_DECODER_LAYERS = 1` and compare the "
        "artifact size and the test scores; raise `TLDR_MAX_NEW_TOKENS` and watch ROUGE fall as summaries lengthen; set "
        "`NUM_BEAMS = 1` and read the greedy scores; or bring your own document–summary pairs through BYOD and read the Lead "
        "baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/fairseq/tree/main/examples/bart\n"
        "- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (Lewis et al., 2019): https://arxiv.org/abs/1910.13461\n"
        "- TLDR: Extreme Summarization of Scientific Documents (Cachola et al., EMNLP Findings 2020; SciTLDR, Apache-2.0): https://arxiv.org/abs/2004.15011\n"
        "- ROUGE: A Package for Automatic Evaluation of Summaries (Lin, 2004): https://aclanthology.org/W04-1013\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}

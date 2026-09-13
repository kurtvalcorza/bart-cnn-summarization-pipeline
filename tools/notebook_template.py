"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "bart_summarization_pipeline",
    "repo_name": "bart-cnn-summarization-pipeline",
    "stem": "bart_summarization",
    "notebook_name": "bart_summarization_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "BARTSummarizationPipeline",
    "weights_key": "bart-large-cnn",
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "BART-large CNN — DIMER abstractive summarization tutorial (standalone)",
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
    "capability": "abstractive summarization of one English document by deterministic beam search (pinned generation defaults; token counts and a truncation flag reported; no score) using the pinned BART-large CNN weights",
    "intro": (
        "At inference the byte-level BPE tokenizer encodes the whole document once, the 12-layer bidirectional encoder of "
        "the 406 M-parameter BART-large reads it, and the 12-layer autoregressive decoder — fine-tuned upstream on "
        "CNN/DailyMail article/highlight pairs — writes a summary token by token under **deterministic beam search**: "
        "`num_beams` 4, `length_penalty` 2.0, `no_repeat_ngram_size` 3, `early_stopping`, and the length bounds of the "
        "snapshot's `generation_config_for_summarization.json` (`max_length` 142 / `min_length` 56, exposed as "
        "`DEFAULT_MAX_NEW_TOKENS` 141 / `DEFAULT_MIN_NEW_TOKENS` 55 because the upstream bounds count the decoder start "
        "token). **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting — "
        "the pinned checkpoint is used as published. What the upstream checkpoint supplies is the encoder-decoder, the "
        "language-model head, the generation config and the tokenizer; what the carried pipeline module adds is manifest "
        "verification, input and generation-setting validation against named ceilings (over-long documents are rejected, "
        "not truncated or chunked), a fixed output contract that reports `generated_tokens`, `input_tokens`, `truncated` "
        "and `stopped_by`, and the `validate_inputs` and `evaluation_report` stage helpers. **The pipeline emits no score, "
        "probability or quality metric** — a summary is free text, and the repository ships no ROUGE helper."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author one synthetic three-paragraph "
        "passage (or upload your own document), stage and digest-verify the immutable upstream snapshot, surface the "
        "pipeline's ceilings and the pinned generation defaults and validate the input into one input manifest, run "
        "`summarize` with the pinned defaults and once more with a shorter length bound and read the token counts and "
        "truncation flag correctly, read from the machine-readable evaluation report why no metric is reported and what "
        "reference data would make the task measurable, and export the summaries alongside their settings plus provenance."
    ),
    "exclusions": (
        "extractive summarization with guaranteed source spans, multi-document or long-document (chunked) summarization, "
        "headline generation, sampling-based or diverse decoding, fine-tuning, classification or question answering (the "
        "`bart-mnli-zero-shot-classification-pipeline` sibling covers zero-shot classification), non-English text, or any "
        "faithfulness or quality score. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (also float32; the pipeline loads the checkpoint in float32 on both). This is a 406 M-parameter model: the model card's CPU smoke loaded and verified the 1.63 GB snapshot in 6.72 s and summarised a 232-token passage with 4 beams in 4.28 s, so the default path runs in about a minute on a hosted CPU runtime once the ~1.6 GB `model.safetensors` download has finished. The pinned `torch==2.14.0` install and that download are the largest transfers of the run; allow ~2 GB of free RAM for the weights.",
        "- **Knowledge:** basic Python; what beam search over a decoder does and why it is deterministic; why an abstractive summary can contain statements the source does not (hallucination) and why ROUGE against references is needed to measure quality.",
        "- **Data:** the default sample is one synthetic three-paragraph passage authored in code (a fictional town-council report), so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 text file holding the document to summarise (at most `MAX_TEXT_CHARS` characters and `MAX_INPUT_TOKENS` BPE tokens; longer documents are rejected, not truncated). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**, written in this cell: a fictional three-paragraph report about a town "
                "council approving a water-mains project (the model card's smoke passage), given the identifier `doc`. It "
                "is prose in the expository register the CNN/DailyMail fine-tune expects, but it is not news, carries no "
                "reference summary, and proves nothing about quality — a run on it is a plumbing check, never benchmark "
                "evidence. `SUMMARY_MAX_NEW_TOKENS` and `NUM_BEAMS` are Colab form parameters checked against the carried module "
                "in Section 5; their defaults are the pinned generation config.\n\n"
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 text file holding the document to "
                "summarise, at most `MAX_TEXT_CHARS` characters and at most `MAX_INPUT_TOKENS` BPE tokens including "
                "`<s>`/`</s>` (a longer document is rejected by the pipeline, not truncated or chunked). The upload stays "
                "inside this runtime. If you also hold a reference summary, keep it outside the notebook — Section 7 "
                "explains what to compute with it."
            ),
            "code": (
                "import hashlib\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SUMMARY_MAX_NEW_TOKENS = 141  # @param {{type:\"integer\"}}\n"
                "NUM_BEAMS = 4  # @param {{type:\"integer\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    document = uploaded[sample_name].decode('utf-8').strip()\n"
                "    if not document:\n"
                "        raise ValueError(f'{{sample_name}}: the uploaded file is empty')\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    document = (\n"
                "        'The town council of Millbrook voted on Tuesday evening to approve a three-year plan to replace the '\n"
                "        'aging water mains beneath the historic district. The plan, which had been debated for more than a '\n"
                "        'year, will cost an estimated 4.2 million dollars and is scheduled to begin in the spring. Council '\n"
                "        'members said the decision was driven by a series of pipe failures last winter that left several '\n"
                "        'streets without water for days.\\n\\n'\n"
                "        'Under the approved schedule, crews will work one block at a time so that no more than two streets are '\n"
                "        'closed on any given day. The public works director told residents that most of the disruption would '\n"
                "        'fall in the first eighteen months, with paving and landscaping to follow. Businesses along Main Street '\n"
                "        'will receive advance notice of closures and a dedicated contact for complaints.\\n\\n'\n"
                "        'Funding will come from a combination of a state infrastructure grant and a modest increase in water '\n"
                "        'rates, which the council set at three percent per year for the duration of the project. Two members '\n"
                "        'voted against the rate increase, arguing that the grant alone should have covered the work, but the '\n"
                "        'majority said delaying the project any further would only raise its cost.'\n"
                "    )\n"
                "    sample_name = 'synthetic_millbrook_council_report'\n"
                "    sample_kind = 'synthetic (authored in this cell; the model card smoke passage)'\n"
                "document_id = 'doc'\n"
                "sample_sha256 = hashlib.sha256(document.encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'chars': len(document), 'paragraphs': len(document.split('\\n\\n')), 'max_new_tokens': SUMMARY_MAX_NEW_TOKENS, 'num_beams': NUM_BEAMS, 'text_sha256': sample_sha256}})\n"
                "print(document[:300] + (' …' if len(document) > 300 else ''))"
            ),
        },
        {
            "md": (
                "## 5. Validate the input → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it takes the batch of documents `summarize` "
                "will be called on (one here) and the generation settings, and runs the method's own checks — "
                "`_check_text` and `_check_settings` — so a rejection here is a rejection there. `MAX_TEXT_CHARS` is the "
                "character guard applied before tokenisation; `MAX_INPUT_TOKENS` (1024, the checkpoint's position limit) "
                "is applied after tokenisation and **rejects** longer documents rather than truncating or chunking them, so "
                "it is enforced inside the pipeline and cannot be observed at this stage; `MAX_NEW_TOKENS`, `MAX_NUM_BEAMS`, "
                "`LENGTH_PENALTY_RANGE` and `MAX_NO_REPEAT_NGRAM_SIZE` bound the generation settings, whose defaults "
                "(`DEFAULT_*`) are the pinned `generation_config_for_summarization.json`; `DECISION_RULE` names the "
                "decoding rule. The manifest records the exact settings that will be used and is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates with "
                "`num_beams` one above the ceiling and records the pipeline's own error message as a finding. The notebook "
                "never trims or alters the document."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_INPUT_TOKENS': MAX_INPUT_TOKENS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'MAX_NUM_BEAMS': MAX_NUM_BEAMS, 'LENGTH_PENALTY_RANGE': LENGTH_PENALTY_RANGE, 'MAX_NO_REPEAT_NGRAM_SIZE': MAX_NO_REPEAT_NGRAM_SIZE}}\n"
                "print(ceilings)\n"
                "print({{'defaults_from': GENERATION_CONFIG_FILE, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DEFAULT_MIN_NEW_TOKENS': DEFAULT_MIN_NEW_TOKENS, 'DEFAULT_NUM_BEAMS': DEFAULT_NUM_BEAMS, 'DEFAULT_LENGTH_PENALTY': DEFAULT_LENGTH_PENALTY, 'DEFAULT_NO_REPEAT_NGRAM_SIZE': DEFAULT_NO_REPEAT_NGRAM_SIZE, 'EARLY_STOPPING': EARLY_STOPPING, 'DECISION_RULE': DECISION_RULE}})\n"
                "input_manifest = validate_inputs([document], max_new_tokens=SUMMARY_MAX_NEW_TOKENS, num_beams=NUM_BEAMS, names=[document_id])\n"
                "# Demonstrate a ceiling rejection; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs([document], max_new_tokens=SUMMARY_MAX_NEW_TOKENS, num_beams=MAX_NUM_BEAMS + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'num-beams-ceiling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "print({{'token_ceiling': f'MAX_INPUT_TOKENS={{MAX_INPUT_TOKENS}} is checked by the pipeline after tokenisation and rejects, never truncates or chunks'}})"
            ),
        },
        {
            "md": (
                "## 6. Summarize and read the output correctly\n\n"
                "**Input/output contract.** `summarize(text, max_new_tokens=..., min_new_tokens=..., num_beams=..., "
                "length_penalty=..., no_repeat_ngram_size=...)` takes one string and returns `summary` (free text), "
                "`generated_tokens` (content tokens emitted, excluding `<s>`/`</s>`/padding), `input_tokens` (encoder "
                "tokens including `<s>`/`</s>`), `truncated` (`True` when the decoder hit `max_new_tokens` before emitting "
                "end-of-sequence), `stopped_by` (`eos` or `max_new_tokens`), the full `generation` settings with the "
                "`decision_rule`, the device and the model identity. **Decoding semantics:** deterministic beam search — "
                "`num_beams` partial summaries are kept at each step and the finished beam with the highest "
                "length-penalised log-probability is returned; `do_sample` is fixed to `False`, so the same document on "
                "the same device gives the same summary and no seed is needed; `min_new_tokens` is a floor that forbids "
                "end-of-sequence before that many tokens, which on a short input forces the model to keep writing. **No "
                "score is emitted:** nothing in the result says whether the summary is faithful or good, and the pipeline "
                "applies no threshold. The model card's CPU smoke on this passage produced 81 tokens, stopped by `eos`, "
                "reproducing the first two source sentences verbatim and paraphrasing the third — one observation of the "
                "model's largely extractive behaviour on short expository input, not an expected value; CPU and CUDA "
                "kernels can pick different beams when two are close. The cell runs the pinned defaults and then a "
                "shorter bound (`max_new_tokens` 60, `min_new_tokens` 20) so the effect of the length settings is visible."
            ),
            "code": (
                "import time\n\n"
                "started = time.perf_counter()\n"
                "result = pipe.summarize(document, max_new_tokens=SUMMARY_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "summarize_elapsed = time.perf_counter() - started\n"
                "checks = {{\n"
                "    'summary_is_non_empty_text': isinstance(result['summary'], str) and bool(result['summary'].strip()),\n"
                "    'generated_within_bound': 1 <= result['generated_tokens'] <= SUMMARY_MAX_NEW_TOKENS <= MAX_NEW_TOKENS,\n"
                "    'input_within_ceiling': 1 <= result['input_tokens'] <= MAX_INPUT_TOKENS,\n"
                "    'truncated_matches_stopped_by': result['truncated'] == (result['stopped_by'] != 'eos'),\n"
                "    'settings_echoed': result['generation']['max_new_tokens'] == SUMMARY_MAX_NEW_TOKENS and result['generation']['num_beams'] == NUM_BEAMS,\n"
                "    'deterministic_decoding': result['generation']['do_sample'] is False,\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'summarize output failed a sanity check: {{checks}}')\n"
                "print({{key: value for key, value in result.items() if key != 'summary'}})\n"
                "print({{'seconds': round(summarize_elapsed, 3), 'checks': checks, 'no_score': 'the pipeline emits no probability or quality score; truncated is a length flag'}})\n"
                "print('summary (pinned defaults):')\n"
                "print(result['summary'])\n"
                "started = time.perf_counter()\n"
                "short_result = pipe.summarize(document, max_new_tokens=60, min_new_tokens=20, num_beams=NUM_BEAMS)\n"
                "short_elapsed = time.perf_counter() - started\n"
                "print({{'short_run': {{key: short_result[key] for key in ('generated_tokens', 'truncated', 'stopped_by')}}, 'seconds': round(short_elapsed, 3)}})\n"
                "print('summary (max_new_tokens=60, min_new_tokens=20):')\n"
                "print(short_result['summary'])"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report — here, one that "
                "says nothing is measurable. The repository ships **no ROUGE or other metric helper and reports no "
                "performance measure**, so the verdict is always `not-measurable` and the report states what would make "
                "the task measurable: one or more reference summaries per document from the deployment domain over enough "
                "documents to state a dispersion, scored with the caller's own ROUGE-1/2/L implementation, plus a "
                "faithfulness check against the source (ROUGE rewards overlap, not truth), excluding or re-running outputs "
                "whose `truncated` flag is true. Supplying a reference does **not** change the verdict — one reference is "
                "not a dispersion and no scorer is shipped — so the helper records that in `reason` instead. The upstream "
                "card's ROUGE numbers on CNN/DailyMail are upstream-reported and nothing here reproduces them. The sanity "
                "checks printed in Section 6 remain falsifiable plumbing checks, not results. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, None, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: the repository ships no ROUGE helper and the sample carries no reference summary; score your own references with your own ROUGE implementation and check faithfulness against the source.')"
            ),
        },
        {
            "md": (
                "## 8. Export summaries alongside their settings, and provenance\n\n"
                "Two further files are written under `outputs/` beside the input manifest and the evaluation report. The "
                "summaries go to CSV (`outputs/{stem}_summaries.csv`) with one row per run — `document_id`, `run`, "
                "`max_new_tokens`, `min_new_tokens`, `num_beams`, `length_penalty`, `no_repeat_ngram_size`, "
                "`generated_tokens`, `input_tokens`, `truncated`, `stopped_by`, `summary` — so every summary stays attached "
                "to the settings that produced it. One JSON record (`outputs/{stem}_result.json`) preserves both results in "
                "full (summary, counts, flags, generation settings, decision rule, seconds), the sanity checks, the ceilings "
                "in force, the input manifest, the evaluation report, the sample identity and digest, the notebook's source "
                "(repository, revision, embedded module digest, generator), the model identifier, the immutable model "
                "revision, the model licence, the verified snapshot summary, and the runtime identity (Python, `torch`, "
                "`transformers`, device, dtype). No credentials are involved in any step, so none can reach the export."
            ),
            "code": (
                "import csv\n\n"
                "runs = [('pinned-defaults', result, summarize_elapsed), ('short-bound', short_result, short_elapsed)]\n"
                "with open('outputs/{stem}_summaries.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['document_id', 'run', 'max_new_tokens', 'min_new_tokens', 'num_beams', 'length_penalty', 'no_repeat_ngram_size', 'generated_tokens', 'input_tokens', 'truncated', 'stopped_by', 'summary'])\n"
                "    for run_name, run_result, _ in runs:\n"
                "        generation = run_result['generation']\n"
                "        writer.writerow([document_id, run_name, generation['max_new_tokens'], generation['min_new_tokens'], generation['num_beams'], generation['length_penalty'], generation['no_repeat_ngram_size'], run_result['generated_tokens'], run_result['input_tokens'], run_result['truncated'], run_result['stopped_by'], run_result['summary']])\n"
                "payload = {{\n"
                "    'document_id': document_id,\n"
                "    'runs': [\n"
                "        {{'run': run_name, 'seconds': round(seconds, 3), **{{key: value for key, value in run_result.items() if key not in ('device', 'source', 'model_id', 'model_revision')}}}}\n"
                "        for run_name, run_result, seconds in runs\n"
                "    ],\n"
                "    'decoding': 'deterministic beam search (do_sample=False); no score, probability or quality metric is emitted; truncated is a length flag; no threshold shipped',\n"
                "    'sanity_checks': checks,\n"
                "    'summaries_file': 'outputs/{stem}_summaries.csv',\n"
                "    'ceilings': ceilings,\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'chars': len(document), 'text_sha256': sample_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': snapshot['path'], 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes')}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The summaries are the output of deterministic beam search over a decoder fine-tuned on news highlights: the "
        "highest length-penalised beam is returned, some token is emitted at every step, `min_new_tokens` forces a "
        "minimum length, and nothing in the result says whether a sentence is faithful to the source — an abstractive "
        "summarizer can state what the document does not and drop what matters most. The pipeline emits no score and "
        "applies no threshold; the caller owns any acceptance rule and must set it on reference summaries from their own "
        "domain. On the synthetic sample the run is plumbing evidence only; the evaluation report is `not-measurable` "
        "because no metric can be computed without references and a scorer, and a real evaluation needs ROUGE against "
        "references over enough documents to state a dispersion plus a faithfulness check the repository does not ship. "
        "Documents over 1024 BPE tokens are rejected, not truncated or chunked; the checkpoint is English only, writes in "
        "the news-highlight register whatever the input, and carries whatever associations CNN/DailyMail and the BART "
        "pre-training corpus contain, which neither the upstream card nor this repository has audited; the pipeline "
        "exposes no sampling, no fine-tuning and no logits. Inference is deterministic on a fixed device and dtype, but "
        "CPU and CUDA kernels can pick different beams when two are close.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot, validate the demonstrated input and settings against the "
        "enforced ceilings, execute the public `summarize` path under two length bounds, and emit the shown "
        "machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, summary quality or faithfulness on any domain, a usable acceptance rule, safety "
        "for high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments.** Lower `SUMMARY_MAX_NEW_TOKENS` until `truncated` turns true and read how the summary ends mid-thought; "
        "set `NUM_BEAMS` to 1 (greedy) and compare the wording; paste a document whose facts you know and count the "
        "statements the summary makes that the source does not — that is the faithfulness check the evaluation report "
        "asks for; assemble a dozen documents with reference summaries of your own and compute ROUGE with a bootstrap "
        "interval. None of these turns the sample result into evidence of production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/bart-cnn-summarization-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/fairseq/tree/main/examples/bart\n"
        "- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension: https://arxiv.org/abs/1910.13461\n"
        "- Get To The Point: Summarization with Pointer-Generator Networks (the CNN/DailyMail summarization split): https://arxiv.org/abs/1704.04368"
    ),
}

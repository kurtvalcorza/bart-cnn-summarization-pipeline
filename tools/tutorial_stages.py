"""Stage runner for the standalone BART-large CNN summarization tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated environment).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage data [--byod PATH] [--split-seed 42] ...

Nothing is installed into the notebook kernel, so a hosted runtime's preloaded packages are never replaced and no
restart is needed. Each stage is a separate process and starts from files only: the verified snapshot and the pinned
corpus cache under ``--weights``, the splits and settings written by ``data`` and ``inference``, the adapter artifact
written by ``adapt`` and the JSON records of earlier stages. Every stage that runs the model builds it from the pinned
files, so a stage re-run after a change always starts from the frozen model and never from an earlier adaptation
(BART-M3). Learner-facing exports go to ``RUN_DIR/outputs``; hand-off state goes to ``RUN_DIR/state``. On failure a
stage writes ``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook re-raises.

Stages: weights → runtime → data → inference → frozen → adapt → evaluate → new → reload → bundle, plus the optional
``activity``. The package API does the work; this runner only sequences it and adds the tutorial's own checks (the
workflow minimum and the token ceiling of a dataset at Section 5, the count-based truncation check, the default-path
expectation, the decoder-tensor parity of the reload, the activity).
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

STEM = "bart_summarization"
SNAPSHOT_KEY = "bart-large-cnn"
CORPUS_CACHE = "scitldr"
LOCK = "requirements.txt"
ADAPTER_DIR = f"{STEM}_adapter"
N_NEW = 4  # documents summarised in Section 10
N_PARITY = 4  # test documents whose summaries the reload must reproduce in Section 11
VAL_FRACTION = 0.15  # BYOD split fractions: the package's split_dataset defaults
TEST_FRACTION = 0.2
DEFAULT_DATA = {"source": "sample", "split_seed": 42}
DEFAULT_GENERATION = {"max_new_tokens": 48, "min_new_tokens": 0, "num_beams": 4}
DEFAULT_TRAINING = {"epochs": 2, "learning_rate": 3e-5, "batch_size": 8, "trainable_decoder_layers": 2}
VERSION_CHECKS = ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub", "numpy")
DOCUMENT = (
    "The town council of Millbrook voted on Tuesday evening to approve a three-year plan to replace the "
    "aging water mains beneath the historic district. The plan, which had been debated for more than a "
    "year, will cost an estimated 4.2 million dollars and is scheduled to begin in the spring. Council "
    "members said the decision was driven by a series of pipe failures last winter that left several "
    "streets without water for days.\n\n"
    "Under the approved schedule, crews will work one block at a time so that no more than two streets are "
    "closed on any given day. The public works director told residents that most of the disruption would "
    "fall in the first eighteen months, with paving and landscaping to follow. Businesses along Main Street "
    "will receive advance notice of closures and a dedicated contact for complaints.\n\n"
    "Funding will come from a combination of a state infrastructure grant and a modest increase in water "
    "rates, which the council set at three percent per year for the duration of the project. Two members "
    "voted against the rate increase, arguing that the grant alone should have covered the work, but the "
    "majority said delaying the project any further would only raise its cost."
)
# Files each stage writes. A stage first removes its own files and those of every stage that reads them
# (DEPENDANTS), so no record from an earlier configuration can be read as a result of the current one.
STAGE_FILES: dict[str, tuple[str, ...]] = {
    "data": ("data.json", "splits.json", f"{STEM}_dataset_manifest.json", f"{STEM}_train.csv"),
    "inference": ("generation.json", f"{STEM}_input_manifest.json", f"{STEM}_inference.json"),
    "frozen": (f"{STEM}_frozen_test.json",),
    "adapt": ("adapt.json", "reference.json", f"{STEM}_training_history.json", ADAPTER_DIR),
    "evaluate": (f"{STEM}_evaluation_report.json",),
    "new": (f"{STEM}_new_documents.json", f"{STEM}_summaries.csv"),
    "reload": (f"{STEM}_reload_parity.json",),
    "bundle": (f"{STEM}_result.json",),
    "activity": ("activity",),
}
DEPENDANTS: dict[str, tuple[str, ...]] = {
    "data": tuple(STAGE_FILES),
    "inference": ("inference", "frozen", "adapt", "evaluate", "new", "reload", "bundle", "activity"),
    "frozen": ("frozen", "evaluate", "bundle", "activity"),
    "adapt": ("adapt", "evaluate", "new", "reload", "bundle", "activity"),
    "evaluate": ("evaluate", "bundle", "activity"),
    "new": ("new", "bundle"),
    "reload": ("reload", "bundle"),
    "bundle": ("bundle",),
}


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``, the snapshot and corpus cache under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.out = root / "outputs"
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot(self) -> Path:
        return self.weights / SNAPSHOT_KEY

    @property
    def corpus_cache(self) -> Path:
        return self.weights / CORPUS_CACHE

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 5, or from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def read_output(self, name: str, needed_by: str) -> Any:
        path = self.out / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 5, or from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def invalidate(self, stage: str) -> list[str]:
        """Remove the files of ``stage`` and of every stage that depends on them (state and outputs)."""
        removed = []
        for later in DEPENDANTS[stage]:
            for name in STAGE_FILES[later]:
                for base in (self.state, self.out):
                    path = base / name
                    if path.is_dir():
                        shutil.rmtree(path)
                        removed.append(name)
                    elif path.is_file():
                        path.unlink()
                        removed.append(name)
        return removed


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_versions(text: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", text, re.M)}


def device_name() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def rounded(metrics: Mapping[str, Any], keys: Sequence[str] = ("rouge1", "rouge2", "rougeL", "mean_summary_words")) -> dict[str, Any]:
    return {k: round(float(metrics[k]), 2) for k in keys}


# --------------------------------------------------------------------------------------------------
# the dataset contract of the workflow (BART-M4, BART-m5): minimum, token ceiling, de-duplication
# --------------------------------------------------------------------------------------------------


def split_sizes(n_unique: int, val_fraction: float = VAL_FRACTION, test_fraction: float = TEST_FRACTION) -> dict[str, int]:
    """The split sizes ``split_dataset`` produces for ``n_unique`` distinct sources (same arithmetic)."""
    n_test = max(1, round(n_unique * test_fraction))
    n_val = round(n_unique * val_fraction)
    return {"train": n_unique - n_test - n_val, "validation": n_val, "test": n_test}


def workflow_minimum(min_train: int, val_fraction: float = VAL_FRACTION, test_fraction: float = TEST_FRACTION) -> int:
    """The smallest number of distinct sources whose split gives at least ``min_train`` training records and at least
    one validation and one test record (the validation record selects the epoch; the test records are scored)."""
    n = 1
    while True:
        sizes = split_sizes(n, val_fraction, test_fraction)
        if sizes["train"] >= min_train and sizes["validation"] >= 1 and sizes["test"] >= 1:
            return n
        n += 1


def prepare_byod(records: Sequence[Mapping[str, Any]], seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Validate a BYOD dataset against the workflow before any model runs: the record contract, de-duplication (the
    number of dropped duplicate sources is reported), the workflow minimum and the seeded source-disjoint split.

    Raises ``ValueError`` naming the count, the split and the minimum, so a dataset that the package accepts but the
    workflow cannot complete is refused here, in Section 5, and not inside a later stage."""
    from bart_summarization_pipeline import MIN_RECORDS, split_dataset, validate_dataset

    checked = validate_dataset(records)["records"]
    unique = len({r["source"].lower() for r in checked})
    dropped = len(checked) - unique
    minimum = workflow_minimum(MIN_RECORDS)
    sizes = split_sizes(unique)
    if unique < minimum:
        raise ValueError(
            f"{len(checked)} records, {unique} with distinct sources ({dropped} duplicate sources dropped): the split "
            f"would give train {sizes['train']} / validation {sizes['validation']} / test {sizes['test']}, but the "
            f"workflow needs at least {MIN_RECORDS} training records and 1 validation record, i.e. at least {minimum} "
            f"records with distinct sources. Add records, then re-run from Section 5."
        )
    splits = split_dataset(checked, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=seed)
    report = {"records": len(checked), "distinct_sources": unique, "dropped_duplicate_sources": dropped, "workflow_minimum": minimum, "split_fractions": {"validation": VAL_FRACTION, "test": TEST_FRACTION}}
    return splits, report


def check_token_ceilings(splits: Mapping[str, Sequence[Mapping[str, Any]]], count_tokens: Callable[[str], int]) -> dict[str, Any]:
    """Count encoder tokens for every source. A validation, test or new source over ``MAX_INPUT_TOKENS`` would fail
    inside the model at Section 7, 8, 9 or 10, so it is refused here naming the split and the id; a training source over
    ``MAX_TRAIN_SOURCE_TOKENS`` is truncated during training only, which is reported, not refused."""
    from bart_summarization_pipeline import MAX_INPUT_TOKENS, MAX_TRAIN_SOURCE_TOKENS

    report: dict[str, Any] = {"MAX_INPUT_TOKENS": MAX_INPUT_TOKENS, "MAX_TRAIN_SOURCE_TOKENS": MAX_TRAIN_SOURCE_TOKENS}
    for name, records in splits.items():
        counts = {r["id"]: count_tokens(r["source"]) for r in records}
        if not counts:
            continue
        report[name] = {"max_tokens": max(counts.values())}
        if name == "train":
            report[name]["truncated_during_training"] = sum(n > MAX_TRAIN_SOURCE_TOKENS for n in counts.values())
            continue
        over = sorted(i for i, n in counts.items() if n > MAX_INPUT_TOKENS)
        if over:
            raise ValueError(f"{name} record {over[0]!r} has {counts[over[0]]} tokens; MAX_INPUT_TOKENS is {MAX_INPUT_TOKENS} and inference never truncates. Shorten or remove it, then re-run from Section 5 ({len(over)} {name} record(s) are over the ceiling)")
    return report


def forced_special_tokens(snapshot: Path) -> int:
    """How many of the ``max_new_tokens`` steps the pinned config spends on forced tokens (BOS first, EOS at the cap)."""
    config = {}
    for name in ("config.json", "generation_config.json"):  # generate() reads the generation config; it wins
        path = snapshot / name
        if path.is_file():
            config.update(json.loads(path.read_text(encoding="utf-8")))
    return int(config.get("forced_bos_token_id") is not None) + int(config.get("forced_eos_token_id") is not None)


def is_default(data: Mapping[str, Any], generation: Mapping[str, Any], training: Mapping[str, Any]) -> bool:
    """The default configuration, on which the notebook's stated expectations were measured."""
    return {"source": data["source"], "split_seed": data["split_seed"]} == DEFAULT_DATA and dict(generation) == DEFAULT_GENERATION and dict(training) == DEFAULT_TRAINING


# --------------------------------------------------------------------------------------------------
# loading: every model is built from the verified files (BART-M3)
# --------------------------------------------------------------------------------------------------


def frozen_pipeline(run: Run) -> Any:
    """A new pipeline from the verified snapshot: the pinned base model with no adaptation."""
    from bart_summarization_pipeline import BARTSummarizationPipeline

    pipe = BARTSummarizationPipeline.from_pretrained(weights_dir=run.snapshot, device=device_name())
    if pipe.adapter is not None:
        raise RuntimeError("a freshly built pipeline is not the frozen model; refusing to continue")
    return pipe


def adapted_pipeline(run: Run, stage: str) -> tuple[Any, dict[str, Any]]:
    """A new pipeline rebuilt from the base files and the exported adapter, checked against the current data and settings."""
    from bart_summarization_pipeline import BARTSummarizationPipeline

    adapt = run.read_state("adapt.json", stage)
    data = run.read_state("data.json", stage)
    generation = run.read_state("generation.json", stage)
    if adapt["dataset_digest"] != data["digest"] or adapt["generation"] != generation:
        raise RuntimeError("the adapter was trained before the data or the generation settings changed; re-run from Section 8")
    pipe = BARTSummarizationPipeline.from_artifact(run.out / ADAPTER_DIR, weights_dir=run.snapshot, device=device_name())
    return pipe, adapt


def load_splits(run: Run, stage: str) -> dict[str, list[dict[str, Any]]]:
    return run.read_state("splits.json", stage)


def decoder_digest(pipe: Any) -> str:
    """SHA-256 over every decoder-block tensor of the loaded model (name, shape, dtype and bytes, in name order)."""
    digest = hashlib.sha256()
    state = pipe._model.state_dict()
    for name in sorted(k for k in state if k.startswith("model.decoder.layers.")):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}|".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    """Section 3: install the carried manifest, fetch absent files at the pinned revision, verify every file."""
    from bart_summarization_pipeline import (
        MODEL_ID,
        MODEL_LICENSE,
        MODEL_REVISION,
        stage_missing_files,
        verify_snapshot,
    )
    from bart_summarization_pipeline.pipeline import MANIFEST_NAME

    carried = json.loads((run.root / "weights" / SNAPSHOT_KEY / MANIFEST_NAME).read_text(encoding="utf-8"))
    if (carried["modelId"], carried["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity pinned in the carried pipeline.py")
    run.snapshot.mkdir(parents=True, exist_ok=True)
    (run.snapshot / MANIFEST_NAME).write_text(json.dumps(carried, indent=2), encoding="utf-8")
    print({"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE, "files": len(carried["files"]), "total_bytes": carried["totalBytes"]})
    fetched = stage_missing_files(run.snapshot, allow_download=True)
    print({"weights_dir": str(run.snapshot), "fetched": fetched})
    verified = verify_snapshot(run.snapshot)
    print({"verified_files": len(verified["files"]), "revision": verified["revision"]})
    run.write_state("weights.json", {"files": verified["files"], "total_bytes": verified["totalBytes"], "fetched": fetched})


def stage_runtime(run: Run) -> None:
    """Section 4: versions in the isolated environment against the carried lock; device; ceilings and pinned defaults."""
    import importlib.metadata

    import torch

    from bart_summarization_pipeline import (
        DECISION_RULE,
        DEFAULT_LENGTH_PENALTY,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_MIN_NEW_TOKENS,
        DEFAULT_NO_REPEAT_NGRAM_SIZE,
        DEFAULT_NUM_BEAMS,
        EARLY_STOPPING,
        GENERATION_CONFIG_FILE,
        LENGTH_PENALTY_RANGE,
        MAX_INPUT_TOKENS,
        MAX_NEW_TOKENS,
        MAX_NO_REPEAT_NGRAM_SIZE,
        MAX_NUM_BEAMS,
        MAX_TEXT_CHARS,
    )

    locked = lock_versions((run.root / LOCK).read_text(encoding="utf-8"))
    installed = {name: importlib.metadata.version(name) for name in VERSION_CHECKS}
    mismatched = {name: {"installed": installed[name], "locked": locked.get(name)} for name in VERSION_CHECKS if locked.get(name) != installed[name].split("+")[0]}
    if mismatched:
        raise RuntimeError(f"the isolated environment does not match the carried lock: {mismatched}")
    device = device_name()
    record = {"python": platform.python_version(), "versions": installed, "versions_match_lock": True, "device": device, "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None, "float": "float32"}
    run.write_output("runtime.json", record)
    print({k: v for k, v in record.items() if k != "versions"})
    print({"versions": installed})
    print({"ceilings": {"MAX_TEXT_CHARS": MAX_TEXT_CHARS, "MAX_INPUT_TOKENS": MAX_INPUT_TOKENS, "MAX_NEW_TOKENS": MAX_NEW_TOKENS, "MAX_NUM_BEAMS": MAX_NUM_BEAMS, "LENGTH_PENALTY_RANGE": LENGTH_PENALTY_RANGE, "MAX_NO_REPEAT_NGRAM_SIZE": MAX_NO_REPEAT_NGRAM_SIZE}})
    print({"pinned_defaults_from": GENERATION_CONFIG_FILE, "DEFAULT_MAX_NEW_TOKENS": DEFAULT_MAX_NEW_TOKENS, "DEFAULT_MIN_NEW_TOKENS": DEFAULT_MIN_NEW_TOKENS, "DEFAULT_NUM_BEAMS": DEFAULT_NUM_BEAMS, "DEFAULT_LENGTH_PENALTY": DEFAULT_LENGTH_PENALTY, "DEFAULT_NO_REPEAT_NGRAM_SIZE": DEFAULT_NO_REPEAT_NGRAM_SIZE, "EARLY_STOPPING": EARLY_STOPPING})
    print({"decision_rule": DECISION_RULE})


def stage_data(run: Run) -> None:
    """Section 5: the pinned SciTLDR sample or BYOD; record contract, workflow minimum, token ceilings, disjoint split,
    training CSV, refusal probes. Nothing here runs the model; the tokenizer only counts tokens."""
    from transformers import BartTokenizerFast

    from bart_summarization_pipeline import (
        CORPUS_FILES,
        CORPUS_LICENSE,
        CORPUS_NAME,
        CORPUS_RELEASE,
        MAX_INPUT_TOKENS,
        MIN_RECORDS,
        build_sample_dataset,
        check_split_disjoint,
        fetch_corpus,
        filter_records,
        load_byod_dataset,
        read_corpus,
        validate_dataset,
        write_dataset_csv,
    )

    seed = run.options.split_seed
    byod_path = None
    if run.options.byod:
        original = Path(run.options.byod).resolve()
        if not original.is_file():
            raise FileNotFoundError(f"BYOD file not found: {original}")
        byod_path = run.state / "byod_incoming" / original.name
        byod_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, byod_path)
    if byod_path is not None:
        splits, byod_report = prepare_byod(load_byod_dataset(byod_path), seed)
        new_records = [{**r, "id": f"new-{i:02d}"} for i, r in enumerate(splits["test"][:N_NEW])]
        source, data_source = "byod", f"BYOD ({byod_path.name})"
        new_kind = "BYOD test records (already scored in Section 9; not unseen documents)"
        raw = {"byod": byod_report["records"]}
    else:
        corpus = read_corpus(fetch_corpus(cache_dir=run.corpus_cache))
        splits = build_sample_dataset(corpus, seed=seed)
        used = {r["source"].lower() for part in splits.values() for r in part}
        new_records = [{**r, "id": f"new-{i:02d}"} for i, r in enumerate([r for r in filter_records(corpus["dev"]) if r["source"].lower() not in used][:N_NEW])]
        source, data_source = "sample", f"{CORPUS_NAME} ({CORPUS_RELEASE}; {CORPUS_LICENSE})"
        new_kind = "unseen SciTLDR dev abstracts (in none of the splits)"
        raw = {name: len(part) for name, part in corpus.items()}
        byod_report = None
    manifests = {name: validate_dataset(part, min_records=MIN_RECORDS if name == "train" else 1) for name, part in splits.items()}
    disjoint = check_split_disjoint(splits)
    if source == "sample":
        check_split_disjoint({**splits, "new": new_records})  # the Section 10 documents are in none of the splits
    tokenizer = BartTokenizerFast.from_pretrained(str(run.snapshot), local_files_only=True)
    ceilings = check_token_ceilings({**splits, "new": new_records}, lambda text: len(tokenizer(text, truncation=False)["input_ids"]))
    # Accepted: only now are the earlier configuration's results cleared, so a refused input leaves them intact.
    removed = run.invalidate("data")
    if byod_path is not None:
        accepted = run.state / "byod" / byod_path.name
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(byod_path, accepted)
        byod_path = accepted
    print({"data_source": data_source, "raw_records": raw, "splits": disjoint, "new_documents": len(new_records), "new_documents_are": new_kind})
    if byod_report:
        print({"byod": byod_report})
    else:
        print({"file_sha256": {k: v[2][:12] + "..." for k, v in CORPUS_FILES.items()}})
    for name, manifest in manifests.items():
        print({name: {"n": manifest["n_records"], "unique_sources": manifest["unique_sources"], "references_per_record": manifest["references_per_record"], "source_chars": manifest["source_chars"], "target_words": manifest["target_words"], "digest": manifest["digest"][:16] + "..."}})
    print({"token_ceilings": ceilings})
    train = splits["train"]
    print({"example": {"id": train[0]["id"], "source": train[0]["source"][:200] + "...", "targets": train[0]["targets"]}})
    write_dataset_csv(train, run.out / f"{STEM}_train.csv")
    count = lambda text: len(tokenizer(text, truncation=False)["input_ids"])  # noqa: E731
    base_source = splits["test"][0]["source"]
    long_source = " ".join([base_source] * (MAX_INPUT_TOKENS // max(1, count(base_source)) + 2))
    probes: dict[str, Callable[[], Any]] = {
        "duplicate id": lambda: validate_dataset([{**r, "id": "same"} for r in train[:8]]),
        "empty reference list": lambda: validate_dataset([{**train[0], "targets": []}, *train[1:8]]),
        "missing field": lambda: validate_dataset([{"id": r["id"], "source": r["source"]} for r in train[:8]]),
        "too few for this workflow": lambda: prepare_byod(train[: workflow_minimum(MIN_RECORDS) - 1], seed),
        f"over-long test source ({count(long_source)} tokens)": lambda: check_token_ceilings({"test": [{"id": "long-test", "source": long_source}]}, count),
    }
    for name, probe in probes.items():
        try:
            probe()
            print({"probe": name, "verdict": "accepted"})
        except (TypeError, ValueError) as exc:
            print({"probe": name, "rejected": str(exc)[:240]})
    record = {
        "source": source,
        "byod_path": str(byod_path) if byod_path else None,
        "data_source": data_source,
        "split_seed": seed,
        "digest": hashlib.sha256(json.dumps({k: m["digest"] for k, m in manifests.items()}, sort_keys=True).encode()).hexdigest(),
        "split_digests": {k: m["digest"] for k, m in manifests.items()},
        "splits": disjoint,
        "new_documents_are": new_kind,
        "byod": byod_report,
        "token_ceilings": ceilings,
    }
    run.write_state("splits.json", {**splits, "new": new_records})
    run.write_state("data.json", record)
    run.write_output(f"{STEM}_dataset_manifest.json", {**record, "manifests": {k: {kk: vv for kk, vv in m.items() if kk != "records"} for k, m in manifests.items()}})
    if removed:
        print({"cleared_results_of_an_earlier_configuration": sorted(set(removed) - set(STAGE_FILES["data"]))})


def stage_inference(run: Run) -> None:
    """Section 6: the inference contract on a synthetic news-style document, with the pinned defaults and the TL;DR
    settings every later score uses; the truncation flag is checked against a token count (BART-M2)."""
    from bart_summarization_pipeline import (
        DEFAULT_MAX_NEW_TOKENS,
        MAX_INPUT_TOKENS,
        MAX_NUM_BEAMS,
        validate_inputs,
    )

    generation = {"max_new_tokens": run.options.max_new_tokens, "min_new_tokens": run.options.min_new_tokens, "num_beams": run.options.num_beams}
    run.read_state("data.json", "inference")
    input_manifest = validate_inputs([DOCUMENT], names=["doc"], **generation)
    try:
        validate_inputs([DOCUMENT], num_beams=MAX_NUM_BEAMS + 1)
    except ValueError as exc:
        input_manifest["findings"].append({"input": "num-beams-ceiling-probe", "verdict": "rejected", "message": str(exc)})
    run.invalidate("inference")
    forced = forced_special_tokens(run.snapshot)
    pipe = frozen_pipeline(run)
    rows = {}
    for label, settings in (("pinned_news_defaults", {}), ("tldr_settings", generation)):
        started = time.perf_counter()
        result = pipe.summarize(DOCUMENT, **settings)
        cap = result["generation"]["max_new_tokens"]
        checks = {
            "summary_is_non_empty_text": isinstance(result["summary"], str) and bool(result["summary"].strip()),
            "content_tokens_within_cap": 1 <= result["generated_tokens"] <= cap - forced,
            "input_within_ceiling": 1 <= result["input_tokens"] <= MAX_INPUT_TOKENS,
            # Independent of the flag's own logic: a summary holding every content token the cap allows was cut.
            "full_length_summary_flagged_truncated": result["truncated"] or result["generated_tokens"] < cap - forced,
            "flag_matches_stopped_by": result["truncated"] == (result["stopped_by"] == "max_new_tokens"),
            "deterministic_decoding": result["generation"]["do_sample"] is False,
        }
        if not all(checks.values()):
            raise RuntimeError(f"summarize output failed a sanity check ({label}): {checks}")
        rows[label] = {"max_new_tokens": cap, "generated_tokens": result["generated_tokens"], "content_tokens_at_cap": cap - forced, "truncated": result["truncated"], "stopped_by": result["stopped_by"], "input_tokens": result["input_tokens"], "seconds": round(time.perf_counter() - started, 2), "checks": checks, "summary": result["summary"]}
        print({k: v for k, v in rows[label].items() if k not in ("summary", "checks")} | {"settings": label})
        print(f"summary ({label.replace('_', ' ')}):")
        print(result["summary"])
    print({"no_score": "the pipeline emits no probability or quality score; truncated is a length flag", "forced_tokens_per_summary": forced, "pinned_default_max_new_tokens": DEFAULT_MAX_NEW_TOKENS, "findings": len(input_manifest["findings"])})
    run.write_output(f"{STEM}_input_manifest.json", input_manifest)
    run.write_output(f"{STEM}_inference.json", {"document": "synthetic three-paragraph news-style document authored for this notebook", "forced_tokens_per_summary": forced, "results": rows})
    run.write_state("generation.json", generation)


def stage_frozen(run: Run) -> None:
    """Section 7: Lead-1 and Lead-3 (no model, no length cap) and the frozen model on the test split under the TL;DR settings."""
    from bart_summarization_pipeline import lead_baseline

    run.invalidate("frozen")
    data = run.read_state("data.json", "frozen")
    generation = run.read_state("generation.json", "frozen")
    test = load_splits(run, "frozen")["test"]
    lead1, lead3 = lead_baseline(test, n_sentences=1), lead_baseline(test, n_sentences=3)
    print({"lead1_baseline": rounded(lead1), "lead3_baseline": rounded(lead3), "note": "Lead baselines copy sentences; no generation settings or token cap apply to them"})
    pipe = frozen_pipeline(run)
    started = time.perf_counter()
    frozen_test = pipe.evaluate(test, **generation)
    seconds = round(time.perf_counter() - started, 1)
    print({"frozen_model_test": {**rounded(frozen_test), "mean_reference_words": round(frozen_test["mean_reference_words"], 1), "n": frozen_test["n"], "adapted": frozen_test["adapted"], "verdict": frozen_test["verdict"]}, "seconds": seconds, "device": pipe.device})
    print({"cut_at_the_cap": f"{frozen_test['hit_token_ceiling']} of {frozen_test['n']} frozen summaries reached max_new_tokens = {generation['max_new_tokens']} and were cut"})
    examples = []
    for record in test[:2]:
        item = pipe.summarize(record["source"], **generation)
        examples.append({"id": record["id"], "frozen": item["summary"], "truncated": item["truncated"], "reference": record["targets"][0]})
        print(examples[-1])
    run.write_output(f"{STEM}_frozen_test.json", {"dataset_digest": data["digest"], "generation": generation, "baselines": {"lead1": lead1, "lead3": lead3}, "frozen_test": frozen_test, "examples": examples, "seconds": seconds})


def _training_options(run: Run) -> dict[str, Any]:
    return {"epochs": run.options.epochs, "learning_rate": run.options.learning_rate, "batch_size": run.options.batch_size, "trainable_decoder_layers": run.options.trainable_decoder_layers}


def _progress(entry: dict[str, Any]) -> None:
    row = {"epoch": entry["epoch"], "train_loss": None if entry["train_loss"] is None else round(entry["train_loss"], 4)}
    if entry.get("val"):
        row.update({f"val_{k}": v for k, v in rounded(entry["val"]).items()})
        row["val_cut_at_cap"] = entry["val"]["hit_token_ceiling"]
    if "note" in entry:
        row["note"] = entry["note"]
    print(row, flush=True)


def stage_adapt(run: Run) -> None:
    """Section 8: bounded fine-tuning of a freshly built frozen pipeline; adapter export; reference for the reload check."""
    run.invalidate("adapt")
    data = run.read_state("data.json", "adapt")
    generation = run.read_state("generation.json", "adapt")
    splits = load_splits(run, "adapt")
    training = _training_options(run)
    pipe = frozen_pipeline(run)
    print({"starts_from": "the pinned base model, built from the verified files in this process", "adapted": pipe.adapter is not None, "device": pipe.device})
    started = time.perf_counter()
    result = pipe.adapt(splits["train"], splits["validation"], epochs=training["epochs"], lr=training["learning_rate"], batch_size=training["batch_size"], trainable_decoder_layers=training["trainable_decoder_layers"], eval_generation=generation, progress=_progress)
    seconds = round(time.perf_counter() - started, 1)
    print({"trainable_parameters": result["n_trainable"], "total_parameters": result["n_total"], "best_epoch": result["best_epoch"], "selection": result["selection"], "seconds": seconds, "device": pipe.device})
    artifact = pipe.save_artifact(run.out / ADAPTER_DIR, metadata={"tutorial": STEM, "data_source": data["data_source"], "dataset_digest": data["digest"]})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    print({"artifact": str(artifact), "format": manifest["format"], "tensors": len(manifest["tensors"]), "bytes": manifest["files"][0]["bytes"], "sha256": manifest["files"][0]["sha256"][:16] + "..."})
    reference = {"decoder_digest": decoder_digest(pipe), "summaries": [pipe.summarize(r["source"], **generation)["summary"] for r in splits["test"][:N_PARITY]]}
    run.write_state("reference.json", reference)
    run.write_output(f"{STEM}_training_history.json", {"training": training, "generation": generation, "history": result["history"], "best_epoch": result["best_epoch"], "seconds": seconds})
    run.write_state("adapt.json", {"training": training, "generation": generation, "dataset_digest": data["digest"], "seconds": seconds, "adaptation": {k: v for k, v in result.items() if k not in ("history", "trainable_names")}})


def stage_evaluate(run: Run) -> None:
    """Section 9: the adapter, rebuilt from files in a new process, scored on the held-out test split; four-way table."""
    from bart_summarization_pipeline import MODEL_ID, MODEL_KEY, MODEL_REVISION

    run.invalidate("evaluate")
    data = run.read_state("data.json", "evaluate")
    generation = run.read_state("generation.json", "evaluate")
    frozen = run.read_output(f"{STEM}_frozen_test.json", "evaluate")
    if frozen["dataset_digest"] != data["digest"] or frozen["generation"] != generation:
        raise RuntimeError("the frozen scores were computed for other data or generation settings; re-run from Section 7")
    splits = load_splits(run, "evaluate")
    pipe, adapt = adapted_pipeline(run, "evaluate")
    adapted_test = pipe.evaluate(splits["test"], **generation)
    adapted_val = pipe.evaluate(splits["validation"], **generation)
    frozen_test, baselines = frozen["frozen_test"], frozen["baselines"]
    comparison = {
        metric: {"lead1": round(baselines["lead1"][metric], 2), "lead3": round(baselines["lead3"][metric], 2), "frozen": round(frozen_test[metric], 2), "adapted": round(adapted_test[metric], 2)}
        for metric in ("rouge1", "rouge2", "rougeL", "mean_summary_words")
    }
    comparison["cut_at_cap"] = {"lead1": None, "lead3": None, "frozen": frozen_test["hit_token_ceiling"], "adapted": adapted_test["hit_token_ceiling"]}
    comparison["delta_adapted_vs_frozen"] = {metric: round(adapted_test[metric] - frozen_test[metric], 2) for metric in ("rouge1", "rouge2", "rougeL")}
    for metric, row in comparison.items():
        print({metric: row})
    examples = []
    for record in splits["test"][:2]:
        item = pipe.summarize(record["source"], **generation)
        examples.append({"id": record["id"], "adapted": item["summary"], "truncated": item["truncated"], "reference": record["targets"][0]})
        print(examples[-1])
    outcome = {"adapted_rougeL_above_frozen": adapted_test["rougeL"] > frozen_test["rougeL"], "delta_rougeL": comparison["delta_adapted_vs_frozen"]["rougeL"], "adapted": adapted_test["adapted"], "default_configuration": is_default(data, generation, adapt["training"])}
    report = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY},
        "data_source": data["data_source"],
        "dataset_digests": data["split_digests"],
        "splits": data["splits"],
        "generation": generation,
        "baselines": baselines,
        "baseline_note": "Lead baselines copy the first sentences of each source; no generation settings or token cap apply to them",
        "frozen_test": frozen_test,
        "validation_metrics": adapted_val,
        "test_metrics": adapted_test,
        "comparison": comparison,
        "outcome": outcome,
        "adaptation": adapt["adaptation"],
        "training": adapt["training"],
        "history": run.read_output(f"{STEM}_training_history.json", "evaluate")["history"],
        "adaptation_seconds": adapt["seconds"],
        "evidence": "tutorial sample-sanity numbers from one seeded split of one corpus; no dispersion estimate; not a benchmark",
    }
    run.write_output(f"{STEM}_evaluation_report.json", report)
    verdict = "the adapted model scores above the frozen model on ROUGE-L" if outcome["adapted_rougeL_above_frozen"] else "the adapted model does NOT score above the frozen model on ROUGE-L"
    print({"outcome": verdict, **outcome, "report": f"outputs/{STEM}_evaluation_report.json"})
    if outcome["default_configuration"] and not outcome["adapted_rougeL_above_frozen"]:
        # Default-path verification only: on the default sample and settings the recorded runs improve ROUGE-L.
        raise RuntimeError(f"default configuration, but the expected ROUGE-L improvement did not occur: {comparison['rougeL']}")


def stage_new(run: Run) -> None:
    """Section 10: the adapted model summarises the documents set aside in Section 5; single-document report; CSV."""
    from bart_summarization_pipeline import evaluation_report

    run.invalidate("new")
    data = run.read_state("data.json", "new")
    generation = run.read_state("generation.json", "new")
    records = load_splits(run, "new")["new"]
    pipe, _adapt = adapted_pipeline(run, "new")
    metrics = pipe.evaluate(records, **generation)
    rows = []
    for record in records:
        item = pipe.summarize(record["source"], **generation)
        rows.append({"id": record["id"], "summary": item["summary"], "reference": record["targets"][0], "generated_tokens": item["generated_tokens"], "input_tokens": item["input_tokens"], "truncated": item["truncated"], "stopped_by": item["stopped_by"]})
        print({k: rows[-1][k] for k in ("id", "summary", "reference", "truncated")})
    single = evaluation_report(pipe.summarize(records[0]["source"], **generation), records[0]["targets"], sample_kind="one of " + data["new_documents_are"])
    print({"documents_are": data["new_documents_are"], "n": metrics["n"], **rounded(metrics, ("rouge1", "rougeL")), "verdict": metrics["verdict"], "single_document_report_verdict": single["verdict"]})
    with open(run.out / f"{STEM}_summaries.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run.write_output(f"{STEM}_new_documents.json", {"documents_are": data["new_documents_are"], "metrics": metrics, "summaries": rows, "single_document_report": single})


def stage_reload(run: Run) -> None:
    """Section 11: rebuild from the exported files in another new process; every decoder tensor and the summaries must
    equal the trained model's (BART-M3)."""
    run.invalidate("reload")
    generation = run.read_state("generation.json", "reload")
    splits = load_splits(run, "reload")
    reference = run.read_state("reference.json", "reload")
    manifest = json.loads((run.out / ADAPTER_DIR / "manifest.json").read_text(encoding="utf-8"))
    print({"artifact": str(run.out / ADAPTER_DIR), "format": manifest["format"], "tensors": len(manifest["tensors"]), "bytes": manifest["files"][0]["bytes"], "sha256": manifest["files"][0]["sha256"][:16] + "..."})
    pipe, _adapt = adapted_pipeline(run, "reload")
    digest_equal = decoder_digest(pipe) == reference["decoder_digest"]
    after = [pipe.summarize(r["source"], **generation)["summary"] for r in splits["test"][:N_PARITY]]
    identical = sum(a == b for a, b in zip(reference["summaries"], after, strict=True))
    passed = digest_equal and identical == len(after)
    parity = {"decoder_tensors_identical": digest_equal, "identical_summaries": identical, "of": len(after), "reload_verification": "PASSED" if passed else "FAILED", "reloaded_best_epoch": pipe.adapter["best_epoch"], "compared": "every decoder-block tensor (SHA-256) and the summaries of the first test documents, against the trained model in the adapt stage"}
    run.write_output(f"{STEM}_reload_parity.json", parity)
    print({"reload_parity": parity})
    if not passed:
        raise RuntimeError(f"the reloaded adapter does not reproduce the trained model: {parity}")


def stage_bundle(run: Run) -> None:
    """Section 12: one result record linking every output, with provenance and the SHA-256 of each file."""
    from bart_summarization_pipeline import (
        CORPUS_BASE_URL,
        CORPUS_FILES,
        CORPUS_LICENSE,
        CORPUS_NAME,
        CORPUS_RELEASE,
        MODEL_ID,
        MODEL_KEY,
        MODEL_LICENSE,
        MODEL_REVISION,
        WEIGHT_FILE,
    )

    run.invalidate("bundle")
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8"))
    weights = run.read_state("weights.json", "bundle")
    data = run.read_state("data.json", "bundle")
    report = run.read_output(f"{STEM}_evaluation_report.json", "bundle")
    manifest = json.loads((run.out / ADAPTER_DIR / "manifest.json").read_text(encoding="utf-8"))
    weight_entry = next(entry for entry in weights["files"] if entry["path"] == WEIGHT_FILE)
    payload = {
        "notebook_source": source,
        "repository_revision": source["revision"],
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "model_license": MODEL_LICENSE},
        "snapshot": {"files": len(weights["files"]), "total_bytes": weights["total_bytes"], "fetched_this_run": weights["fetched"], "weight_file": WEIGHT_FILE, "weight_format": "safetensors, digest-verified", "weight_sha256": weight_entry["sha256"], "remote_code_executed": False},
        "corpus": {"name": CORPUS_NAME, "release": CORPUS_RELEASE, "base_url": CORPUS_BASE_URL, "files": {k: {"name": v[0], "bytes": v[1], "sha256": v[2]} for k, v in CORPUS_FILES.items()}, "license": CORPUS_LICENSE},
        "runtime": run.read_output("runtime.json", "bundle"),
        "environment": "isolated hash-locked uv environment; nothing installed into the notebook kernel",
        "data_source": data["data_source"],
        "splits": data["splits"],
        "generation": report["generation"],
        "training": report["training"],
        "comparison": report["comparison"],
        "outcome": report["outcome"],
        "artifact": {"dir": ADAPTER_DIR, "sha256": manifest["files"][0]["sha256"], "bytes": manifest["files"][0]["bytes"], "tensors": len(manifest["tensors"])},
        "reload_parity": run.read_output(f"{STEM}_reload_parity.json", "bundle"),
    }
    files = sorted(p for p in run.out.rglob("*") if p.is_file() and "activity" not in p.relative_to(run.out).parts and p.name != f"{STEM}_result.json")
    payload["files"] = {p.relative_to(run.out).as_posix(): sha256_file(p) for p in files}
    run.write_output(f"{STEM}_result.json", payload)
    print({"outputs_directory": str(run.out)})
    for path in sorted(p for p in run.out.rglob("*") if p.is_file()):
        print(f"  - {path.relative_to(run.out).as_posix()} ({path.stat().st_size / 1024:.1f} KB)")


def stage_activity(run: Run) -> None:
    """Section 13 (optional): change one thing -- the summary-length cap -- and re-score the frozen and the adapted model
    on the test split; writes only under outputs/activity/ and checks that no canonical output changed."""
    generation = run.read_state("generation.json", "activity")
    report = run.read_output(f"{STEM}_evaluation_report.json", "activity")
    test = load_splits(run, "activity")["test"]
    changed = {**generation, "max_new_tokens": run.options.max_new_tokens}
    if changed["max_new_tokens"] == generation["max_new_tokens"]:
        raise ValueError(f"the activity changes the summary-length cap: choose an ACTIVITY_MAX_NEW_TOKENS other than the canonical {generation['max_new_tokens']}")
    canonical = {p.relative_to(run.out).as_posix(): sha256_file(p) for p in sorted(run.out.rglob("*")) if p.is_file() and "activity" not in p.relative_to(run.out).parts}
    print({"changed_variable": "max_new_tokens (the summary-length cap)", "canonical": generation["max_new_tokens"], "activity": changed["max_new_tokens"], "unchanged": {k: v for k, v in generation.items() if k != "max_new_tokens"}})
    frozen = frozen_pipeline(run).evaluate(test, **changed)
    adapted = adapted_pipeline(run, "activity")[0].evaluate(test, **changed)
    rows = {}
    for name, before, after in (("frozen", report["frozen_test"], frozen), ("adapted", report["test_metrics"], adapted)):
        rows[name] = {"rougeL_canonical": round(before["rougeL"], 2), "rougeL_activity": round(after["rougeL"], 2), "mean_summary_words_canonical": round(before["mean_summary_words"], 1), "mean_summary_words_activity": round(after["mean_summary_words"], 1), "cut_at_cap_canonical": before["hit_token_ceiling"], "cut_at_cap_activity": after["hit_token_ceiling"]}
        print({name: rows[name]})
    record = {
        "changed_variable": "max_new_tokens",
        "canonical_max_new_tokens": generation["max_new_tokens"],
        "activity_max_new_tokens": changed["max_new_tokens"],
        "per_model": rows,
        "gap_rougeL": {"canonical": round(report["test_metrics"]["rougeL"] - report["frozen_test"]["rougeL"], 2), "activity": round(adapted["rougeL"] - frozen["rougeL"], 2)},
    }
    out = run.out / "activity"
    out.mkdir(exist_ok=True)
    (out / f"{STEM}_activity_max_new_tokens_{changed['max_new_tokens']}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    after_files = {p.relative_to(run.out).as_posix(): sha256_file(p) for p in sorted(run.out.rglob("*")) if p.is_file() and "activity" not in p.relative_to(run.out).parts}
    if after_files != canonical:
        raise RuntimeError("the activity changed a canonical output; it must write only under outputs/activity/")
    print({"gap_rougeL_adapted_minus_frozen": record["gap_rougeL"]})


STAGES = {
    "weights": stage_weights,
    "runtime": stage_runtime,
    "data": stage_data,
    "inference": stage_inference,
    "frozen": stage_frozen,
    "adapt": stage_adapt,
    "evaluate": stage_evaluate,
    "new": stage_new,
    "reload": stage_reload,
    "bundle": stage_bundle,
    "activity": stage_activity,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the pinned snapshot and the corpus cache")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--byod", default="", help="data: a BYOD .csv/.json/.jsonl dataset instead of the pinned sample")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_DATA["split_seed"], help="data: seed of the sample draw or the BYOD split")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_GENERATION["max_new_tokens"], help="inference/activity: the summary-length cap")
    parser.add_argument("--min-new-tokens", type=int, default=DEFAULT_GENERATION["min_new_tokens"], help="inference: minimum new tokens")
    parser.add_argument("--num-beams", type=int, default=DEFAULT_GENERATION["num_beams"], help="inference: beam width")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING["epochs"], help="adapt: training epochs")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAINING["learning_rate"], help="adapt: AdamW learning rate")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING["batch_size"], help="adapt: batch size")
    parser.add_argument("--trainable-decoder-layers", type=int, default=DEFAULT_TRAINING["trainable_decoder_layers"], help="adapt: how many of the last decoder blocks train")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    root = options.root.resolve()
    carried_src = root / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    run = Run(root, options.weights.resolve(), options)
    error_file = run.state / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        STAGES[options.stage](run)
    except Exception as exc:  # the notebook re-raises this message in the kernel
        traceback.print_exc()
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    print({"stage": options.stage, "status": "ok", "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

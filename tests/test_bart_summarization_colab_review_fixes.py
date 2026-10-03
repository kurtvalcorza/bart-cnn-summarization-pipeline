"""Regression tests for the Notebook Review Framework v1 findings on tutorials/bart_summarization_colab.ipynb
(review PR #8, BART-M1..M5 and BART-m1..m7).

Everything here runs within CI's install budget (pytest, ruff, numpy; no torch, no transformers) except the tests
that take the `tiny_snapshot` fixture, which skip cleanly without torch/transformers and run the carried stage runner
end to end on a tiny randomly initialised BART built from the pinned snapshot's config and tokenizer (a stand-in for
control flow, not for BART-large-CNN's numerics).
"""
# ruff: noqa: E501  -- assertion messages and record literals are kept on one line

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

from bart_summarization_pipeline import MIN_RECORDS, BARTSummarizationPipeline, split_dataset
from bart_summarization_pipeline.pipeline import _stopped_by

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_review_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stages = _load("tutorial_stages")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = json.loads((ROOT / "tutorials" / TEMPLATE["notebook_name"]).read_text(encoding="utf-8"))
MARKDOWN = "\n".join(c["source"] for c in NOTEBOOK["cells"] if c["cell_type"] == "markdown")
CODE = [c for c in NOTEBOOK["cells"] if c["cell_type"] == "code"]
KERNEL = [c["source"] for c in CODE if not c["metadata"].get("dimer", {}).get("embedded_sources")]
RUNNER = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")


def _record(i: int, words: int = 30) -> dict:
    source = " ".join(f"word{i}x{k}" for k in range(words)) + f". Sentence two of document {i}. Sentence three ends it."
    return {"id": f"r{i:03d}", "source": source, "targets": [f"document {i} in one short sentence"]}


def _run(tmp_path: Path, weights: Path | None = None, **options) -> stages.Run:
    defaults = vars(stages.parse_args(["--root", str(tmp_path), "--weights", str(weights or tmp_path / "w"), "--stage", "data"]))
    return stages.Run(tmp_path, weights or tmp_path / "w", argparse.Namespace(**{**defaults, **options}))


# ---------------------------------------------------------------------------------------------------- BART-M1


def test_m1_kernel_installs_nothing_and_never_asks_for_a_restart():
    kernel = "\n".join(KERNEL)
    assert "pip install" not in kernel and "'-m', 'pip'" not in kernel and "importlib" not in kernel
    assert "--require-hashes" in kernel and "'--managed-python', '--python', '3.12.12'" in kernel
    prose = MARKDOWN.lower().replace("no runtime restart", "").replace("no restart", "")
    assert "restart" not in prose


def test_m1_release_record_is_candidate_and_names_the_two_passes():
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status and "2 passes" in status
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "restart after the install is expected" not in record
    assert "**PASSED** — 11/11 code cells ok (1 restart after install cell)" not in record
    assert record.count("2 passes") >= 2 and "**Candidate.**" in record
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "| **Candidate**" in registry and "Release-grade** — clean-runtime" not in registry


# ---------------------------------------------------------------------------------------------------- BART-M2


@pytest.mark.parametrize(
    ("ids", "cap", "expected"),
    [
        ([2, 0, 11, 12, 2], 4, "max_new_tokens"),  # forced BOS + 2 content + forced EOS at the cap: cut
        ([2, 0, 11, 2], 4, "eos"),  # ended on its own one step before the cap
        ([2, 0, 11, 12, 13], 4, "max_new_tokens"),  # no EOS at all (no forced EOS configured)
        ([2, 0, 2], 48, "eos"),
        ([2, *range(3, 51)], 48, "max_new_tokens"),
    ],
)
def test_m2_stopped_by_is_decided_from_the_length_not_the_presence_of_eos(ids, cap, expected):
    assert _stopped_by(ids, 2, cap) == expected


def test_m2_summarize_reports_truncated_from_stopped_by(forbid_model_imports):
    pipe = BARTSummarizationPipeline(lambda text, settings: ("cut short", settings["max_new_tokens"] - 2, "max_new_tokens"), lambda text: 12)
    result = pipe.summarize("A document.", max_new_tokens=48, min_new_tokens=0, num_beams=4)
    assert result["truncated"] is True and result["stopped_by"] == "max_new_tokens"


def test_m2_inference_stage_checks_the_flag_against_a_token_count():
    assert '"full_length_summary_flagged_truncated": result["truncated"] or result["generated_tokens"] < cap - forced,' in RUNNER
    assert "truncated_matches_stopped_by" not in RUNNER + "\n".join(KERNEL)
    assert "**How `truncated` is decided.**" in MARKDOWN and "cut_at_the_cap" in RUNNER


@pytest.mark.parametrize("cap", [6, 10, 16])
def test_m2_tiny_stand_in_reports_a_capped_summary_as_truncated(tiny_snapshot, cap):
    """The review's probe P3: the stand-in with the snapshot's forced_eos_token_id returned stopped_by 'eos' at caps
    6/10/16; it must now report the cut."""
    pipe = BARTSummarizationPipeline.from_pretrained(weights_dir=tiny_snapshot, device="cpu")
    doc = "The town council of Millbrook voted on Tuesday evening to approve a three-year plan to replace the aging water mains. " * 3
    result = pipe.summarize(doc, max_new_tokens=cap, min_new_tokens=0, num_beams=4)
    assert result["truncated"] is True and result["stopped_by"] == "max_new_tokens"
    assert result["generated_tokens"] <= cap - 2
    assert stages.forced_special_tokens(tiny_snapshot) == 2


# ---------------------------------------------------------------------------------------------------- BART-M3


def test_m3_adapt_refuses_an_already_adapted_pipeline(forbid_model_imports):
    pipe = BARTSummarizationPipeline(lambda text, settings: ("s", 1, "eos"), lambda text: 3)
    pipe.adapter = {"trainable_names": ["model.decoder.layers.11.fc1.weight"]}
    with pytest.raises(ValueError, match="already adapted"):
        pipe.adapt([_record(i) for i in range(8)])


def test_m3_every_model_is_built_from_the_verified_files():
    assert RUNNER.count("Pipeline.from_pretrained(") == 1 and RUNNER.count("Pipeline.from_artifact(") == 1
    tree = ast.parse(RUNNER)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("stage_"):
            body = ast.unparse(node)
            if ".evaluate(" in body or ".adapt(" in body or ".summarize(" in body:
                assert "frozen_pipeline(run)" in body or "adapted_pipeline(run" in body, node.name


def test_m3_rerunning_a_section_clears_every_later_result(tmp_path, forbid_model_imports):
    run = _run(tmp_path)
    for names in stages.STAGE_FILES.values():
        for name in names:
            (run.state / name).write_text("{}", encoding="utf-8")
    removed = run.invalidate("data")
    assert set(removed) == {n for names in stages.STAGE_FILES.values() for n in names}
    for name in ("generation.json", "adapt.json", "reference.json", "splits.json"):
        (run.state / name).write_text("{}", encoding="utf-8")
    assert set(run.invalidate("adapt")) == {"adapt.json", "reference.json"}
    assert (run.state / "generation.json").exists() and (run.state / "splits.json").exists()


def test_m3_a_stale_adapter_is_refused_before_any_model_work(tmp_path, forbid_model_imports):
    run = _run(tmp_path)
    run.write_state("data.json", {"digest": "new"})
    run.write_state("generation.json", {"max_new_tokens": 48})
    run.write_state("adapt.json", {"dataset_digest": "old", "generation": {"max_new_tokens": 48}})
    with pytest.raises(RuntimeError, match="re-run from Section 8"):
        stages.adapted_pipeline(run, "evaluate")


def test_m3_stage_runner_journeys_on_the_tiny_stand_in(tiny_snapshot, tmp_path):
    """Default-then-rerun journey of the review (probe P4), through the carried stage runner: a rerun from Section 5
    gives a frozen score equal to the first pass with `adapted: False`; the second adaptation starts from the base
    (its epoch-0 validation score equals the first one's); after a TRAINABLE_DECODER_LAYERS = 1 rerun the reload
    reproduces every decoder tensor, the artifact holds only layer 11 and layer 10 is the base's."""
    import shutil

    import torch
    from safetensors.torch import load_file

    weights = tmp_path / "weights"
    shutil.copytree(tiny_snapshot, weights / "bart-large-cnn")
    byod = tmp_path / "pairs.jsonl"
    byod.write_text("".join(json.dumps(_record(i)) + "\n" for i in range(20)), encoding="utf-8")
    root = tmp_path / "run"
    (root / "state").mkdir(parents=True)
    (root / "state" / "weights.json").write_text(json.dumps({"files": [], "total_bytes": 0, "fetched": []}), encoding="utf-8")

    def stage(name, *options):
        code = stages.main(["--root", str(root), "--weights", str(weights), "--stage", name, *map(str, options)])
        error = root / "state" / f"{name}.error.json"
        assert code == 0, error.read_text(encoding="utf-8") if error.exists() else name

    out = root / "outputs"
    gen = ("--max-new-tokens", 8, "--num-beams", 2)
    stage("data", "--byod", byod)
    data = json.loads((root / "state" / "data.json").read_text(encoding="utf-8"))
    assert data["splits"] == {"test": 4, "validation": 3, "train": 13} and data["new_documents_are"].startswith("BYOD test records")
    stage("inference", *gen)
    stage("frozen")
    first_frozen = json.loads((out / "bart_summarization_frozen_test.json").read_text(encoding="utf-8"))["frozen_test"]
    stage("adapt", "--epochs", 1, "--batch-size", 4)
    first_history = json.loads((out / "bart_summarization_training_history.json").read_text(encoding="utf-8"))["history"]
    stage("evaluate")
    report = json.loads((out / "bart_summarization_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["outcome"]["default_configuration"] is False  # BYOD: a negative result is printed, not raised (BART-m1)
    # rerun from Section 5, as the notebook instructs
    stage("data", "--byod", byod)
    assert not (out / "bart_summarization_frozen_test.json").exists() and not (out / "bart_summarization_adapter").exists()
    stage("inference", *gen)
    stage("frozen")
    second_frozen = json.loads((out / "bart_summarization_frozen_test.json").read_text(encoding="utf-8"))["frozen_test"]
    assert second_frozen["adapted"] is False
    assert {k: v for k, v in second_frozen.items() if k != "seconds"} == {k: v for k, v in first_frozen.items() if k != "seconds"}
    stage("adapt", "--epochs", 1, "--batch-size", 4, "--trainable-decoder-layers", 1)
    second_history = json.loads((out / "bart_summarization_training_history.json").read_text(encoding="utf-8"))["history"]
    assert second_history[0]["note"] == "frozen model" and second_history[0]["val"] == first_history[0]["val"]
    stage("reload")
    parity = json.loads((out / "bart_summarization_reload_parity.json").read_text(encoding="utf-8"))
    assert parity["decoder_tensors_identical"] is True and parity["reload_verification"] == "PASSED"
    tensors = load_file(str(out / "bart_summarization_adapter" / "adapter.safetensors"))
    assert tensors and all(name.startswith("model.decoder.layers.11.") for name in tensors)
    base = BARTSummarizationPipeline.from_pretrained(weights_dir=weights / "bart-large-cnn", device="cpu")._model.state_dict()
    reloaded = BARTSummarizationPipeline.from_artifact(out / "bart_summarization_adapter", weights_dir=weights / "bart-large-cnn", device="cpu")._model.state_dict()
    layer10 = [k for k in base if k.startswith("model.decoder.layers.10.")]
    assert layer10 and all(torch.equal(base[k], reloaded[k]) for k in layer10)
    # the optional activity changes only the cap and writes only under outputs/activity/
    stage("evaluate")
    stage("activity", "--max-new-tokens", 12)
    assert list((out / "activity").iterdir())


# ---------------------------------------------------------------------------------------------------- BART-M4


def test_m4_workflow_minimum_matches_split_dataset():
    assert stages.workflow_minimum(MIN_RECORDS) == 12
    for n in range(1, 61):
        sizes = stages.split_sizes(n)
        records = [_record(i) for i in range(n)]
        if n < MIN_RECORDS:
            continue
        try:
            splits = split_dataset(records, seed=0)
        except ValueError:
            assert sizes["train"] < MIN_RECORDS
            continue
        assert {k: len(v) for k, v in splits.items()} == sizes


@pytest.mark.parametrize("n", [8, 11])
def test_m4_a_byod_file_below_the_workflow_minimum_is_refused_in_section_5(n, forbid_model_imports):
    with pytest.raises(ValueError, match=r"at least 12 records with distinct sources") as excinfo:
        stages.prepare_byod([_record(i) for i in range(n)], seed=42)
    assert "train" in str(excinfo.value) and "validation" in str(excinfo.value)


def test_m4_a_20_record_byod_file_runs_through_section_5_and_reports_duplicates(forbid_model_imports):
    records = [_record(i) for i in range(20)] + [{**_record(3), "id": "dup-3"}]
    splits, report = stages.prepare_byod(records, seed=42)
    assert report["dropped_duplicate_sources"] == 1 and report["distinct_sources"] == 20
    assert {k: len(v) for k, v in splits.items()} == {"test": 4, "validation": 3, "train": 13}


def test_m4_an_over_ceiling_test_source_is_refused_in_section_5_with_its_id(forbid_model_imports):
    splits = {"train": [_record(i) for i in range(3)], "validation": [_record(5)], "test": [_record(6), {**_record(7), "id": "too-long", "source": "x " * 1300}]}

    def count(text):
        return len(text.split())

    with pytest.raises(ValueError, match=r"test record 'too-long' has 1300 tokens; MAX_INPUT_TOKENS is 1024"):
        stages.check_token_ceilings(splits, count)
    long_train = {"train": [{**_record(1), "source": "y " * 2000}], "test": [_record(2)]}
    assert stages.check_token_ceilings(long_train, count)["train"]["truncated_during_training"] == 1


def test_m4_documented_minimum_is_the_enforced_one():
    assert "**12 records with distinct sources**" in MARKDOWN
    assert "this workflow needs at least 12 records with distinct sources" in MARKDOWN
    assert "a dataset needs 8..20,000 records" not in MARKDOWN


# ---------------------------------------------------------------------------------------------------- BART-M5 and the validator


def test_m5_guided_layer_and_the_validator_pass():
    validator = _load("validate_release_assets")
    validator.validate_notebooks()
    for marker in ("## How to use this notebook", "## Roadmap", "<summary><strong>Glossary</strong>", "## Troubleshooting", "## Conclude with evidence", "**Predict → Change one thing → Run → Observe → Explain.**"):
        assert marker in MARKDOWN
    assert "after this notebook you should be able to (1)" in MARKDOWN


def test_m5_infrastructure_cells_are_titled_and_collapsed():
    infra = [c for c in CODE if c["source"].startswith("# @title Infrastructure: ")]
    assert len(infra) == 4
    assert all(c["metadata"].get("cellView") == "form" and c["metadata"]["jupyter"]["source_hidden"] is True for c in infra)


# ---------------------------------------------------------------------------------------------------- minors


def test_m1_minor_no_result_asserts_and_only_the_default_must_improve():
    assert all("assert " not in source for source in KERNEL) and "assert " not in RUNNER
    data = {"source": "sample", "split_seed": 42}
    assert stages.is_default(data, stages.DEFAULT_GENERATION, stages.DEFAULT_TRAINING)
    assert not stages.is_default({**data, "source": "byod"}, stages.DEFAULT_GENERATION, stages.DEFAULT_TRAINING)
    assert not stages.is_default(data, {**stages.DEFAULT_GENERATION, "max_new_tokens": 96}, stages.DEFAULT_TRAINING)
    assert not stages.is_default(data, stages.DEFAULT_GENERATION, {**stages.DEFAULT_TRAINING, "trainable_decoder_layers": 1})


def test_m2_minor_pretraining_overlap_is_stated():
    assert "**Pretraining overlap.**" in MARKDOWN and "**cannot be ruled out**" in MARKDOWN


def test_m3_minor_timings_name_their_environment():
    for claim in ("about twelve minutes of model time", "40 s per training epoch", "about a minute of training", "in a few minutes on CPU"):
        assert claim not in MARKDOWN
    assert "Kaggle T4 run of the previous, in-kernel version" in MARKDOWN and "a local pre-flight of the previous version, not a supported runtime" in MARKDOWN and "(an estimate)" in MARKDOWN


def test_m4_minor_lead_baselines_are_not_said_to_use_the_tldr_settings():
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    for text in (MARKDOWN, registry):
        assert "applied identically to the Lead baselines" not in text and "all under the TL;DR-length settings" not in text
    assert "no generation settings or token cap apply to them" in MARKDOWN


def _obtain_upload(namespace: dict):
    install = next(c["source"] for c in CODE if c["source"].startswith("# @title Infrastructure: install"))
    function = next(node for node in ast.parse(install).body if isinstance(node, ast.FunctionDef) and node.name == "obtain_upload")
    exec(ast.unparse(function), namespace)  # noqa: S102 - the notebook's own helper
    return namespace["obtain_upload"]


def test_m5_minor_byod_path_works_without_google_colab_and_an_empty_upload_says_what_to_do(tmp_path, monkeypatch):
    path = tmp_path / "pairs.csv"
    path.write_text("id,source,target\n", encoding="utf-8")
    obtain = _obtain_upload({"Path": Path, "ROOT": tmp_path})
    monkeypatch.setitem(sys.modules, "google.colab", None)
    assert obtain(str(path), (".csv", ".json", ".jsonl"), "BYOD_PATH") == path.resolve()
    with pytest.raises(RuntimeError, match="set BYOD_PATH to its path"):
        obtain("", (".csv", ".json", ".jsonl"), "BYOD_PATH")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {})
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError, match="No file was uploaded"):
        obtain("", (".csv", ".json", ".jsonl"), "BYOD_PATH")
    assert "BYOD_PATH = ''  # @param {type:\"string\"}" in "\n".join(KERNEL)


def test_m6_minor_section_10_says_where_byod_documents_come_from():
    assert 'new_kind = "BYOD test records (already scored in Section 9; not unseen documents)"' in RUNNER
    assert "Under BYOD there are no spare documents, so they are the first four of your **test** records" in MARKDOWN


def test_m7_minor_no_doubled_braces_in_markdown():
    assert "{{" not in MARKDOWN and "}}" not in MARKDOWN
    assert "[A-Za-z0-9_.:-]{1,64}" in MARKDOWN
    assert re.search(r"`\{id, source, targets\}`", MARKDOWN)

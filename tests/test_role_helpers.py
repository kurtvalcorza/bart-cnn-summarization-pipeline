"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from bart_summarization_pipeline import (
    DECISION_RULE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_NUM_BEAMS,
    GENERATION_CONFIG_FILE,
    INPUT_SCHEMA,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MAX_NUM_BEAMS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)

DOC = "First paragraph of a report.\n\nSecond paragraph with more detail.\n\nThird paragraph closing."


def _result(n: int = 40, stopped_by: str = "eos") -> dict:
    return {
        "summary": "A short summary.",
        "generated_tokens": n,
        "input_tokens": 120,
        "truncated": stopped_by != "eos",
        "stopped_by": stopped_by,
        "generation": {"num_beams": 4, "decision_rule": DECISION_RULE},
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([DOC, "One line."], num_beams=2, names=["d1", "d2"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["text_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["input_tokens"] == [1, MAX_INPUT_TOKENS]
    assert manifest["schema"]["max_new_tokens"] == [1, MAX_NEW_TOKENS]
    assert manifest["schema"]["num_beams"] == [1, MAX_NUM_BEAMS]
    assert manifest["schema"]["defaults_from"] == GENERATION_CONFIG_FILE
    assert manifest["inputs"] == [
        {"id": "d1", "chars": len(DOC), "paragraphs": 3},
        {"id": "d2", "chars": 9, "paragraphs": 1},
    ]
    assert manifest["generation"]["num_beams"] == 2
    assert manifest["generation"]["max_new_tokens"] == DEFAULT_MAX_NEW_TOKENS
    assert manifest["generation"]["min_new_tokens"] == DEFAULT_MIN_NEW_TOKENS
    assert manifest["generation"]["do_sample"] is False
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_and_settings() -> None:
    manifest = validate_inputs(["A document."])
    assert [entry["id"] for entry in manifest["inputs"]] == ["doc-0"]
    assert manifest["generation"]["num_beams"] == DEFAULT_NUM_BEAMS
    assert manifest["generation"]["decision_rule"] == DECISION_RULE


def test_validate_inputs_rejects_like_the_core_method() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs("a bare string")
    with pytest.raises(ValueError, match="at least one"):
        validate_inputs([])
    with pytest.raises(ValueError, match="is empty"):
        validate_inputs(["  "])
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        validate_inputs(["x" * (MAX_TEXT_CHARS + 1)])
    with pytest.raises(TypeError, match="max_new_tokens must be an int"):
        validate_inputs([DOC], max_new_tokens=True)
    with pytest.raises(ValueError, match=f"between 1 and {MAX_NEW_TOKENS}"):
        validate_inputs([DOC], max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(ValueError, match="exceeds max_new_tokens"):
        validate_inputs([DOC], max_new_tokens=5, min_new_tokens=6)
    with pytest.raises(ValueError, match=f"between 1 and {MAX_NUM_BEAMS}"):
        validate_inputs([DOC], num_beams=MAX_NUM_BEAMS + 1)
    with pytest.raises(TypeError, match="length_penalty must be a float"):
        validate_inputs([DOC], length_penalty=None)
    with pytest.raises(ValueError, match="length_penalty must be within"):
        validate_inputs([DOC], length_penalty=99.0)
    with pytest.raises(ValueError, match="no_repeat_ngram_size"):
        validate_inputs([DOC], no_repeat_ngram_size=-1)
    with pytest.raises(ValueError, match="names must have one entry per text"):
        validate_inputs([DOC], names=["a", "b"])


def test_evaluation_report_is_not_measurable_without_references() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_generated_tokens"] == 40
    assert report["truncated"] is False
    assert report["sample_kind"] == "synthetic"
    assert "no reference summary" in report["reason"]
    assert "ROUGE-1/2/L" in report["needs"]
    assert "faithfulness" in report["needs"]
    assert DECISION_RULE in report["score_semantics"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_scores_one_document_as_sample_sanity() -> None:
    result = {**_result(141, "max_new_tokens"), "summary": "the council approved the plan"}
    report = evaluation_report(result, ["the council approved the plan", "other"], sample_kind="BYOD upload")
    assert report["verdict"] == "sample-sanity"
    assert {m["id"]: m["value"] for m in report["metrics"]} == {
        "rouge1": 100.0,
        "rouge2": 100.0,
        "rougeL": 100.0,
    }
    assert report["truncated"] is True
    assert report["sample_kind"] == "BYOD upload"
    assert "single document" in report["reason"]
    with pytest.raises(ValueError, match="at least one non-empty"):
        evaluation_report(result, [" "])

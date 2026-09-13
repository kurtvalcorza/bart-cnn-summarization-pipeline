import hashlib
import json
import re
from pathlib import Path

import pytest

from bart_summarization_pipeline import (
    DECISION_RULE,
    DEFAULT_LENGTH_PENALTY,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_NUM_BEAMS,
    DEFAULT_WEIGHTS_DIR,
    LENGTH_PENALTY_RANGE,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MAX_NO_REPEAT_NGRAM_SIZE,
    MAX_NUM_BEAMS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    BARTSummarizationPipeline,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]
DOC = "The council met on Tuesday. It approved the budget. Members praised the plan."


def _fake_runner(text: str, settings: dict) -> tuple[str, int, str]:
    words = text.split()[: settings["max_new_tokens"]]
    stopped_by = "eos" if len(words) < settings["max_new_tokens"] else "max_new_tokens"
    return " ".join(words), len(words), stopped_by


def _count_tokens(text: str) -> int:
    return len(text.split()) + 2


def _pipeline(runner=_fake_runner, counter=_count_tokens) -> BARTSummarizationPipeline:
    return BARTSummarizationPipeline(runner, counter, "cpu", "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / "model.safetensors").write_bytes(payload)
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "facebook/bart-large-cnn"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY


def test_defaults_match_pinned_generation_config_when_present():
    """The defaults are the snapshot's generation_config_for_summarization.json (max/min_length minus 1)."""
    path = DEFAULT_WEIGHTS_DIR / "generation_config_for_summarization.json"
    if not path.is_file():
        pytest.skip("local snapshot not staged")
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["num_beams"] == DEFAULT_NUM_BEAMS
    assert config["length_penalty"] == DEFAULT_LENGTH_PENALTY
    assert config["no_repeat_ngram_size"] == DEFAULT_NO_REPEAT_NGRAM_SIZE
    assert config["max_length"] - 1 == DEFAULT_MAX_NEW_TOKENS
    assert config["min_length"] - 1 == DEFAULT_MIN_NEW_TOKENS
    assert config["early_stopping"] is True


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    result = verify_snapshot(_write_snapshot(tmp_path).parent)
    assert result["revision"] == MODEL_REVISION and result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    manifest["files"][0]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_size_missing_file_and_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        BARTSummarizationPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    assert len(verify_snapshot(tmp_path)["files"]) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_summarize_rejects_bad_inputs():
    pipe = _pipeline()
    with pytest.raises(TypeError):
        pipe.summarize(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        pipe.summarize("   ")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        pipe.summarize("a" * (MAX_TEXT_CHARS + 1))
    with pytest.raises(TypeError, match="max_new_tokens"):
        pipe.summarize(DOC, max_new_tokens=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_new_tokens"):
        pipe.summarize(DOC, max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        pipe.summarize(DOC, max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(ValueError, match="min_new_tokens"):
        pipe.summarize(DOC, min_new_tokens=-1)
    with pytest.raises(ValueError, match="exceeds max_new_tokens"):
        pipe.summarize(DOC, max_new_tokens=10, min_new_tokens=11)
    with pytest.raises(TypeError, match="num_beams"):
        pipe.summarize(DOC, num_beams=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_beams"):
        pipe.summarize(DOC, num_beams=MAX_NUM_BEAMS + 1)
    with pytest.raises(TypeError, match="length_penalty"):
        pipe.summarize(DOC, length_penalty="2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length_penalty"):
        pipe.summarize(DOC, length_penalty=LENGTH_PENALTY_RANGE[1] + 1)
    with pytest.raises(ValueError, match="no_repeat_ngram_size"):
        pipe.summarize(DOC, no_repeat_ngram_size=MAX_NO_REPEAT_NGRAM_SIZE + 1)


def test_summarize_rejects_input_over_token_ceiling():
    with pytest.raises(ValueError, match="MAX_INPUT_TOKENS"):
        _pipeline(counter=lambda _: MAX_INPUT_TOKENS + 1).summarize(DOC)


def test_summarize_output_fields_and_pinned_defaults():
    result = _pipeline().summarize(DOC)
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert result["summary"] == DOC and result["generated_tokens"] == 13
    assert result["input_tokens"] == 15
    assert result["truncated"] is False and result["stopped_by"] == "eos"
    assert result["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "min_new_tokens": DEFAULT_MIN_NEW_TOKENS,
        "num_beams": DEFAULT_NUM_BEAMS,
        "length_penalty": DEFAULT_LENGTH_PENALTY,
        "no_repeat_ngram_size": DEFAULT_NO_REPEAT_NGRAM_SIZE,
        "early_stopping": True,
        "do_sample": False,
        "decision_rule": DECISION_RULE,
    }
    assert result["device"] == "cpu" and result["source"] == "injected"


def test_summarize_reports_truncation_at_max_new_tokens():
    result = _pipeline().summarize(DOC, max_new_tokens=5, min_new_tokens=0, length_penalty=1)
    assert result["generated_tokens"] == 5
    assert result["truncated"] is True and result["stopped_by"] == "max_new_tokens"
    assert result["generation"]["length_penalty"] == 1.0


def test_summarize_rejects_backend_contract_violation():
    with pytest.raises(RuntimeError, match="runner must return"):
        _pipeline(runner=lambda t, s: (t, "3", "eos")).summarize(DOC)

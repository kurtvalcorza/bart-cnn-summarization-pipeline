# ruff: noqa: E501  -- the stand-in config line is kept on one line
import builtins

import pytest

MODEL_LIBRARIES = {"torch", "transformers", "timm", "gliner", "safetensors", "huggingface_hub"}


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.fixture(scope="session")
def tiny_snapshot(tmp_path_factory):
    """A tiny randomly initialised BART with the pinned snapshot's own config (12 decoder layers, forced BOS/EOS,
    generation defaults) and its real tokenizer files, staged with a manifest so the carried `from_pretrained`,
    `summarize`, `evaluate`, `adapt`, `save_artifact` and `from_artifact` run unmodified. It exercises control flow
    and transformers' generation logic, not BART-large-CNN's numerics. Skipped where torch/transformers are absent
    (CI installs neither)."""
    import hashlib
    import json
    import shutil
    from pathlib import Path

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from bart_summarization_pipeline import MODEL_ID, MODEL_KEY, MODEL_REVISION

    source = Path(__file__).resolve().parents[1] / "weights" / MODEL_KEY
    snapshot = tmp_path_factory.mktemp("tiny") / MODEL_KEY
    snapshot.mkdir()
    config = transformers.BartConfig.from_pretrained(str(source))
    config.update({"d_model": 16, "encoder_layers": 1, "decoder_layers": 12, "encoder_ffn_dim": 32, "decoder_ffn_dim": 32, "encoder_attention_heads": 2, "decoder_attention_heads": 2})
    torch.manual_seed(0)
    transformers.BartForConditionalGeneration(config).save_pretrained(str(snapshot), safe_serialization=True)
    for name in ("generation_config.json", "generation_config_for_summarization.json", "merges.txt", "tokenizer.json", "vocab.json"):
        shutil.copy(source / name, snapshot / name)
    files = sorted(p.name for p in snapshot.iterdir())
    entries = [{"path": f, "bytes": (snapshot / f).stat().st_size, "sha256": hashlib.sha256((snapshot / f).read_bytes()).hexdigest()} for f in files]
    manifest = {"format": "dimer_hf_snapshot", "formatVersion": 1, "modelKey": MODEL_KEY, "modelId": MODEL_ID, "revision": MODEL_REVISION, "files": entries, "totalBytes": sum(e["bytes"] for e in entries)}
    (snapshot / "dimer-base-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return snapshot

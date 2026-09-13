"""Abstractive summarization with the pinned ``facebook/bart-large-cnn`` checkpoint.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. One task method, ``summarize``: deterministic beam search
whose defaults are the values in the snapshot's ``generation_config_for_summarization.json``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "facebook/bart-large-cnn"
MODEL_REVISION = "37f520fa929c961707657b28798b30c003dd100b"
MODEL_LICENSE = "mit"
MODEL_KEY = "bart-large-cnn"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Ceilings. 1024 is max_position_embeddings in the snapshot config.json; longer inputs are rejected, not cut.
MAX_INPUT_TOKENS = 1024
MAX_TEXT_CHARS = 40_000  # pre-tokenisation guard on the input string; ~4 chars per BPE token on English text
MAX_NEW_TOKENS = 512  # ceiling on decoder steps per call
MAX_NUM_BEAMS = 8
MAX_NO_REPEAT_NGRAM_SIZE = 10
LENGTH_PENALTY_RANGE = (-5.0, 5.0)
# Defaults read from the snapshot's generation_config_for_summarization.json (identical to
# generation_config.json): num_beams 4, length_penalty 2.0, no_repeat_ngram_size 3, early_stopping true,
# max_length 142, min_length 56. Upstream's max_length/min_length count the decoder start token, so the
# equivalent *new*-token bounds are one lower.
DEFAULT_NUM_BEAMS = 4
DEFAULT_LENGTH_PENALTY = 2.0
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_MAX_NEW_TOKENS = 141  # pinned max_length 142 - 1
DEFAULT_MIN_NEW_TOKENS = 55  # pinned min_length 56 - 1
EARLY_STOPPING = True
DECISION_RULE = (
    "deterministic beam search (do_sample=False, early_stopping=True): the highest length-penalised "
    "log-probability beam is returned; no sampling, no seed"
)
GENERATION_CONFIG_FILE = "generation_config_for_summarization.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _check_settings(
    max_new_tokens: Any, min_new_tokens: Any, num_beams: Any, length_penalty: Any, no_repeat_ngram_size: Any
) -> dict[str, Any]:
    """Raise TypeError/ValueError naming the first violated generation ceiling; return the settings."""
    for name, value, low, high in (
        ("max_new_tokens", max_new_tokens, 1, MAX_NEW_TOKENS),
        ("min_new_tokens", min_new_tokens, 0, MAX_NEW_TOKENS),
        ("num_beams", num_beams, 1, MAX_NUM_BEAMS),
        ("no_repeat_ngram_size", no_repeat_ngram_size, 0, MAX_NO_REPEAT_NGRAM_SIZE),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}, got {value}")
    if min_new_tokens > max_new_tokens:
        raise ValueError(f"min_new_tokens {min_new_tokens} exceeds max_new_tokens {max_new_tokens}")
    if isinstance(length_penalty, bool) or not isinstance(length_penalty, int | float):
        raise TypeError("length_penalty must be a float")
    if not LENGTH_PENALTY_RANGE[0] <= length_penalty <= LENGTH_PENALTY_RANGE[1]:
        raise ValueError(f"length_penalty must be within {LENGTH_PENALTY_RANGE}, got {length_penalty}")
    return {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "num_beams": num_beams,
        "length_penalty": float(length_penalty),
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": EARLY_STOPPING,
        "do_sample": False,
        "decision_rule": DECISION_RULE,
    }


def _check_text(text: Any, name: str = "text") -> str:
    if not isinstance(text, str):
        raise TypeError(f"{name} must be str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError(f"{name} is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"{name} has {len(text)} chars; ceiling is MAX_TEXT_CHARS={MAX_TEXT_CHARS}")
    return text


def _check_input_tokens(n_input: int) -> int:
    """The encoder-token ceiling, applied once the tokenizer has counted."""
    if n_input > MAX_INPUT_TOKENS:
        raise ValueError(f"input is {n_input} tokens; ceiling is MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}")
    return n_input


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one non-empty str: the document to summarise (English news-style prose)",
    "text_chars": [1, MAX_TEXT_CHARS],
    "input_tokens": [1, MAX_INPUT_TOKENS],
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "min_new_tokens": [0, MAX_NEW_TOKENS],
    "num_beams": [1, MAX_NUM_BEAMS],
    "length_penalty": list(LENGTH_PENALTY_RANGE),
    "no_repeat_ngram_size": [0, MAX_NO_REPEAT_NGRAM_SIZE],
    "defaults_from": GENERATION_CONFIG_FILE,
    "decision_rule": DECISION_RULE,
    "preprocessing": (
        "byte-level BPE encoding with <s>/</s> added and no truncation: an input over MAX_INPUT_TOKENS is "
        "rejected with a ValueError naming the count, never cut"
    ),
}


def validate_inputs(
    texts: Sequence[str],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
    num_beams: int = DEFAULT_NUM_BEAMS,
    length_penalty: float = DEFAULT_LENGTH_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``summarize`` would: both route through ``_check_text``
    and ``_check_settings``. ``summarize`` takes one text per call, so ``texts`` is the batch the notebook
    will loop over and every entry is validated with the same settings. The encoder-token ceiling
    (``MAX_INPUT_TOKENS``) needs the loaded tokenizer and is enforced inside ``summarize``.
    """
    if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of str, not a single string")
    if not texts:
        raise ValueError("texts must hold at least one item")
    checked = [_check_text(text, f"texts[{i}]") for i, text in enumerate(texts)]
    settings = _check_settings(
        max_new_tokens, min_new_tokens, num_beams, length_penalty, no_repeat_ngram_size
    )
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per text")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[i] if names else f"doc-{i}",
                "chars": len(text),
                "paragraphs": len(text.split("\n\n")),
            }
            for i, text in enumerate(checked)
        ],
        "generation": settings,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], references: Sequence[str] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even though no metric exists here.

    The repository ships no ROUGE (or any) metric helper, so the verdict is always ``not-measurable``
    (EVAL9). ``references`` exists for interface parity with the fleet's other pipelines and is recorded in
    ``reason`` rather than scored: ROUGE needs a scorer and enough referenced documents to state a
    dispersion, and a proxy such as compression ratio or copy rate would misrepresent a plumbing check
    as a quality measurement.
    """
    generation = result.get("generation", {})
    supplied = references is not None
    return {
        "task": "abstractive summarization of one English document (CNN/DailyMail fine-tune)",
        "score_semantics": (
            "the pipeline emits no probability, confidence or score: generated_tokens, input_tokens, "
            "truncated and stopped_by are counts and flags, and "
            f"{generation.get('decision_rule', DECISION_RULE)} produces some token at every step with no "
            "minimum-probability cut-off and no shipped acceptance threshold"
        ),
        "sample_kind": sample_kind,
        "n_generated_tokens": int(result.get("generated_tokens", 0)),
        "truncated": bool(result.get("truncated", False)),
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": (
            "the repository ships no ROUGE or other metric helper and a summary has no ground truth here"
            + (
                "; reference summaries were supplied but no metric helper exists to score them, and one "
                "reference is not a dispersion"
                if supplied
                else "; the evaluated sample has no reference summary"
            )
        ),
        "needs": (
            "one or more reference summaries per document from the deployment domain over enough documents "
            "to state a dispersion, scored with the caller's own ROUGE-1/2/L implementation (the upstream "
            "card reports ROUGE on CNN/DailyMail; nothing here reproduces it), plus a faithfulness check "
            "against the source, excluding or re-running outputs whose truncated flag is true; no proxy such "
            "as compression ratio or copy rate substitutes for that"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@dataclass
class BARTSummarizationPipeline:
    """``_runner(text, settings)`` -> ``(summary_text, generated_tokens, stopped_by)``;
    ``_count_tokens(text)`` -> encoder token count incl. <s>/</s>. Both injectable so tests run offline."""

    _runner: Callable[[str, dict[str, Any]], tuple[str, int, str]]
    _count_tokens: Callable[[str], int]
    device: str = "cpu"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BARTSummarizationPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, kwargs, source = str(root), {"local_files_only": True}, "local-snapshot"
        elif allow_download:
            location, kwargs, source = MODEL_ID, {}, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizerFast

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = BartTokenizerFast.from_pretrained(
            location, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = BartForConditionalGeneration.from_pretrained(
            location, revision=MODEL_REVISION, dtype=torch.float32, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()
        special = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}

        def count_tokens(text: str) -> int:
            return len(tokenizer(text, truncation=False)["input_ids"])

        def runner(text: str, settings: dict[str, Any]) -> tuple[str, int, str]:
            enc = tokenizer(text, return_tensors="pt", truncation=False).to(resolved_device)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=settings["max_new_tokens"],
                    min_new_tokens=settings["min_new_tokens"],
                    num_beams=settings["num_beams"],
                    length_penalty=settings["length_penalty"],
                    no_repeat_ngram_size=settings["no_repeat_ngram_size"],
                    early_stopping=settings["early_stopping"],
                    do_sample=False,
                )
            ids = out[0].tolist()
            content = [t for t in ids[1:] if t not in special]  # ids[0] is the decoder start token
            stopped_by = "eos" if tokenizer.eos_token_id in ids[1:] else "max_new_tokens"
            return tokenizer.decode(content, skip_special_tokens=True).strip(), len(content), stopped_by

        return cls(runner, count_tokens, resolved_device, source)

    def summarize(
        self,
        text: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        num_beams: int = DEFAULT_NUM_BEAMS,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    ) -> dict[str, Any]:
        """Summarise one document by deterministic beam search; defaults are the pinned generation config."""
        text = _check_text(text)
        settings = _check_settings(
            max_new_tokens, min_new_tokens, num_beams, length_penalty, no_repeat_ngram_size
        )
        n_input = _check_input_tokens(self._count_tokens(text))
        summary, n_generated, stopped_by = self._runner(text, settings)
        if not (isinstance(summary, str) and isinstance(n_generated, int) and isinstance(stopped_by, str)):
            raise RuntimeError("runner must return (str, int, str)")
        return {
            "summary": summary,
            "generated_tokens": n_generated,
            "input_tokens": n_input,
            "truncated": stopped_by != "eos",
            "stopped_by": stopped_by,
            "generation": settings,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

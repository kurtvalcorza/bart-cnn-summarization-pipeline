"""Abstractive summarization with the pinned ``facebook/bart-large-cnn`` checkpoint.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. One task method, ``summarize``: deterministic beam search
whose defaults are the values in the snapshot's ``generation_config_for_summarization.json``.

The adaptation contract (``evaluate``, ``adapt``, ``save_artifact``, ``from_artifact``) fine-tunes the last
decoder blocks on a validated ``{id, source, targets}`` dataset with validation-ROUGE-L epoch selection and
exports the trained tensors as a safetensors adapter bound to the pinned base weights. The inference contract
above is unchanged by it.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = (
    "40041830399afb5348525ef8354b007ecec4286fdf3524f7e6b54377e17096cb"  # manifest digest of WEIGHT_FILE
)
PARAMETER_COUNT = 406_290_432
DECODER_LAYERS = 12  # config.json decoder_layers
DEFAULT_TRAINABLE_DECODER_LAYERS = 2  # the last two decoder blocks (33,593,344 parameters)
MAX_TRAIN_SOURCE_TOKENS = 512  # source truncation ceiling during adaptation (never at inference)
MAX_TRAIN_TARGET_TOKENS = 64  # target truncation ceiling during adaptation
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.bart-large-cnn.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"


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
    """Evaluation stage: a machine-readable report for one ``summarize`` result.

    With ``references`` (one or more reference summaries for the same document) the report carries the
    ROUGE-1/2/L F1 of that single summary against its best-matching reference (`metrics.py`) with the
    verdict ``sample-sanity`` — one document is a plumbing check, not a quality measurement; the corpus-level
    stage is ``BARTSummarizationPipeline.evaluate``. Without references the verdict is ``not-measurable``.
    """
    from .metrics import rouge_scores

    generation = result.get("generation", {})
    supplied = references is not None
    metrics: list[dict[str, Any]] = []
    if supplied:
        refs = [str(r) for r in references if str(r).strip()]
        if not refs:
            raise ValueError("references must hold at least one non-empty summary")
        scores = rouge_scores(str(result.get("summary", "")), refs)
        metrics = [
            {"id": name, "value": 100.0 * value, "estimation": "single document, best of the references"}
            for name, value in scores.items()
        ]
    return {
        "task": "abstractive summarization of one English document (CNN/DailyMail fine-tune)",
        "score_semantics": (
            "the pipeline emits no probability, confidence or score: generated_tokens, input_tokens, "
            "truncated and stopped_by are counts and flags, and "
            f"{generation.get('decision_rule', DECISION_RULE)} produces some token at every step with no "
            "minimum-probability cut-off and no shipped acceptance threshold; ROUGE, when references are "
            "supplied, is n-gram agreement with those references (own implementation), not faithfulness"
        ),
        "sample_kind": sample_kind,
        "n_generated_tokens": int(result.get("generated_tokens", 0)),
        "truncated": bool(result.get("truncated", False)),
        "metrics": metrics,
        "baselines": [],
        "verdict": "sample-sanity" if supplied else "not-measurable",
        "reason": (
            "ROUGE-1/2/L are computed for one document against its reference summaries with the repository's "
            "own implementation; a single document states no dispersion and is not a quality measurement"
            if supplied
            else "no reference summary was supplied, so ROUGE cannot be computed; "
            "a summary has no ground truth here"
        ),
        "needs": (
            "one or more reference summaries per document from the deployment domain over enough documents "
            "to state a dispersion, scored with `evaluate` (ROUGE-1/2/L, own implementation; the upstream "
            "card reports ROUGE on CNN/DailyMail and nothing here reproduces it), plus a faithfulness check "
            "against "
            "the source, excluding or re-running outputs whose truncated flag is true; no proxy such as "
            "compression ratio or copy rate substitutes for that"
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
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

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

        return cls(runner, count_tokens, resolved_device, source, _model=model, _tokenizer=tokenizer)

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

    # ---- adaptation -----------------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._tokenizer

    def evaluate(self, records: Sequence[Mapping[str, Any]], **generation: Any) -> dict[str, Any]:
        """Summarise every record's source and score it against its references (ROUGE-1/2/L F1)."""
        from .metrics import summary_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        hypotheses, truncated, settings = [], 0, None
        for record in checked:
            result = self.summarize(record["source"], **generation)
            hypotheses.append(result["summary"])
            truncated += result["truncated"]
            settings = result["generation"]
        metrics = summary_metrics(hypotheses, [r["targets"] for r in checked])
        metrics.update(
            {
                "hit_token_ceiling": truncated,
                "generation": settings,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _trainable_names(self, trainable_decoder_layers: int) -> list[str]:
        if (
            not isinstance(trainable_decoder_layers, int)
            or not 1 <= trainable_decoder_layers <= DECODER_LAYERS
        ):
            raise ValueError(f"trainable_decoder_layers must be an int in 1..{DECODER_LAYERS}")
        model, _ = self._require_model()
        first = DECODER_LAYERS - trainable_decoder_layers
        prefixes = tuple(f"model.decoder.layers.{k}." for k in range(first, DECODER_LAYERS))
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 2,
        lr: float = 3e-5,
        batch_size: int = 8,
        trainable_decoder_layers: int = DEFAULT_TRAINABLE_DECODER_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
        eval_generation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised fine-tuning on a validated summarization dataset.

        Only the last `trainable_decoder_layers` decoder blocks train (2 by default; the encoder, the shared
        embeddings, the earlier decoder blocks and the tied output projection stay frozen). Teacher-forced
        cross-entropy on the first reference summary, AdamW at a fixed learning rate with gradient clipping
        at 1.0, sources truncated to MAX_TRAIN_SOURCE_TOKENS and targets to MAX_TRAIN_TARGET_TOKENS
        **during training only**. Epoch 0 records the frozen model's validation ROUGE; every epoch is scored
        on the validation split with `eval_generation` (the pipeline defaults unless given), and the epoch
        with the highest validation ROUGE-L is kept."""
        from .samples import validate_dataset

        if not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        names = self._trainable_names(trainable_decoder_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        generation = dict(eval_generation or {})
        import torch

        torch.manual_seed(seed)
        model, tokenizer = self._require_model()
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = torch.device(self.device)

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            return {
                k: v
                for k, v in self.evaluate(val_checked, **generation).items()
                if k in ("rouge1", "rouge2", "rougeL", "n", "mean_summary_words", "hit_token_ceiling")
            }

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_rouge = entry["val"]["rougeL"] if entry["val"] else -math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_checked[i] for i in order[start : start + batch_size]]
                    encoded = tokenizer(
                        [r["source"] for r in batch],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MAX_TRAIN_SOURCE_TOKENS,
                    )
                    labels = tokenizer(
                        text_target=[r["targets"][0] for r in batch],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MAX_TRAIN_TARGET_TOKENS,
                    )["input_ids"]
                    labels[labels == tokenizer.pad_token_id] = -100
                    out = model(
                        input_ids=encoded["input_ids"].to(device),
                        attention_mask=encoded["attention_mask"].to(device),
                        labels=labels.to(device),
                    )
                    optimiser.zero_grad(set_to_none=True)
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(float(out.loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
                history.append(entry)
                if progress:
                    progress(entry)
                current = entry["val"]["rougeL"] if entry["val"] else math.inf
                if current > best_rouge or not entry["val"]:
                    best_rouge = current
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the base
            # exactly as it was, with every parameter frozen again.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_decoder_layers": trainable_decoder_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation ROUGE-L" if val_checked else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "max_train_source_tokens": MAX_TRAIN_SOURCE_TOKENS,
            "max_train_target_tokens": MAX_TRAIN_TARGET_TOKENS,
            "eval_generation": generation,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted decoder tensors as safetensors with a manifest naming the pinned base."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def _check_artifact_manifest(self, root: Path, manifest: Mapping[str, Any]) -> Path:
        """Refuse an artifact whose manifest is not exactly the one this pipeline writes: the supported
        format and version, the pinned base (id, revision, weight file, digest), exactly one file entry
        named `adapter.safetensors` that resolves inside the artifact directory, and a recorded
        `trainable_decoder_layers` in range. Nothing is deserialised here. The digest check that follows
        detects corruption or drift of the weights relative to the adjacent manifest; it is not
        authenticity against an actor who can replace both files."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not the supported "
                f"{ARTIFACT_FORMAT_VERSION!r}"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file", WEIGHT_FILE) != WEIGHT_FILE:
            raise ValueError("artifact was adapted from a different base weight file")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        adapter = manifest.get("adapter")
        layers = adapter.get("trainable_decoder_layers") if isinstance(adapter, Mapping) else None
        if isinstance(layers, bool) or not isinstance(layers, int):
            raise ValueError("artifact manifest does not record an integer trainable_decoder_layers")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = self._check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        # The exact tensor set the recorded configuration implies — no subset, no extra, no other layer.
        expected = sorted(self._trainable_names(manifest["adapter"]["trainable_decoder_layers"]))
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith("model.decoder.layers."):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable decoder tensor of the base model"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, "
                    f"base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BARTSummarizationPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline

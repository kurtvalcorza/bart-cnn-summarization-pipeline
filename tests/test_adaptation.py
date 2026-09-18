"""Offline tests for the summarization-dataset contract, the pinned corpus reader, ROUGE and the Lead
baseline, BYOD loaders, CSV export, artifact-manifest rejections and adapt() argument validation. Nothing
here imports torch or transformers; the corpus is three crafted JSON-Lines files served through an
injected fetcher."""

from __future__ import annotations

import hashlib
import json

import pytest

from bart_summarization_pipeline import (
    ARTIFACT_FORMAT,
    DECODER_LAYERS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    WEIGHT_SHA256,
    BARTSummarizationPipeline,
    build_sample_dataset,
    check_split_disjoint,
    dataset_digest,
    fetch_corpus,
    fetch_sample_dataset,
    filter_records,
    lead_baseline,
    lead_sentences,
    load_byod_dataset,
    read_corpus,
    rouge_l,
    rouge_n,
    split_dataset,
    summary_metrics,
    validate_dataset,
    write_dataset_csv,
)
from bart_summarization_pipeline import pipeline as pl
from bart_summarization_pipeline import samples as sm

PAPERS = [
    (
        "p01",
        [
            "We study graph neural networks for molecule property prediction.",
            "Our model beats three baselines on two benchmarks.",
            "Code is released.",
        ],
        ["A graph neural network for molecule property prediction that beats three baselines."],
    ),
    (
        "p02",
        [
            "Transformers are hard to train on small data.",
            "We propose a curriculum that orders examples by length.",
            "Accuracy improves by four points.",
        ],
        ["A length curriculum that improves transformer training on small data."],
    ),
    (
        "p03",
        [
            "Reinforcement learning agents forget old tasks.",
            "We add a replay buffer with prioritised sampling.",
            "Forgetting drops by half.",
        ],
        ["Prioritised replay halves forgetting in continual reinforcement learning."],
    ),
    (
        "p04",
        [
            "Speech recognition degrades with accents.",
            "We fine-tune on accented data with adapters.",
            "Word error rate falls.",
        ],
        ["Adapters fine-tuned on accented speech lower word error rate."],
    ),
    (
        "p05",
        [
            "Image captioning models hallucinate objects.",
            "We penalise captions naming absent objects.",
            "Hallucination rate halves.",
        ],
        ["An object-grounded penalty halves caption hallucination."],
    ),
    (
        "p06",
        [
            "Sparse attention scales to long documents.",
            "We route tokens to experts by locality-sensitive hashing.",
            "Memory drops fourfold.",
        ],
        ["Hash-routed sparse attention cuts memory fourfold on long documents."],
    ),
    (
        "p07",
        [
            "Tabular data resists deep learning.",
            "We pretrain a transformer on synthetic tables.",
            "It matches gradient boosting.",
        ],
        ["A transformer pretrained on synthetic tables matches gradient boosting."],
    ),
    (
        "p08",
        [
            "Machine translation for low-resource languages lacks data.",
            "We back-translate monolingual text.",
            "BLEU rises by six.",
        ],
        ["Back-translation raises BLEU by six for low-resource translation."],
    ),
    (
        "p09",
        [
            "Protein structure prediction is costly.",
            "We distil a large model into a small one.",
            "Speed improves ten times.",
        ],
        ["A distilled protein model is ten times faster."],
    ),
    (
        "p10",
        [
            "Robots grasp unfamiliar objects poorly.",
            "We learn grasps from simulated point clouds.",
            "Success rate reaches ninety percent.",
        ],
        ["Simulated point-cloud training reaches ninety percent grasp success."],
    ),
    (
        "p11",
        [
            "Recommender systems amplify popularity bias.",
            "We reweight the loss by item frequency.",
            "Long-tail recall improves.",
        ],
        ["Frequency reweighting improves long-tail recall in recommendation."],
    ),
    (
        "p12",
        [
            "Code models struggle with long files.",
            "We add retrieval over the repository.",
            "Completion accuracy improves.",
        ],
        ["Repository retrieval improves long-file code completion."],
    ),
]


def _rows(prefix="", papers=PAPERS, n_targets=1):
    """Crafted SciTLDR rows; the prefix also goes into the first sentence so the three members are
    source-disjoint like the real release."""
    return [
        {
            "paper_id": f"{prefix}{pid}",
            "source": [f"{prefix.upper()}{src[0]}", *src[1:]] if prefix else src,
            "target": tgts * n_targets,
            "title": "t",
        }
        for pid, src, tgts in papers
    ]


def _records(prefix="r"):
    return [
        {"id": f"{prefix}{i:03d}", "source": " ".join(src), "targets": list(tgts)}
        for i, (_pid, src, tgts) in enumerate(PAPERS)
    ]


def _files():
    def jsonl(rows):
        return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")

    return {
        "train": jsonl(_rows("tr-")),
        "dev": jsonl(_rows("dv-", n_targets=2)),
        "test": jsonl(_rows("te-", n_targets=3)),
    }


def _pin(monkeypatch, files):
    monkeypatch.setattr(
        sm,
        "CORPUS_FILES",
        {k: (f"{k}.jsonl", len(v), hashlib.sha256(v).hexdigest()) for k, v in files.items()},
    )
    monkeypatch.setattr(sm, "CORPUS_PAPERS", {k: 12 for k in files})
    monkeypatch.setattr(sm, "MIN_SAMPLE_SOURCE_CHARS", 50)


def _echo_runner(text, settings):
    words = text.split()[:5]
    return " ".join(words), len(words), "eos"


def _pipeline_without_model():
    return BARTSummarizationPipeline(_echo_runner, lambda text: len(text.split()) + 2)


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_BASE_URL.startswith("https://raw.githubusercontent.com/allenai/scitldr/5ccad9c0")
    assert {k: v[1] for k, v in sm.CORPUS_FILES.items()} == {
        "train": 3_155_015,
        "dev": 1_124_865,
        "test": 1_204_107,
    }
    assert all(len(v[2]) == 64 for v in sm.CORPUS_FILES.values())
    assert sum(SAMPLE_SPLIT.values()) == 450 and set(SAMPLE_SPLIT) == {"train", "validation", "test"}


def test_fetch_corpus_verifies_each_file_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    calls = []

    def fetcher(url):
        calls.append(url)
        return files[url.rsplit("/", 1)[1].removesuffix(".jsonl")]

    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files
    assert len(calls) == 3 and all(u.startswith(sm.CORPUS_BASE_URL) for u in calls)
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_flattens_and_checks_counts(monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    corpus = read_corpus(files)
    assert len(corpus["train"]) == 12 and corpus["train"][0]["id"] == "train-tr-p01"
    assert corpus["train"][0]["source"].startswith("TR-We study graph neural networks")
    assert len(corpus["test"][0]["targets"]) == 3 and corpus["dev"][0]["paper_id"] == "dv-p01"
    with pytest.raises(ValueError, match="missing the test file"):
        read_corpus({"train": files["train"], "dev": files["dev"]})
    monkeypatch.setattr(sm, "CORPUS_PAPERS", {"train": 99, "dev": 12, "test": 12})
    with pytest.raises(ValueError, match="expected 99"):
        read_corpus(files)


def test_filter_and_sample_split_are_seeded_and_disjoint(monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    corpus = read_corpus(files)
    noisy = [
        *corpus["train"],
        {**corpus["train"][0], "id": "dup"},
        {**corpus["train"][1], "id": "short", "source": "Tiny."},
        {**corpus["train"][2], "id": "none", "targets": []},
    ]
    assert len(filter_records(noisy)) == 12
    sizes = {"train": 8, "validation": 3, "test": 4}
    splits = build_sample_dataset(corpus, seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == sizes
    assert splits["train"][0]["id"] == "train-0000" and len(splits["test"][0]["targets"]) == 3
    assert check_split_disjoint(splits) == sizes
    assert build_sample_dataset(corpus, seed=1, sizes=sizes) == splits
    assert build_sample_dataset(corpus, seed=2, sizes=sizes) != splits
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(corpus, sizes={"train": 100, "validation": 1, "test": 1})


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    splits = fetch_sample_dataset(
        cache_dir=tmp_path,
        fetcher=lambda url: files[url.rsplit("/", 1)[1].removesuffix(".jsonl")],
        sizes={"train": 8, "validation": 2, "test": 2},
    )
    assert validate_dataset(splits["train"])["n_records"] == 8


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(forbid_model_imports):
    report = validate_dataset(_records())
    assert report["n_records"] == 12 and report["unique_sources"] == 12
    assert report["references_per_record"] == {"min": 1, "max": 1}
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    single = validate_dataset(
        [{"id": r["id"], "source": r["source"], "target": r["targets"][0]} for r in _records()]
    )
    assert single["records"][0]["targets"] == _records()[0]["targets"]
    good = _records()
    for bad, message in (
        (good[:7], "8..20000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "source": " "}, *good[1:]], "source is empty"),
        ([{**good[0], "source": "x" * 40_001}, *good[1:]], "MAX_TEXT_CHARS"),
        ([{**good[0], "targets": []}, *good[1:]], "non-empty list"),
        ([{**good[0], "targets": "one string"}, *good[1:]], "non-empty list"),
        ([{**good[0], "targets": [""]}, *good[1:]], "non-empty string"),
        ([{**good[0], "targets": ["x" * 2_001]}, *good[1:]], "MAX_TARGET_CHARS"),
        ([{"id": "a", "source": "b"}, *good[1:]], "missing 'targets'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)


def test_split_dataset_deduplicates_and_is_seeded(forbid_model_imports):
    records = [*_records(), {**_records()[0], "id": "dup"}]
    splits = split_dataset(records, val_fraction=0.1, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == 12 and len(splits["test"]) == 2
    assert check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.1, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(records, val_fraction=0.0, test_fraction=0.9)


# --- metrics and baseline -----------------------------------------------------------------------------


def test_rouge_has_the_expected_extremes_and_multi_reference_max(forbid_model_imports):
    assert rouge_n("the cat sat", "the cat sat", 1) == 1.0 and rouge_l("the cat sat", "the cat sat") == 1.0
    assert rouge_n("dogs", "the cat sat", 1) == 0.0 and rouge_n("the cat sat", "the cat sat", 2) == 1.0
    assert rouge_l("cat the sat", "the cat sat") == pytest.approx(2 / 3)
    metrics = summary_metrics(["the cat sat"], [["dogs bark", "the cat sat"]])
    assert metrics["rouge1"] == 100.0 and metrics["rougeL"] == 100.0 and metrics["n"] == 1
    with pytest.raises(ValueError, match="reference lists"):
        summary_metrics(["a"], [["a"], ["b"]])
    with pytest.raises(ValueError, match="at least one reference"):
        summary_metrics(["a"], [[]])


def test_lead_baseline_scores_the_first_sentences(forbid_model_imports):
    records = _records()
    assert (
        lead_sentences(records[0]["source"], 1)
        == "We study graph neural networks for molecule property prediction."
    )
    lead1 = lead_baseline(records)
    assert 0.0 < lead1["rouge1"] < 100.0 and "lead-1" in lead1["baseline"]
    lead3 = lead_baseline(records, n_sentences=3)
    assert lead3["mean_summary_words"] > lead1["mean_summary_words"]
    with pytest.raises(ValueError, match="at least 1"):
        lead_sentences("a. b.", 0)


# --- BYOD loaders and CSV -----------------------------------------------------------------------------


def test_byod_csv_json_jsonl_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records()
    csv_path = write_dataset_csv(records, tmp_path / "data.csv")
    assert load_byod_dataset(csv_path) == records
    (tmp_path / "data.json").write_text(
        json.dumps([{"id": r["id"], "source": r["source"], "target": r["targets"][0]} for r in records]),
        encoding="utf-8",
    )
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.jsonl") == records
    (tmp_path / "bad.csv").write_text("id,source\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "obj.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of records"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.csv.bak").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="csv, .json or .jsonl"):
        load_byod_dataset(tmp_path / "data.csv.bak")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.csv")


# --- adaptation and artifacts without a model ---------------------------------------------------------


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(_records(), batch_size=0)
    with pytest.raises(ValueError, match="trainable_decoder_layers"):
        pipe.adapt(_records(), trainable_decoder_layers=DECODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    # evaluate() only needs the summarize path, so it works with an injected runner (first five words)
    metrics = pipe.evaluate(_records(), num_beams=1)
    assert (
        metrics["n"] == 12 and metrics["verdict"] == "measured-small-sample" and metrics["adapted"] is False
    )
    assert metrics["generation"]["num_beams"] == 1 and 0.0 < metrics["rouge1"] < 100.0
    with pytest.raises(ValueError, match="num_beams"):
        pipe.evaluate(_records(), num_beams=0)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["model.decoder.layers.11.fc1.weight"],
        "adapter": {},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)

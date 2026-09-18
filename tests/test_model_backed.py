"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a
referenced evaluation, a one-epoch adaptation of the last decoder layer on a dozen abstracts, and the
artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import json

import pytest

from bart_summarization_pipeline import DEFAULT_WEIGHTS_DIR, WEIGHT_FILE, BARTSummarizationPipeline

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

PAPERS = [
    (
        "We study graph neural networks for molecule property prediction. "
        "Our model beats three baselines on two benchmarks. "
        "Code is released.",
        "A graph neural network for molecule property prediction that beats three baselines.",
    ),
    (
        "Transformers are hard to train on small data. "
        "We propose a curriculum that orders examples by length. "
        "Accuracy improves by four points.",
        "A length curriculum that improves transformer training on small data.",
    ),
    (
        "Reinforcement learning agents forget old tasks. "
        "We add a replay buffer with prioritised sampling. "
        "Forgetting drops by half.",
        "Prioritised replay halves forgetting in continual reinforcement learning.",
    ),
    (
        "Speech recognition degrades with accents. "
        "We fine-tune on accented data with adapters. "
        "Word error rate falls.",
        "Adapters fine-tuned on accented speech lower word error rate.",
    ),
    (
        "Image captioning models hallucinate objects. "
        "We penalise captions naming absent objects. "
        "Hallucination rate halves.",
        "An object-grounded penalty halves caption hallucination.",
    ),
    (
        "Sparse attention scales to long documents. "
        "We route tokens to experts by locality-sensitive hashing. "
        "Memory drops fourfold.",
        "Hash-routed sparse attention cuts memory fourfold on long documents.",
    ),
    (
        "Tabular data resists deep learning. "
        "We pretrain a transformer on synthetic tables. "
        "It matches gradient boosting.",
        "A transformer pretrained on synthetic tables matches gradient boosting.",
    ),
    (
        "Machine translation for low-resource languages lacks data. "
        "We back-translate monolingual text. "
        "BLEU rises by six.",
        "Back-translation raises BLEU by six for low-resource translation.",
    ),
    (
        "Protein structure prediction is costly. "
        "We distil a large model into a small one. "
        "Speed improves ten times.",
        "A distilled protein model is ten times faster.",
    ),
    (
        "Robots grasp unfamiliar objects poorly. "
        "We learn grasps from simulated point clouds. "
        "Success rate reaches ninety percent.",
        "Simulated point-cloud training reaches ninety percent grasp success.",
    ),
    (
        "Recommender systems amplify popularity bias. "
        "We reweight the loss by item frequency. "
        "Long-tail recall improves.",
        "Frequency reweighting improves long-tail recall in recommendation.",
    ),
    (
        "Code models struggle with long files. "
        "We add retrieval over the repository. "
        "Completion accuracy improves.",
        "Repository retrieval improves long-file code completion.",
    ),
]
RECORDS = [{"id": f"p{i:02d}", "source": s, "targets": [t]} for i, (s, t) in enumerate(PAPERS)]
GEN = {"max_new_tokens": 32, "min_new_tokens": 0, "num_beams": 2}


@pytest.fixture(scope="module")
def pipe():
    return BARTSummarizationPipeline.from_pretrained(device="cpu")


def test_evaluate_scores_references(pipe):
    metrics = pipe.evaluate(RECORDS[:3], **GEN)
    assert metrics["n"] == 3 and 0.0 < metrics["rouge1"] <= 100.0 and metrics["adapted"] is False


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(
        RECORDS[:8], RECORDS[8:11], epochs=1, trainable_decoder_layers=1, batch_size=4, eval_generation=GEN
    )
    assert result["n_trainable"] == 16_796_672 and result["history"][0]["note"] == "frozen model"
    assert all(name.startswith("model.decoder.layers.11.") for name in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"])
    reloaded = BARTSummarizationPipeline.from_artifact(artifact, device="cpu")
    a = [pipe.summarize(r["source"], **GEN)["summary"] for r in RECORDS[:2]]
    b = [reloaded.summarize(r["source"], **GEN)["summary"] for r in RECORDS[:2]]
    assert a == b
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]

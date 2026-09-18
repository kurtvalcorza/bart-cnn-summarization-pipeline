"""Reference-based summarization metrics (ROUGE-1/2/L F1, own implementation) and the Lead baseline.

ROUGE follows the `rouge-score` package's recipe without stemming: lower-case, keep runs of ASCII letters
and digits as tokens, unigram/bigram overlap F1 for ROUGE-1/2, longest-common-subsequence F1 for ROUGE-L
(sentence-level, the whole summary as one sequence). With several references per document the best score
over the references is taken (the SciTLDR convention), then averaged over documents and reported in percent.
Values are close to, but not identical with, `rouge-score` — no stemming, no bootstrap — and neither is a
human judgement of faithfulness.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
METRIC_DEFINITIONS = {
    "rouge1": (
        "unigram overlap F1 between the summary and the best-matching reference, averaged over documents; "
        "percent"
    ),
    "rouge2": (
        "bigram overlap F1 between the summary and the best-matching reference, averaged over documents; "
        "percent"
    ),
    "rougeL": (
        "longest-common-subsequence F1 (whole summary as one token sequence) against the best-matching "
        "reference, averaged over documents; percent"
    ),
    "tokenisation": (
        "lower-cased runs of ASCII letters and digits; no stemming; "
        "rouge-score-style, not rouge-score-identical"
    ),
}


def rouge_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _f1(overlap: int, n_hyp: int, n_ref: int) -> float:
    if overlap == 0 or n_hyp == 0 or n_ref == 0:
        return 0.0
    precision, recall = overlap / n_hyp, overlap / n_ref
    return 2 * precision * recall / (precision + recall)


def rouge_n(hypothesis: str, reference: str, n: int) -> float:
    hyp = rouge_tokens(hypothesis)
    ref = rouge_tokens(reference)
    hyp_grams = Counter(tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1))
    ref_grams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
    overlap = sum((hyp_grams & ref_grams).values())
    return _f1(overlap, sum(hyp_grams.values()), sum(ref_grams.values()))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(hypothesis: str, reference: str) -> float:
    hyp = rouge_tokens(hypothesis)
    ref = rouge_tokens(reference)
    return _f1(_lcs_length(hyp, ref), len(hyp), len(ref))


def rouge_scores(hypothesis: str, references: Sequence[str]) -> dict[str, float]:
    """Best ROUGE-1/2/L F1 over the references for one summary (fractions in 0..1)."""
    if not references:
        raise ValueError("at least one reference is required")
    return {
        "rouge1": max(rouge_n(hypothesis, r, 1) for r in references),
        "rouge2": max(rouge_n(hypothesis, r, 2) for r in references),
        "rougeL": max(rouge_l(hypothesis, r) for r in references),
    }


def summary_metrics(hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Corpus ROUGE-1/2/L F1 in percent over parallel summaries and reference lists."""
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} summaries but {len(references)} reference lists")
    if not hypotheses:
        raise ValueError("no summaries to score")
    per_doc = [rouge_scores(h, r) for h, r in zip(hypotheses, references, strict=True)]
    return {
        "n": len(hypotheses),
        "rouge1": 100.0 * sum(d["rouge1"] for d in per_doc) / len(per_doc),
        "rouge2": 100.0 * sum(d["rouge2"] for d in per_doc) / len(per_doc),
        "rougeL": 100.0 * sum(d["rougeL"] for d in per_doc) / len(per_doc),
        "mean_summary_words": sum(len(h.split()) for h in hypotheses) / len(hypotheses),
        "mean_reference_words": sum(len(r[0].split()) for r in references) / len(references),
        "definitions": dict(METRIC_DEFINITIONS),
    }


def lead_sentences(text: str, n_sentences: int = 1) -> str:
    """The first `n_sentences` sentences of a document (a naive `.!?` split)."""
    if n_sentences < 1:
        raise ValueError("n_sentences must be at least 1")
    return " ".join(_SENTENCE_RE.split(text.strip())[:n_sentences])


def lead_baseline(records: Sequence[Mapping[str, Any]], *, n_sentences: int = 1) -> dict[str, Any]:
    """Lead-N: the first N sentences of the source submitted as the summary — the classic extractive floor."""
    result = summary_metrics(
        [lead_sentences(r["source"], n_sentences) for r in records], [r["targets"] for r in records]
    )
    result["baseline"] = (
        f"lead-{n_sentences} (the first {n_sentences} sentence(s) of the source as the summary)"
    )
    return result

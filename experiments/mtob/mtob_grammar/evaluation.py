from __future__ import annotations

import string
import math
import re
from collections import Counter

import cer
import sacrebleu
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a


def paper_clean(text: str) -> str:
    return text.replace(" =", "").translate(str.maketrans("", "", string.punctuation))


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _f1(overlap: int, predicted: int, reference: int) -> float:
    if not predicted or not reference or not overlap:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2 * precision * recall / (precision + recall)


def _lcs_sequence_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for index, right in enumerate(b, start=1):
            current.append(previous[index - 1] + 1 if left == right else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    if not predictions:
        return totals
    for prediction, reference in zip(predictions, references, strict=True):
        p = re.findall(r"[a-z0-9]+", prediction.casefold())
        r = re.findall(r"[a-z0-9]+", reference.casefold())
        for n, key in ((1, "rouge1"), (2, "rouge2")):
            pg, rg = _ngrams(p, n), _ngrams(r, n)
            overlap = sum((pg & rg).values())
            totals[key] += _f1(overlap, sum(pg.values()), sum(rg.values()))
        lcs = _lcs_sequence_length(p, r)
        totals["rougeL"] += _f1(lcs, len(p), len(r))
        totals["rougeLsum"] += _f1(lcs, len(p), len(r))
    return {key: value / len(predictions) for key, value in totals.items()}


def paper_bleu(predictions: list[str], references: list[str]) -> dict[str, object]:
    tokenizer = Tokenizer13a()
    translations = [tokenizer(value).split() for value in predictions]
    reference_corpus = [[tokenizer(value).split()] for value in references]
    matches_by_order = [0] * 4
    possible_by_order = [0] * 4
    translation_length = 0
    reference_length = 0
    for translation, refs in zip(translations, reference_corpus, strict=True):
        translation_length += len(translation)
        reference_length += len(refs[0])
        merged_ref: Counter[tuple[str, ...]] = Counter()
        for ref in refs:
            ref_counts = Counter()
            for order in range(1, 5):
                ref_counts.update(_ngrams(ref, order))
            merged_ref |= ref_counts
        translation_counts: Counter[tuple[str, ...]] = Counter()
        for order in range(1, 5):
            translation_counts.update(_ngrams(translation, order))
        overlap = translation_counts & merged_ref
        for ngram, count in overlap.items():
            matches_by_order[len(ngram) - 1] += count
        for order in range(1, 5):
            possible_by_order[order - 1] += max(len(translation) - order + 1, 0)
    precisions = [
        matches / possible if possible else 0.0
        for matches, possible in zip(matches_by_order, possible_by_order, strict=True)
    ]
    geo_mean = math.exp(sum(math.log(value) for value in precisions) / 4) if min(precisions) > 0 else 0.0
    ratio = translation_length / reference_length if reference_length else 0.0
    bp = 1.0 if ratio > 1.0 else (math.exp(1.0 - 1.0 / ratio) if ratio > 0 else 0.0)
    return {
        "bleu": geo_mean * bp,
        "precisions": precisions,
        "brevity_penalty": bp,
        "length_ratio": ratio,
        "translation_length": translation_length,
        "reference_length": reference_length,
    }


def evaluate(predictions: list[str], references: list[str]) -> dict[str, object]:
    cleaned_predictions = [paper_clean(value) for value in predictions]
    cleaned_references = [paper_clean(value) for value in references]
    character = cer.calculate_cer_corpus(
        [prediction.split() for prediction in cleaned_predictions],
        [reference.split() for reference in cleaned_references],
    )
    chrf = sacrebleu.corpus_chrf(cleaned_predictions, [cleaned_references])
    return {
        "num_examples": len(references),
        "chrf": {"score": float(chrf.score), "char_order": 6, "word_order": 0, "beta": 2},
        "bleu": paper_bleu(cleaned_predictions, cleaned_references),
        "rouge": rouge_scores(cleaned_predictions, cleaned_references),
        "character": {"cer_score": float(character["mean"])},
        "cleaning": "remove ' =' and ASCII punctuation, matching released MTOB evaluator",
        "metric_implementation": "MTOB Evaluate metrics: ROUGE, BLEU, chrF, and CharacTER",
    }

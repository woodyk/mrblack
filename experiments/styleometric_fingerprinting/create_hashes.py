#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: fingerprint_builder.py
# Description: Generate stable stylometric fingerprints and similarity-preserving hash
# Author: Ms. White
# Created: 2025-05-17

import json
import hashlib
import math
from typing import List, Dict


def normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val - min_val == 0:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def round5(x: float) -> float:
    return round(x, 5)


def extract_vector(data: Dict) -> List[float]:
    vector = []

    # Basic stats
    vector.append(round5(data["basic_stats"]["avg_word_length"]))
    vector.append(round5(data["basic_stats"]["avg_sentence_length"]))

    # POS & lexical
    vector.append(round5(data["part_of_speech"]["lexical_density"]))
    vector.append(round5(data["part_of_speech"]["noun_to_verb_ratio"]))

    # Frequency analysis
    vector.append(round5(data["frequency_analysis"]["hapax_percentage"]))

    # Readability composite
    r = data["readability"]
    readability = (r["flesch_kincaid_grade"] + r["gunning_fog_index"] + r["smog_index"]) / 3
    vector.append(round5(readability))

    # Sentiment
    vector.append(round5(data["sentiment"]["overall_polarity"]))
    vector.append(round5(data["sentiment"]["polarity_variance"]))
    vector.append(round5(data["sentiment"]["overall_subjectivity"]))

    # POS tag entropy
    pos_counts = data["part_of_speech"]["distribution"].values()
    total = sum(pos_counts)
    entropy = -sum((c / total) * math.log2(c / total) for c in pos_counts if c > 0)
    vector.append(round5(entropy))

    # Function word frequency mean
    function_words = [data["part_of_speech"]["distribution"].get(p, 0)
                      for p in ["DT", "IN", "CC", "TO", "PRP", "PRP$", "WDT"]]
    vector.append(round5(sum(function_words) / len(function_words)))

    # TF-IDF entropy
    tfidf = data["summarization"]["tfidf_top_terms"]
    tfidf_flat = [score for block in tfidf for (_, score) in block]
    total_tfidf = sum(tfidf_flat)
    tfidf_entropy = -sum((x / total_tfidf) * math.log2(x / total_tfidf) for x in tfidf_flat if x > 0)
    vector.append(round5(tfidf_entropy))

    # Named entity count normalized
    total_entities = data["named_entities"]["total_entities"]
    word_count = data["basic_stats"]["word_count"]
    vector.append(round5(total_entities / word_count))

    # Cohesion
    vector.append(round5(data["cohesion"]["cohesion_score"]))

    # Paragraph length variance
    para_len = data["basic_stats"]["avg_paragraph_length"]
    sentence_len = data["basic_stats"]["avg_sentence_length"]
    para_var = abs(para_len - sentence_len) / para_len
    vector.append(round5(para_var))

    # N-gram repetition score
    trigrams = data["frequency_analysis"]["most_common_trigrams"]
    ngram_score = sum(freq for _, freq in trigrams) / len(trigrams) if trigrams else 0.0
    vector.append(round5(ngram_score))

    return vector


def compute_sha_fingerprint(vector: List[float]) -> str:
    raw = "|".join(f"{x:.5f}" for x in vector)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def quantize_vector(vector: List[float], levels: int = 16) -> List[int]:
    quantized = []
    for val in vector:
        norm = max(0.0, min(val, 1.0))  # Clamp
        q = int(norm * (levels - 1))
        quantized.append(q)
    return quantized


def custom_similarity_hash(vector: List[float], levels: int = 16) -> str:
    quantized = quantize_vector(vector, levels)
    bits = ''.join(f"{q:04b}" for q in quantized)  # 4 bits per feature
    hex_prefix = f"{int(bits, 2):016x}"  # 64-bit vector → 16 hex chars
    padded = hashlib.sha256(hex_prefix.encode()).hexdigest()[:32]  # make it SHA-like
    return padded


def main(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vector = extract_vector(data)
    sha_fp = compute_sha_fingerprint(vector)
    sim_hash = custom_similarity_hash(vector)

    print("Stylometric Vector:\n", vector)
    print("\nFingerprint (SHA-256, 128-bit):", sha_fp)
    print("Similarity-Preserving Hash:", sim_hash)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fingerprint_builder.py <analysis.json>")
    else:
        main(sys.argv[1])


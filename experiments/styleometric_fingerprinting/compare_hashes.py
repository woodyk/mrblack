#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: compare_fingerprints.py
# Description: Compare two similarity-preserving stylometric hashes using Hamming similarity
# Author: Ms. White
# Created: 2025-05-17

import sys
import hashlib


def hash_to_quantized_bits(sim_hash: str) -> str:
    """
    Extract the original 64-bit quantized fingerprint from the padded SHA-like hash.
    Assumes first 16 hex chars were the encoded bits before SHA-256 padding.
    """
    first_16_hex = sim_hash[:16]
    bitstring = bin(int(first_16_hex, 16))[2:].zfill(64)
    return bitstring


def compare_simhashes(hash1: str, hash2: str) -> float:
    """Compare two similarity-preserving hashes using Hamming similarity."""
    b1 = hash_to_quantized_bits(hash1)
    b2 = hash_to_quantized_bits(hash2)

    hamming_dist = sum(c1 != c2 for c1, c2 in zip(b1, b2))
    similarity = 1.0 - (hamming_dist / len(b1))
    return round(similarity, 5)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_fingerprints.py <hash1> <hash2>")
        sys.exit(1)

    hash1, hash2 = sys.argv[1], sys.argv[2]
    score = compare_simhashes(hash1, hash2)

    print(f"Hamming Similarity Score: {score:.5f}")
    if score > 0.95:
        print("→ Likely same author or strong stylistic match.")
    elif score > 0.85:
        print("→ Possible stylistic similarity.")
    else:
        print("→ Unlikely same author.")

if __name__ == "__main__":
    main()


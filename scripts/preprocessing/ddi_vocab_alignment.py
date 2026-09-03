#!/usr/bin/env python3
"""Vocabulary-aware alignment helpers for medication adjacency matrices."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def ordered_vocabulary_codes(vocab: Mapping[str, Any], key: str = "med_voc") -> list[str]:
    if key not in vocab:
        raise KeyError(f"vocabulary is missing {key!r}")
    idx2word = vocab[key].idx2word
    if isinstance(idx2word, Mapping):
        missing = [index for index in range(len(idx2word)) if index not in idx2word]
        if missing:
            raise ValueError(f"{key}.idx2word is not contiguous; missing indexes: {missing[:5]}")
        codes = [str(idx2word[index]) for index in range(len(idx2word))]
    else:
        codes = [str(code) for code in idx2word]
    if len(codes) != len(set(codes)):
        raise ValueError(f"{key} contains duplicate codes")
    return codes


def align_square_matrix_by_codes(
    matrix: np.ndarray,
    source_codes: Sequence[str],
    target_codes: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reindex a source-ordered square matrix into target vocabulary order."""
    array = np.asarray(matrix)
    source = [str(code) for code in source_codes]
    target = [str(code) for code in target_codes]
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"matrix must be square, received shape={array.shape}")
    if array.shape[0] != len(source):
        raise ValueError(
            f"matrix dimension {array.shape[0]} does not match source vocabulary {len(source)}"
        )
    if len(source) != len(set(source)) or len(target) != len(set(target)):
        raise ValueError("source and target vocabulary codes must be unique")
    source_set = set(source)
    target_set = set(target)
    if source_set != target_set:
        missing = sorted(target_set - source_set)
        extra = sorted(source_set - target_set)
        raise ValueError(
            "source and target vocabulary sets differ: "
            f"missing_from_source={missing[:10]}, extra_in_source={extra[:10]}"
        )
    source_index = {code: index for index, code in enumerate(source)}
    permutation = np.asarray([source_index[code] for code in target], dtype=np.int64)
    aligned = array[np.ix_(permutation, permutation)]
    audit = {
        "source_vocabulary_size": len(source),
        "target_vocabulary_size": len(target),
        "exact_code_set_match": True,
        "same_index_order_before_alignment": source == target,
        "positions_changed": int(np.count_nonzero(permutation != np.arange(len(permutation)))),
        "target_to_source_permutation": permutation.tolist(),
        "nonzero_entries_before": int(np.count_nonzero(array)),
        "nonzero_entries_after": int(np.count_nonzero(aligned)),
    }
    return aligned, audit

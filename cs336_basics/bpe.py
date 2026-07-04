from __future__ import annotations

import os

import regex as re

from .utils import find_chunk_boundaries, split_on_special_tokens

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_PATTERN = re.compile(PAT)

Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]
WordCounts = dict[Word, int]
PairCounts = dict[Pair, int]


def create_bytepair(vocab: WordCounts) -> PairCounts:
    pairs: PairCounts = {}
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def pretokenize(content: str, word_counts: WordCounts | None = None) -> WordCounts:
    if word_counts is None:
        word_counts = {}

    for match in PRETOKEN_PATTERN.finditer(content):
        word = tuple(bytes([b]) for b in match.group().encode("utf-8"))
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts


def get_best_pair(pairs: PairCounts) -> tuple[Pair, int]:
    if not pairs:
        raise ValueError("Cannot choose a merge from an empty pair table.")
    return max(pairs.items(), key=lambda item: (item[1], item[0]))


def create_merge(word_counts: WordCounts, pair: Pair) -> WordCounts:
    merged_counts: WordCounts = {}
    first, second = pair

    for word, count in word_counts.items():
        merged_word: list[bytes] = []
        i = 0

        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                merged_word.append(first + second)
                i += 2
            else:
                merged_word.append(word[i])
                i += 1

        merged_word_tuple = tuple(merged_word)
        merged_counts[merged_word_tuple] = merged_counts.get(merged_word_tuple, 0) + count

    return merged_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256 + i: token.encode("utf-8") for i, token in enumerate(special_tokens)})

    merges: list[Pair] = []
    word_counts: WordCounts = {}

    with open(input_path, "rb") as file:
        if special_tokens:
            boundaries = find_chunk_boundaries(file, 4, special_tokens[0].encode("utf-8"))
        else:
            file.seek(0, os.SEEK_END)
            boundaries = [0, file.tell()]

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            for document in split_on_special_tokens(chunk, special_tokens):
                pretokenize(document, word_counts)

    next_vocab_index = len(vocab)
    while next_vocab_index < vocab_size:
        pairs = create_bytepair(word_counts)
        if not pairs:
            break

        best_pair, _ = get_best_pair(pairs)
        word_counts = create_merge(word_counts, best_pair)
        merges.append(best_pair)
        vocab[next_vocab_index] = best_pair[0] + best_pair[1]
        next_vocab_index += 1

    return vocab, merges

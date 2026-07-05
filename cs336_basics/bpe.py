from __future__ import annotations

import multiprocessing as mp
import os

import regex as re

from .utils import find_chunk_boundaries, split_on_special_tokens

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_PATTERN = re.compile(PAT)

Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]
WordCounts = dict[Word, int]
PairCounts = dict[Pair, int]
ChunkTask = tuple[str, int, int, tuple[str, ...]]


def create_bytepair(vocab: WordCounts) -> PairCounts:
    pairs: PairCounts = {}
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def pretokenize_sequence(content: str) -> list[Word]:
    return [tuple(bytes([b]) for b in match.group().encode("utf-8")) for match in PRETOKEN_PATTERN.finditer(content)]


def pretokenize(content: str, word_counts: WordCounts | None = None) -> WordCounts:
    if word_counts is None:
        word_counts = {}

    for match in PRETOKEN_PATTERN.finditer(content):
        word = tuple(bytes([b]) for b in match.group().encode("utf-8"))
        word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def _pretokenize_chunk(task: ChunkTask) -> WordCounts:
    input_path, start, end, special_tokens = task
    word_counts: WordCounts = {}

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    for document in split_on_special_tokens(chunk, list(special_tokens)):
        pretokenize(document, word_counts)

    return word_counts


def _pretokenize_chunks(tasks: list[ChunkTask], num_processes: int):
    if num_processes == 1 or len(tasks) <= 1:
        yield from map(_pretokenize_chunk, tasks)
        return

    with mp.Pool(processes=num_processes) as pool:
        yield from pool.imap_unordered(_pretokenize_chunk, tasks)


def get_best_pair(pairs: PairCounts) -> tuple[Pair, int]:
    if not pairs:
        raise ValueError("Cannot choose a merge from an empty pair table.")
    return max(pairs.items(), key=lambda item: (item[1], item[0]))


def create_merge(word_counts: WordCounts, pair: Pair) -> WordCounts:
    merged_counts: WordCounts = {}
    first, second = pair

    for word, count in word_counts.items():
        merged_word: list[bytes] | None = None
        i = 0

        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                if merged_word is None:
                    merged_word = list(word[:i])
                merged_word.append(first + second)
                i += 2
            else:
                if merged_word is not None:
                    merged_word.append(word[i])
                i += 1

        merged_word_tuple = word if merged_word is None else tuple(merged_word)
        merged_counts[merged_word_tuple] = merged_counts.get(merged_word_tuple, 0) + count

    return merged_counts


def merge_word(word: Word, pair: Pair) -> Word:
    first, second = pair
    merged: list[bytes] = []
    i = 0

    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            merged.append(first + second)
            i += 2
        else:
            merged.append(word[i])
            i += 1

    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[Pair]]:
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1.")

    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256 + i: token.encode("utf-8") for i, token in enumerate(special_tokens)})

    merges: list[Pair] = []
    word_counts: WordCounts = {}
    input_path_str = os.fspath(input_path)

    with open(input_path, "rb") as file:
        if special_tokens:
            boundaries = find_chunk_boundaries(file, num_processes, special_tokens[0].encode("utf-8"))
        else:
            file.seek(0, os.SEEK_END)
            boundaries = [0, file.tell()]

    tasks: list[ChunkTask] = [
        (input_path_str, start, end, tuple(special_tokens))
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]

    for partial_counts in _pretokenize_chunks(tasks, num_processes):
        for word, count in partial_counts.items():
            word_counts[word] = word_counts.get(word, 0) + count

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

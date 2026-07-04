import os
import sys

import regex as re

from .utils import find_chunk_boundaries, split_on_special_tokens

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def create_bytepair(vocab: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pairs: dict[tuple[bytes, bytes], int] = {}
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            x = (word[i], word[i + 1])
            if x in pairs:
                pairs[x] += freq
            else:
                pairs[x] = freq
    return pairs


def pretokenize(content: str, v_in) -> dict[tuple[bytes, ...], int]:
    for match in re.finditer(PAT, content):
        char_tuple = tuple(bytes([b]) for b in match.group().encode("utf-8"))
        if char_tuple in v_in:
            v_in[char_tuple] += 1
        else:
            v_in[char_tuple] = 1
    return v_in


def get_best_pair(pairs: dict[tuple[bytes, bytes], int]) -> tuple[tuple[bytes, bytes], int]:
    # @TODO need to add if pairs is None check here!
    return max(pairs.items(), key=lambda item: (item[1], item[0]))


def create_merge(v_in: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
    v_out: dict[tuple[bytes, ...], int] = {}
    first, second = pair
    for word, count in v_in.items():
        new_word = []
        i = 0

        while i < len(word):
            # check if there is a combination of first, second pair, if yes append, else skip
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                # collapse
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        v_out[tuple(new_word)] = count

    return v_out


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    first_special_token = special_tokens[0].encode("utf-8")
    # init
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {i: bytes([i - 1]) for i in range(1, 257)}
    vocab[0] = b"<|endoftext|>"
    cur_vocab_index = 257

    v_accumulator: dict[tuple[bytes, ...], int] = {}
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, first_special_token)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            for doc in split_on_special_tokens(chunk, special_tokens):
                v_accumulator = pretokenize(doc, v_accumulator)

    while cur_vocab_index < vocab_size:
        pairs = create_bytepair(v_accumulator)
        best = get_best_pair(pairs)
        v_accumulator = create_merge(v_accumulator, best[0])
        merges.append(best[0])
        first, second = best[0]
        vocab[cur_vocab_index] = first + second
        cur_vocab_index += 1

    # done
    return vocab, merges


# Using sys.argv to grab the path from the command line
file_path = sys.argv[1]


def f(x):
    return x * x


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    train_bpe(file_path, 500, special_tokens)

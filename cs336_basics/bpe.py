import os
import sys

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def create_bytepair(vocab: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Returns:
        dict[tuple[int, int], int]
    """
    pairs: dict[tuple[bytes, bytes], int] = {}
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            x = (word[i], word[i + 1])
            if x in pairs:
                pairs[x] += freq
            else:
                pairs[x] = freq
    return pairs


def create_pretokenizer(input_path) -> dict[tuple[bytes, ...], int]:
    """
    Returns:
        dict[tuple[bytes, ...], int]
    """
    vocab: dict[tuple[bytes, ...], int] = {}
    for match in re.finditer(PAT, input_path):
        group = match.group()
        x = group.encode("utf-8")

        char_tuple = tuple(bytes([b]) for b in x)
        if char_tuple in vocab:
            vocab[char_tuple] += 1
        else:
            vocab[char_tuple] = 1

    return vocab


def get_largest_pairs(pairs) -> dict[tuple[bytes, bytes], int]:
    return {k: v for k, v in pairs.items() if v == max(pairs.values())}


def get_lexicographical_largest_pair(pairs: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    return max(pairs)


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
    # init
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"

    # print(sorted_items)
    pretokenized = create_pretokenizer(input_path)
    for _ in range(vocab_size):
        pairs = create_bytepair(pretokenized)
        best = get_lexicographical_largest_pair(get_largest_pairs(pairs))
        pretokenized = create_merge(pretokenized, best)

    # done
    return vocab, merges


# Using sys.argv to grab the path from the command line
file_path = sys.argv[1]


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
        train_bpe(content, 20, special_tokens)

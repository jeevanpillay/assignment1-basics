import os
import sys

import regex as re

from .utils import find_chunk_boundaries

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


def create_pretokenizer(input_path) -> dict[tuple[bytes, ...], int]:
    vocab: dict[tuple[bytes, ...], int] = {}
    for match in re.finditer(PAT, input_path):
        char_tuple = tuple(bytes([b]) for b in match.group().encode("utf-8"))
        if char_tuple in vocab:
            vocab[char_tuple] += 1
        else:
            vocab[char_tuple] = 1
    return vocab


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
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, first_special_token)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            v = create_pretokenizer("".join(chunk.split(special_tokens[0])))
            while cur_vocab_index < vocab_size:
                sorted_v = sorted(v.items(), key=lambda item: item[1], reverse=True)
                for item in sorted_v:
                    print(item)
                print("next")

                pairs = create_bytepair(v)
                # print("p", sorted(pairs.items(), key=lambda item: item[1], reverse=True))
                if not pairs:
                    break
                best = get_best_pair(pairs)
                # print("best", best)
                v = create_merge(v, best[0])
                merges.append(best[0])
                first, second = best[0]
                vocab[cur_vocab_index] = first + second
                cur_vocab_index += 1

    print(vocab, merges)

    # done
    return vocab, merges


# Using sys.argv to grab the path from the command line
file_path = sys.argv[1]


def f(x):
    return x * x


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    train_bpe(file_path, 500, special_tokens)

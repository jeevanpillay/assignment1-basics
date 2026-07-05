from collections.abc import Iterable, Iterator

from .bpe import Pair, WordCounts, pretokenize


class TestIterator(Iterator):
    def __init__(self, start: int) -> None:
        self.current = start

    def __next__(self):
        raise NotImplementedError()


class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[Pair]
    special_tokens: list[str] | None

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # do something

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        # pre-tokenizer sequence
        v = pretokenize(text)
        print(v)
        for item in self.merges:
            # check if item exists in "v"
            first, second = item
            i = 0
            new_v: WordCounts = {}
            for word, count in v.items():
                merged_word: list[bytes] | None = None
                while i < len(word):
                    if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                        # should merge
                        if merged_word is None:
                            merged_word = list(word[:i])
                        merged_word.append(first + second)
                        i += 2
                    else:
                        # skip
                        if merged_word is not None:
                            merged_word.append(word[i])
                        i += 1

                merged_word_tuple = word if merged_word is None else tuple(merged_word)
                new_v[merged_word_tuple] = count
            v = new_v

        print("v", v)
        for token_pos in self.vocab:
            for word in v:
                for item in word:
                    if self.vocab[token_pos] == item:
                        out.append(token_pos)

        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return TestIterator(1)

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

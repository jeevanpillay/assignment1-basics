from collections.abc import Iterable, Iterator

from .bpe import Pair, merge_word, pretokenize_sequence


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
        words = pretokenize_sequence(text)

        for pair in self.merges:
            words = [merge_word(word, pair) for word in words]

        token_to_id = {token: token_id for token_id, token in self.vocab.items()}

        out: list[int] = []
        for word in words:
            for token in word:
                out.append(token_to_id[token])

        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return TestIterator(1)

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

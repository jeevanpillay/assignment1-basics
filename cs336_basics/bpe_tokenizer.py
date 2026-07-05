import json
import re
from collections.abc import Iterable, Iterator

from cs336_basics.utils import split_with_special_tokens

from .bpe import Pair, merge_word, pretokenize_sequence


class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[Pair]
    special_tokens: list[str] | None

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = merges

        if special_tokens and "" in special_tokens:
            raise ValueError("special_tokens cannot include the empty string")

        self.special_tokens = list(dict.fromkeys(special_tokens)) if special_tokens else None

        if self.special_tokens:
            next_id = max(self.vocab.keys(), default=-1) + 1

            for special_token in self.special_tokens:
                token = special_token.encode("utf-8")

                if token not in self.vocab.values():
                    self.vocab[next_id] = token
                    next_id += 1

            pattern_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_token_pattern = re.compile("(" + "|".join(re.escape(token) for token in pattern_tokens) + ")")
        else:
            self.special_token_pattern = None

        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.special_token_to_id = {
            token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens or []
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {int(token_id): bytes(token_bytes) for token_id, token_bytes in raw_vocab.items()}

        with open(merges_filepath, encoding="utf-8") as f:
            raw_merges = json.load(f)

        merges: list[Pair] = [(bytes(first), bytes(second)) for first, second in raw_merges]

        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self, text: str) -> list[int]:
        words = pretokenize_sequence(text)

        for pair in self.merges:
            words = [merge_word(word, pair) for word in words]

        out: list[int] = []
        for word in words:
            for token in word:
                out.append(self.token_to_id[token])

        return out

    def encode(self, text: str) -> list[int]:
        if self.special_token_pattern is None:
            return self._encode_chunk(text)
        out: list[int] = []
        for piece in split_with_special_tokens(text, self.special_tokens):
            if piece in self.special_token_to_id:
                out.append(self.special_token_to_id[piece])
            else:
                out.extend(self._encode_chunk(piece))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        special_tokens = self.special_tokens or []

        for chunk in iterable:
            buffer += chunk

            guard_end = len(buffer)

            for special_token in special_tokens:
                max_prefix_len = min(len(special_token), len(buffer))
                for prefix_len in range(1, max_prefix_len + 1):
                    if buffer.endswith(special_token[:prefix_len]):
                        guard_end = min(guard_end, len(buffer) - prefix_len)

            split_at = -1
            for index in range(guard_end - 1, -1, -1):
                if buffer[index].isspace():
                    split_at = index
                    break

            if split_at <= 0:
                continue

            yield from self.encode(buffer[:split_at])
            buffer = buffer[split_at:]

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")

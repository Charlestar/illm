"""Incremental UTF-8 detokenization for the two pinned tokenizer contracts.

Decode bytes, not separately decoded token strings. Incomplete UTF-8 stays in
the codec until another token arrives or terminal flush emits replacement text.
"""

from __future__ import annotations

import codecs
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from illm.model.spec import ModelError
from illm.tokenizer.reference import ByteTokenizer, LocalTokenizer, TokenizerAdapter


def _bytelevel_inverse() -> dict[str, int]:
    # The pinned ByteLevel alphabet: printable bytes retain their codepoints;
    # the remaining bytes map in ascending order to U+0100 and above.
    visible = [*range(33, 127), *range(161, 173), *range(174, 256)]
    result = {chr(value): value for value in visible}
    for index, value in enumerate(value for value in range(256) if value not in visible):
        result[chr(256 + index)] = value
    return result


class DetokenizerFactory:
    """Validate/cache token-to-byte mapping once; mutable codec is per request."""

    def __init__(self, tokenizer: TokenizerAdapter, tokenizer_json: Path | None = None) -> None:
        self._tokenizer = tokenizer
        mapping: dict[int, bytes]
        if isinstance(tokenizer, ByteTokenizer):
            mapping = {index + tokenizer.byte_offset: bytes([index]) for index in range(256)}
            mapping.update({tokenizer.pad_token_id: b"", tokenizer.bos_token_id: b"", tokenizer.eos_token_id: b""})
        elif isinstance(tokenizer, LocalTokenizer) and tokenizer_json is not None:
            value = json.loads(tokenizer_json.read_text(encoding="utf-8"))
            if value.get("decoder", {}).get("type") != "ByteLevel" or value.get("model", {}).get("type") != "BPE":
                raise ModelError("UNSUPPORTED", "streaming supports only the pinned ByteLevel BPE decoder")
            inverse = _bytelevel_inverse()
            mapping = {}
            for token, index in value["model"]["vocab"].items():
                try:
                    mapping[index] = bytes(inverse[character] for character in token)
                except KeyError as exc:
                    raise ModelError("UNSUPPORTED", "BPE vocabulary is not in the ByteLevel alphabet") from exc
            for token in value.get("added_tokens", []):
                mapping[token["id"]] = b"" if token["special"] else token["content"].encode("utf-8")
        else:
            raise ModelError("UNSUPPORTED", "tokenizer has no supported streaming byte contract")
        self._mapping: Mapping[int, bytes] = MappingProxyType(mapping)
        self._vocab_size = int(tokenizer.to_dict()["vocab_size"])

    def new(self) -> IncrementalDetokenizer:
        return IncrementalDetokenizer(self._tokenizer, self._mapping, self._vocab_size)


class IncrementalDetokenizer:
    def __init__(self, tokenizer: TokenizerAdapter, mapping: Mapping[int, bytes], vocab_size: int) -> None:
        self._tokenizer = tokenizer
        self._mapping = mapping
        self._vocab_size = vocab_size
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._tokens: list[int] = []
        self._published = ""
        self._finished = False

    def push(self, token_id: int) -> str:
        if self._finished:
            raise ModelError("INVALID_ARGUMENT", "detokenizer is already finalized")
        if type(token_id) is not int or not 0 <= token_id < self._vocab_size:
            raise ModelError("INVALID_ARGUMENT", "token ID is outside streaming vocabulary")
        # Native tokenizers ignores unused model-vocabulary IDs, as does its
        # canonical whole-prefix decode. They are not replacement characters.
        delta = self._decoder.decode(self._mapping.get(token_id, b""), final=False)
        self._tokens.append(token_id)
        self._published += delta
        return delta

    def finish(self) -> str:
        if self._finished:
            raise ModelError("INVALID_ARGUMENT", "detokenizer may be finalized only once")
        self._finished = True
        delta = self._decoder.decode(b"", final=True)
        self._published += delta
        if self._published != self._tokenizer.decode(self._tokens):
            raise ModelError("UNSUPPORTED", "incremental decode differs from canonical tokenizer decode")
        return delta

    @property
    def text(self) -> str:
        return self._published

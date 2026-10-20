"""Deterministic tiny byte tokenizer and explicit input identity."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Protocol, Sequence, cast

from illm.model.spec import ModelError, digest


class TokenizerAdapter(Protocol):
    @property
    def revision(self) -> str: ...

    def encode(self, text: str, *, add_bos: bool | None = None) -> tuple[int, ...]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...

    def input_identity(self, token_ids: Sequence[int], *, add_bos: bool) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


class _EncodedTokens(Protocol):
    @property
    def ids(self) -> list[int]: ...


class _TokenizerBackend(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> _EncodedTokens: ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str: ...


class _TokenizerFactory(Protocol):
    def from_file(self, path: str) -> _TokenizerBackend: ...


@dataclass(frozen=True, slots=True)
class ByteTokenizer:
    vocab_size: int = 259
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    byte_offset: int = 3
    encoding: str = "utf-8"
    template_version: str = "raw-text-no-chat-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if any(type(getattr(self, field)) is not int for field in (
            "vocab_size", "pad_token_id", "bos_token_id", "eos_token_id", "byte_offset", "schema_version",
        )) or self.to_dict() != {
            "vocab_size": 259, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
            "byte_offset": 3, "encoding": "utf-8", "template_version": "raw-text-no-chat-v1", "schema_version": 1,
        }:
            raise ModelError("INVALID_ARGUMENT", "unsupported byte tokenizer contract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size, "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id, "eos_token_id": self.eos_token_id,
            "byte_offset": self.byte_offset, "encoding": self.encoding,
            "template_version": self.template_version, "schema_version": self.schema_version,
        }

    @property
    def revision(self) -> str:
        return "sha256:" + digest(self.to_dict())

    def encode(self, text: str, *, add_bos: bool | None = None) -> tuple[int, ...]:
        if add_bos is None:
            add_bos = True
        if not isinstance(text, str) or type(add_bos) is not bool:
            raise ModelError("INVALID_ARGUMENT", "encode requires text and explicit bool add_bos")
        try:
            encoded = tuple(value + self.byte_offset for value in text.encode(self.encoding, errors="strict"))
        except UnicodeEncodeError as exc:
            raise ModelError("INVALID_ARGUMENT", "text contains invalid Unicode surrogates") from exc
        if add_bos:
            encoded = (self.bos_token_id, *encoded)
        if not encoded:
            raise ModelError("INVALID_ARGUMENT", "empty input requires explicit BOS injection")
        return encoded

    def validate_ids(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        result = tuple(token_ids)
        if any(type(value) is not int or not 0 <= value < self.vocab_size for value in result):
            raise ModelError("INVALID_ARGUMENT", "token ID is outside the vocabulary")
        return result

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        if type(skip_special_tokens) is not bool:
            raise ModelError("INVALID_ARGUMENT", "skip_special_tokens must be bool")
        raw = bytearray()
        for token in self.validate_ids(token_ids):
            if token < self.byte_offset:
                if not skip_special_tokens:
                    raw.extend(("<pad>", "<bos>", "<eos>")[token].encode())
            else:
                raw.append(token - self.byte_offset)
        return raw.decode(self.encoding, errors="replace")

    def input_identity(self, token_ids: Sequence[int], *, add_bos: bool) -> str:
        if type(add_bos) is not bool:
            raise ModelError("INVALID_ARGUMENT", "add_bos must be bool")
        return digest({"schema_version": 1, "tokenizer_revision": self.revision,
                       "template_version": self.template_version, "add_bos": add_bos,
                       "token_ids": list(self.validate_ids(token_ids))})


class LocalTokenizer:
    """Pinned tokenizer.json adapter, raw text only; no implicit Hub downloads."""

    def __init__(self, path: Path, *, revision: str, vocab_size: int, bos_token_id: int) -> None:
        # tokenizers' native extension ships no typing marker. Keep its untyped
        # boundary local and describe only the two operations this adapter uses.
        factory = cast(_TokenizerFactory, importlib.import_module("tokenizers").Tokenizer)
        self._backend = factory.from_file(str(path))
        self._revision = revision
        self._vocab_size = vocab_size
        self._bos_token_id = bos_token_id

    @property
    def revision(self) -> str:
        return self._revision

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "kind": "hf-tokenizers-json", "revision": self.revision,
                "template_version": "raw-text-no-chat-v1", "default_add_bos": False,
                "empty_input_policy": "bos_if_empty", "vocab_size": self._vocab_size}

    def _validate_ids(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        result = tuple(token_ids)
        if any(type(token) is not int or not 0 <= token < self._vocab_size for token in result):
            raise ModelError("INVALID_ARGUMENT", "token ID is outside model vocabulary")
        return result

    def encode(self, text: str, *, add_bos: bool | None = None) -> tuple[int, ...]:
        if not isinstance(text, str) or add_bos is not None and type(add_bos) is not bool:
            raise ModelError("INVALID_ARGUMENT", "encode requires text and bool add_bos")
        try:
            encoded = tuple(self._backend.encode(text, add_special_tokens=False).ids)
        except Exception as exc:
            raise ModelError("INVALID_ARGUMENT", "tokenizer could not encode input") from exc
        if add_bos or not encoded:
            encoded = (self._bos_token_id, *encoded)
        return self._validate_ids(encoded)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        if type(skip_special_tokens) is not bool:
            raise ModelError("INVALID_ARGUMENT", "skip_special_tokens must be bool")
        return str(self._backend.decode(list(self._validate_ids(token_ids)), skip_special_tokens=skip_special_tokens))

    def input_identity(self, token_ids: Sequence[int], *, add_bos: bool) -> str:
        if type(add_bos) is not bool:
            raise ModelError("INVALID_ARGUMENT", "add_bos must be bool")
        return digest({**self.to_dict(), "add_bos": add_bos, "token_ids": list(self._validate_ids(token_ids))})

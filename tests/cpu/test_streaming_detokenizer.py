from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import pytest

from illm.model.spec import ModelError
from illm.tokenizer.reference import ByteTokenizer, LocalTokenizer
from illm.tokenizer.streaming import DetokenizerFactory


@pytest.mark.parametrize("raw", [b"", b"hello", "你好😀".encode(), b"\xe4\xbd", b"\xff\xe4a\x80",
                                  b"\xef\xbf\xbd", bytes(range(256))])
def test_byte_decode_every_prefix_including_terminal_flush(raw: bytes) -> None:
    tokenizer = ByteTokenizer()
    ids = tuple(value + 3 for value in raw)
    for size in range(len(ids) + 1):
        decoder = DetokenizerFactory(tokenizer).new()
        deltas = [decoder.push(token) for token in ids[:size]]
        assert "".join((*deltas, decoder.finish())) == tokenizer.decode(ids[:size])


def test_partial_utf8_is_empty_until_complete_or_terminal() -> None:
    factory = DetokenizerFactory(ByteTokenizer())
    decoder = factory.new()
    assert decoder.push(0xE4 + 3) == ""
    assert decoder.push(0xBD + 3) == ""
    assert decoder.push(0xA0 + 3) == "你"
    assert decoder.finish() == ""
    cancelled = factory.new()
    assert cancelled.push(0xE4 + 3) == ""
    assert cancelled.finish() == "�"


def test_special_tokens_preserve_pending_bytes_and_finalization_is_single() -> None:
    decoder = DetokenizerFactory(ByteTokenizer()).new()
    assert decoder.push(0xE4 + 3) == ""
    assert decoder.push(2) == ""
    assert decoder.push(0xBD + 3) == ""
    assert decoder.push(0) == ""
    assert decoder.push(0xA0 + 3) == "你"
    assert decoder.finish() == ""
    with pytest.raises(ModelError):
        decoder.finish()
    with pytest.raises(ModelError):
        decoder.push(4)


@pytest.mark.parametrize("token", [True, -1, 259, 2.5])
def test_invalid_ids_do_not_mutate_decoder(token: Any) -> None:
    decoder = DetokenizerFactory(ByteTokenizer()).new()
    with pytest.raises(ModelError):
        decoder.push(token)
    assert decoder.push(100) == "a"
    assert decoder.finish() == ""


def test_native_bytelevel_bpe_matches_canonical_prefixes(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    native = cast(Any, importlib.import_module("tokenizers"))
    vocab = {"<s>": 0, "a": 1, "ä": 2, "½": 3, "ł": 4, "ï": 5, "¿": 6, "½a": 7, "Ġ": 8}
    backend = native.Tokenizer(native.models.BPE(vocab=vocab, merges=[]))
    backend.decoder = native.decoders.ByteLevel()
    backend.add_special_tokens(["<s>"])
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    tokenizer = LocalTokenizer(path, revision="test-native-bytelevel", vocab_size=12, bos_token_id=0)
    factory = DetokenizerFactory(tokenizer, path)
    for ids in ((2, 3, 4), (2, 0, 3, 4), (5, 6, 3), (8, 1, 2), (2, 1), (7, 11, 0)):
        for size in range(len(ids) + 1):
            decoder = factory.new()
            deltas = [decoder.push(token) for token in ids[:size]]
            assert "".join((*deltas, decoder.finish())) == tokenizer.decode(ids[:size])

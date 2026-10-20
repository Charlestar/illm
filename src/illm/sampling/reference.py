"""Canonical V0.1 CPU sampler with explicit arithmetic and request ownership.

Binary32 uses IEEE round-to-nearest/even. Temperature remains binary64 until
score rounding; exp_fp32(x) is round32(math.exp(round32(x))). All binary64
reductions are token-ordered left folds, never sum/fsum. CDF prefixes saturate
at one, and the final positive token terminates at one. This module has no
framework, device, global-RNG, filesystem, or event-export dependency.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from illm.canonical import sha256_json


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


class SamplingError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _uint(value: object, bits: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << bits):
        raise SamplingError("INVALID_ARGUMENT", f"{name} must be uint{bits}")
    return value


def fp32(value: float) -> float:
    """Round to IEEE binary32, round-to-nearest/even; overflow becomes infinity."""
    try:
        return float(struct.unpack("<f", struct.pack("<f", value))[0])
    except OverflowError:
        return math.copysign(math.inf, value)


def _ordered_sum(values: Sequence[float]) -> float:
    result = 0.0
    for value in values:
        result = result + value
    return result


def _digest(value: Mapping[str, Any]) -> str:
    return sha256_json(dict(value))


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    mode: str = "greedy"
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    compute_dtype: str = "fp32"
    processor_order_version: int = 1
    rng_algorithm: str = "philox4x32-10"
    rng_contract_version: int = 1
    actual_seed: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SamplingError("INVALID_ARGUMENT", f"{name} must be finite")
            try:
                converted = float(value)
            except OverflowError as exc:
                raise SamplingError("INVALID_ARGUMENT", f"{name} must be finite") from exc
            if not math.isfinite(converted):
                raise SamplingError("INVALID_ARGUMENT", f"{name} must be finite")
            object.__setattr__(self, name, 0.0 if converted == 0.0 else converted)
        if not isinstance(self.mode, str) or self.mode not in {"greedy", "sample"}:
            raise SamplingError("INVALID_ARGUMENT", "mode must be greedy or sample")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise SamplingError("INVALID_ARGUMENT", "top_k must be a nonnegative integer")
        if not 0.0 < self.top_p <= 1.0:
            raise SamplingError("INVALID_ARGUMENT", "top_p must be in (0, 1]")
        if self.mode == "greedy" and (self.temperature != 0.0 or self.top_k != 0 or self.top_p != 1.0):
            raise SamplingError("INVALID_ARGUMENT", "greedy conflicts with sampling parameters")
        if self.mode == "sample" and self.temperature <= 0.0:
            raise SamplingError("INVALID_ARGUMENT", "sample temperature must be positive")
        if (
            type(self.schema_version) is not int or self.schema_version != 1
            or self.compute_dtype != "fp32"
            or type(self.processor_order_version) is not int or self.processor_order_version != 1
            or self.rng_algorithm != "philox4x32-10"
            or type(self.rng_contract_version) is not int or self.rng_contract_version != 1
        ):
            raise SamplingError("UNSUPPORTED_SAMPLING_PARAMETER", "sampling numerical contract is unsupported")
        _uint(self.actual_seed, 64, "actual_seed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "compute_dtype": self.compute_dtype,
            "processor_order_version": self.processor_order_version,
            "rng_algorithm": self.rng_algorithm,
            "rng_contract_version": self.rng_contract_version,
            "actual_seed": self.actual_seed,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SamplingSpec:
        if not isinstance(value, Mapping):
            raise SamplingError("INVALID_ARGUMENT", "SamplingSpec must be an object")
        if any(not isinstance(key, str) for key in value):
            raise SamplingError("INVALID_ARGUMENT", "sampling parameter names must be strings")
        fields = set(cls().to_dict())
        unknown = set(value) - fields
        if unknown:
            raise SamplingError("UNSUPPORTED_SAMPLING_PARAMETER", f"unknown sampling parameters: {sorted(unknown)}")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True, order=True)
class CounterAddress:
    absolute_output_pos: int
    draw_kind: int = 0
    draw_index: int = 0

    def __post_init__(self) -> None:
        _uint(self.absolute_output_pos, 64, "absolute_output_pos")
        _uint(self.draw_kind, 32, "draw_kind")
        _uint(self.draw_index, 32, "draw_index")

    def to_bytes(self) -> bytes:
        return struct.pack("<QII", self.absolute_output_pos, self.draw_kind, self.draw_index)

    def to_words(self) -> tuple[int, int, int, int]:
        return struct.unpack("<IIII", self.to_bytes())

    def to_dict(self) -> dict[str, int]:
        return {
            "absolute_output_pos": self.absolute_output_pos,
            "draw_kind": self.draw_kind,
            "draw_index": self.draw_index,
        }


def philox4x32_10(
    counter: Sequence[int], key: Sequence[int],
) -> tuple[int, int, int, int]:
    """Random123 Philox4x32, exactly ten rounds, unsigned word arithmetic."""
    if len(counter) != 4 or len(key) != 2:
        raise SamplingError("INVALID_ARGUMENT", "Philox requires four counter and two key words")
    c0, c1, c2, c3 = (_uint(value, 32, "counter word") for value in counter)
    k0, k1 = (_uint(value, 32, "key word") for value in key)
    for round_index in range(10):
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & UINT32_MAX,
            p1 & UINT32_MAX,
            ((p0 >> 32) ^ c3 ^ k1) & UINT32_MAX,
            p0 & UINT32_MAX,
        )
        if round_index != 9:
            k0 = (k0 + 0x9E3779B9) & UINT32_MAX
            k1 = (k1 + 0xBB67AE85) & UINT32_MAX
    return c0, c1, c2, c3


@dataclass(frozen=True, slots=True)
class PhiloxDraw:
    address: CounterAddress
    seed: int
    lanes: tuple[int, int, int, int]
    uniform: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address.to_dict(),
            "key_bytes_le": struct.pack("<Q", self.seed).hex(),
            "counter_bytes_le": self.address.to_bytes().hex(),
            "lanes_uint32": list(self.lanes),
            "uniform": self.uniform,
        }


def philox_draw(seed: int, address: CounterAddress) -> PhiloxDraw:
    _uint(seed, 64, "seed")
    lanes = philox4x32_10(address.to_words(), (seed & UINT32_MAX, seed >> 32))
    return PhiloxDraw(address, seed, lanes, lanes[0] * (2.0 ** -32))


@dataclass(frozen=True, slots=True)
class SamplingState:
    request_id: str
    sampling_spec_digest: str
    next_output_position: int = 0
    rng_draws_consumed: int = 0
    committed_draw_addresses: tuple[CounterAddress, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise SamplingError("INVALID_ARGUMENT", "request_id must be a nonempty string")
        if not isinstance(self.sampling_spec_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sampling_spec_digest):
            raise SamplingError("INVALID_ARGUMENT", "state must bind a canonical SamplingSpec digest")
        _uint(self.next_output_position, 64, "next_output_position")
        _uint(self.rng_draws_consumed, 64, "rng_draws_consumed")
        if not isinstance(self.committed_draw_addresses, tuple) or any(
            not isinstance(address, CounterAddress) for address in self.committed_draw_addresses
        ):
            raise SamplingError("INVALID_ARGUMENT", "committed draw addresses must be an immutable tuple")
        if (
            self.rng_draws_consumed != len(self.committed_draw_addresses)
            or len(set(self.committed_draw_addresses)) != len(self.committed_draw_addresses)
            or tuple(sorted(self.committed_draw_addresses)) != self.committed_draw_addresses
            or any(
                address.absolute_output_pos >= self.next_output_position
                or address.draw_kind != 0 or address.draw_index != 0
                for address in self.committed_draw_addresses
            )
        ):
            raise SamplingError("INVALID_ARGUMENT", "sampling state has invalid committed draw addresses")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "sampling_spec_digest": self.sampling_spec_digest,
            "next_output_position": self.next_output_position,
            "rng_draws_consumed": self.rng_draws_consumed,
            "committed_draw_addresses": [address.to_dict() for address in self.committed_draw_addresses],
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def for_request(cls, request_id: str, spec: SamplingSpec) -> SamplingState:
        return cls(request_id=request_id, sampling_spec_digest=spec.digest)


@dataclass(frozen=True, slots=True)
class SamplingDecision:
    request_id: str
    token_id: int
    output_position: int
    mode: str
    spec_digest: str
    base_state_digest: str
    logits_digest: str
    draw: PhiloxDraw | None
    top_k_ids_by_score: tuple[int, ...] = ()
    temperature_scores_by_token: tuple[tuple[int, float], ...] = ()
    weights_by_token: tuple[tuple[int, float], ...] = ()
    fp32_probabilities_by_token: tuple[tuple[int, float], ...] = ()
    normalized_probabilities_by_token: tuple[tuple[int, float], ...] = ()
    top_p_ids_by_score: tuple[int, ...] = ()
    retained_probabilities_by_token: tuple[tuple[int, float], ...] = ()
    cdf_by_token: tuple[tuple[int, float], ...] = ()
    normalization_z: float | None = None
    normalization_t: float | None = None
    normalization_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "token_id": self.token_id,
            "output_position": self.output_position,
            "mode": self.mode,
            "spec_digest": self.spec_digest,
            "base_state_digest": self.base_state_digest,
            "logits_digest": self.logits_digest,
            "draw": None if self.draw is None else self.draw.to_dict(),
            "top_k_ids_by_score": list(self.top_k_ids_by_score),
            "temperature_scores_by_token": list(self.temperature_scores_by_token),
            "weights_by_token": list(self.weights_by_token),
            "fp32_probabilities_by_token": list(self.fp32_probabilities_by_token),
            "normalized_probabilities_by_token": list(self.normalized_probabilities_by_token),
            "top_p_ids_by_score": list(self.top_p_ids_by_score),
            "retained_probabilities_by_token": list(self.retained_probabilities_by_token),
            "cdf_by_token": list(self.cdf_by_token),
            "normalization_z": self.normalization_z,
            "normalization_t": self.normalization_t,
            "normalization_s": self.normalization_s,
        }


def select_cdf(cdf_by_token: Sequence[tuple[int, float]], uniform: float) -> int:
    """Select the first half-open CDF interval, excluding zero-width intervals."""
    if not isinstance(uniform, (int, float)) or isinstance(uniform, bool) or not 0.0 <= uniform < 1.0:
        raise SamplingError("INVALID_ARGUMENT", "uniform must be in [0, 1)")
    if not isinstance(cdf_by_token, (tuple, list)) or not cdf_by_token:
        raise SamplingError("NUMERICAL_ERROR", "CDF must be a nonempty sequence")
    previous = 0.0
    previous_id = -1
    validated: list[tuple[int, float]] = []
    for item in cdf_by_token:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise SamplingError("NUMERICAL_ERROR", "CDF entry must contain token ID and probability")
        token_id, cumulative = item
        if isinstance(cumulative, bool) or not isinstance(cumulative, (int, float)):
            raise SamplingError("NUMERICAL_ERROR", "CDF probability must be numeric and not bool")
        try:
            probability = float(cumulative)
        except OverflowError as exc:
            raise SamplingError("NUMERICAL_ERROR", "CDF probability overflows FP64") from exc
        if (
            type(token_id) is not int or token_id <= previous_id
            or not math.isfinite(probability) or not previous <= probability <= 1.0
        ):
            raise SamplingError("NUMERICAL_ERROR", "CDF ordering or probability is invalid")
        previous = probability
        previous_id = token_id
        validated.append((token_id, previous))
    if previous != 1.0:
        raise SamplingError("NUMERICAL_ERROR", "CDF must terminate at 1")
    for token_id, cumulative in validated:
        if uniform < cumulative:
            return token_id
    raise SamplingError("NUMERICAL_ERROR", "uniform is outside CDF coverage")


def prepare_sample(
    logits: Sequence[float], spec: SamplingSpec, state: SamplingState,
) -> SamplingDecision:
    """Compute a proposal without consuming or mutating any request RNG state."""
    if state.sampling_spec_digest != spec.digest:
        raise SamplingError("INVALID_ARGUMENT", "SamplingSpec does not match immutable request state binding")
    if spec.mode == "greedy" and state.rng_draws_consumed != 0:
        raise SamplingError("INVALID_ARGUMENT", "greedy request state cannot contain RNG draws")
    if spec.mode == "sample" and (
        state.rng_draws_consumed != state.next_output_position
        or any(address.absolute_output_pos != index for index, address in enumerate(state.committed_draw_addresses))
    ):
        raise SamplingError("INVALID_ARGUMENT", "sample state must cover every committed output position")
    if state.next_output_position == UINT64_MAX:
        raise SamplingError("INVALID_ARGUMENT", "output-position counter exhausted; next token cannot commit")
    if not logits:
        raise SamplingError("NUMERICAL_ERROR", "logits contain no finite candidates")
    raw: list[float] = []
    for value in logits:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise SamplingError("INVALID_ARGUMENT", "raw logits must be numbers")
        try:
            converted = fp32(float(value))
        except OverflowError as exc:
            raise SamplingError("NUMERICAL_ERROR", "raw logits overflow FP32") from exc
        if not math.isfinite(converted):
            raise SamplingError("NUMERICAL_ERROR", "raw logits contain NaN or infinity")
        raw.append(converted)
    if spec.mode == "greedy":
        token_id = max(range(len(raw)), key=lambda index: (raw[index], -index))
        return SamplingDecision(
            request_id=state.request_id, token_id=token_id, draw=None, output_position=state.next_output_position,
            mode=spec.mode, spec_digest=spec.digest, base_state_digest=state.digest,
            logits_digest=_digest({"fp32_raw_logits": raw}),
        )

    scores = [fp32(value / spec.temperature) for value in raw]
    if not all(math.isfinite(value) for value in scores):
        raise SamplingError("NUMERICAL_ERROR", "temperature produced non-finite scores")
    score_order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    top_k_ids = score_order[:min(spec.top_k, len(scores))] if spec.top_k else score_order
    token_order = sorted(top_k_ids)
    maximum = max(scores[index] for index in token_order)
    weights = [(index, fp32(math.exp(fp32(scores[index] - maximum)))) for index in token_order]
    z = _ordered_sum([value for _, value in weights])
    if not math.isfinite(z) or z <= 0.0:
        raise SamplingError("NUMERICAL_ERROR", "softmax Z is non-finite or nonpositive")
    probabilities = [(index, fp32(weight / z)) for index, weight in weights]
    t = _ordered_sum([value for _, value in probabilities])
    if not math.isfinite(t) or t <= 0.0:
        raise SamplingError("NUMERICAL_ERROR", "softmax T is non-finite or nonpositive")
    normalized = [(index, value / t) for index, value in probabilities]
    normalized_map = dict(normalized)
    probability_order = sorted(token_order, key=lambda index: (-normalized_map[index], index))
    retained = list(probability_order)
    if spec.top_p < 1.0:
        retained = []
        cumulative = 0.0
        for index, token_id in enumerate(probability_order):
            retained.append(token_id)
            cumulative = cumulative + normalized_map[token_id]
            if index == len(probability_order) - 1:
                cumulative = 1.0
            if cumulative >= spec.top_p:
                break
    retained_order = sorted(retained)
    s = _ordered_sum([normalized_map[index] for index in retained_order])
    if not math.isfinite(s) or s <= 0.0:
        raise SamplingError("NUMERICAL_ERROR", "retained normalization S is invalid")
    final_probabilities = [(index, normalized_map[index] / s) for index in retained_order]
    last_positive = next(index for index, value in reversed(final_probabilities) if value > 0.0)
    cdf: list[tuple[int, float]] = []
    cumulative = 0.0
    for token_id, value in final_probabilities:
        # FP64 left-fold can overshoot 1 by rounding before the final item.
        # Saturation freezes valid probability prefixes; validation stays strict.
        cumulative = min(1.0, cumulative + value)
        if token_id == last_positive:
            cumulative = 1.0
        cdf.append((token_id, cumulative))
    draw = philox_draw(spec.actual_seed, CounterAddress(state.next_output_position))
    return SamplingDecision(
        request_id=state.request_id, token_id=select_cdf(cdf, draw.uniform), draw=draw,
        top_k_ids_by_score=tuple(top_k_ids), weights_by_token=tuple(weights),
        temperature_scores_by_token=tuple(enumerate(scores)),
        fp32_probabilities_by_token=tuple(probabilities),
        normalized_probabilities_by_token=tuple(normalized),
        top_p_ids_by_score=tuple(retained), retained_probabilities_by_token=tuple(final_probabilities),
        cdf_by_token=tuple(cdf), normalization_z=z, normalization_t=t, normalization_s=s,
        output_position=state.next_output_position, mode=spec.mode,
        spec_digest=spec.digest, base_state_digest=state.digest,
        logits_digest=_digest({"fp32_raw_logits": raw}),
    )


def commit_sample(
    state: SamplingState, decision: SamplingDecision, *,
    spec: SamplingSpec, logits: Sequence[float],
) -> SamplingState:
    """Commit against trusted request inputs; caller-supplied proposals are checked.

    The reference recomputes the pure proposal at commit. This binds token,
    candidate diagnostics and Philox key/address/lanes/uniform to the frozen
    request/spec/current logits without inventing a signing/trust mechanism.
    """
    if not isinstance(decision, SamplingDecision):
        raise SamplingError("INVALID_ARGUMENT", "sampling decision has invalid type")
    if decision.request_id != state.request_id:
        raise SamplingError("INVALID_ARGUMENT", "sampling decision belongs to another request")
    if state.digest != decision.base_state_digest or state.next_output_position != decision.output_position:
        raise SamplingError("INVALID_ARGUMENT", "sampling proposal is stale or already committed")
    if type(decision.token_id) is not int or not 0 <= decision.token_id < len(logits):
        raise SamplingError("INVALID_ARGUMENT", "selected token ID is outside trusted logits vocabulary")
    expected = prepare_sample(logits, spec, state)
    if decision.draw is not None and (
        not isinstance(decision.draw, PhiloxDraw)
        or type(decision.draw.seed) is not int or decision.draw.seed != spec.actual_seed
        or not isinstance(decision.draw.lanes, tuple) or len(decision.draw.lanes) != 4
        or any(type(lane) is not int or not 0 <= lane <= UINT32_MAX for lane in decision.draw.lanes)
        or type(decision.draw.uniform) is not float or not 0 <= decision.draw.uniform < 1
    ):
        raise SamplingError("INVALID_ARGUMENT", "Philox draw diagnostics are malformed")
    try:
        actual_identity = _digest(decision.to_dict())
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise SamplingError("INVALID_ARGUMENT", "sampling decision diagnostics are malformed") from exc
    if actual_identity != _digest(expected.to_dict()):
        raise SamplingError("INVALID_ARGUMENT", "sampling decision differs from trusted prepared result")
    addresses = state.committed_draw_addresses
    consumed = state.rng_draws_consumed
    if decision.mode == "sample":
        expected_address = CounterAddress(state.next_output_position)
        if decision.draw is None or decision.draw.address != expected_address or expected_address in addresses:
            raise SamplingError("INVALID_ARGUMENT", "sampling proposal has invalid draw identity")
        addresses = (*addresses, expected_address)
        consumed += 1
    elif decision.mode != "greedy" or decision.draw is not None:
        raise SamplingError("INVALID_ARGUMENT", "greedy proposal cannot consume RNG")
    return SamplingState(state.request_id, state.sampling_spec_digest,
                         state.next_output_position + 1, consumed, addresses)

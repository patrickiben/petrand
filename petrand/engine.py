"""
petrand.engine
==============
Core entropy engine: pools raw sensor events, runs the continuous health tests
(modeled on NIST SP 800-90B RCT/APT), keeps an entropy credit, and extracts seeds
through an HMAC-SHA-512 conditioner.

Credit rule: each event is credited H bits, where H is the min-entropy per sample
that the health tests are calibrated for (``min_entropy_per_sample``), capped at
``max_event_entropy_cap``. An event whose ``estimated_entropy_bits`` is zero or less
is mixed into the pool but earns no credit (used for simulated frames). H defaults to
a 0.5-bit placeholder; replace it with an assessed value (see paper/RESEARCH_PLAN.md).

Conditioning: HMAC-SHA-256 for seeds of up to 32 bytes and HMAC-SHA-512 for longer
seeds, keyed with the domain tag. The default credit target is 320 bits, the output
length plus 64 bits that FIPS 140-3 IG D.K Resolution 19 and SP 800-90C use for full
entropy; full entropy also needs an assessed H and the other conditions listed there.
"""

import os
import time
import struct
import hashlib
import hmac
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Set, Sequence
from petrand.health import NISTHealthMonitor, HealthError


def combine_entropy_estimates(
    estimates: Sequence[float],
    mode: str = "conservative",
    discount: float = 0.5,
    cap: Optional[float] = 4.0,
) -> float:
    """
    Combines min-entropy estimates across multiple concurrent physical sensors.

    Modes
    -----
    - "conservative": max(estimates) — safe against arbitrarily correlated channels
    - "discounted"  : discount * sum(estimates) — accounts for partial cross-channel coupling
    - "additive"    : sum(estimates) — valid only for physically orthogonal noise sources
    """
    clean = [max(0.0, float(e)) for e in estimates]
    if not clean:
        return 0.0

    if mode == "conservative":
        combined = max(clean)
    elif mode == "discounted":
        combined = float(discount) * sum(clean)
    elif mode == "additive":
        combined = sum(clean)
    else:
        raise ValueError(f"Unsupported entropy combination mode: {mode}")

    if cap is not None:
        combined = min(combined, float(cap))
    return combined


def intervals_to_von_neumann_bits(intervals: Sequence[int]) -> bytes:
    """
    Pairwise comparison extractor on inter-arrival intervals (e.g. microseconds),
    in the spirit of Von Neumann de-biasing. Unbiased only if each pair is
    exchangeable (e.g. i.i.d. intervals). Compares non-overlapping pairs (dt_{2i}, dt_{2i+1}):
      - dt_{2i} < dt_{2i+1} -> 0
      - dt_{2i} > dt_{2i+1} -> 1
      - equal               -> discard
    Packs complete 8-bit octets.
    """
    out = bytearray()
    acc = 0
    bit_count = 0

    for i in range(0, len(intervals) - 1, 2):
        a = intervals[i]
        b = intervals[i + 1]
        if a == b:
            continue
        bit = 0 if a < b else 1
        acc = (acc << 1) | bit
        bit_count += 1
        if bit_count == 8:
            out.append(acc)
            acc = 0
            bit_count = 0

    return bytes(out)


# One lock for every caller that polls a sensor (server handlers and the background
# harvester), so two threads never drive the same sensor at once.
SENSOR_POLL_LOCK = threading.RLock()


def fill_until_ready(engine: Any, sensor: Any, timeout_s: float = 10.0) -> bool:
    """
    Polls `sensor` into `engine` until the engine is ready, the engine halts, or
    `timeout_s` seconds pass. Returns True only when ready. A health alarm halts the
    engine (see engine.halt_reason) and makes this return False instead of raising.
    """
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while not engine.is_ready:
        if sensor is None or engine.is_halted or time.monotonic() > deadline:
            return False
        with SENSOR_POLL_LOCK:
            ev = sensor.poll()
        if ev is not None:
            try:
                engine.ingest_event(ev)
            except (HealthError, EngineHaltedError):
                return False
    return True


@dataclass
class RawPhysicalEvent:
    """
    Represents an atomic physical perturbation event (e.g. wall impact, optical delta).
    """
    event_id: int
    timestamp_ns: int
    channel: str
    raw_symbol: int               # Raw sample fed to the health tests (must not mix in host-clock bits)
    sensor_payload: bytes         # High-resolution digitized bytes (audio PCM, frame bytes, etc.)
    estimated_entropy_bits: float # Zero or less: mix into the pool without credit; otherwise credited at H
    ball_id: str = "Ball_A"       # Originating ball ID ("Ball_A" or "Ball_B")


class EngineHaltedError(RuntimeError):
    """Raised when attempting to operate an engine in a halted/failed health state."""
    pass


class InsufficientEntropyError(RuntimeError):
    """Raised when requesting a seed before the required entropy credit has accumulated."""
    pass


class PetRandEngine:
    """
    Entropy engine with health gating modeled on NIST SP 800-90B and an HMAC-SHA-512
    conditioner. Supports the two-ball alternating lifecycle. Not a certified module.
    """

    def __init__(
        self,
        required_entropy_bits: float = 320.0,
        max_event_entropy_cap: float = 4.0,
        min_entropy_per_sample: float = 0.5,
        domain_tag: bytes = b"PetRand-MC-v1.0",
        max_pool_bytes: int = 1_048_576,  # 1 MB cap
        active_ball_id: str = "Ball_A",
        require_startup_test: bool = True,
        startup_samples: int = 1024,
        run_post_self_test: bool = False,
    ):
        self.required_bits = float(required_entropy_bits)
        if not (0.0 < self.required_bits <= 512.0):
            raise ValueError("required_entropy_bits must be in (0, 512]: a full pool folds into a 512-bit digest")
        self.max_event_cap = float(max_event_entropy_cap)
        self.domain_tag = bytes(domain_tag)
        self.max_pool_bytes = int(max_pool_bytes)
        self.active_ball_id = active_ball_id

        self.post_self_test_status = "not run"
        if run_post_self_test:
            from petrand.self_test import SelfTestEngine
            SelfTestEngine.run_all_self_tests()  # raises on any failure
            self.post_self_test_status = "passed"

        self.monitor = NISTHealthMonitor(
            min_entropy_per_sample=min_entropy_per_sample,
            require_startup_test=require_startup_test,
            startup_samples=startup_samples,
        )

        self._pool = bytearray()
        self._entropy_credit = 0.0
        self._lifetime_entropy_bits = 0.0
        self._engine_start_time = time.time()
        self._last_event_time: Optional[float] = None
        self._counter = 0
        self._total_events = 0
        self._is_halted = False
        self._halt_reason: Optional[str] = None
        self._is_swap_paused = False
        self._pool_folds = 0
        self._events_in_pool = 0
        self._extra_inputs_in_pool = 0

        self._contributing_balls: Set[str] = set()
        self._lock = threading.RLock()

    def _zeroize_pool(self) -> None:
        """Actively overwrites raw entropy pool buffer in RAM with zeros before clearing."""
        n = len(self._pool)
        if n > 0:
            self._pool[:] = b"\x00" * n
        self._pool.clear()

    @property
    def is_halted(self) -> bool:
        with self._lock:
            return self._is_halted

    @property
    def is_swap_paused(self) -> bool:
        with self._lock:
            return self._is_swap_paused

    @property
    def halt_reason(self) -> Optional[str]:
        with self._lock:
            return self._halt_reason

    @property
    def entropy_credit(self) -> float:
        with self._lock:
            return self._entropy_credit

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return (
                (not self._is_halted)
                and (not self._is_swap_paused)
                and self.monitor.is_operational
                and (self._entropy_credit >= self.required_bits)
            )

    def pause_for_swap(self) -> None:
        """
        Pauses health monitoring during physical ball swapping.
        Prevents false stuck-source alarms when the ball is removed from the arena.
        """
        with self._lock:
            self._is_swap_paused = True

    def resume_after_swap(self, new_ball_id: str) -> None:
        """
        Resumes harvesting after ball swap is complete and resets the health monitor baseline.
        """
        with self._lock:
            self.active_ball_id = new_ball_id
            self.monitor.reset()
            self._is_swap_paused = False

    def ingest_event(self, event: RawPhysicalEvent) -> None:
        """
        Ingests a raw physical event. Validates health first, then updates entropy pool.
        """
        with self._lock:
            if self._is_halted:
                raise EngineHaltedError(f"Engine is halted: {self._halt_reason}")

            if self._is_swap_paused:
                # Do not credit or test events during active physical swap
                return

            # 1. NIST SP 800-90B Health Check on the raw symbol
            try:
                self.monitor.validate_sample(event.raw_symbol)
            except HealthError as exc:
                self._is_halted = True
                self._halt_reason = str(exc)
                # Zeroize and purge pool immediately on health failure
                self._zeroize_pool()
                self._entropy_credit = 0.0
                self._events_in_pool = 0
                self._extra_inputs_in_pool = 0
                raise

            self._last_event_time = time.time()

            # 2. Conservative entropy budgeting (only once operational / startup passed)
            if not self.monitor.is_operational:
                self._total_events += 1
                return

            # Credit H bits per raw sample (the H the health-test cutoffs assume); a claim of
            # zero or less marks an event that is mixed in without credit.
            claim = float(event.estimated_entropy_bits)
            capped_credit = 0.0 if claim <= 0.0 else min(self.max_event_cap, self.monitor.H)

            # 3. Pack event structure into pool (including ball_id provenance)
            ball_bytes = event.ball_id.encode("utf-8")
            channel_bytes = event.channel.encode("utf-8")
            header = struct.pack(
                "<QIHfH",
                event.timestamp_ns & 0xFFFFFFFFFFFFFFFF,
                event.event_id & 0xFFFFFFFF,
                len(channel_bytes),
                capped_credit,
                len(ball_bytes),
            )
            entry = header + channel_bytes + ball_bytes + event.sensor_payload

            # Pool full: fold it into its SHA-512 digest and cap the credit at the 512 bits a
            # digest can hold, then keep going. Credit never refers to discarded data, and a
            # full pool can no longer stall the engine.
            if len(self._pool) + len(entry) > self.max_pool_bytes:
                digest = hashlib.sha512(bytes(self._pool)).digest()
                self._zeroize_pool()
                self._pool.extend(digest)
                self._entropy_credit = min(self._entropy_credit, 8.0 * len(digest))
                self._pool_folds += 1

            self._pool.extend(entry)
            self._events_in_pool += 1
            self._entropy_credit += capped_credit
            self._lifetime_entropy_bits += capped_credit
            self._total_events += 1
            self._contributing_balls.add(event.ball_id)

    def mix_uncredited(self, data: bytes, label: str = "extra") -> None:
        """
        Mixes extra input, such as a Zoomies! round, into the pool with no entropy credit.
        It is not the primary noise source, so it skips the health tests and the sample count:
        it can add unpredictability but never raises the credit, and stuck or repeated input
        cannot halt the engine. Only a SHA-512 digest of the input enters the pool.
        """
        with self._lock:
            if self._is_halted or self._is_swap_paused:
                return
            label_bytes = label.encode("utf-8")[:64]
            entry = (struct.pack("<QH", time.time_ns() & 0xFFFFFFFFFFFFFFFF, len(label_bytes))
                     + label_bytes + hashlib.sha512(bytes(data)).digest())
            if len(self._pool) + len(entry) > self.max_pool_bytes:
                digest = hashlib.sha512(bytes(self._pool)).digest()
                self._zeroize_pool()
                self._pool.extend(digest)
                self._entropy_credit = min(self._entropy_credit, 8.0 * len(digest))
                self._pool_folds += 1
            self._pool.extend(entry)
            self._extra_inputs_in_pool += 1

    def extract_seed(self, seed_bytes: int = 32, consume: bool = True) -> bytes:
        """
        Extracts a conditioned seed from the pool with HMAC keyed by the domain tag:
        HMAC-SHA-256 for seeds of up to 32 bytes, HMAC-SHA-512 for longer seeds, so the
        default 32-byte seed is a full HMAC-SHA-256 output (no truncation). HMAC with an
        approved hash is a vetted conditioning function in NIST SP 800-90B; the output is
        only as good as the assessed entropy of the raw samples behind the credit.
        """
        if seed_bytes < 16 or seed_bytes > 64:
            raise ValueError("seed_bytes must be between 16 and 64")

        with self._lock:
            if self._is_halted:
                raise EngineHaltedError(f"Engine is halted: {self._halt_reason}")

            if self._is_swap_paused:
                raise RuntimeError("Cannot extract seed while swap is in progress")

            if self._entropy_credit < self.required_bits:
                raise InsufficientEntropyError(
                    f"Insufficient entropy: have {self._entropy_credit:.1f} bits, "
                    f"need {self.required_bits:.1f} bits"
                )

            # Conditioning: HMAC-SHA-256 (up to 32 bytes) or HMAC-SHA-512, keyed with the domain tag
            ctx = hmac.new(self.domain_tag, digestmod=hashlib.sha256 if seed_bytes <= 32 else hashlib.sha512)
            ctx.update(self._counter.to_bytes(8, "little"))
            ctx.update(struct.pack("<d", self._entropy_credit))
            ctx.update(self._pool)
            extracted_seed = ctx.digest()[:seed_bytes]

            if consume:
                # Clear the pool (best effort in Python) and reset credit
                self._counter += 1
                self._zeroize_pool()
                self._entropy_credit = 0.0
                self._events_in_pool = 0
                self._extra_inputs_in_pool = 0
                self._contributing_balls.clear()

            return extracted_seed

    def extract_seed_with_report(self, seed_bytes: int = 32) -> Dict[str, Any]:
        """
        Extracts a seed (consuming the pool) and reports, from the same locked state,
        how it was made: samples pooled, credit spent, conditioner and seed number.
        """
        with self._lock:
            report = {
                "samples": self._events_in_pool,
                "uncredited_extra_inputs": self._extra_inputs_in_pool,
                "health": "passed",   # extraction needs the start-up check passed and no alarm
                "credit_bits": round(self._entropy_credit, 2),
                "h_bits_per_sample": self.monitor.H,
                "conditioner": "HMAC-SHA-256" if seed_bytes <= 32 else "HMAC-SHA-512",
                "seed_number": self._counter + 1,
            }
            seed = self.extract_seed(seed_bytes=seed_bytes, consume=True)
            return {"seed": seed, "report": report}

    def reset_health(self) -> None:
        """Manually resets health monitor state after resolving physical failure."""
        with self._lock:
            self.monitor.reset()
            self._zeroize_pool()
            self._entropy_credit = 0.0
            self._events_in_pool = 0
            self._extra_inputs_in_pool = 0
            self._is_halted = False
            self._halt_reason = None
            self._is_swap_paused = False

    def run_on_demand_self_test(self) -> Dict[str, Any]:
        """
        On-demand self-test: known-answer tests of the health tests and the conditioner.
        """
        from petrand.self_test import SelfTestEngine
        return SelfTestEngine.run_all_self_tests()

    def get_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            elapsed = max(1e-3, now - self._engine_start_time)
            rate_bps = self._lifetime_entropy_bits / elapsed
            seconds_since_last = (now - self._last_event_time) if self._last_event_time is not None else 0.0
            return {
                "halted": self._is_halted,
                "halt_reason": self._halt_reason,
                "swap_paused": self._is_swap_paused,
                "active_ball_id": self.active_ball_id,
                "contributing_balls": list(self._contributing_balls),
                "pool_bytes": len(self._pool),
                "entropy_credit_bits": self._entropy_credit,
                "required_entropy_bits": self.required_bits,
                "estimated_entropy_rate_bps": round(rate_bps, 2),
                "seconds_since_last_event": round(seconds_since_last, 2),
                "counter": self._counter,
                "total_events_processed": self._total_events,
                "ready": self.is_ready,
                "credit_cap_bits_per_event": min(self.max_event_cap, self.monitor.H),
                "events_in_pool": self._events_in_pool,
                "extra_inputs_in_pool": self._extra_inputs_in_pool,
                "pool_folds": self._pool_folds,
                "conditioner": "HMAC-SHA-256 (32-byte seeds); HMAC-SHA-512 (longer)",
                "post_self_test": self.post_self_test_status,
                "is_operational": self.monitor.is_operational,
                "startup_remaining": self.monitor.get_telemetry().get("startup_samples_remaining", 0),
            }

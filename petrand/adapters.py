"""
petrand.adapters
================
Adapters from the entropy engine to scientific computing tools:
- NumPy SeedSequence & parallel Generator spawning (PCG64 / Philox / SFC64)
- SciPy Randomized Quasi-Monte Carlo (Scrambled Sobol)
- Block-reseeded Monte Carlo streams with automatic sensor replenishment
- Seed ledger with a SHA-256 provenance manifest (records whether the source was physical)
"""

import os
import sys
import json
import time
import random
import secrets
import base64
import struct
import hashlib
import threading
from collections import deque
from datetime import datetime, timezone
from typing import List, Generator, Optional, Dict, Any, Sequence, TypeVar
import numpy as np
from petrand.engine import PetRandEngine, fill_until_ready
from petrand.sensors import BaseSensor, SyntheticArenaSimulator

T = TypeVar("T")


def _petrand_version() -> str:
    try:
        from petrand import __version__
        return __version__
    except Exception:
        return "unknown"


def uint64_to_uniform_float64(u64: np.ndarray) -> np.ndarray:
    """
    Standard high-quality conversion of 64-bit unsigned integers to uniform float64
    in [0.0, 1.0) using the 53-bit mantissa IEEE 754 shift.
    """
    return (u64 >> 11) * (2.0 ** -53)


class MonteCarloSeedFactory:
    """
    Bridges PetRandEngine to NumPy BitGenerators and SciPy QMC samplers.
    Optionally binds a physical sensor to automatically harvest entropy when
    multi-block streams or multi-seed ledgers require replenishment.
    """

    def __init__(self, engine: PetRandEngine, sensor: Optional[BaseSensor] = None):
        self.engine = engine
        self.sensor = sensor

    def _ensure_ready(self, sensor: Optional[BaseSensor] = None) -> None:
        """Replenishes engine from bound or provided sensor if not yet ready."""
        active_sensor = sensor or self.sensor
        if active_sensor is None:
            return
        while not self.engine.is_ready:
            ev = active_sensor.poll()
            if ev is not None:
                self.engine.ingest_event(ev)

    def extract_seed(self, seed_bytes: int = 32, consume: bool = True) -> bytes:
        """Ensures entropy readiness via bound sensor (if any) and extracts a conditioned seed."""
        self._ensure_ready()
        return self.engine.extract_seed(seed_bytes=seed_bytes, consume=consume)

    def spawn_numpy_generators(
        self,
        n_workers: int = 1,
        bit_generator: str = "pcg64",
        seed_bytes: int = 32,
    ) -> List[np.random.Generator]:
        """
        Derives n_workers mathematically independent, non-overlapping NumPy Generators
        from a single seed block via SeedSequence.spawn().
        """
        self._ensure_ready()
        seed = self.engine.extract_seed(seed_bytes=seed_bytes)
        words = np.frombuffer(seed, dtype=np.uint32).tolist()
        root_ss = np.random.SeedSequence(words)
        child_seeds = root_ss.spawn(n_workers)

        generators = []
        for ss in child_seeds:
            if bit_generator.lower() == "pcg64":
                bg = np.random.PCG64(ss)
            elif bit_generator.lower() == "philox":
                bg = np.random.Philox(ss)
            elif bit_generator.lower() == "sfc64":
                bg = np.random.SFC64(ss)
            else:
                raise ValueError(f"Unsupported bit_generator: {bit_generator}")
            generators.append(np.random.Generator(bg))

        return generators

    def get_rqmc_sobol_sampler(
        self,
        dimension: int,
        scramble: bool = True,
        seed_bytes: int = 32,
    ):
        """
        Returns a SciPy Scrambled Sobol sampler randomized by physical entropy.
        """
        from scipy.stats import qmc

        self._ensure_ready()
        seed = self.engine.extract_seed(seed_bytes=seed_bytes)

        # Try full Generator seeding if supported by the SciPy version
        try:
            words = np.frombuffer(seed, dtype=np.uint32).tolist()
            ss = np.random.SeedSequence(words)
            rng = np.random.Generator(np.random.PCG64(ss))
            return qmc.Sobol(d=dimension, scramble=scramble, seed=rng)
        except Exception:
            # Fallback to 32-bit integer run seed
            seed32 = int.from_bytes(seed[:4], "little")
            return qmc.Sobol(d=dimension, scramble=scramble, seed=seed32)

    def block_reseeded_stream(
        self,
        total_samples: int,
        block_size: int = 500_000,
        dimension: int = 1,
        bit_generator: str = "pcg64",
        sensor: Optional[BaseSensor] = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Yields batches of float64 samples where every batch is initialized from
        a freshly harvested seed. Useful for batch variance estimation.
        """
        remaining = total_samples
        while remaining > 0:
            current_count = min(block_size, remaining)
            self._ensure_ready(sensor=sensor)
            rng = self.spawn_numpy_generators(n_workers=1, bit_generator=bit_generator)[0]

            if dimension == 1:
                batch = rng.random(current_count, dtype=np.float64)
            else:
                batch = rng.random((current_count, dimension), dtype=np.float64)

            remaining -= current_count
            yield batch

    def write_seed_ledger(
        self,
        bin_path: str,
        manifest_path: str,
        num_seeds: int = 10,
        seed_bytes: int = 32,
        sensor: Optional[BaseSensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Harvests num_seeds into a binary ledger file and writes a cryptographic
        manifest JSON with SHA-256 digests for auditing and exact reproducibility.
        """
        seed_hashes = []
        full_file_hasher = hashlib.sha256()

        with open(bin_path, "wb") as f_bin:
            for _ in range(num_seeds):
                self._ensure_ready(sensor=sensor)
                seed = self.engine.extract_seed(seed_bytes=seed_bytes)
                f_bin.write(seed)
                full_file_hasher.update(seed)
                seed_hashes.append(hashlib.sha256(seed).hexdigest())

        src = sensor or self.sensor
        source_info = src.describe() if hasattr(src, "describe") else {"kind": "unknown", "physical": False}
        manifest = {
            "format": "PetRand-Seed-Ledger-v1.5",
            "petrand_version": _petrand_version(),
            "source": source_info,
            "credit_h_bits_per_sample": self.engine.monitor.H,
            "credit_h_assessed": False,
            "conditioner": "HMAC-SHA-256" if seed_bytes <= 32 else "HMAC-SHA-512",
            "seed_is_full_conditioner_output": seed_bytes in (32, 64),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed_bytes": seed_bytes,
            "num_seeds": num_seeds,
            "required_entropy_bits_per_seed": self.engine.required_bits,
            "seed_sha256_digests": seed_hashes,
            "binary_file_sha256": full_file_hasher.hexdigest(),
            "active_ball_id": self.engine.active_ball_id,
            "domain_tag": self.engine.domain_tag.decode("utf-8", errors="replace"),
            "health_telemetry": self.engine.monitor.get_telemetry(),
            "environment": {
                "numpy_version": np.__version__,
                "python_version": sys.version.split()[0],
            },
            "metadata": metadata or {},
        }

        with open(manifest_path, "w", encoding="utf-8") as f_man:
            json.dump(manifest, f_man, indent=2)

        return seed_hashes


class SeedSpooler:
    """
    Background seed spooler: keeps a ring buffer of conditioned 256-bit seeds in memory
    so scripts and APIs can pop one without waiting. Optional on-disk spool files store
    seeds in plain text; leave spool_path unset for anything secret.
    """

    def __init__(
        self,
        engine: Optional[PetRandEngine] = None,
        sensor: Optional[BaseSensor] = None,
        capacity: int = 32,
        spool_path: Optional[str] = None,
    ):
        self.engine = engine or PetRandEngine(required_entropy_bits=256.0)
        self.sensor = sensor or SyntheticArenaSimulator()
        self.capacity = max(1, int(capacity))
        self.spool_path = spool_path
        self._lock = threading.Lock()
        self._queue: deque = deque(maxlen=self.capacity)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.total_spooled = 0
        self.total_popped = 0

        if self.spool_path and os.path.exists(self.spool_path):
            self._load_from_disk()

    def _harvest_one_seed(self) -> Dict[str, Any]:
        fill_until_ready(self.engine, self.sensor, timeout_s=30.0)
        seed_bytes = self.engine.extract_seed(seed_bytes=32)
        self.total_spooled += 1
        entry = {
            "seed_hex": seed_bytes.hex(),
            "sha256_receipt": hashlib.sha256(seed_bytes).hexdigest()[:16],
            "minted_ns": time.time_ns(),
            "source_ball": self.engine.active_ball_id,
            "source": self.sensor.describe() if hasattr(self.sensor, "describe") else {"kind": "unknown", "physical": False},
        }
        return entry

    def fill_to_capacity(self) -> int:
        """Synchronously fills the spool buffer up to `self.capacity`."""
        added = 0
        while len(self._queue) < self.capacity:
            entry = self._harvest_one_seed()
            with self._lock:
                if len(self._queue) < self.capacity:
                    self._queue.append(entry)
                    added += 1
                else:
                    break
        if added > 0 and self.spool_path:
            self._persist_to_disk()
        return added

    def pop_seed(self) -> Dict[str, Any]:
        """
        Pops a pre-conditioned 256-bit seed in sub-millisecond time.
        If the spool is empty, harvests one immediately and triggers refill.
        """
        with self._lock:
            if self._queue:
                entry = self._queue.popleft()
                self.total_popped += 1
                if self.spool_path:
                    self._persist_to_disk_unlocked()
                return entry

        entry = self._harvest_one_seed()
        with self._lock:
            self.total_popped += 1
        return entry

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            buffered = len(self._queue)
            preview = self._queue[0]["sha256_receipt"] if self._queue else None
        return {
            "available_seeds": buffered,
            "capacity": self.capacity,
            "fill_ratio": round(buffered / max(1, self.capacity), 3),
            "total_spooled": self.total_spooled,
            "total_popped": self.total_popped,
            "next_seed_sha256_prefix": preview,
            "spool_path": self.spool_path,
            "daemon_running": bool(self._thread and self._thread.is_alive()),
        }

    def _persist_to_disk_unlocked(self) -> None:
        if not self.spool_path:
            return
        try:
            parent = os.path.dirname(os.path.abspath(self.spool_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = f"{self.spool_path}.tmp"
            payload = {
                "format": "PetRand-Seed-Spool-v1",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "capacity": self.capacity,
                "total_spooled": self.total_spooled,
                "total_popped": self.total_popped,
                "seeds": list(self._queue),
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self.spool_path)
        except Exception:
            pass

    def _persist_to_disk(self) -> None:
        with self._lock:
            self._persist_to_disk_unlocked()

    def _load_from_disk(self) -> None:
        try:
            with open(self.spool_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                for item in data.get("seeds", []):
                    if isinstance(item, dict) and len(item.get("seed_hex", "")) == 64:
                        self._queue.append(item)
        except Exception:
            pass

    def start_background_daemon(self, interval_sec: float = 0.02) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _worker():
            while not self._stop_event.is_set():
                if len(self._queue) < self.capacity:
                    try:
                        entry = self._harvest_one_seed()
                        with self._lock:
                            if len(self._queue) < self.capacity:
                                self._queue.append(entry)
                                self._persist_to_disk_unlocked()
                    except Exception:
                        pass
                time.sleep(interval_sec)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


class PetRandRandom(random.Random):
    """
    `random.Random` subclass whose output is BLAKE2b in counter mode over seeds from
    PetRandEngine (or a SeedSpooler). With no sensor given it runs on the simulator,
    so its unpredictability comes from the OS-seeded NumPy generator inside it.
    """

    def __init__(
        self,
        engine: Optional[PetRandEngine] = None,
        sensor: Optional[BaseSensor] = None,
        spooler: Optional[SeedSpooler] = None,
        reseed_interval_blocks: int = 64,
    ):
        self.engine = engine or PetRandEngine(required_entropy_bits=256.0)
        self.sensor = sensor or SyntheticArenaSimulator()
        self.spooler = spooler
        self.reseed_interval_blocks = max(1, int(reseed_interval_blocks))
        self._root_seed: bytes = b""
        self._counter: int = 0
        self._buffer: bytearray = bytearray()
        super().__init__()

    def _pull_hardware_seed(self) -> bytes:
        if self.spooler is not None:
            entry = self.spooler.pop_seed()
            return bytes.fromhex(entry["seed_hex"])
        fill_until_ready(self.engine, self.sensor, timeout_s=30.0)
        return self.engine.extract_seed(seed_bytes=32)

    def seed(self, a: Any = None, version: int = 2) -> None:
        if a is None:
            self._root_seed = self._pull_hardware_seed()
        elif isinstance(a, bytes):
            self._root_seed = hashlib.blake2b(a, digest_size=32).digest()
        else:
            self._root_seed = hashlib.blake2b(str(a).encode("utf-8"), digest_size=32).digest()
        self._counter = 0
        self._buffer.clear()

    def _refill_bytes(self, needed: int) -> None:
        while len(self._buffer) < needed:
            if not self._root_seed or (self._counter > 0 and self._counter % self.reseed_interval_blocks == 0):
                self._root_seed = self._pull_hardware_seed()
            block = hashlib.blake2b(
                self._root_seed + struct.pack("<Q", self._counter),
                digest_size=64,
                person=b"PetRandStdlibv1",
            ).digest()
            self._counter += 1
            self._buffer.extend(block)

    def randbytes(self, n: int) -> bytes:
        if n <= 0:
            return b""
        self._refill_bytes(n)
        out = bytes(self._buffer[:n])
        del self._buffer[:n]
        return out

    def getrandbits(self, k: int) -> int:
        if k < 0:
            raise ValueError("number of bits must be non-negative")
        if k == 0:
            return 0
        num_bytes = (k + 7) // 8
        raw = int.from_bytes(self.randbytes(num_bytes), "little")
        extra_bits = num_bytes * 8 - k
        return raw >> extra_bits

    def random(self) -> float:
        # 53-bit IEEE 754 double precision uniform float in [0.0, 1.0)
        return (self.getrandbits(53)) * (2.0 ** -53)


class PetRandSecrets:
    """
    `secrets`-style API (token_bytes, token_hex, token_urlsafe, randbelow, choice,
    compare_digest) over PetRandRandom. For real keys, use Python's own `secrets`
    module unless the physical source has been assessed.
    """

    def __init__(self, rng: Optional[PetRandRandom] = None):
        self.rng = rng or PetRandRandom(reseed_interval_blocks=1)

    def token_bytes(self, nbytes: int = 32) -> bytes:
        return self.rng.randbytes(nbytes)

    def token_hex(self, nbytes: int = 32) -> str:
        return self.token_bytes(nbytes).hex()

    def token_urlsafe(self, nbytes: int = 32) -> str:
        tok = base64.urlsafe_b64encode(self.token_bytes(nbytes)).rstrip(b"=")
        return tok.decode("ascii")

    def randbelow(self, exclusive_upper_bound: int) -> int:
        if exclusive_upper_bound <= 0:
            raise ValueError("Upper bound must be positive.")
        return self.rng._randbelow(exclusive_upper_bound)

    def choice(self, seq: Sequence[T]) -> T:
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        return seq[self.randbelow(len(seq))]

    @staticmethod
    def compare_digest(a: Any, b: Any) -> bool:
        return secrets.compare_digest(a, b)

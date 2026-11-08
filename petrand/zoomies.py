"""
petrand.zoomies
=================
Zoomies!: a motion game that uses human movement (mouse, touch, keys, phone motion,
microphone) as an experimental entropy input. A rhythm detector rejects steady, scripted
motion; a round that scores at least 75% hashes the motion into a 256-bit seed. How much
min-entropy hand motion carries has not been measured, so each motion sample is credited
a 0.5-bit placeholder.
"""

import os
import json
import math
import time
import struct
import hashlib
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from petrand.engine import PetRandEngine, RawPhysicalEvent
from petrand.toolkit import EverydaySeedToolkit
from petrand.http_guard import REQUEST_TIMEOUT_S, read_body, request_refusal


# Unassessed placeholder for the min-entropy of one motion sample (mouse, touch or IMU).
PLACEHOLDER_BITS_PER_MOTION_SAMPLE = 0.5

# ---------------------------------------------------------------------------
# Move cards: one prompt per round
# ---------------------------------------------------------------------------

FORM_CARDS: List[Dict[str, Any]] = [
    {
        "id": "spiral_whisk",
        "form_number": "MODE 01",
        "form_title": "SPIRAL WHISK",
        "verb": "WHISK!",
        "host": "CURVATURE CHANNEL",
        "narrator_intro": "Stir in fast, jagged spirals across the playfield.",
        "micro_action": "JAGGED SPIRALS",
        "target_bonus": "curvature",
        "badge": "🌀",
        "accent": "#f97316",
    },
    {
        "id": "scatter_snap",
        "form_number": "MODE 02",
        "form_title": "SCATTER SNAP",
        "verb": "FLAIL!",
        "host": "JERK CHANNEL",
        "narrator_intro": "Whip sharply across all 8 compass directions.",
        "micro_action": "8-WAY SNAP",
        "target_bonus": "jerk",
        "badge": "⚡",
        "accent": "#ef4444",
    },
    {
        "id": "vortex_orbit",
        "form_number": "MODE 03",
        "form_title": "VORTEX ORBIT",
        "verb": "SWIRL!",
        "host": "SPECTRAL CHANNEL",
        "narrator_intro": "Orbit the ring, then abruptly shatter the loop.",
        "micro_action": "ORBIT & SHATTER",
        "target_bonus": "spectral",
        "badge": "🎯",
        "accent": "#a855f7",
    },
    {
        "id": "wide_pendulum",
        "form_number": "MODE 04",
        "form_title": "WIDE PENDULUM",
        "verb": "SWEEP!",
        "host": "JERK CHANNEL",
        "narrator_intro": "Sweep in wide arcs with sudden reversals.",
        "micro_action": "ARC & REVERSE",
        "target_bonus": "jerk",
        "badge": "〰️",
        "accent": "#38bdf8",
    },
    {
        "id": "staccato_burst",
        "form_number": "MODE 05",
        "form_title": "STACCATO BURST",
        "verb": "STAMP!",
        "host": "SPECTRAL CHANNEL",
        "narrator_intro": "Dart in rapid staccato bursts and sharp stops.",
        "micro_action": "RAPID STOPS",
        "target_bonus": "spectral",
        "badge": "💥",
        "accent": "#eab308",
    },
    {
        "id": "zigzag_cut",
        "form_number": "MODE 06",
        "form_title": "ZIGZAG CUT",
        "verb": "SLASH!",
        "host": "CURVATURE CHANNEL",
        "narrator_intro": "Slice sharp diagonal corners across the frame.",
        "micro_action": "SHARP CORNERS",
        "target_bonus": "curvature",
        "badge": "📐",
        "accent": "#ec4899",
    },
    {
        "id": "seismic_shake",
        "form_number": "MODE 07",
        "form_title": "SEISMIC SHAKE",
        "verb": "QUAKE!",
        "host": "ACOUSTIC CHANNEL",
        "narrator_intro": "Jolt erratically while tapping random keys.",
        "micro_action": "ERRATIC JOLTS",
        "target_bonus": "acoustic",
        "badge": "🌋",
        "accent": "#10b981",
    },
    {
        "id": "fractal_scribble",
        "form_number": "MODE 08",
        "form_title": "FRACTAL SCRIBBLE",
        "verb": "SCRIBBLE!",
        "host": "CURVATURE CHANNEL",
        "narrator_intro": "Scribble unpredictable loops and tight turns.",
        "micro_action": "CHAOTIC SCRIBBLE",
        "target_bonus": "curvature",
        "badge": "✏️",
        "accent": "#6366f1",
    },
]


# Earlier card ids, still accepted by the API
class ZoomiesScorer:
    """
    Mathematical evaluator that scores a human's physical motion trajectory
    for genuine multi-axis chaos, rejects periodic/rhythmic cheating, and
    hashes the motion into a 256-bit seed with keyed BLAKE2b. The seed's entropy is
    unassessed; see PLACEHOLDER_BITS_PER_MOTION_SAMPLE.
    """

    UNLOCK_THRESHOLD = 75.0  # Minimum chaos score (%) needed to unlock a seed

    MAX_SAMPLES = 10000  # a 4-second round from a 1,000 Hz mouse is about 4,000

    @classmethod
    def evaluate_request(cls, data: Any) -> Dict[str, Any]:
        """Check a round sent by a page, then score it. Raises ValueError on bad input."""
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")

        def finite(v: Any, what: str) -> float:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{what} must be a finite number")
            return float(v)

        samples = data.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError("samples must be a list")
        if len(samples) > cls.MAX_SAMPLES:
            raise ValueError(f"at most {cls.MAX_SAMPLES:,} samples per round")
        clean: List[Dict[str, float]] = []
        for s in samples:
            if not isinstance(s, dict):
                raise ValueError("each sample must be an object")
            clean.append({k: finite(s[k], f"sample field {k}") for k in ("t", "x", "y", "z", "audio")
                          if s.get(k) is not None})
        audio = data.get("audio_bursts")
        if audio is not None:
            if not isinstance(audio, list) or len(audio) > cls.MAX_SAMPLES:
                raise ValueError("audio_bursts must be a list of numbers")
            audio = [finite(a, "audio_bursts entry") for a in audio]
        try:
            key_mashes = max(0, min(100000, int(data.get("key_mashes", 0) or 0)))
        except (TypeError, ValueError):
            raise ValueError("key_mashes must be a whole number")
        form_id = data.get("form_id")
        form_id = form_id[:64] if isinstance(form_id, str) else "vortex_orbit"
        area = None
        raw_area = data.get("area")
        if isinstance(raw_area, dict):
            aw, ah = raw_area.get("w"), raw_area.get("h")
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 50 <= v <= 20000
                   for v in (aw, ah)):
                area = (float(aw), float(ah))
        return cls.evaluate_trajectory(samples=clean, form_id=form_id, audio_bursts=audio,
                                       key_mashes=key_mashes, area=area)

    @classmethod
    def evaluate_trajectory(
        cls,
        samples: List[Dict[str, float]],
        form_id: str = "vortex_orbit",
        audio_bursts: Optional[List[float]] = None,
        key_mashes: int = 0,
        area: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Score one round with the chaos meter (score_motion, mirrored in the page's JavaScript)
        and, if it unlocks, derive the round's seed from everything recorded in it.
        """
        card = next((c for c in FORM_CARDS if c["id"] == form_id), FORM_CARDS[0])
        m = score_motion(samples, area=area, target_bonus=card.get("target_bonus"))
        status, score, metrics = m["status"], m["score"], m["metrics"]
        success = status == "SEED_UNLOCKED"
        feedback = {
            "STILLNESS_DETECTED": "No motion yet. Move inside the play area!",
            "FROZEN": "Almost no movement. Move big and wild across the play area!",
            "RHYTHMIC_LOOP_DETECTED": "Steady rhythm rejected: break it with sudden, uneven moves.",
            "SEED_UNLOCKED": f"Seed unlocked ({score:.1f}% chaos).",
        }.get(status)
        if feedback is None:
            feedback = ("Move bigger: use the whole play area." if m.get("too_small")
                        else f"Reached {score:.1f}% ({cls.UNLOCK_THRESHOLD:.0f}% needed).")
        result: Dict[str, Any] = {
            "unlocked": success,
            "status": status,
            "feedback": feedback,
            "form_id": card["id"],
            "form_title": card["form_title"],
            "chaos_score": score,
            "threshold": cls.UNLOCK_THRESHOLD,
            "sample_count": m["sample_count"],
            "kept_samples": m["kept"],
            # Placeholder: 0.5 bit for each movement big enough to count, up to 256 bits.
            # Human-motion entropy has not been assessed.
            "estimated_entropy_bits": round(min(256.0, m["kept"] * PLACEHOLDER_BITS_PER_MOTION_SAMPLE), 1) if success else 0.0,
            "metrics": metrics,
            "seed_hex": None,
            "seed_label": "No seed",
        }
        if not success:
            return result

        # The seed: everything recorded in the round, hashed with BLAKE2b and keyed with the time
        hasher = hashlib.blake2b(
            digest_size=32,
            person=b"ZoomiesSeedV2",
            key=hashlib.sha256(f"{card['id']}:{len(samples)}:{time.time_ns()}".encode("utf-8")).digest(),
        )
        for key in ("t", "x", "y", "z", "audio"):
            hasher.update(np.array([float(s.get(key, 0.0)) for s in samples], dtype=np.float64).tobytes())
        if audio_bursts:
            hasher.update(np.array([float(a) for a in audio_bursts], dtype=np.float64).tobytes())
        hasher.update(struct.pack("<Id", min(int(key_mashes), 0xFFFFFFFF), float(score)))
        seed_bytes = hasher.digest()
        seed_hex = seed_bytes.hex()

        u64_int = struct.unpack("<Q", seed_bytes[:8])[0]
        result.update({
            "seed_hex": seed_hex,
            "seed_label": f"Seed #{struct.unpack('<H', seed_bytes[:2])[0]:04d}",
            "derived_rolls": {
                "uniform_0_1": round(float((u64_int >> 11) * (1.0 / (1 << 53))), 12),
                "d20": int((u64_int % 20) + 1),   # bias below 1e-18 from a 64-bit value
                "6d6": unbiased_dice(seed_bytes[8:], 6, 6),
                "uint64": u64_int,
            },
            "toolkit": EverydaySeedToolkit.evaluate_all(seed_hex),
        })
        return result

    @classmethod
    def generate_synthetic_trajectory(
        cls,
        mode: str = "chaotic",
        num_samples: int = 180,
        seed: int = 42,
        area: Tuple[float, float] = (640.0, 400.0),
    ) -> List[Dict[str, float]]:
        """
        Test motions in a play area of the given size. "chaotic" is a wild human: fast, smooth
        strokes between random targets (minimum-jerk profiles), the way people flick a mouse
        or finger. The others are what the meter must reject: "static", "jitter" (1 px),
        "still_phone" (1.5 px), "lazy" (slow and small), "sine", "circle", "back_and_forth".
        """
        rng = np.random.Generator(np.random.PCG64(seed))
        w, h = float(area[0]), float(area[1])
        rate = 100.0
        if mode == "chaotic" or mode == "lazy":
            reach, stroke_ms = ((0.05, 0.95), (90.0, 320.0)) if mode == "chaotic" else ((0.4, 0.6), (400.0, 900.0))
            dt0 = 1000.0 / float(rng.uniform(60.0, 125.0))
            t, x, y = 0.0, float(rng.uniform(0.3 * w, 0.7 * w)), float(rng.uniform(0.3 * h, 0.7 * h))
            out = [{"t": t, "x": x, "y": y, "z": 0.0, "audio": 0.0}]
            while t < 4000.0:
                tx, ty = float(rng.uniform(reach[0] * w, reach[1] * w)), float(rng.uniform(reach[0] * h, reach[1] * h))
                steps = max(2, int(float(rng.uniform(*stroke_ms)) / dt0))
                for j in range(1, steps + 1):
                    u = j / steps
                    e = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
                    t += dt0 * float(rng.uniform(0.85, 1.15))
                    out.append({"t": t, "x": x + (tx - x) * e + float(rng.normal(0, 0.4)),
                                "y": y + (ty - y) * e + float(rng.normal(0, 0.4)),
                                "z": float(rng.normal(0, 20.0)), "audio": float(abs(rng.normal(0.05, 0.1)))})
                x, y = tx, ty
                t += float(rng.uniform(0.0, 60.0))
            return [p for p in out if p["t"] <= 4000.0]
        out = []
        for i in range(num_samples if mode in ("static", "sine") else int(4 * rate)):
            t = i * 1000.0 / rate
            if mode == "static":
                x, y = w / 2, h / 2
            elif mode in ("jitter", "still_phone"):
                amp = 1.0 if mode == "jitter" else 1.5
                x, y = w / 2 + float(rng.uniform(-amp, amp)), h / 2 + float(rng.uniform(-amp, amp))
            elif mode == "sine":
                t = i * 16.0
                x, y = w / 2 + 0.4 * w * math.sin(2.0 * math.pi * 2.5 * t / 1000.0), h / 2
            elif mode == "circle":
                a = 2.0 * math.pi * 1.2 * t / 1000.0
                x, y = w / 2 + 0.4 * h * math.cos(a), h / 2 + 0.4 * h * math.sin(a)
            elif mode == "back_and_forth":
                ph = (2.0 * t / 1000.0) % 1.0
                tri = 1.0 - abs(2.0 * ph - 1.0)
                x, y = 0.1 * w + 0.8 * w * tri, 0.15 * h + 0.7 * h * tri
            else:
                raise ValueError(f"unknown mode {mode!r}")
            out.append({"t": t, "x": x, "y": y, "z": 0.0, "audio": 0.0})
        return out


# ---------------------------------------------------------------------------
# The chaos meter (mirrored line for line by scoreMotion in the page's JavaScript)
# ---------------------------------------------------------------------------
# It judges motion in the play area's own units:
#   1. Wiggles smaller than one step (1.2% of the play-area diagonal, at least 4 px) are
#      dropped, so jitter, tremor and a phone lying still never count as motion.
#   2. Size: the moves must span much of the play area (coverage) and travel far (path).
#   3. Unpredictability: a mix of directions, irregular turns and changing speed.
#   4. Rhythm: the kept path is resampled at 50 Hz and its velocity spectrum checked; a
#      strong peak (circles, sweeps, steady shaking) is rejected.
# Calibrated on simulated rounds (2026-11-08): fast wild strokes unlock about 99% of the
# time, mixed-pace strokes about 93%, slower strokes about 44%; lazy small motion, 1 px
# jitter, a still phone and periodic bots never do.

METER = {
    "MIN_SAMPLES": 12, "STEP_FRAC": 0.012, "STEP_MIN_PX": 4.0, "MIN_KEPT": 8,
    "COVER_FULL": 0.6, "PATH_FULL": 4.5, "GATE_COVER": 0.5, "GATE_PATH": 0.4,
    "W_SIZE": 0.40, "W_RHYTHM": 0.20, "W_DIR": 0.25, "W_SPEED": 0.15,
    "RESAMPLE_HZ": 50.0, "PEAK_MAX": 0.35, "TOP3_MAX": 0.60, "THRESHOLD": 75.0,
}
BONUS_CHANNEL = {"jerk": "speed", "spectral": "rhythm", "curvature": "direction", "acoustic": "speed"}


def _zero_metrics() -> Dict[str, float]:
    return {"size_score": 0.0, "rhythm_free_score": 0.0, "direction_score": 0.0, "speed_change_score": 0.0,
            "coverage": 0.0, "path_diagonals": 0.0, "spectral_flatness": 0.0, "peak_ratio": 0.0, "top3_ratio": 0.0}


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def score_motion(samples: List[Dict[str, float]], area: Optional[Tuple[float, float]] = None,
                 target_bonus: Optional[str] = None) -> Dict[str, Any]:
    """The chaos meter. Returns status, score (0-100), metrics, sample_count and kept."""
    M_ = METER
    n = len(samples)
    w = float(area[0]) if area and area[0] and area[0] > 0 else 640.0
    h = float(area[1]) if area and area[1] and area[1] > 0 else 400.0
    diag = math.hypot(w, h)
    if n < M_["MIN_SAMPLES"]:
        return {"status": "STILLNESS_DETECTED", "score": 0.0, "metrics": _zero_metrics(), "sample_count": n, "kept": 0}

    # 1. Drop wiggles smaller than one step; keep times strictly increasing
    step = max(M_["STEP_MIN_PX"], M_["STEP_FRAC"] * diag)
    kx = [float(samples[0].get("x", 0.0))]
    ky = [float(samples[0].get("y", 0.0))]
    kt = [float(samples[0].get("t", 0.0))]
    for s in samples[1:]:
        x, y = float(s.get("x", 0.0)), float(s.get("y", 0.0))
        if math.hypot(x - kx[-1], y - ky[-1]) >= step:
            kx.append(x)
            ky.append(y)
            kt.append(max(float(s.get("t", 0.0)), kt[-1] + 0.001))
    k = len(kx)
    if k < M_["MIN_KEPT"]:
        return {"status": "FROZEN", "score": 0.0, "metrics": _zero_metrics(), "sample_count": n, "kept": k}

    X, Y, T = np.array(kx), np.array(ky), np.array(kt)
    dx, dy = np.diff(X), np.diff(Y)
    seg = np.hypot(dx, dy)

    # 2. Size
    span_w = min(1.0, float(X.max() - X.min()) / w)
    span_h = min(1.0, float(Y.max() - Y.min()) / h)
    coverage = min(1.0, math.sqrt((span_w / M_["COVER_FULL"]) * (span_h / M_["COVER_FULL"])))
    path_diag = float(seg.sum()) / diag
    path = min(1.0, path_diag / M_["PATH_FULL"])
    size = math.sqrt(coverage * path)

    # 3. Unpredictability
    ang = np.arctan2(dy, dx)
    bins = np.clip(np.floor((ang + math.pi) / (2.0 * math.pi) * 16.0).astype(int), 0, 15)
    pr = np.bincount(bins, minlength=16) / float(len(bins))
    pr = pr[pr > 0]
    sector = float(-(pr * np.log2(pr)).sum() / 4.0)
    turns = np.array([_wrap_angle(float(a)) for a in np.diff(ang)]) if len(ang) > 1 else np.zeros(1)
    turn = min(1.0, float(np.std(turns)) / 1.15)
    direction = 0.6 * sector + 0.4 * turn
    speeds = seg / np.maximum(np.diff(T), 1.0)
    cv = float(np.std(speeds)) / max(1e-9, float(np.mean(speeds)))
    speed = min(1.0, cv / 0.8)

    # 4. Rhythm: resample the kept path at 50 Hz, then the velocity spectrum (bins 1..K)
    m = max(8, int((T[-1] - T[0]) / 1000.0 * M_["RESAMPLE_HZ"]))
    grid = np.linspace(T[0], T[-1], m)
    vx, vy = np.diff(np.interp(grid, T, X)), np.diff(np.interp(grid, T, Y))
    MM = len(vx)
    K = min(32, MM // 2)
    tt = np.arange(MM)
    psd = np.zeros(K)
    for kk in range(1, K + 1):
        c, sn = np.cos(2.0 * math.pi * kk * tt / MM), np.sin(2.0 * math.pi * kk * tt / MM)
        psd[kk - 1] = float(vx @ c) ** 2 + float(vx @ sn) ** 2 + float(vy @ c) ** 2 + float(vy @ sn) ** 2
    tot = float(psd.sum())
    if tot > 1e-9 and K > 1:
        pn = psd / tot
        peak, top3 = float(pn.max()), float(np.sort(pn)[-3:].sum())
        pp = pn[pn > 1e-12]
        flat = float(-(pp * np.log2(pp)).sum() / math.log2(K))
    else:
        peak, top3, flat = 1.0, 1.0, 0.0
    rhythmic = peak > M_["PEAK_MAX"] or top3 > M_["TOP3_MAX"]

    ch = {"size": size, "rhythm": flat, "direction": direction, "speed": speed}
    if target_bonus in BONUS_CHANNEL:
        ch[BONUS_CHANNEL[target_bonus]] = min(1.0, ch[BONUS_CHANNEL[target_bonus]] * 1.10)
    score = 100.0 * (M_["W_SIZE"] * ch["size"] + M_["W_RHYTHM"] * ch["rhythm"]
                     + M_["W_DIR"] * ch["direction"] + M_["W_SPEED"] * ch["speed"])
    too_small = coverage < M_["GATE_COVER"] or path < M_["GATE_PATH"]
    if too_small:
        status, score = "INSUFFICIENT_CHAOS", min(74.9, score)
    elif rhythmic:
        status, score = "RHYTHMIC_LOOP_DETECTED", min(64.9, score)
    else:
        status = "SEED_UNLOCKED" if score >= M_["THRESHOLD"] else "INSUFFICIENT_CHAOS"
    metrics = {
        "size_score": round(100.0 * ch["size"], 1), "rhythm_free_score": round(100.0 * ch["rhythm"], 1),
        "direction_score": round(100.0 * ch["direction"], 1), "speed_change_score": round(100.0 * ch["speed"], 1),
        "coverage": round(coverage, 3), "path_diagonals": round(path_diag, 2), "spectral_flatness": round(flat, 3),
        "peak_ratio": round(peak, 3), "top3_ratio": round(top3, 3),
    }
    return {"status": status, "score": round(score, 2), "metrics": metrics, "sample_count": n, "kept": k,
            "too_small": too_small}


def unbiased_dice(data: bytes, count: int, sides: int) -> List[int]:
    """Dice from bytes by rejection (no modulo bias); extends the bytes by hashing if needed."""
    limit = 256 - (256 % sides)
    rolls: List[int] = []
    buf = bytes(data)
    while len(rolls) < count:
        for b in buf:
            if b < limit:
                rolls.append(b % sides + 1)
                if len(rolls) == count:
                    break
        buf = hashlib.sha256(buf).digest()
    return rolls


# ---------------------------------------------------------------------------
# Zoomies! page (served at /zoomies and exported as a standalone file)
# ---------------------------------------------------------------------------

ZOOMIES_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" href="data:,">
<title>Zoomies! // PetRand motion game</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0d0718; --card: #170f2a; --card-2: #110a20; --line: #31244f;
    --text: #f8fafc; --muted: #c9bfe3;
    --orange: #ff8c1a; --pink: #ec4899; --yellow: #facc15; --green: #4ade80;
    --blue: #38bdf8; --purple: #a855f7; --red: #f87171;
    --accent: #f97316;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 16px; overflow-x: hidden; }
  .wrap { max-width: 1180px; margin: 0 auto; display: grid; gap: 14px; }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  header.top { display: grid; gap: 12px; }
  .header-row { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo { width: 40px; height: 40px; border-radius: 50%; background: radial-gradient(circle at 35% 35%, #ffd39e, var(--orange) 55%, #b45309); box-shadow: 0 3px 0 #7c2d12; flex: none; }
  h1.brand-title { font-size: 1.7rem; font-weight: 950; letter-spacing: -0.02em; line-height: 1; }
  h1.brand-title span { color: var(--yellow); }
  .tagline { color: var(--muted); font-size: 0.88rem; margin-top: 3px; }
  .btn { appearance: none; background: #241838; color: var(--text); border: 1px solid #4b3a73; border-radius: 10px; padding: 9px 14px; font-weight: 800; font-size: 0.88rem; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; justify-content: center; gap: 6px; }
  .btn:hover { background: #31214d; }
  .btn:focus-visible, .pill-btn:focus-visible, summary:focus-visible, a:focus-visible { outline: 3px solid var(--blue); outline-offset: 2px; }
  .btn-small { padding: 5px 10px; font-size: 0.78rem; border-radius: 8px; }
  .inputs { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .inputs .lbl { font-size: 0.74rem; color: var(--muted); font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; margin-right: 2px; }
  .pill-btn { appearance: none; background: #1b1230; color: var(--muted); border: 1px solid #3f2f63; border-radius: 999px; padding: 5px 11px; font-size: 0.78rem; font-weight: 800; cursor: pointer; }
  .pill-btn.active { color: #052e16; background: var(--green); border-color: var(--green); }
  .pill-static { cursor: default; }
  .visually-hidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
  .layout { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 14px; align-items: start; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
  .card { background: var(--card); border: 3px solid #2a1d45; border-radius: 16px; padding: 16px; min-width: 0; }
  .play-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
  .round-pill { background: var(--yellow); color: #1c1300; font-weight: 950; border-radius: 8px; padding: 3px 10px; font-size: 0.85rem; box-shadow: 0 3px 0 #a16207; }
  .move { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .move-badge { font-size: 1.6rem; line-height: 1; }
  .move-num { font-size: 0.72rem; color: var(--muted); font-weight: 800; letter-spacing: 0.05em; }
  .move-name { font-size: 1.05rem; font-weight: 950; letter-spacing: 0.01em; }
  .move-tip { font-size: 0.86rem; color: var(--muted); margin-bottom: 10px; }
  .arena-wrap { position: relative; width: 100%; height: 300px; border-radius: 14px; overflow: hidden; border: 3px solid #000; background: #08040f; touch-action: none; }
  @media (max-width: 520px) { .arena-wrap { height: 260px; } }
  canvas#chaosCanvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
  .arena-overlay { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; pointer-events: none; text-align: center; padding-bottom: 34px; }
  .mega-verb { font-size: clamp(2.4rem, 8vw, 4.2rem); font-weight: 950; color: var(--yellow); letter-spacing: 0.02em; text-shadow: 0 4px 0 #000, 0 0 18px rgba(250, 204, 21, 0.25); }
  .micro-prompt { font-size: 0.9rem; font-weight: 900; color: #ffffff; letter-spacing: 0.08em; background: rgba(0, 0, 0, 0.55); padding: 3px 10px; border-radius: 6px; margin-bottom: 6px; }
  .warning-toast { display: none; position: absolute; top: 10px; left: 50%; transform: translateX(-50%); background: var(--red); color: #1f0000; font-weight: 950; padding: 6px 12px; border-radius: 8px; font-size: 0.86rem; white-space: nowrap; box-shadow: 0 3px 0 #7f1d1d; }
  .controls-bar { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; margin-top: 12px; }
  .btn-jump { appearance: none; border: none; border-radius: 14px; background: var(--orange); color: #1a0d00; font-size: 1.25rem; font-weight: 950; padding: 14px; cursor: pointer; box-shadow: 0 6px 0 #9a4a00; letter-spacing: 0.02em; }
  .btn-jump:hover { background: #ff9d3d; }
  .btn-jump:active { transform: translateY(3px); box-shadow: 0 3px 0 #9a4a00; }
  .demo-link { appearance: none; background: none; border: none; color: var(--blue); font-weight: 800; font-size: 0.84rem; cursor: pointer; text-decoration: underline; margin-top: 8px; }
  .meter-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .meter-title { font-weight: 950; font-size: 1rem; }
  .meter-val { font-size: 2.1rem; font-weight: 950; color: var(--yellow); line-height: 1; }
  .meter-track { position: relative; height: 18px; background: #24173b; border-radius: 9px; overflow: hidden; margin-top: 8px; border: 2px solid #000; }
  .meter-fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--pink), var(--orange), var(--yellow)); transition: width 0.15s; }
  .meter-threshold { position: absolute; top: 0; bottom: 0; left: 75%; width: 3px; background: #ffffff; }
  .meter-threshold-label { font-size: 0.78rem; color: var(--muted); margin-top: 4px; text-align: right; font-weight: 800; }
  .result { margin-top: 14px; background: var(--card-2); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
  .status-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
  .result-tag { font-size: 0.78rem; font-weight: 950; letter-spacing: 0.05em; border: 2px solid var(--muted); border-radius: 8px; padding: 2px 8px; }
  .feedback { font-size: 0.88rem; color: var(--muted); margin: 6px 0 8px; }
  .seed-hex { background: #08040f; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font-size: 0.8rem; word-break: break-all; color: #6ee7b7; min-height: 40px; }
  .rolls-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 10px; text-align: center; }
  .roll-pill { background: #08040f; border: 1px solid var(--line); border-radius: 10px; padding: 8px 4px; min-width: 0; }
  .roll-pill .k { font-size: 0.7rem; color: var(--muted); font-weight: 800; }
  .roll-pill .v { font-weight: 950; color: var(--green); font-size: 1rem; overflow-wrap: anywhere; }
  .capsule-box { margin-top: 10px; }
  .capsule-title { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 0.78rem; color: var(--muted); font-weight: 800; }
  .capsule-pass { font-size: 1rem; font-weight: 950; color: var(--yellow); overflow-wrap: anywhere; margin-top: 4px; }
  .actions { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
  details.more { margin-top: 12px; }
  details.more > summary { cursor: pointer; font-weight: 900; font-size: 0.9rem; color: var(--blue); }
  .metrics-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
  .metric-box { background: var(--card-2); border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; }
  .metric-label { display: flex; justify-content: space-between; gap: 6px; font-size: 0.74rem; color: var(--muted); font-weight: 800; }
  .metric-num { font-size: 1.1rem; font-weight: 950; }
  .mini-bar { height: 6px; background: #24173b; border-radius: 3px; overflow: hidden; margin-top: 4px; }
  .mini-fill { height: 100%; width: 0%; transition: width 0.15s; }
  .history-list { margin-top: 8px; display: grid; gap: 4px; }
  .history-item { display: flex; justify-content: space-between; gap: 8px; font-size: 0.74rem; flex-wrap: wrap; }
  .credit-note { font-size: 0.78rem; color: var(--muted); margin-top: 8px; }
  footer.honesty-bar { color: var(--muted); font-size: 0.8rem; text-align: center; }
  .hidden { display: none; }
  @media (prefers-reduced-motion: reduce) { .btn-jump:active { transform: none; } .meter-fill, .mini-fill { transition: none; } }
</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <div class="header-row">
    <div class="brand">
      <div class="logo" aria-hidden="true"></div>
      <div>
        <h1 class="brand-title">Zoomies<span>!</span></h1>
        <p class="tagline">Move like a cat with the zoomies for 4 seconds to unlock a random seed.</p>
      </div>
    </div>
    __CONSOLE_LINK__
  </div>
<div class="inputs" role="group" aria-label="Inputs">
  <span class="lbl">Inputs</span>
  <span class="pill-btn active pill-static" id="pillPointer" title="Mouse, trackpad or touch: always on">Mouse / touch</span>
  <span class="pill-btn active pill-static" id="pillKeys" title="Key presses during a round count as moves: always on">Keys</span>
  <button class="pill-btn" id="pillMotion" onclick="enableDeviceMotion()" aria-pressed="false" title="Use this phone's motion sensors">Phone motion</button>
  <button class="pill-btn" id="pillAudio" onclick="toggleMicAudio()" aria-pressed="false" title="Add microphone loudness">Microphone</button>
</div>
</header>

<main class="layout">
  <section class="card" aria-label="Play">
    <div class="play-top">
      <div class="move">
        <div class="move-badge" id="formBadge" aria-hidden="true">🎯</div>
        <div>
          <div class="move-num" id="formNum">MOVE 3 OF 8</div>
          <div class="move-name" id="formTitle">VORTEX ORBIT</div>
        </div>
      </div>
      <div style="display: flex; gap: 8px; align-items: center;">
        <span class="round-pill" id="stageCounter">ROUND 1</span>
        <button class="btn btn-small" id="btnNextCard" onclick="cycleFormCard()">Next move</button>
      </div>
    </div>
    <p class="move-tip" id="formNarrator">Orbit the ring, then abruptly shatter the loop.</p>
    <span id="hostChannel" class="hidden">SPECTRAL</span>

    <div class="arena-wrap" id="arenaWrap">
      <canvas id="chaosCanvas" role="img" aria-label="Play area: move your mouse, finger or phone here"></canvas>
      <div class="arena-overlay">
        <div class="micro-prompt" id="microPrompt">ORBIT &amp; SHATTER</div>
        <div class="mega-verb" id="megaVerb">SWIRL!</div>
      </div>
      <div class="warning-toast" id="warningToast" role="status">Too rhythmic! Break your rhythm.</div>
    </div>

    <div class="controls-bar">
      <button class="btn-jump" id="btnStartJump" onclick="startRound()">GO! (4 s)</button>
      <span class="mono" id="timerDisplay" style="align-self: center; font-weight: 900; color: var(--muted); min-width: 4.5ch; text-align: right;">4.0 s</span>
    </div>
    <button class="demo-link" id="btnSimBurst" onclick="runInstantAutoDemo()">No mouse? Watch a demo round</button>
  </section>

  <section class="card" aria-label="Chaos meter and your seed">
    <div class="meter-head">
      <span class="meter-title">Chaos meter</span>
      <span class="meter-val mono" id="improbValue">0.00%</span>
    </div>
    <div class="meter-track" id="meterTrack" role="meter" aria-label="Chaos meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div class="meter-fill" id="improbFill"></div><div class="meter-threshold"></div></div>
    <div class="meter-threshold-label">Unlocks at 75.0%</div>

    <p id="announce" class="visually-hidden" aria-live="polite"></p>
    <div class="result">
      <div class="status-row">
        <span class="result-tag" id="resultTag">READY</span>
        <button class="btn btn-small" onclick="copySeedHex(this)">Copy seed</button>
      </div>
      <p class="feedback" id="statusFeedback">Press GO, then move wildly until time runs out.</p>
      <div class="seed-hex mono" id="seedHexBox">No seed yet.</div>
      <div class="rolls-row">
        <div class="roll-pill"><div class="k">Number 0–1</div><div class="v mono" id="rollFloat">—</div></div>
        <div class="roll-pill"><div class="k">D20</div><div class="v mono" id="rollD20">—</div></div>
        <div class="roll-pill"><div class="k">6 dice</div><div class="v mono" id="roll6d6">—</div></div>
      </div>
      <div class="capsule-box">
        <div class="capsule-title"><span>Passphrase</span><span id="vjPin" class="mono" style="color: var(--green);">PIN: —</span></div>
        <div class="capsule-pass mono" id="vjPassphrase">— unlock a round for a passphrase —</div>
      </div>
    </div>

    <details class="more">
      <summary>How the meter scores you</summary>
      <div class="metrics-grid">
        <div class="metric-box"><div class="metric-label"><span>Size of moves</span><span id="valJerk">0×</span></div><div class="metric-num" style="color: var(--accent);" id="bigJerk">0%</div><div class="mini-bar"><div class="mini-fill" id="barJerk" style="background: var(--accent);"></div></div></div>
        <div class="metric-box"><div class="metric-label"><span>No steady beat</span><span id="valSpec">0.00</span></div><div class="metric-num" style="color: var(--purple);" id="bigSpec">0%</div><div class="mini-bar"><div class="mini-fill" id="barSpec" style="background: var(--purple);"></div></div></div>
        <div class="metric-box"><div class="metric-label"><span>Direction mix</span><span id="valDisp">16 bins</span></div><div class="metric-num" style="color: var(--blue);" id="bigDisp">0%</div><div class="mini-bar"><div class="mini-fill" id="barDisp" style="background: var(--blue);"></div></div></div>
        <div class="metric-box"><div class="metric-label"><span>Speed changes</span><span id="valSamples">0 moves</span></div><div class="metric-num" style="color: var(--green);" id="bigJitt">0%</div><div class="mini-bar"><div class="mini-fill" id="barJitt" style="background: var(--green);"></div></div></div>
      </div>
      <p class="credit-note">Big moves across the play area, no steady beat, many directions and changing speed score high. Wiggles too small to see don't count.</p>
      <p class="credit-note">Entropy credit for this round: <strong id="entropyCreditBadge" class="mono">0.0 bits</strong>. That is a placeholder of 0.5 bit for each move big enough to count, up to 256; nobody has measured how random hand motion really is.</p>
      <div class="actions"><button class="btn btn-small" onclick="downloadLedgerJson()">Download my seeds (.json)</button></div>
      <div class="history-list" id="historyList"></div>
    </details>
  </section>
</main>

<footer class="honesty-bar">Experimental human-motion entropy game, not a certified cryptographic module.</footer>
</div>

<script>
const FORM_CARDS = __FORM_CARDS_JSON__;
const WORDS_256 = __WORDS_JSON__;   // the console's passphrase words (8 bits each)

let currentCardIdx = 0;
let stageCount = 1;
let speedMult = 1.0;
let isJumping = false;
let jumpStartTime = 0;
let JUMP_DURATION_MS = 4000;
let lastPanX = 0.0;
let samples = [];
let keyMashCount = 0;
let jumpHistory = [];
let particles = [];
let trail = [];          // the pointer's own recent path, drawn as the ribbon
let lastSparkAt = 0;
let cssW = 500, cssH = 245;
let lastSecondTick = 5;

// Optional WebAudio synthesis & Microphone analyser
let audioCtx = null;
let micAnalyser = null;
let micDataArray = null;
let micActive = false;
let imuActive = false;

const canvas = document.getElementById('chaosCanvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = Math.max(2, window.devicePixelRatio || 1);
  cssW = Math.max(200, rect.width);
  cssH = Math.max(150, rect.height);
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function selectFormCard(idx) {
  currentCardIdx = ((idx % FORM_CARDS.length) + FORM_CARDS.length) % FORM_CARDS.length;
  const c = FORM_CARDS[currentCardIdx];
  document.getElementById('formBadge').textContent = c.badge;
  document.getElementById('formNum').textContent = `MOVE ${currentCardIdx + 1} OF ${FORM_CARDS.length}`;
  document.getElementById('formTitle').textContent = c.form_title;
  document.getElementById('formNarrator').textContent = c.narrator_intro;
  document.getElementById('megaVerb').textContent = c.verb || 'FLAIL!';
  document.getElementById('hostChannel').textContent = (c.host || 'SPECTRAL').toUpperCase();
  if (!isJumping) {
    document.getElementById('microPrompt').textContent = c.micro_action;
  }
}

function cycleFormCard() {
  selectFormCard(currentCardIdx + 1);
  playTone(587.33, 0.06, 'square');
}

selectFormCard(2); // Start on VORTEX ORBIT

// Sound effects: short square-wave blips, panned with the pointer
function ensureAudioCtx() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) audioCtx = new AudioContextClass();
  }
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
}

function playTone(freq, dur, type='square', gainVal=0.11, panVal=null) {
  try {
    ensureAudioCtx();
    if (!audioCtx) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(gainVal, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
    if (audioCtx.createStereoPanner) {
      const panner = audioCtx.createStereoPanner();
      const p = panVal !== null ? panVal : lastPanX;
      panner.pan.setValueAtTime(Math.max(-1.0, Math.min(1.0, p)), audioCtx.currentTime);
      osc.connect(gain);
      gain.connect(panner);
      panner.connect(audioCtx.destination);
    } else {
      osc.connect(gain);
      gain.connect(audioCtx.destination);
    }
    osc.start();
    osc.stop(audioCtx.currentTime + dur);
  } catch (e) {}
}

function playResultChord(success) {
  if (success) {
    // Unlocked: quick rising arpeggio
    [523.25, 659.25, 783.99, 1046.50, 1318.51].forEach((f, i) => {
      setTimeout(() => playTone(f, 0.18, 'square', 0.13), i * 50);
    });
  } else {
    // Missed: low buzz
    playTone(160, 0.22, 'sawtooth', 0.15);
    setTimeout(() => playTone(115, 0.35, 'sawtooth', 0.15), 130);
  }
}

// Microphone Acoustic Turbulence
let micStream = null, micSource = null;
async function toggleMicAudio() {
  const btn = document.getElementById('pillAudio');
  if (micActive) {
    // Turning it off releases the microphone, so the browser's mic indicator goes off too
    micActive = false;
    if (micStream) micStream.getTracks().forEach((t) => t.stop());
    if (micSource) micSource.disconnect();
    micStream = null; micSource = null; micAnalyser = null;
    btn.classList.remove('active'); btn.setAttribute('aria-pressed', 'false');
    btn.textContent = 'Microphone';
    return;
  }
  try {
    ensureAudioCtx();
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micSource = audioCtx.createMediaStreamSource(micStream);
    micAnalyser = audioCtx.createAnalyser();
    micAnalyser.fftSize = 256;
    micDataArray = new Uint8Array(micAnalyser.frequencyBinCount);
    micSource.connect(micAnalyser);
    micActive = true;
    btn.classList.add('active'); btn.setAttribute('aria-pressed', 'true');
    btn.textContent = 'Microphone: on';
  } catch (err) {
    btn.textContent = 'Microphone: blocked';
  }
}

function readMicFlux() {
  if (!micActive || !micAnalyser || !micDataArray) return 0.0;
  micAnalyser.getByteTimeDomainData(micDataArray);
  let sum = 0;
  for (let i = 0; i < micDataArray.length; i++) {
    const v = (micDataArray[i] - 128) / 128.0;
    sum += v * v;
  }
  return Math.sqrt(sum / micDataArray.length);
}

// Smartphone 6-Axis IMU
let motionOn = false, motionHandler = null, motionSeenAt = 0;
async function enableDeviceMotion() {
  const btn = document.getElementById('pillMotion');
  if (motionOn) {
    window.removeEventListener('devicemotion', motionHandler);
    motionOn = false; imuActive = false;
    btn.classList.remove('active'); btn.setAttribute('aria-pressed', 'false');
    btn.textContent = 'Phone motion';
    return;
  }
  try {
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      const perm = await DeviceMotionEvent.requestPermission();
      if (perm !== 'granted') { btn.textContent = 'Phone motion: blocked'; return; }
    }
  } catch (e) { btn.textContent = 'Phone motion: blocked'; return; }
  if (!motionHandler) {
    motionHandler = (ev) => {
      const acc = ev.accelerationIncludingGravity || ev.acceleration;
      if (!acc || acc.x === null) return;
      motionSeenAt = performance.now();
      imuActive = true;
      const rot = ev.rotationRate;
      const ax = (acc.x || 0) * 18 + (rot ? (rot.alpha || 0) * 0.2 : 0);
      const ay = (acc.y || 0) * 18 + (rot ? (rot.beta || 0) * 0.2 : 0);
      const az = (acc.z || 0) * 18 + (rot ? (rot.gamma || 0) * 0.2 : 0);
      recordSample(cssW / 2 + ax * 4, cssH / 2 + ay * 4, az);
    };
  }
  window.addEventListener('devicemotion', motionHandler);
  motionOn = true;
  btn.classList.add('active'); btn.setAttribute('aria-pressed', 'true');
  btn.textContent = 'Phone motion: on';
  const started = performance.now();
  setTimeout(() => {
    // Computers without motion sensors, and phones on a plain-http address, send no readings
    if (motionOn && motionSeenAt < started) btn.textContent = 'Phone motion: no sensor found';
  }, 1500);
}

// Pointer / Touch / Keyboard input
function handlePointerEvent(clientX, clientY, pressureOrZ=0) {
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;

  recordSample(x, y, pressureOrZ);
}

canvas.addEventListener('mousemove', (e) => {
  handlePointerEvent(e.clientX, e.clientY, (e.buttons || 0) * 25.0);
});
canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  // Only the first finger counts: alternating between two fingers would read as a steady beat
  if (e.touches.length) handlePointerEvent(e.touches[0].clientX, e.touches[0].clientY, 30.0);
}, { passive: false });

window.addEventListener('keydown', (e) => {
  if (!isJumping) return;
  keyMashCount++;
  const rx = ((e.keyCode * 73) % Math.max(100, cssW));
  const ry = ((e.keyCode * 131) % Math.max(100, cssH));
  recordSample(rx, ry, e.keyCode * 1.7);
});

// People who ask their system for reduced motion get a calmer play area: one steady particle color,
// no drift, and a still ring. The game and its scoring work the same.
const reducedMotion = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };

function recordSample(x, y, z) {
  const now = performance.now();
  const aud = readMicFlux();
  lastPanX = Math.max(-1.0, Math.min(1.0, 2.0 * (x / Math.max(1.0, cssW)) - 1.0));
  if (isJumping) {
    samples.push({ t: now, x: x, y: y, z: z + aud * 100.0, audio: aud });
  }
  // Visuals stay under the WCAG flash limit even with the wildest input (validation/flash_check.py):
  // at most one new spark per frame, colors that turn slowly (a full rainbow every 6 s), gentle drift,
  // and a ribbon along the pointer's own path rather than through drifting sparks.
  trail.push({ x: x, y: y, t: now });
  if (trail.length > 48) trail.shift();
  if (now - lastSparkAt < 16) return;
  lastSparkAt = now;
  const calm = reducedMotion.matches;
  const hue = calm ? 195 : (now * 0.06) % 360;
  particles.push({
    x: x, y: y,
    vx: calm ? 0 : (Math.random() - 0.5) * 1.2,
    vy: calm ? 0 : (Math.random() - 0.5) * 1.2,
    r: calm ? 4 : Math.random() * 4 + 3,
    alpha: 1.0,
    color: `hsl(${hue}, 90%, 62%)`
  });
  if (particles.length > 160) particles.shift();
}

let roundSerial = 0;     // a result is shown only if no newer round has started since
let demoRound = false;   // demo rounds are scored in the page and never make a seed

function startRound(isDemo = false) {
  if (isJumping) return;   // Space or Enter on the focused GO button must not restart a round
  ensureAudioCtx();
  roundSerial++;
  demoRound = !!isDemo;
  samples = [];
  keyMashCount = 0;
  lastSecondTick = 5;
  isJumping = true;
  jumpStartTime = performance.now();
  const c = FORM_CARDS[currentCardIdx];
  document.getElementById('megaVerb').textContent = c.verb || 'FLAIL!';
  document.getElementById('microPrompt').textContent = c.micro_action;
  document.getElementById('btnStartJump').textContent = 'GO GO GO!';
  document.getElementById('warningToast').style.display = 'none';
  playTone(784, 0.09, 'square', 0.12);
}

// Client-side live telemetry + fallback evaluator
// ---- The chaos meter: the same rules, constants and order as score_motion() on the server ----
const METER = { MIN_SAMPLES: 12, STEP_FRAC: 0.012, STEP_MIN_PX: 4.0, MIN_KEPT: 8, COVER_FULL: 0.6, PATH_FULL: 4.5,
  GATE_COVER: 0.5, GATE_PATH: 0.4, W_SIZE: 0.40, W_RHYTHM: 0.20, W_DIR: 0.25, W_SPEED: 0.15,
  RESAMPLE_HZ: 50.0, PEAK_MAX: 0.35, TOP3_MAX: 0.60, THRESHOLD: 75.0 };
const BONUS_CHANNEL = { jerk: 'speed', spectral: 'rhythm', curvature: 'direction', acoustic: 'speed' };

function playArea() { return { w: cssW, h: cssH }; }
function zeroMetrics() {
  return { size_score: 0, rhythm_free_score: 0, direction_score: 0, speed_change_score: 0,
           coverage: 0, path_diagonals: 0, spectral_flatness: 0, peak_ratio: 0, top3_ratio: 0 };
}
function wrapAngle(a) { const tau = 2 * Math.PI; return ((((a + Math.PI) % tau) + tau) % tau) - Math.PI; }
function meanOf(a) { let s = 0; for (const v of a) s += v; return s / a.length; }
function stdOf(a) { const m = meanOf(a); let s = 0; for (const v of a) s += (v - m) * (v - m); return Math.sqrt(s / a.length); }
function roundTo(v, d) { const f = Math.pow(10, d); return Math.round(v * f) / f; }

function scoreMotion(sampleArr, area, targetBonus) {
  const n = sampleArr.length;
  const w = (area && area.w > 0) ? area.w : 640.0, h = (area && area.h > 0) ? area.h : 400.0;
  const diag = Math.hypot(w, h);
  if (n < METER.MIN_SAMPLES) return { status: 'STILLNESS_DETECTED', score: 0, metrics: zeroMetrics(), sample_count: n, kept: 0 };

  // 1. Drop wiggles smaller than one step; keep times strictly increasing
  const step = Math.max(METER.STEP_MIN_PX, METER.STEP_FRAC * diag);
  const X = [+sampleArr[0].x || 0], Y = [+sampleArr[0].y || 0], T = [+sampleArr[0].t || 0];
  for (let i = 1; i < n; i++) {
    const x = +sampleArr[i].x || 0, y = +sampleArr[i].y || 0;
    if (Math.hypot(x - X[X.length - 1], y - Y[Y.length - 1]) >= step) {
      X.push(x); Y.push(y); T.push(Math.max(+sampleArr[i].t || 0, T[T.length - 1] + 0.001));
    }
  }
  const k = X.length;
  if (k < METER.MIN_KEPT) return { status: 'FROZEN', score: 0, metrics: zeroMetrics(), sample_count: n, kept: k };

  const dx = [], dy = [], seg = [];
  for (let i = 1; i < k; i++) { dx.push(X[i] - X[i - 1]); dy.push(Y[i] - Y[i - 1]); seg.push(Math.hypot(dx[i - 1], dy[i - 1])); }

  // 2. Size
  const spanW = Math.min(1, (Math.max(...X) - Math.min(...X)) / w);
  const spanH = Math.min(1, (Math.max(...Y) - Math.min(...Y)) / h);
  const coverage = Math.min(1, Math.sqrt((spanW / METER.COVER_FULL) * (spanH / METER.COVER_FULL)));
  let pathLen = 0; for (const v of seg) pathLen += v;
  const pathDiag = pathLen / diag;
  const path = Math.min(1, pathDiag / METER.PATH_FULL);
  const size = Math.sqrt(coverage * path);

  // 3. Unpredictability
  const ang = dx.map((v, i) => Math.atan2(dy[i], v));
  const counts = new Array(16).fill(0);
  for (const a of ang) counts[Math.min(15, Math.max(0, Math.floor((a + Math.PI) / (2 * Math.PI) * 16)))]++;
  let sector = 0;
  for (const c of counts) if (c > 0) { const q = c / ang.length; sector -= q * Math.log2(q); }
  sector /= 4.0;
  const turns = [];
  for (let i = 1; i < ang.length; i++) turns.push(wrapAngle(ang[i] - ang[i - 1]));
  const turn = Math.min(1, (turns.length ? stdOf(turns) : 0) / 1.15);
  const direction = 0.6 * sector + 0.4 * turn;
  const speeds = seg.map((v, i) => v / Math.max(T[i + 1] - T[i], 1.0));
  const speed = Math.min(1, (stdOf(speeds) / Math.max(1e-9, meanOf(speeds))) / 0.8);

  // 4. Rhythm: resample the kept path at 50 Hz, then the velocity spectrum (bins 1..K)
  const m = Math.max(8, Math.floor((T[k - 1] - T[0]) / 1000.0 * METER.RESAMPLE_HZ));
  const gx = [], gy = [];
  let j = 0;
  for (let i = 0; i < m; i++) {
    const tg = T[0] + (T[k - 1] - T[0]) * (m > 1 ? i / (m - 1) : 0);
    while (j < k - 2 && T[j + 1] < tg) j++;
    const f = Math.min(1, Math.max(0, (tg - T[j]) / (T[j + 1] - T[j])));
    gx.push(X[j] + (X[j + 1] - X[j]) * f); gy.push(Y[j] + (Y[j + 1] - Y[j]) * f);
  }
  const vx = [], vy = [];
  for (let i = 1; i < m; i++) { vx.push(gx[i] - gx[i - 1]); vy.push(gy[i] - gy[i - 1]); }
  const MM = vx.length, K = Math.min(32, Math.floor(MM / 2));
  const psd = [];
  let tot = 0;
  for (let kk = 1; kk <= K; kk++) {
    let rx = 0, ix = 0, ry = 0, iy = 0;
    for (let t = 0; t < MM; t++) {
      const c = Math.cos(2 * Math.PI * kk * t / MM), sn = Math.sin(2 * Math.PI * kk * t / MM);
      rx += vx[t] * c; ix += vx[t] * sn; ry += vy[t] * c; iy += vy[t] * sn;
    }
    const pw = rx * rx + ix * ix + ry * ry + iy * iy;
    psd.push(pw); tot += pw;
  }
  let peak = 1, top3 = 1, flat = 0;
  if (tot > 1e-9 && K > 1) {
    const pn = psd.map((v) => v / tot);
    peak = Math.max(...pn);
    top3 = pn.slice().sort((a, b) => a - b).slice(-3).reduce((a, b) => a + b, 0);
    flat = 0;
    for (const q of pn) if (q > 1e-12) flat -= q * Math.log2(q);
    flat /= Math.log2(K);
  }
  const rhythmic = peak > METER.PEAK_MAX || top3 > METER.TOP3_MAX;

  const ch = { size: size, rhythm: flat, direction: direction, speed: speed };
  if (BONUS_CHANNEL[targetBonus]) ch[BONUS_CHANNEL[targetBonus]] = Math.min(1, ch[BONUS_CHANNEL[targetBonus]] * 1.10);
  let score = 100 * (METER.W_SIZE * ch.size + METER.W_RHYTHM * ch.rhythm + METER.W_DIR * ch.direction + METER.W_SPEED * ch.speed);
  const tooSmall = coverage < METER.GATE_COVER || path < METER.GATE_PATH;
  let status;
  if (tooSmall) { status = 'INSUFFICIENT_CHAOS'; score = Math.min(74.9, score); }
  else if (rhythmic) { status = 'RHYTHMIC_LOOP_DETECTED'; score = Math.min(64.9, score); }
  else status = score >= METER.THRESHOLD ? 'SEED_UNLOCKED' : 'INSUFFICIENT_CHAOS';
  return {
    status: status, score: roundTo(score, 2), sample_count: n, kept: k, too_small: tooSmall,
    metrics: {
      size_score: roundTo(100 * ch.size, 1), rhythm_free_score: roundTo(100 * ch.rhythm, 1),
      direction_score: roundTo(100 * ch.direction, 1), speed_change_score: roundTo(100 * ch.speed, 1),
      coverage: roundTo(coverage, 3), path_diagonals: roundTo(pathDiag, 2), spectral_flatness: roundTo(flat, 3),
      peak_ratio: roundTo(peak, 3), top3_ratio: roundTo(top3, 3)
    }
  };
}

// The same result shape as the server's /api/v1/zoomies/evaluate (without a seed)
function evaluateClientSide(sampleArr, formCard, area) {
  const m = scoreMotion(sampleArr, area, formCard && formCard.target_bonus);
  const ok = m.status === 'SEED_UNLOCKED';
  const feedback = {
    STILLNESS_DETECTED: 'No motion yet. Move inside the play area!',
    FROZEN: 'Almost no movement. Move big and wild across the play area!',
    RHYTHMIC_LOOP_DETECTED: 'Steady rhythm rejected: break it with sudden, uneven moves.',
    SEED_UNLOCKED: `Seed unlocked (${m.score.toFixed(1)}% chaos).`
  }[m.status] || (m.too_small ? 'Move bigger: use the whole play area.' : `Reached ${m.score.toFixed(1)}% (75% needed).`);
  return {
    unlocked: ok, status: m.status, feedback: feedback, chaos_score: m.score,
    sample_count: m.sample_count, kept_samples: m.kept, metrics: m.metrics,
    estimated_entropy_bits: ok ? Math.min(256, m.kept * 0.5) : 0, seed_hex: null, seed_label: 'No seed'
  };
}

// Unbiased picks from hash bytes: values that would favor some outcomes are skipped
function makeByteReader(bytes) {
  let i = 0;
  return {
    pick(sides) {
      const limit = 256 - (256 % sides);
      while (i < bytes.length) { const b = bytes[i++]; if (b < limit) return b % sides; }
      return bytes[(i++) % bytes.length] % sides;   // practically unreachable with 64 bytes
    },
    pin6() {   // 100000-999999 from 3 bytes, skipping the biased top range
      while (i + 3 <= bytes.length) {
        const v = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2]; i += 3;
        if (v < 16200000) return String(100000 + (v % 900000));
      }
      return String(100000 + (((bytes[0] << 16) | (bytes[1] << 8) | bytes[2]) % 900000));
    }
  };
}

async function finalizeRound() {
  isJumping = false;
  document.getElementById('btnStartJump').textContent = 'GO! (4 s)';
  const card = FORM_CARDS[currentCardIdx];
  const serial = roundSerial;
  const roundSamples = samples.slice();
  const area = playArea();

  if (demoRound) {
    // A demo shows how scoring works; its motion is made by the page, so it never makes a seed
    const res = evaluateClientSide(roundSamples, card, area);
    res.demo = true;
    renderEvaluationResult(res);
    return;
  }

  let result = null;
  try {
    const resp = await fetch('/api/v1/zoomies/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ form_id: card.id, samples: roundSamples, key_mashes: keyMashCount, area: area })
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    result = await resp.json();
  } catch (err) {
    result = evaluateClientSide(roundSamples, card, area);
    if (result.unlocked) {
      const rawBuf = new Float64Array(roundSamples.flatMap(s => [s.t, s.x, s.y, s.z || 0, s.audio || 0]));
      const h1 = new Uint8Array(await crypto.subtle.digest('SHA-256', rawBuf.buffer));
      const h2 = new Uint8Array(await crypto.subtle.digest('SHA-256', h1));
      const bytes = new Uint8Array([...h1, ...h2]);
      result.seed_hex = Array.from(h1).map(b => b.toString(16).padStart(2, '0')).join('');
      result.seed_label = `Seed #${String(h1[0] | (h1[1] << 8)).padStart(4, '0')}`;
      const rd = makeByteReader(bytes.slice(2));
      result.derived_rolls = {
        uniform_0_1: ((h1[0] * 256 + h1[1]) / 65536).toFixed(8),
        d20: rd.pick(20) + 1,
        '6d6': [0, 1, 2, 3, 4, 5].map(() => rd.pick(6) + 1)
      };
      const words = [0, 1, 2, 3, 4, 5].map(() => WORDS_256[rd.pick(256)]);
      result.toolkit = { passphrase_bundle: { passphrase: words.join('-'), pin_6_digit: rd.pin6() } };
    }
    result.feedback = 'Offline, scored in your browser: ' + result.feedback;
  }
  if (serial !== roundSerial) return;   // a newer round has started; drop this stale result
  renderEvaluationResult(result);
  playResultChord(result.unlocked);
}

function updateHUDMetrics(evalObj) {
  // Every value is read defensively: a round with no motion returns fewer metrics
  const pct = (v) => Math.max(0, Math.min(100, Number.isFinite(+v) ? +v : 0));
  const m = evalObj.metrics || {};
  const score = pct(evalObj.chaos_score);
  document.getElementById('improbValue').textContent = `${score.toFixed(2)}%`;
  document.getElementById('improbFill').style.width = `${score}%`;
  document.getElementById('meterTrack').setAttribute('aria-valuenow', String(Math.round(score)));

  const sizeS = pct(m.size_score), spec = pct(m.rhythm_free_score), disp = pct(m.direction_score), jitt = pct(m.speed_change_score);
  const num = (v, d) => (Number.isFinite(+v) ? +v : 0).toFixed(d);
  document.getElementById('valJerk').textContent = `${num(m.path_diagonals, 1)}× across`;
  document.getElementById('bigJerk').textContent = `${sizeS.toFixed(0)}%`;
  document.getElementById('barJerk').style.width = `${sizeS}%`;

  document.getElementById('valSpec').textContent = `flat ${num(m.spectral_flatness, 2)}`;
  document.getElementById('bigSpec').textContent = `${spec.toFixed(0)}%`;
  document.getElementById('barSpec').style.width = `${spec}%`;

  document.getElementById('bigDisp').textContent = `${disp.toFixed(0)}%`;
  document.getElementById('barDisp').style.width = `${disp}%`;

  document.getElementById('valSamples').textContent = `${evalObj.kept_samples || 0} moves`;
  document.getElementById('bigJitt').textContent = `${jitt.toFixed(0)}%`;
  document.getElementById('barJitt').style.width = `${jitt}%`;

  // The rhythm warning shows for at least 1.5 s and at most once every 2.5 s, so it never blinks
  const warn = document.getElementById('warningToast');
  const nowT = performance.now();
  if (evalObj.status === 'RHYTHMIC_LOOP_DETECTED' && evalObj.sample_count > 22) {
    if (warn.style.display !== 'block' && nowT - toastShownAt > 2500) { warn.style.display = 'block'; toastShownAt = nowT; }
  } else if (warn.style.display === 'block' && nowT - toastShownAt > 1500) {
    warn.style.display = 'none';
  }
}
let toastShownAt = -1e9;

function announce(msg) {
  const el = document.getElementById('announce');
  el.textContent = '';
  setTimeout(() => { el.textContent = msg; }, 30);
}

function statusLabel(status) {
  return ({
    RHYTHMIC_LOOP_DETECTED: 'TOO RHYTHMIC',
    INSUFFICIENT_CHAOS: 'NOT WILD ENOUGH',
    STILLNESS_DETECTED: 'NO MOTION',
    FROZEN: 'NO MOTION'
  })[status] || 'TRY AGAIN';
}

function renderEvaluationResult(res) {
  updateHUDMetrics(res);
  document.getElementById('entropyCreditBadge').textContent = `${res.estimated_entropy_bits} bits`;
  document.getElementById('statusFeedback').textContent = res.feedback;
  const uTag = document.getElementById('resultTag');
  const seedBox = document.getElementById('seedHexBox');

  if (res.demo) {
    document.getElementById('megaVerb').textContent = 'DEMO';
    uTag.textContent = 'DEMO';
    uTag.style.borderColor = '#38bdf8';
    seedBox.textContent = 'Demo round: no seed. Press GO and move to make your own.';
    seedBox.style.color = '#7dd3fc';
    for (const id of ['rollFloat', 'rollD20', 'roll6d6']) document.getElementById(id).textContent = '—';
    document.getElementById('vjPin').textContent = 'PIN: —';
    document.getElementById('vjPassphrase').textContent = '— unlock a round for a passphrase —';
    document.getElementById('statusFeedback').textContent =
      `Demo: the page drew that motion itself and scored it ${res.chaos_score.toFixed(1)}%. Demos never make a seed.`;
    document.getElementById('entropyCreditBadge').textContent = '0 bits';
    document.getElementById('microPrompt').textContent = 'YOUR TURN';
    announce('Demo round: no seed.');
    return;
  }
  if (res.unlocked && res.seed_hex) {
    stageCount++;
    document.getElementById('stageCounter').textContent = `ROUND ${stageCount}`;
    document.getElementById('megaVerb').textContent = 'UNLOCKED!';
    uTag.textContent = 'UNLOCKED';
    uTag.style.borderColor = '#4ade80';
    seedBox.textContent = res.seed_hex;
    seedBox.style.color = '#6ee7b7';
    document.getElementById('microPrompt').textContent = 'SEED UNLOCKED';
    announce(`Seed unlocked at ${Number(res.chaos_score).toFixed(1)}% chaos.`);
    if (res.derived_rolls) {
      document.getElementById('rollFloat').textContent = Number(res.derived_rolls.uniform_0_1).toFixed(4);
      document.getElementById('rollD20').textContent = `${res.derived_rolls.d20}`;
      document.getElementById('roll6d6').textContent = res.derived_rolls['6d6'].join(' ');
    }
    if (res.toolkit && res.toolkit.passphrase_bundle) {
      document.getElementById('vjPassphrase').textContent = res.toolkit.passphrase_bundle.passphrase;
      document.getElementById('vjPin').textContent = `PIN: ${res.toolkit.passphrase_bundle.pin_6_digit}`;
    }
    jumpHistory.unshift({
      timestamp: new Date().toISOString(),
      move: FORM_CARDS[currentCardIdx].form_title,
      result: res.seed_label,
      chaos_percent: res.chaos_score,
      seed_hex: res.seed_hex
    });
    renderHistory();
  } else {
    const label = statusLabel(res.status);
    document.getElementById('megaVerb').textContent = 'AGAIN!';
    uTag.textContent = label;
    uTag.style.borderColor = '#ef4444';
    seedBox.textContent = 'No seed this round.';
    seedBox.style.color = '#f87171';
    // Clear the last round's values so nothing under "No seed this round" looks current
    for (const id of ['rollFloat', 'rollD20', 'roll6d6']) document.getElementById(id).textContent = '—';
    document.getElementById('vjPin').textContent = 'PIN: —';
    document.getElementById('vjPassphrase').textContent = '— unlock a round for a passphrase —';
    document.getElementById('microPrompt').textContent =
      res.status === 'RHYTHMIC_LOOP_DETECTED' ? 'BREAK YOUR RHYTHM' : (label === 'NO MOTION' ? 'MOVE!' : 'WILDER!');
    announce(`No seed: ${res.feedback}`);
  }
}

function renderHistory() {
  const el = document.getElementById('historyList');
  el.innerHTML = jumpHistory.slice(0, 6).map(h => `
    <div class="history-item">
      <span style="color:#facc15;">${h.move} (${h.chaos_percent}%)</span>
      <span style="color:#6ee7b7;">${h.seed_hex.slice(0, 20)}...</span>
    </div>
  `).join('');
}

function runInstantAutoDemo() {
  // Fast strokes between random targets, the way people flick a mouse; scored as a demo only
  startRound(true);
  if (!demoRound) return;
  const t0 = performance.now();
  let t = 0, x = cssW * (0.3 + 0.4 * Math.random()), y = cssH * (0.3 + 0.4 * Math.random());
  const dt0 = 10;
  while (t < 3600) {
    const tx = cssW * (0.05 + 0.9 * Math.random()), ty = (cssH - 30) * (0.05 + 0.9 * Math.random());
    const steps = Math.max(2, Math.floor((90 + 230 * Math.random()) / dt0));
    for (let j = 1; j <= steps; j++) {
      const u = j / steps, e = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5;
      t += dt0 * (0.85 + 0.3 * Math.random());
      samples.push({ t: t0 + t, x: x + (tx - x) * e, y: y + (ty - y) * e, z: 0, audio: 0 });
    }
    x = tx; y = ty;
    t += 60 * Math.random();
  }
  for (const pt of samples.filter((_, i) => i % 3 === 0)) {   // draw the trail without recording it again
    particles.push({ x: pt.x, y: pt.y, vx: 0, vy: 0, r: 4, alpha: 1.0, color: 'hsl(195, 95%, 62%)' });
  }
  if (particles.length > 220) particles.splice(0, particles.length - 220);
  finalizeRound();
}

async function copySeedHex(btn) {
  const txt = document.getElementById('seedHexBox').textContent.trim();
  if (!/^[0-9a-f]{64}$/i.test(txt)) return;
  const old = btn ? btn.textContent : '';
  try {
    await navigator.clipboard.writeText(txt);
    if (btn) btn.textContent = 'Copied';
  } catch (e) {
    if (btn) btn.textContent = 'Copy failed';
  }
  if (btn) setTimeout(() => { btn.textContent = old; }, 1200);
}

function downloadLedgerJson() {
  const blob = new Blob([JSON.stringify(jumpHistory, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'zoomies_seeds.json';
  a.click();
}

// Countdown bar along the bottom of the play area
function drawCountdown(progressOneToZero, secondsLeft) {
  const barH = 10, pad = 14, y = cssH - 20;
  ctx.fillStyle = 'rgba(8, 4, 15, 0.92)';
  ctx.fillRect(0, cssH - 34, cssW, 34);
  ctx.fillStyle = '#24173b';
  ctx.fillRect(pad, y - barH / 2, cssW - 2 * pad, barH);
  const w = (cssW - 2 * pad) * Math.max(0, Math.min(1, progressOneToZero));
  ctx.fillStyle = progressOneToZero > 0.5 ? '#4ade80' : (progressOneToZero > 0.25 ? '#facc15' : '#f87171');
  ctx.fillRect(pad, y - barH / 2, w, barH);
}

// Main 60fps Render & Live Assessment Loop
function animate() {
  ctx.fillStyle = 'rgba(6, 3, 13, 0.26)';
  ctx.fillRect(0, 0, cssW, cssH);

  // Slowly turning target ring in the center (still under reduced motion)
  const cx = cssW / 2, cy = (cssH - 30) / 2;
  const t = reducedMotion.matches ? 0 : performance.now() * 0.0025;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(t);
  ctx.strokeStyle = 'rgba(250, 204, 21, 0.25)';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(0, 0, 52 + Math.sin(t * 2) * 7, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  // Ribbon along the pointer's recent path (the last 0.6 s); jumps across the screen, such as key
  // presses, are not joined, so the ribbon never flickers as long lines across the play area
  const nowR = performance.now();
  while (trail.length && nowR - trail[0].t > 600) trail.shift();
  if (trail.length > 1) {
    const maxJump = 0.25 * Math.hypot(cssW, cssH);
    ctx.beginPath();
    for (let i = 0; i < trail.length; i++) {
      const p = trail[i];
      if (i === 0 || Math.hypot(p.x - trail[i - 1].x, p.y - trail[i - 1].y) > maxJump) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
    ctx.lineWidth = 3.0;
    ctx.stroke();
  }

  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx; p.y += p.vy;
    p.alpha -= 0.018;
    if (p.alpha <= 0) {
      particles.splice(i, 1);
      continue;
    }
    ctx.fillStyle = p.color;
    ctx.globalAlpha = Math.max(0, p.alpha);
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1.0;

  let timeRatio = 1.0;
  let secondsLeft = Math.ceil(JUMP_DURATION_MS / 1000);
  if (isJumping) {
    const elapsed = performance.now() - jumpStartTime;
    const rem = Math.max(0, (JUMP_DURATION_MS - elapsed) / 1000.0);
    timeRatio = Math.max(0, rem / (JUMP_DURATION_MS / 1000.0));
    secondsLeft = Math.max(1, Math.ceil(rem));
    if (secondsLeft !== lastSecondTick) {
      lastSecondTick = secondsLeft;
      playTone(secondsLeft <= 1 ? 880 : 660, 0.05, 'square', secondsLeft <= 1 ? 0.10 : 0.06);
    }
    document.getElementById('timerDisplay').textContent = `${rem.toFixed(1)} s`;
    if (samples.length >= 12 && (samples.length % 4 === 0)) {
      const liveEval = evaluateClientSide(samples, FORM_CARDS[currentCardIdx], playArea());
      updateHUDMetrics(liveEval);
    }
    if (elapsed >= JUMP_DURATION_MS) {
      finalizeRound();
    }
  }

  drawCountdown(timeRatio, secondsLeft);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
</script>
</body>
</html>
"""


def render_zoomies_html(console_link: bool = False) -> str:
    """The self-contained game page. The '← Console' link appears only when the console serves it;
    on the standalone game server or as a file there is no console to go back to."""
    from petrand.toolkit import EFF_VIVID_WORDS_256
    link = '<a class="btn btn-small" href="/">← Console</a>' if console_link else ""
    return (ZOOMIES_HTML.replace("__FORM_CARDS_JSON__", json.dumps(FORM_CARDS))
            .replace("__WORDS_JSON__", json.dumps(EFF_VIVID_WORDS_256))
            .replace("__CONSOLE_LINK__", link))


def write_standalone_html(out_path: str) -> str:
    """Writes a zero-dependency standalone HTML file that can be opened directly in any browser."""
    html_content = render_zoomies_html()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return os.path.abspath(out_path)


class ZoomiesHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the standalone Zoomies! game server."""

    engine: Optional[PetRandEngine] = None

    timeout = REQUEST_TIMEOUT_S  # drop idle or half-sent connections

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _refused(self, is_post: bool) -> bool:
        """Same guard as the console (petrand.http_guard). True when the request was refused."""
        refusal = request_refusal(self, is_post)
        if refusal:
            self._send_json(refusal[0], {"error": refusal[1]})
            return True
        return False

    def _json_body(self) -> Optional[Any]:
        """Read and parse a capped JSON body; on failure send the error and return None."""
        raw_body, err = read_body(self)
        if err:
            self._send_json(err[0], {"error": err[1]})
            return None
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON payload"})
            return None

    def _send_html(self, status_code: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self._refused(is_post=False):
            return
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/zoomies"):
            self._send_html(200, render_zoomies_html())
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path == "/api/v1/zoomies/cards":
            self._send_json(200, {"cards": FORM_CARDS})
        else:
            self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self) -> None:
        if self._refused(is_post=True):
            return
        path = self.path.split("?")[0]
        if path == "/api/v1/zoomies/evaluate":
            data = self._json_body()
            if data is None:
                return
            try:
                res = ZoomiesScorer.evaluate_request(data)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            # An unlocked round is mixed into this server's pool without credit
            if self.engine is not None and res["unlocked"] and res["seed_hex"]:
                self.engine.mix_uncredited(bytes.fromhex(res["seed_hex"]), label=f"zoomies_{res['form_id']}")
            self._send_json(200, res)
        else:
            self._send_json(404, {"error": "POST endpoint not found"})


def start_zoomies_server(
    host: str = "127.0.0.1",
    port: int = 8085,
    engine: Optional[PetRandEngine] = None,
) -> HTTPServer:
    """Starts the standalone Zoomies! game server."""
    handler_cls = type(
        "BoundZoomiesHandler",
        (ZoomiesHTTPHandler,),
        {"engine": engine},
    )
    # Threaded, so one slow or idle connection cannot freeze the game
    server = ThreadingHTTPServer((host, port), handler_cls)
    server.daemon_threads = True
    return server

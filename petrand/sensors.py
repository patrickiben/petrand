"""
petrand.sensors
===============
Sensors for the entropy engine. Every sensor reports ``source_kind`` and
``is_physical`` so the CLI, dashboard and ledger can say whether a seed came from
real hardware or from a simulator:
- CameraMotionSensor: overhead webcam tracking (falls back to a synthetic camera,
  marked ``camera-simulator``, when no camera is available)
- AudioImpactSensor: microphone impact capture with debouncing and clipping guards
- SyntheticArenaSimulator: toy billiard simulator for demos and tests (not physical)

Raw symbols are functions of the measurement only. They never mix in host-clock bits,
which would let the health tests and entropy estimates credit computer timing jitter
as if it came from the ball.
"""

import time
import math
import struct
import hashlib
from typing import Optional, Tuple, Any
import numpy as np
from petrand.engine import RawPhysicalEvent
from petrand.optical import OverheadCameraTracker
from petrand.lifecycle import TwoBallManager


class BaseSensor:
    """Base interface for all entropy sensors."""
    source_kind: str = "unknown"

    @property
    def is_physical(self) -> bool:
        """True only when samples come from real hardware (not a simulator or fallback)."""
        return False

    def describe(self) -> dict:
        return {"kind": self.source_kind, "physical": bool(self.is_physical)}

    def poll(self) -> Optional[RawPhysicalEvent]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CameraMotionSensor(BaseSensor):
    """
    Overhead webcam sensor utilizing OverheadCameraTracker.
    Tracks ball position, computes delta velocity and collision rebounds,
    and extracts sensor photon shot noise. Integrates with TwoBallManager.
    """
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 320,
        height: int = 240,
        fps: int = 60,
        ball_manager: Optional[TwoBallManager] = None,
        motion_threshold: float = 60.0,
        entropy_credit_per_collision: float = 2.5,
        entropy_credit_per_motion: float = 0.5,
    ):
        self.tracker = OverheadCameraTracker(
            camera_index=camera_index,
            width=width,
            height=height,
            fps=fps,
        )
        self.ball_manager = ball_manager or TwoBallManager()
        self.motion_threshold = motion_threshold
        self.entropy_credit_collision = entropy_credit_per_collision
        self.entropy_credit_motion = entropy_credit_per_motion
        self.event_counter = 0

    @property
    def is_hardware_camera(self) -> bool:
        return self.tracker.is_hardware_camera

    @property
    def source_kind(self) -> str:
        return "webcam" if self.tracker.is_hardware_camera else "camera-simulator"

    @property
    def is_physical(self) -> bool:
        return self.tracker.is_hardware_camera

    def poll(self) -> Optional[RawPhysicalEvent]:
        active_ball = self.ball_manager.current_ball
        res = self.tracker.process_frame(active_ball)

        if not res.detected or res.motion_energy < self.motion_threshold:
            return None

        # Record speed telemetry for battery monitoring
        self.ball_manager.record_motion(res.speed)

        self.event_counter += 1
        now_ns = res.timestamp_ns

        # Credit calculation: collisions provide higher entropy than regular motion
        if res.is_collision:
            credit = self.entropy_credit_collision
            channel = "camera_collision"
            raw_symbol = (int(res.speed * 10) ^ int(res.cx * 100)) & 0xFFFF
        else:
            credit = self.entropy_credit_motion
            channel = "camera_motion"
            raw_symbol = (int(res.motion_energy) ^ int(res.cx * 10) ^ int(res.cy * 10)) & 0xFFFF

        payload = struct.pack(
            "<Qdddf",
            now_ns,
            res.cx,
            res.cy,
            res.speed,
            res.motion_energy,
        ) + res.shot_noise_bytes[:128]

        return RawPhysicalEvent(
            event_id=self.event_counter,
            timestamp_ns=now_ns,
            channel=channel,
            raw_symbol=raw_symbol,
            sensor_payload=payload,
            estimated_entropy_bits=credit,
            ball_id=active_ball.ball_id,
        )

    def close(self) -> None:
        self.tracker.close()


class AudioImpactSensor(BaseSensor):
    """
    Acoustic collision sensor using sounddevice (physical when a microphone is available).
    Samples broadband acoustic transients when the ball strikes arena boundaries.
    Includes refractory debouncing (prevents double-counting ringdown echoes)
    and silence / ADC clipping health counters.
    """
    def __init__(
        self,
        samplerate: int = 44_100,
        block_seconds: float = 0.05,
        rms_threshold: float = 0.01,
        clipping_threshold: float = 0.98,
        refractory_ms: float = 30.0,
        entropy_credit_per_thud: float = 1.5,
        ball_id: str = "Ball_A",
        ball_manager: Optional[TwoBallManager] = None,
    ):
        self.samplerate = samplerate
        self.block_seconds = block_seconds
        self.rms_threshold = rms_threshold
        self.clipping_threshold = clipping_threshold
        self.refractory_ns = int(refractory_ms * 1_000_000)
        self.entropy_credit = entropy_credit_per_thud
        self.ball_id = ball_id
        self.ball_manager = ball_manager
        self.event_counter = 0
        self.last_trigger_ns = 0
        self.silence_events = 0
        self.clipping_events = 0

        try:
            import sounddevice as sd
            self.sd = sd
        except ImportError:
            self.sd = None

    @property
    def source_kind(self) -> str:
        return "microphone" if self.sd is not None else "microphone-unavailable"

    @property
    def is_physical(self) -> bool:
        return self.sd is not None

    def poll(self) -> Optional[RawPhysicalEvent]:
        if self.sd is None:
            return None

        now_ns = time.time_ns()
        if (now_ns - self.last_trigger_ns) < self.refractory_ns:
            return None

        frames = int(self.samplerate * self.block_seconds)
        audio = self.sd.rec(
            frames=frames,
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        self.sd.wait()
        samples = audio.ravel().astype("float64") / 32768.0
        rms = float((samples**2).mean() ** 0.5)
        peak = float(abs(samples).max()) if len(samples) > 0 else 0.0

        if rms < self.rms_threshold:
            self.silence_events += 1
            return None

        if peak >= self.clipping_threshold:
            # Reject saturated/clipped ADC blocks (potential acoustic tampering or overdrive)
            self.clipping_events += 1
            return None

        now_ns = time.time_ns()
        self.last_trigger_ns = now_ns
        self.event_counter += 1
        raw_symbol = int(rms * 10000) & 0xFFFF

        active_id = self.ball_manager.current_ball.ball_id if self.ball_manager is not None else self.ball_id
        payload = struct.pack("<Qd", now_ns, rms) + audio.tobytes()[:256]

        return RawPhysicalEvent(
            event_id=self.event_counter,
            timestamp_ns=now_ns,
            channel="audio",
            raw_symbol=raw_symbol,
            sensor_payload=payload,
            estimated_entropy_bits=self.entropy_credit,
            ball_id=active_id,
        )


class SyntheticArenaSimulator(BaseSensor):
    """
    Toy simulator for demos and tests: a constant-speed ball in a 1.2 m arena with
    specular reflections from the walls and three circular scatterers, a 90-degree
    motor turn every 0.6 to 1.0 s, and small random perturbations drawn from a NumPy
    generator.

    It is not an entropy source. All of its unpredictability comes from that generator,
    which is seeded from the operating system unless ``rng_seed`` is given. A fixed
    ``rng_seed`` reproduces the raw symbols; seeds still differ between runs because
    event timestamps go into the pool. It does not model friction, spin or firmware.
    """
    source_kind = "simulator"
    def __init__(
        self,
        arena_size: Tuple[float, float] = (1.2, 1.2),  # meters
        ball_radius: float = 0.025,                   # 50mm (2.0 in) diameter
        speed: float = 0.8,                           # m/s
        ball_id: str = "Ball_A",
        ball_manager: Optional[TwoBallManager] = None,
        sim_stuck: bool = False,
        rng_seed: Optional[int] = None,
    ):
        self._rng = np.random.default_rng(rng_seed)
        self.width, self.height = arena_size
        self.r = ball_radius
        self.speed = speed
        self.ball_id = ball_id
        self.ball_manager = ball_manager
        self.sim_stuck = sim_stuck

        # Ball state
        self.x = self.width / 2.0
        self.y = self.height / 2.0
        self.theta = 0.7854  # 45 degrees

        # Internal motor & physics timers
        self.sim_time = 0.0
        self.next_turn_sim_time = 1.5
        self.last_impact_sim_time = 0.0
        self.last_impact_time_ns = time.time_ns()
        self.event_counter = 0

        # Circular obstacles inside the arena (Sinai billiard scatterers)
        self.obstacles = [
            (0.4, 0.4, 0.08),
            (0.8, 0.7, 0.10),
            (0.3, 0.9, 0.06),
        ]

    def poll(self) -> Optional[RawPhysicalEvent]:
        now_ns = time.time_ns()
        active_id = self.ball_manager.current_ball.ball_id if self.ball_manager is not None else self.ball_id

        if self.sim_stuck:
            # Emulates a ball trapped under furniture (constant symbol output)
            self.event_counter += 1
            return RawPhysicalEvent(
                event_id=self.event_counter,
                timestamp_ns=now_ns,
                channel="sim_stuck",
                raw_symbol=42,  # Frozen constant value to trigger RCT
                sensor_payload=b"\x00" * 32,
                estimated_entropy_bits=0.01,
                ball_id=active_id,
            )

        dt = 0.016  # ~60Hz step
        self.sim_time += dt

        # Small random perturbation standing in for surface and motor irregularities
        micro_noise = float(self._rng.uniform(-0.005, 0.005))
        self.x += (self.speed * math.cos(self.theta) + micro_noise) * dt
        self.y += (self.speed * math.sin(self.theta) + micro_noise) * dt

        impact_detected = False
        impact_channel = "sim_motion"

        # 1. Wall collisions
        if self.x <= self.r:
            self.x = self.r
            self.theta = math.pi - self.theta + (micro_noise * 10)
            impact_detected = True
        elif self.x >= self.width - self.r:
            self.x = self.width - self.r
            self.theta = math.pi - self.theta + (micro_noise * 10)
            impact_detected = True

        if self.y <= self.r:
            self.y = self.r
            self.theta = -self.theta + (micro_noise * 10)
            impact_detected = True
        elif self.y >= self.height - self.r:
            self.y = self.height - self.r
            self.theta = -self.theta + (micro_noise * 10)
            impact_detected = True

        # 2. Obstacle scattering collisions
        for ox, oy, oradius in self.obstacles:
            dx = self.x - ox
            dy = self.y - oy
            dist = math.hypot(dx, dy)
            min_dist = self.r + oradius
            if dist < min_dist and dist > 1e-9:
                nx, ny = dx / dist, dy / dist
                dot = math.cos(self.theta) * nx + math.sin(self.theta) * ny
                vx = math.cos(self.theta) - 2 * dot * nx
                vy = math.sin(self.theta) - 2 * dot * ny
                self.theta = math.atan2(vy, vx) + (micro_noise * 5)
                self.x = ox + nx * min_dist
                self.y = oy + ny * min_dist
                impact_detected = True
                impact_channel = "sim_obstacle"

        # 3. Autonomous internal MCU direction reversal
        if self.sim_time >= self.next_turn_sim_time:
            self.theta += (math.pi / 2.0) + (micro_noise * 20)
            self.next_turn_sim_time = self.sim_time + 0.6 + (abs(micro_noise) * 80)
            impact_detected = True
            impact_channel = "sim_motor"

        if impact_detected:
            if self.ball_manager is not None:
                self.ball_manager.record_motion(self.speed, event_count=1)

            self.event_counter += 1
            sim_delta_us = int((self.sim_time - self.last_impact_sim_time) * 1_000_000)
            self.last_impact_time_ns = now_ns
            self.last_impact_sim_time = self.sim_time

            # Raw symbol from simulated quantities only (no host-clock bits)
            delta_t_us = sim_delta_us & 0xFFFFFFFFFFFFFFFF
            angle_quant = int((self.theta % (2.0 * math.pi)) * 10000)
            coord_quant = int((self.x * 1000) + (self.y * 1000))
            raw_symbol = (delta_t_us ^ angle_quant ^ coord_quant) & 0xFFFF

            payload = struct.pack("<QddQ", now_ns, self.x, self.y, delta_t_us)

            return RawPhysicalEvent(
                event_id=self.event_counter,
                timestamp_ns=now_ns,
                channel=impact_channel,
                raw_symbol=raw_symbol,
                sensor_payload=payload,
                estimated_entropy_bits=2.0,
                ball_id=active_id,
            )

        return None

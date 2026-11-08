"""
petrand.lifecycle
=================
Two-ball cycling state machine and battery depletion watchdog.
Manages continuous 24/7 entropy harvesting by alternating between two balls
(Ball A and Ball B) while one charges via USB-C.
"""

import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable


class BallStatus(Enum):
    ACTIVE = "active"                   # Ball is active in the arena
    LOW_BATTERY = "low_battery"         # Runtime or speed indicates battery is depleted
    SWAP_IN_PROGRESS = "swapping"       # Ball is being swapped; health monitoring paused
    PAUSED = "paused"                   # System paused manually


@dataclass
class BallProfile:
    ball_id: str                        # e.g. "Ball_A"
    color_name: str                     # e.g. "orange", "blue"
    hsv_lower: tuple                    # (H, S, V)
    hsv_upper: tuple                    # (H, S, V)
    total_active_seconds: float = 0.0
    total_events_harvested: int = 0
    charge_cycle_count: int = 0


# Preset HSV color profiles for PetGravity and similar motorized toys
COLOR_PRESETS = {
    "orange": ((8, 120, 120), (25, 255, 255)),
    "blue": ((95, 120, 100), (125, 255, 255)),
    "yellow": ((25, 120, 120), (35, 255, 255)),
    "green": ((40, 100, 100), (85, 255, 255)),
    "pink": ((150, 100, 100), (175, 255, 255)),
}


class TwoBallManager:
    """
    State machine that orchestrates alternating between two motorized balls.
    Prevents false NIST health alarms during physical ball swapping.
    """

    def __init__(
        self,
        ball_a_id: str = "Ball_A",
        ball_a_color: str = "orange",
        ball_b_id: str = "Ball_B",
        ball_b_color: str = "blue",
        max_active_duration_minutes: float = 35.0,
        speed_stagnation_threshold: float = 0.05,
    ):
        self.ball_a = BallProfile(
            ball_id=ball_a_id,
            color_name=ball_a_color,
            hsv_lower=COLOR_PRESETS.get(ball_a_color, ((8, 120, 120), (25, 255, 255)))[0],
            hsv_upper=COLOR_PRESETS.get(ball_a_color, ((8, 120, 120), (25, 255, 255)))[1],
        )
        self.ball_b = BallProfile(
            ball_id=ball_b_id,
            color_name=ball_b_color,
            hsv_lower=COLOR_PRESETS.get(ball_b_color, ((95, 120, 100), (125, 255, 255)))[0],
            hsv_upper=COLOR_PRESETS.get(ball_b_color, ((95, 120, 100), (125, 255, 255)))[1],
        )

        self.current_ball: BallProfile = self.ball_a
        self.idle_ball: BallProfile = self.ball_b
        self.status: BallStatus = BallStatus.ACTIVE

        self.max_duration_sec = max_active_duration_minutes * 60.0
        self.stagnation_speed_threshold = speed_stagnation_threshold

        self.ball_started_time = time.time()
        self.last_state_change = time.time()
        self.recent_speeds = []
        self.swap_count = 0

    @property
    def active_duration(self) -> float:
        return time.time() - self.ball_started_time

    @property
    def estimated_battery_percent(self) -> float:
        """Estimates remaining battery based on active runtime."""
        elapsed = self.active_duration
        remaining = max(0.0, 1.0 - (elapsed / self.max_duration_sec))
        return remaining * 100.0

    def record_motion(self, speed: float, event_count: int = 1) -> None:
        """Logs speed and checks for battery depletion or stagnation."""
        if self.status != BallStatus.ACTIVE:
            return

        self.current_ball.total_active_seconds += 0.016
        self.current_ball.total_events_harvested += event_count

        self.recent_speeds.append(speed)
        if len(self.recent_speeds) > 120:  # ~2 seconds at 60 fps
            self.recent_speeds.pop(0)

        # 1. Timer-based depletion check
        if self.active_duration >= self.max_duration_sec:
            self.status = BallStatus.LOW_BATTERY

        # 2. Speed-based stagnation check (ball motor dead / slowed down)
        if len(self.recent_speeds) >= 120:
            avg_speed = sum(self.recent_speeds) / len(self.recent_speeds)
            if avg_speed < self.stagnation_speed_threshold and self.active_duration > 60.0:
                self.status = BallStatus.LOW_BATTERY

    def initiate_swap(self) -> None:
        """
        Transitions system to SWAP_IN_PROGRESS state.
        Health monitors MUST pause while the user removes one ball and inserts the other.
        """
        self.status = BallStatus.SWAP_IN_PROGRESS
        self.last_state_change = time.time()
        self.current_ball.charge_cycle_count += 1

    def complete_swap(self) -> BallProfile:
        """
        Completes the physical swap, switching active ball to the charged one,
        resetting timers and speed history.
        """
        # Swap pointers
        self.current_ball, self.idle_ball = self.idle_ball, self.current_ball
        self.ball_started_time = time.time()
        self.last_state_change = time.time()
        self.recent_speeds.clear()
        self.status = BallStatus.ACTIVE
        self.swap_count += 1
        return self.current_ball

    def set_ball_colors(self, color_a: Optional[str] = None, color_b: Optional[str] = None) -> Dict[str, str]:
        """Updates the HSV color presets of Ball A and/or Ball B dynamically at runtime."""
        if color_a and color_a in COLOR_PRESETS:
            self.ball_a.color_name = color_a
            self.ball_a.hsv_lower, self.ball_a.hsv_upper = COLOR_PRESETS[color_a]
        if color_b and color_b in COLOR_PRESETS:
            self.ball_b.color_name = color_b
            self.ball_b.hsv_lower, self.ball_b.hsv_upper = COLOR_PRESETS[color_b]
        return {
            "ball_a_color": self.ball_a.color_name,
            "ball_b_color": self.ball_b.color_name,
        }

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "active_ball_id": self.current_ball.ball_id,
            "active_ball_color": self.current_ball.color_name,
            "idle_ball_id": self.idle_ball.ball_id,
            "idle_ball_color": self.idle_ball.color_name,
            "ball_a_color": self.ball_a.color_name,
            "ball_b_color": self.ball_b.color_name,
            "active_duration_minutes": round(self.active_duration / 60.0, 1),
            "estimated_battery_percent": round(self.estimated_battery_percent, 1),
            "swap_count": self.swap_count,
            "ball_a_events": self.ball_a.total_events_harvested,
            "ball_b_events": self.ball_b.total_events_harvested,
        }

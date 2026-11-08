"""
Anti-bot audit for the Zoomies! motion game: an AgentEnv-style environment that runs
scripted motion bots (static, sine, figure-eight, circle, sawtooth, tiny jitter, a still
phone) and a synthetic "chaotic human" against the game's chaos meter and rhythm detector.
"""

from __future__ import annotations

import math
import struct
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .engine import RawPhysicalEvent
from .zoomies import ZoomiesScorer


# ============================================================================
# Anti-bot audit environment for the motion game
# ============================================================================

class ZoomiesBotAudit:
    """
    An AgentEnv-style environment (reset and step) wrapping
    ZoomiesScorer to test adversarial spoofing bots vs. genuine human chaos.
    """

    BOT_POLICIES = (
        "static_bot",
        "sine_jiggler_bot",
        "lissajous_bot",
        "uniform_circle_bot",
        "sawtooth_sweep_bot",
        "tiny_jitter_bot",
        "still_phone_bot",
        "back_and_forth_bot",
        "chaotic_human_agent",
    )
    AREA = (500.0, 250.0)  # the play-area size the scripted bots draw in

    def __init__(self, form_id: str = "vortex_orbit", horizon_steps: int = 180) -> None:
        self.form_id = form_id
        self.horizon_steps = horizon_steps
        self.trajectory: List[Dict[str, float]] = []
        self._x = 250.0
        self._y = 125.0
        self._z = 0.0
        self._t = 0.0

    def reset(self, form_id: Optional[str] = None) -> Dict[str, float]:
        if form_id:
            self.form_id = form_id
        self.trajectory = []
        self._x = 250.0
        self._y = 125.0
        self._z = 0.0
        self._t = 0.0
        sample = {"t": self._t, "x": self._x, "y": self._y, "z": self._z, "audio": 0.0}
        self.trajectory.append(sample)
        return sample

    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """
        Execute one RL environment step with continuous action (dx, dy, dz, dt, audio).
        Returns (observation, reward, done, info).
        """
        dt = max(4.0, float(action.get("dt", 18.0)))
        self._t += dt
        self._x = max(0.0, min(500.0, self._x + float(action.get("dx", 0.0))))
        self._y = max(0.0, min(250.0, self._y + float(action.get("dy", 0.0))))
        self._z = max(-200.0, min(200.0, self._z + float(action.get("dz", 0.0))))
        aud = max(0.0, min(1.0, float(action.get("audio", 0.0))))

        obs = {"t": self._t, "x": self._x, "y": self._y, "z": self._z, "audio": aud}
        self.trajectory.append(obs)

        done = len(self.trajectory) >= self.horizon_steps
        if done:
            eval_res = ZoomiesScorer.evaluate_trajectory(
                self.trajectory,
                form_id=self.form_id,
                key_mashes=int(action.get("key_mashes", 6)),
            )
            # RL reward: positive for passing genuine chaos lock, negative penalty if caught cheating
            reward = (
                float(eval_res["chaos_score"])
                if eval_res["unlocked"]
                else -float(100.0 - eval_res["chaos_score"])
            )
            return obs, reward, True, eval_res
        return obs, 0.0, False, {}

    @classmethod
    def generate_policy_trajectory(
        cls,
        policy_name: str,
        num_samples: int = 180,
        seed: int = 42,
    ) -> List[Dict[str, float]]:
        """Generate a 180-sample trajectory for any of the 5 adversarial bots or human benchmark."""
        rng = np.random.default_rng(seed)
        samples: List[Dict[str, float]] = []
        t = 0.0

        for i in range(num_samples):
            if policy_name == "static_bot":
                t += 20.0
                samples.append({"t": t, "x": 250.0, "y": 125.0, "z": 0.0, "audio": 0.0})

            elif policy_name == "sine_jiggler_bot":
                t += 20.0
                samples.append({
                    "t": t,
                    "x": 250.0 + 140.0 * math.sin(2.0 * math.pi * 2.5 * (t / 1000.0)),
                    "y": 125.0 + 90.0 * math.cos(2.0 * math.pi * 2.5 * (t / 1000.0)),
                    "z": 0.0,
                    "audio": 0.0,
                })

            elif policy_name == "lissajous_bot":
                # 2-frequency harmonic figure-8 macro
                t += 20.0
                sec = t / 1000.0
                samples.append({
                    "t": t,
                    "x": 250.0 + 180.0 * math.sin(2.0 * math.pi * 3.0 * sec + math.pi / 4.0),
                    "y": 125.0 + 110.0 * math.sin(2.0 * math.pi * 6.0 * sec),
                    "z": 40.0 * math.cos(2.0 * math.pi * 3.0 * sec),
                    "audio": 0.0,
                })

            elif policy_name == "uniform_circle_bot":
                # Constant-speed circular orbit macro trying to game 16-sector dispersion
                t += 20.0
                angle = (i / float(num_samples)) * 8.0 * math.pi
                samples.append({
                    "t": t,
                    "x": 250.0 + 120.0 * math.cos(angle),
                    "y": 125.0 + 120.0 * math.sin(angle),
                    "z": 0.0,
                    "audio": 0.0,
                })

            elif policy_name == "sawtooth_sweep_bot":
                # Triangle / sawtooth linear sweep macro
                t += 20.0
                phase = (i % 30) / 30.0
                tri = 1.0 - abs(2.0 * phase - 1.0)
                samples.append({
                    "t": t,
                    "x": 50.0 + 400.0 * tri,
                    "y": 30.0 + 190.0 * phase,
                    "z": 0.0,
                    "audio": 0.0,
                })

            elif policy_name in ("tiny_jitter_bot", "still_phone_bot"):
                # 1 px mouse jiggle, or a phone lying still (1.5 px of sensor noise)
                t += 10.0 if policy_name == "tiny_jitter_bot" else 16.0
                amp = 1.0 if policy_name == "tiny_jitter_bot" else 1.5
                samples.append({"t": t, "x": 250.0 + float(rng.uniform(-amp, amp)),
                                "y": 125.0 + float(rng.uniform(-amp, amp)), "z": 0.0, "audio": 0.0})

            elif policy_name == "back_and_forth_bot":
                # Big diagonal scrubbing at a steady 2 Hz beat
                t += 20.0
                ph = (2.0 * t / 1000.0) % 1.0
                tri = 1.0 - abs(2.0 * ph - 1.0)
                samples.append({"t": t, "x": 50.0 + 400.0 * tri, "y": 30.0 + 190.0 * tri, "z": 0.0, "audio": 0.0})

        if policy_name == "chaotic_human_agent":
            # Fast, smooth strokes between random targets, the way people flick a mouse
            return ZoomiesScorer.generate_synthetic_trajectory("chaotic", seed=seed, area=cls.AREA)
        return samples

    @classmethod
    def run_adversarial_audit(cls, form_id: str = "vortex_orbit") -> Dict[str, Any]:
        """
        Run all 5 adversarial spoofing bots + 1 chaotic human benchmark against
        ZoomiesScorer and verify that every bot is rejected while the chaotic
        human trajectory unlocks a 256-bit seed.
        """
        results = []
        all_bots_blocked = True
        human_passed = False

        for policy in cls.BOT_POLICIES:
            traj = cls.generate_policy_trajectory(policy, num_samples=180)
            eval_res = ZoomiesScorer.evaluate_trajectory(
                traj,
                form_id=form_id,
                key_mashes=8 if policy == "chaotic_human_agent" else 0,
                area=cls.AREA,
            )
            is_bot = policy != "chaotic_human_agent"
            passed = bool(eval_res["unlocked"])
            if is_bot and passed:
                all_bots_blocked = False
            if (not is_bot) and passed:
                human_passed = True

            results.append({
                "policy": policy,
                "is_adversarial_bot": is_bot,
                "unlocked": passed,
                "expected_outcome_met": (not passed) if is_bot else passed,
                "status": eval_res["status"],
                "chaos_score": eval_res["chaos_score"],
                "metrics": eval_res["metrics"],
            })

        return {
            "audit_passed": bool(all_bots_blocked and human_passed),
            "all_bots_blocked": all_bots_blocked,
            "chaotic_human_passed": human_passed,
            "policies_tested": len(results),
            "results": results,
            "agent_env_spec": cls.export_environment_spec(),
        }

    @staticmethod
    def export_environment_spec() -> Dict[str, Any]:
        """The environment's action and reward description, for anyone driving it with their own agent."""
        return {
            "environment_id": "petrand-zoomies-v1",
            "endpoint": "/api/v1/zoomies/evaluate",
            "observation_space": {
                "t": "float64 (ms)",
                "x": "float64 [0, 500]",
                "y": "float64 [0, 250]",
                "z": "float64 [-200, 200]",
                "audio": "float64 [0, 1]",
            },
            "action_space": {
                "dx": "float64 [-150, 150]",
                "dy": "float64 [-150, 150]",
                "dz": "float64 [-100, 100]",
                "dt": "float64 [4, 50]",
            },
            "reward_check": {
                "lock_threshold": ZoomiesScorer.UNLOCK_THRESHOLD,
                "anti_bot_gates": [
                    "3D Kinematic Jerk RMS",
                    "FFT Spectral Flatness (Wiener Entropy >= 0.46)",
                    "Top-2 Harmonic Concentration (<= 0.56)",
                    "16-Sector Angular Dispersion (>= 38.0%) & Turn Irregularity",
                    "Speed Coefficient of Variation (CV >= 0.22 to reject constant-speed circle bots)",
                ],
            },
        }

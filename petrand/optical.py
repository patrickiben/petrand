"""
petrand.optical
===============
Dedicated Overhead Webcam Vision Pipeline.
Provides dual-color ball tracking (HSV segmentation), ROI masking, morphological
filtering, rebound/collision detection, frame difference energy, and sensor shot noise extraction.
Works seamlessly with OpenCV (cv2) or with pure NumPy vectorized color segmentation.
"""

import time
import math
import numpy as np
from typing import Optional, Tuple
from petrand.lifecycle import BallProfile, COLOR_PRESETS


class OpticalTrackingResult:
    def __init__(
        self,
        detected: bool,
        cx: float,
        cy: float,
        radius: float,
        speed: float,
        is_collision: bool,
        motion_energy: float,
        shot_noise_bytes: bytes,
        timestamp_ns: int,
        alternate_detected_color: Optional[str] = None,
        lighting_stable: bool = True,
    ):
        self.detected = detected
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.speed = speed
        self.is_collision = is_collision
        self.motion_energy = motion_energy
        self.shot_noise_bytes = shot_noise_bytes
        self.timestamp_ns = timestamp_ns
        self.alternate_detected_color = alternate_detected_color
        self.lighting_stable = lighting_stable


class OverheadCameraTracker:
    """
    Overhead webcam vision processor for the PetRand arena.
    Tracks ball coordinates, computes velocities and rebound angles,
    and extracts sensor photon shot noise with ROI masking and morphological filtering.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 320,
        height: int = 240,
        fps: int = 60,
        roi_rect: Optional[Tuple[int, int, int, int]] = None,  # (x, y, w, h)
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_index = camera_index
        self.roi_rect = roi_rect

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        self.prev_vx: float = 0.0
        self.prev_vy: float = 0.0
        self.prev_time_ns: int = time.time_ns()
        self.baseline_brightness: Optional[float] = None
        self._sim_rng = np.random.Generator(np.random.PCG64())
        self._sim_step = 0

        # Try initializing OpenCV
        try:
            import cv2
            self.cv2 = cv2
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            if not self.cap.isOpened():
                self.cap = None
        except ImportError:
            self.cv2 = None
            self.cap = None

    @property
    def is_hardware_camera(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Sets the Region-Of-Interest bounding box for arena cropping."""
        self.roi_rect = (x, y, w, h)

    def process_frame(self, ball_profile: BallProfile) -> OpticalTrackingResult:
        """
        Grabs a frame, crops to ROI if configured, applies color thresholding
        for the active ball profile, and computes dynamics + shot noise.
        """
        now_ns = time.time_ns()

        if self.is_hardware_camera:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return self._empty_result(now_ns)
        else:
            frame = self._generate_synthetic_frame(ball_profile, now_ns)

        # Apply ROI crop if defined
        if self.roi_rect is not None:
            rx, ry, rw, rh = self.roi_rect
            frame = frame[ry : ry + rh, rx : rx + rw]

        return self._analyze_frame(frame, ball_profile, now_ns)

    @staticmethod
    def _numpy_color_mask(frame: np.ndarray, color_name: str) -> np.ndarray:
        """Vectorized BGR color segmentation fallback when OpenCV is not installed."""
        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]
        if color_name == "orange":
            return (r > 180) & (g > 80) & (g < 200) & (b < 70)
        elif color_name == "blue":
            return (b > 180) & (g > 80) & (r < 70)
        elif color_name == "yellow":
            return (r > 180) & (g > 180) & (b < 80)
        elif color_name == "green":
            return (g > 160) & (r < 100) & (b < 100)
        elif color_name == "pink":
            return (r > 180) & (b > 120) & (g < 120)
        return (r > 150) | (b > 150)

    def _analyze_frame(
        self, frame: np.ndarray, ball_profile: BallProfile, now_ns: int
    ) -> OpticalTrackingResult:
        h, w = frame.shape[:2]
        alternate_color = None

        if self.cv2 is not None:
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)

            # Morphological cleaning kernel
            kernel = np.ones((3, 3), np.uint8)

            # Primary ball color thresholding
            lower = np.array(ball_profile.hsv_lower, dtype=np.uint8)
            upper = np.array(ball_profile.hsv_upper, dtype=np.uint8)
            mask = self.cv2.inRange(hsv, lower, upper)
            mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, kernel)
            mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)
            contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)

            detected = False
            cx, cy, radius = 0.0, 0.0, 0.0
            if contours:
                c = max(contours, key=self.cv2.contourArea)
                if self.cv2.contourArea(c) > 20:
                    (x, y), r = self.cv2.minEnclosingCircle(c)
                    cx, cy, radius = float(x), float(y), float(r)
                    detected = True

            # If primary ball not detected, check alternate ball color (e.g. blue instead of orange)
            if not detected:
                for alt_color, (alt_lower, alt_upper) in COLOR_PRESETS.items():
                    if alt_color != ball_profile.color_name:
                        alt_mask = self.cv2.inRange(hsv, np.array(alt_lower, np.uint8), np.array(alt_upper, np.uint8))
                        alt_cnts, _ = self.cv2.findContours(alt_mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
                        if alt_cnts and self.cv2.contourArea(max(alt_cnts, key=self.cv2.contourArea)) > 30:
                            alternate_color = alt_color
                            break
        else:
            # Pure NumPy pixel segmentation fallback
            gray = np.mean(frame, axis=2).astype(np.uint8)
            mask = self._numpy_color_mask(frame, ball_profile.color_name)
            pixel_count = int(np.count_nonzero(mask))

            detected = False
            cx, cy, radius = 0.0, 0.0, 0.0
            if pixel_count > 15:
                ys, xs = np.nonzero(mask)
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                radius = float(math.sqrt(pixel_count / math.pi))
                detected = True
            else:
                for alt_color in COLOR_PRESETS.keys():
                    if alt_color != ball_profile.color_name:
                        alt_mask = self._numpy_color_mask(frame, alt_color)
                        if int(np.count_nonzero(alt_mask)) > 20:
                            alternate_color = alt_color
                            break

        # Lighting stability check (detect room lamp on/off or shadow cast)
        curr_brightness = float(np.mean(gray))
        lighting_stable = True
        if self.baseline_brightness is None:
            self.baseline_brightness = curr_brightness
        else:
            if abs(curr_brightness - self.baseline_brightness) > 60.0:
                lighting_stable = False
            # Smooth tracking
            self.baseline_brightness = 0.95 * self.baseline_brightness + 0.05 * curr_brightness

        # 1. Motion energy from frame difference
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            diff = gray.copy()
            motion_energy = float(np.sum(diff)) * 0.01
        else:
            diff = np.abs(gray.astype(np.int16) - self.prev_gray.astype(np.int16)).astype(np.uint8)
            self.prev_gray = gray
            motion_energy = float(np.sum(diff))

        # 2. Velocity and Rebound Collision Detection. Real frames use the nominal frame
        # interval so speeds (and the raw symbols built from them) do not pick up host
        # scheduling jitter; synthetic frames keep host time.
        if self.is_hardware_camera:
            dt = 1.0 / max(1, int(self.fps))
        else:
            dt = max(1e-3, (now_ns - self.prev_time_ns) * 1e-9)
        self.prev_time_ns = now_ns

        speed = 0.0
        is_collision = False

        if detected and self.prev_x is not None and self.prev_y is not None:
            vx = (cx - self.prev_x) / dt
            vy = (cy - self.prev_y) / dt
            speed = math.hypot(vx, vy)

            # Check for velocity deflection (rebound angle change > 45 degrees)
            prev_speed = math.hypot(self.prev_vx, self.prev_vy)
            if prev_speed > 10.0 and speed > 10.0:
                dot = (vx * self.prev_vx + vy * self.prev_vy) / (speed * prev_speed)
                dot = max(-1.0, min(1.0, dot))
                angle_delta = math.acos(dot)
                if angle_delta > 0.785:  # ~45 degrees
                    is_collision = True

            # Boundary collision check
            if cx <= radius + 5 or cx >= w - radius - 5:
                is_collision = True
            if cy <= radius + 5 or cy >= h - radius - 5:
                is_collision = True

            self.prev_vx, self.prev_vy = vx, vy
        elif detected:
            self.prev_vx, self.prev_vy = 0.0, 0.0

        if detected:
            self.prev_x, self.prev_y = cx, cy

        # 3. Photon Shot Noise: Extract LSBs from frame difference
        lsb_bits = np.bitwise_and(diff, 1).ravel()
        shot_noise_bytes = np.packbits(lsb_bits[:1024]).tobytes()

        return OpticalTrackingResult(
            detected=detected,
            cx=cx,
            cy=cy,
            radius=radius,
            speed=speed,
            is_collision=is_collision,
            motion_energy=motion_energy,
            shot_noise_bytes=shot_noise_bytes,
            timestamp_ns=now_ns,
            alternate_detected_color=alternate_color,
            lighting_stable=lighting_stable,
        )

    def _generate_synthetic_frame(self, ball_profile: BallProfile, now_ns: int) -> np.ndarray:
        """Simulates overhead camera view with Poisson noise and moving color disk."""
        self._sim_step += 1
        frame = np.full((self.height, self.width, 3), 50, dtype=np.uint8)

        # Microscopic photon shot noise
        noise = self._sim_rng.integers(0, 4, size=(self.height, self.width), dtype=np.uint8)
        frame[:, :, 0] ^= noise

        # Ball center trajectory (advances smoothly even in tight polling loops)
        t = (now_ns * 1e-9) + (self._sim_step * 0.04)
        bx = int(self.width / 2.0 + math.sin(t * 2.5) * (self.width * 0.35))
        by = int(self.height / 2.0 + math.cos(t * 3.1) * (self.height * 0.35))
        r = 10

        y_coords, x_coords = np.ogrid[: self.height, : self.width]
        dist_sq = (x_coords - bx) ** 2 + (y_coords - by) ** 2
        mask = dist_sq <= r**2

        if ball_profile.color_name == "orange":
            frame[mask] = (0, 140, 255)   # BGR orange
        elif ball_profile.color_name == "blue":
            frame[mask] = (255, 140, 0)   # BGR blue
        elif ball_profile.color_name == "green":
            frame[mask] = (0, 220, 0)     # BGR green
        elif ball_profile.color_name == "pink":
            frame[mask] = (180, 80, 255)  # BGR pink
        else:
            frame[mask] = (0, 255, 255)   # BGR yellow

        return frame

    def _empty_result(self, now_ns: int) -> OpticalTrackingResult:
        return OpticalTrackingResult(
            detected=False,
            cx=0.0,
            cy=0.0,
            radius=0.0,
            speed=0.0,
            is_collision=False,
            motion_energy=0.0,
            shot_noise_bytes=b"\x00" * 32,
            timestamp_ns=now_ns,
            lighting_stable=True,
        )

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

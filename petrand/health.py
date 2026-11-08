"""
petrand.health
==============
Continuous health tests on raw entropy samples, following the designs in NIST SP 800-90B:
- Repetition Count Test (RCT) (§4.4.1)
- Adaptive Proportion Test (APT) (§4.4.2)
- Start-up testing over at least 1,024 consecutive samples (§4.3)
These tests catch gross failures (a stuck or badly biased source). They cannot show
that a source has the min-entropy H they are configured for; only an entropy
assessment of the raw samples can do that.
"""

import math
from typing import Optional, List, Dict, Any


class HealthError(RuntimeError):
    """Base exception for entropy health monitoring failures."""
    pass


class RepetitionCountFailure(HealthError):
    """Raised when the source outputs the same value too many consecutive times."""
    pass


class AdaptiveProportionFailure(HealthError):
    """Raised when a single value occurs with abnormally high frequency in a window."""
    pass


class StartupHealthFailure(HealthError):
    """Raised when the entropy source fails initial validation during startup."""
    pass


class NISTStartupHealthTest:
    """
    NIST SP 800-90B §4.3: Startup Health Testing.
    The continuous tests must pass over at least 1,024 consecutive samples (SP 800-90B
    requires at least 1,024) before the source enters the OPERATIONAL state.
    """

    def __init__(self, required_startup_samples: int = 1024):
        self.required_samples = required_startup_samples
        self.samples_collected = 0
        self.passed = False

    def check_sample(self) -> bool:
        """Increments valid sample count. Returns True when startup phase is satisfied."""
        if not self.passed:
            self.samples_collected += 1
            if self.samples_collected >= self.required_samples:
                self.passed = True
        return self.passed

    def reset(self) -> None:
        self.samples_collected = 0
        self.passed = False


class NISTHealthMonitor:
    """
    Continuous online health testing following NIST SP 800-90B §4.4.
    
    Tests are performed directly on raw, unconditioned digitized physical values
    (e.g. raw inter-arrival microseconds or quantized pixel motion values),
    BEFORE any cryptographic hashing or whitening.
    """

    def __init__(
        self,
        min_entropy_per_sample: float = 0.5,
        window_size: int = 512,
        alpha: float = 2.0**(-20),
        require_startup_test: bool = True,
        startup_samples: int = 1024,
    ):
        """
        Parameters
        ----------
        min_entropy_per_sample : float
            Assumed lower-bound min-entropy H in bits per raw symbol (0.0 < H <= 8.0).
        window_size : int
            Window size W for the Adaptive Proportion Test. SP 800-90B uses W = 512 for
            non-binary samples (PetRand's raw symbols) and W = 1,024 for binary samples.
        alpha : float
            Type I error (false-positive alarm) threshold, default 2^-20 (~9.54e-7).
        require_startup_test : bool
            Whether to enforce NIST SP 800-90B §4.3 startup testing before approving samples.
        startup_samples : int
            Number of consecutive passing samples required to satisfy startup test.
        """
        if not (0.0 < min_entropy_per_sample <= 16.0):
            raise ValueError("min_entropy_per_sample must be in (0, 16]: raw symbols are 16-bit")
        if window_size < 16:
            raise ValueError("window_size must be at least 16")

        self.H = float(min_entropy_per_sample)
        self.W = int(window_size)
        self.alpha = float(alpha)

        # 1. Repetition Count Test (RCT) Cutoff: C = 1 + ceil(-log2(alpha) / H)
        neg_log2_alpha = -math.log2(self.alpha)
        self.rct_cutoff = 1 + math.ceil(neg_log2_alpha / self.H)

        # 2. Adaptive Proportion Test (APT) Cutoff calculation
        self.apt_cutoff = self._compute_apt_cutoff()

        # 3. Startup Health Tester
        self.startup_test = NISTStartupHealthTest(required_startup_samples=startup_samples) if require_startup_test else None

        # State tracking
        self.last_sample: Optional[int] = None
        self.rct_consecutive = 0

        self.apt_target: Optional[int] = None
        self.apt_window: List[int] = []
        self.apt_matches: int = 0

        self.total_samples_checked = 0
        self.rct_alarms = 0
        self.apt_alarms = 0

    @property
    def is_operational(self) -> bool:
        """Returns True if startup test is completed (or disabled) and monitor is ready."""
        if self.startup_test is None:
            return True
        return self.startup_test.passed

    def _compute_apt_cutoff(self) -> int:
        """
        SP 800-90B APT cutoff: C = 1 + CRITBINOM(W, 2^-H, 1 - alpha), where CRITBINOM(W, p, q)
        is the smallest k with P(Binomial(W, p) <= k) >= q. The window count B includes the
        window's first sample, and the test fails when B >= C.
        """
        p = 2.0 ** (-self.H)
        W = self.W
        target = 1.0 - self.alpha
        try:
            from scipy.stats import binom
            return min(W, 1 + int(binom.ppf(target, W, p)))
        except ImportError:
            pass
        # Exact fallback: accumulate the binomial pmf in log space
        log_p = math.log(p)
        log_q = math.log1p(-p) if p < 1.0 else float("-inf")
        cdf = 0.0
        for k in range(W + 1):
            log_pmf = (math.lgamma(W + 1) - math.lgamma(k + 1) - math.lgamma(W - k + 1)
                       + k * log_p + (W - k) * log_q)
            cdf += math.exp(log_pmf)
            if cdf >= target:
                return min(W, 1 + k)
        return W

    def validate_sample(self, raw_sample: int) -> None:
        """
        Validates a single incoming raw physical symbol.
        
        Raises
        ------
        RepetitionCountFailure : If the sample repeats consecutively >= rct_cutoff.
        AdaptiveProportionFailure : If the window count B (first sample included) reaches apt_cutoff.
        """
        val = int(raw_sample)
        self.total_samples_checked += 1

        # --- Repetition Count Test (RCT) ---
        if self.last_sample is not None and val == self.last_sample:
            self.rct_consecutive += 1
            if self.rct_consecutive >= self.rct_cutoff:
                self.rct_alarms += 1
                raise RepetitionCountFailure(
                    f"NIST SP 800-90B RCT Alarm: sample {val} repeated {self.rct_consecutive} "
                    f"times consecutively (cutoff: {self.rct_cutoff}, H={self.H:.2f})"
                )
        else:
            self.last_sample = val
            self.rct_consecutive = 1

        # --- Adaptive Proportion Test (APT) ---
        if self.apt_target is None:
            self.apt_target = val
            self.apt_window = []
            self.apt_matches = 1  # B counts the window's first sample (SP 800-90B §4.4.2)
        else:
            self.apt_window.append(val)
            if val == self.apt_target:
                self.apt_matches += 1

            if self.apt_matches >= self.apt_cutoff:
                matches = self.apt_matches
                target = self.apt_target
                self.apt_target = None
                self.apt_window = []
                self.apt_matches = 0
                self.apt_alarms += 1
                raise AdaptiveProportionFailure(
                    f"NIST SP 800-90B APT Alarm: sample {target} matched {matches} "
                    f"times in window {self.W} (cutoff: {self.apt_cutoff}, H={self.H:.2f})"
                )

            if len(self.apt_window) >= self.W - 1:
                self.apt_target = None
                self.apt_window = []
                self.apt_matches = 0

        # --- Startup Health Test Progress ---
        if self.startup_test is not None:
            self.startup_test.check_sample()

    def reset(self) -> None:
        """Resets the monitor state while preserving lifetime telemetry counts."""
        self.last_sample = None
        self.rct_consecutive = 0
        self.apt_target = None
        self.apt_window = []
        self.apt_matches = 0
        if self.startup_test is not None:
            self.startup_test.reset()

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "total_samples_checked": self.total_samples_checked,
            "rct_consecutive": self.rct_consecutive,
            "rct_cutoff": self.rct_cutoff,
            "apt_matches": self.apt_matches,
            "apt_window_progress": len(self.apt_window),
            "apt_cutoff": self.apt_cutoff,
            "rct_alarms": self.rct_alarms,
            "apt_alarms": self.apt_alarms,
            "is_operational": self.is_operational,
            "startup_samples_remaining": (
                0 if (self.startup_test is None or self.startup_test.passed)
                else (self.startup_test.required_samples - self.startup_test.samples_collected)
            ),
        }

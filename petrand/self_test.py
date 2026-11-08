"""
petrand.self_test
=================
Power-on and on-demand self-tests, in the style of FIPS 140-3 known-answer tests
(this package is not a validated module).

Checks that the health tests fire when they should and stay quiet when they should,
and that the hash functions match published test vectors.
"""

import hashlib
import hmac
from typing import Dict, Any
from petrand.health import (
    NISTHealthMonitor,
    RepetitionCountFailure,
    AdaptiveProportionFailure,
)


class KnownAnswerSelfTestError(RuntimeError):
    """Raised when an internal self-test or known-answer test fails."""
    pass


# RFC 4231, Test Case 1: key = 0x0b * 20, data = "Hi There". Checks both conditioners
# used by PetRandEngine.extract_seed (HMAC-SHA-256 and HMAC-SHA-512).
HMAC_SHA256_RFC4231_TC1_HEX = "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
HMAC_SHA512_RFC4231_TC1_KEY = b"\x0b" * 20
HMAC_SHA512_RFC4231_TC1_DATA = b"Hi There"
HMAC_SHA512_RFC4231_TC1_HEX = (
    "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cde"
    "daa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854"
)

# RFC 7693, Appendix A: BLAKE2b-512("abc"). BLAKE2b is still used for sub-seed
# derivation in adapters/benchmarks (not for conditioning).
BLAKE2B_512_RFC7693_ABC_HEX = (
    "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1"
    "7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
)


class SelfTestEngine:
    """
    Known-answer self-tests:
    1. RCT fires at exactly its cutoff C
    2. APT fires on a skewed window
    3. A well-spread stream completes start-up without false alarms
    4. Conditioner and hash KATs: HMAC-SHA-256/512 (RFC 4231 TC1), BLAKE2b-512 (RFC 7693 App. A)
    """

    @classmethod
    def run_all_self_tests(cls) -> Dict[str, Any]:
        results = {}

        # Test 1: RCT KAT
        try:
            cls.test_rct_known_answer()
            results["rct_kat"] = "PASSED [OK]"
        except Exception as exc:
            results["rct_kat"] = f"FAILED: {exc}"
            raise KnownAnswerSelfTestError(f"RCT KAT failed: {exc}")

        # Test 2: APT KAT
        try:
            cls.test_apt_known_answer()
            results["apt_kat"] = "PASSED [OK]"
        except Exception as exc:
            results["apt_kat"] = f"FAILED: {exc}"
            raise KnownAnswerSelfTestError(f"APT KAT failed: {exc}")

        # Test 3: Healthy Stream Non-Interference KAT
        try:
            cls.test_healthy_stream_kat()
            results["healthy_stream_kat"] = "PASSED [OK]"
        except Exception as exc:
            results["healthy_stream_kat"] = f"FAILED: {exc}"
            raise KnownAnswerSelfTestError(f"Healthy Stream KAT failed: {exc}")

        # Test 4: Conditioner and hash KATs (published RFC vectors)
        try:
            cls.test_conditioning_kat()
            results["conditioning_kat"] = "PASSED [OK]"
        except Exception as exc:
            results["conditioning_kat"] = f"FAILED: {exc}"
            raise KnownAnswerSelfTestError(f"Conditioning KAT failed: {exc}")

        results["status"] = "ALL_TESTS_PASSED"
        return results

    @staticmethod
    def test_rct_known_answer() -> None:
        """Verifies that RCT triggers at exactly C consecutive repeats, and not at C-1."""
        monitor = NISTHealthMonitor(min_entropy_per_sample=1.0, require_startup_test=False)
        cutoff = monitor.rct_cutoff  # For H=1.0, alpha=2^-20, cutoff = 1 + ceil(20/1) = 21

        # Feed cutoff - 1 repeats (must pass)
        for _ in range(cutoff - 1):
            monitor.validate_sample(42)

        # Feed the cutoff-th sample (MUST raise RepetitionCountFailure)
        alarm_raised = False
        try:
            monitor.validate_sample(42)
        except RepetitionCountFailure:
            alarm_raised = True

        if not alarm_raised:
            raise KnownAnswerSelfTestError(f"RCT failed to trigger on exactly {cutoff} repeated symbols")

    @staticmethod
    def test_apt_known_answer() -> None:
        """
        Verifies the APT cutoff against SP 800-90B Table 2 (W = 512, H = 0.5 gives 410),
        that a window holding exactly C copies of its first value fires and C - 1 does not,
        and that a skewed small window fires.
        """
        mon = NISTHealthMonitor(min_entropy_per_sample=0.5, window_size=512, require_startup_test=False)
        if mon.apt_cutoff != 410:
            raise KnownAnswerSelfTestError(f"APT cutoff {mon.apt_cutoff} != 410 (W=512, H=0.5)")

        def window_fires(total_targets: int) -> bool:
            m = NISTHealthMonitor(min_entropy_per_sample=0.5, window_size=512, require_startup_test=False)
            seq, need, other = [7], total_targets - 1, 1000
            while len(seq) < 512:
                run = min(5, need)
                seq += [7] * run
                need -= run
                if len(seq) < 512:
                    seq.append(other)
                    other += 1
            try:
                for v in seq[:512]:
                    m.validate_sample(v)
            except AdaptiveProportionFailure:
                return True
            return False

        if not window_fires(410) or window_fires(409):
            raise KnownAnswerSelfTestError("APT did not fire at exactly B = C = 410")

        monitor = NISTHealthMonitor(min_entropy_per_sample=1.0, window_size=64, require_startup_test=False)
        cutoff = monitor.apt_cutoff

        # Set target symbol
        monitor.validate_sample(777)

        # Feed interspersed samples such that target appears >= cutoff times without triggering RCT
        alarm_raised = False
        try:
            for i in range(15):
                for _ in range(5):
                    monitor.validate_sample(777)
                monitor.validate_sample(i + 1)
        except AdaptiveProportionFailure:
            alarm_raised = True

        if not alarm_raised:
            raise KnownAnswerSelfTestError(f"APT failed to trigger when target exceeded cutoff {cutoff}")

    @staticmethod
    def test_healthy_stream_kat() -> None:
        """Verifies that a deterministic well-distributed sequence completes startup and passes without false alarms."""
        monitor = NISTHealthMonitor(
            min_entropy_per_sample=1.0,
            window_size=64,
            require_startup_test=True,
            startup_samples=128,
        )
        for i in range(256):
            # LCG-like full-period permutation mod 256
            symbol = (i * 73 + 19) & 0xFF
            monitor.validate_sample(symbol)

        if not monitor.is_operational:
            raise KnownAnswerSelfTestError("Startup health test failed to transition to OPERATIONAL on healthy stream")

    @staticmethod
    def test_conditioning_kat() -> None:
        """Checks HMAC-SHA-256, HMAC-SHA-512 and BLAKE2b-512 against published RFC test vectors."""
        mac256 = hmac.new(HMAC_SHA512_RFC4231_TC1_KEY, HMAC_SHA512_RFC4231_TC1_DATA, hashlib.sha256).hexdigest()
        if mac256 != HMAC_SHA256_RFC4231_TC1_HEX:
            raise KnownAnswerSelfTestError(
                f"HMAC-SHA-256 mismatch (RFC 4231 TC1): expected {HMAC_SHA256_RFC4231_TC1_HEX}, got {mac256}"
            )
        mac = hmac.new(HMAC_SHA512_RFC4231_TC1_KEY, HMAC_SHA512_RFC4231_TC1_DATA, hashlib.sha512).hexdigest()
        if mac != HMAC_SHA512_RFC4231_TC1_HEX:
            raise KnownAnswerSelfTestError(
                f"HMAC-SHA-512 mismatch (RFC 4231 TC1): expected {HMAC_SHA512_RFC4231_TC1_HEX}, got {mac}"
            )
        b2 = hashlib.blake2b(b"abc").hexdigest()
        if b2 != BLAKE2B_512_RFC7693_ABC_HEX:
            raise KnownAnswerSelfTestError(
                f"BLAKE2b-512 mismatch (RFC 7693 App. A): expected {BLAKE2B_512_RFC7693_ABC_HEX}, got {b2}"
            )

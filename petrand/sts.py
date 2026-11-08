"""
petrand.sts
===========
Four of the fifteen tests in the NIST SP 800-22 Statistical Test Suite:
1. Frequency (Monobit) Test (§2.1)
2. Frequency Test within a Block (§2.2)
3. Runs Test (§2.3)
4. Cumulative Sums (Cusum) Test (§2.13)

Each test is run at alpha = 0.01, so ideal random input fails each one about 1% of the
time. Passing them on hashed (conditioned) output shows the plumbing works; it says
nothing about the entropy of the source. The block size follows the §2.2.7
recommendation (M >= 20, M > 0.01 n, N < 100), and the cumulative sums test runs in
both its forward and backward modes.
"""

import math
from typing import List, Dict, Any, Union
import numpy as np


class NISTStatisticalTestSuite:
    """
    Executes standard NIST SP 800-22 randomness tests on binary bitstreams.
    """

    @classmethod
    def evaluate_bits(cls, bits: Union[np.ndarray, List[int], bytes]) -> Dict[str, Any]:
        """
        Runs the NIST SP 800-22 battery on an array of 0s and 1s or raw bytes.
        """
        if isinstance(bits, bytes):
            bit_array = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
        elif isinstance(bits, list):
            bit_array = np.array(bits, dtype=np.uint8)
        else:
            bit_array = bits

        n = len(bit_array)
        if n < 1000:
            raise ValueError(f"At least 1,000 bits required for NIST STS evaluation (got {n})")

        res_monobit = cls.frequency_monobit_test(bit_array)
        res_block = cls.frequency_block_test(bit_array)
        res_runs = cls.runs_test(bit_array)
        res_cusum = cls.cumulative_sums_test(bit_array)
        res_cusum_back = cls.cumulative_sums_test(bit_array[::-1].copy())

        all_passed = (
            res_monobit["passed"]
            and res_block["passed"]
            and res_runs["passed"]
            and res_cusum["passed"]
            and res_cusum_back["passed"]
        )

        return {
            "total_bits": n,
            "monobit_test": res_monobit,
            "block_frequency_test": res_block,
            "runs_test": res_runs,
            "cumulative_sums_test": res_cusum,
            "cumulative_sums_backward_test": res_cusum_back,
            "all_passed": all_passed,
        }

    @staticmethod
    def frequency_monobit_test(bits: np.ndarray, alpha: float = 0.01) -> Dict[str, Any]:
        """
        NIST SP 800-22 §2.1: Frequency (Monobit) Test.
        Tests whether the number of 1s and 0s are approximately equal.
        """
        n = len(bits)
        s_n = int(np.sum(2 * bits.astype(np.int64) - 1))
        s_obs = abs(s_n) / math.sqrt(n)
        p_value = math.erfc(s_obs / math.sqrt(2.0))

        return {
            "p_value": round(p_value, 6),
            "s_obs": round(s_obs, 4),
            "passed": p_value >= alpha,
        }

    @staticmethod
    def frequency_block_test(bits: np.ndarray, block_size: int = 0, alpha: float = 0.01) -> Dict[str, Any]:
        """
        NIST SP 800-22 §2.2: Frequency Test within a Block.
        Tests whether the frequency of 1s in blocks of size M is approximately M/2.
        """
        n = len(bits)
        # §2.2.7: M >= 20, M > 0.01 n and N = n // M < 100; block_size=0 picks M that way
        m = block_size if block_size > 0 else max(20, n // 99 + 1)
        num_blocks = n // m

        if num_blocks < 1:
            raise ValueError(f"Bitstream length {n} too short for block size {m}")

        # Crop to multiple of block_size
        trimmed = bits[: num_blocks * m].reshape((num_blocks, m))
        proportions = np.mean(trimmed, axis=1)

        chi_sq = float(4.0 * m * np.sum((proportions - 0.5) ** 2))

        try:
            from scipy.special import gammaincc
            p_value = float(gammaincc(num_blocks / 2.0, chi_sq / 2.0))
        except ImportError:
            # Chi-square survival function approximation via normal transform
            z = math.sqrt(2.0 * chi_sq) - math.sqrt(2.0 * num_blocks - 1.0)
            p_value = 0.5 * math.erfc(z / math.sqrt(2.0))

        return {
            "p_value": round(p_value, 6),
            "chi_square": round(chi_sq, 4),
            "block_size": m,
            "num_blocks": num_blocks,
            "passed": p_value >= alpha,
        }

    @staticmethod
    def runs_test(bits: np.ndarray, alpha: float = 0.01) -> Dict[str, Any]:
        """
        NIST SP 800-22 §2.3: Runs Test.
        Tests whether the number of runs of 1s and 0s of various lengths is as expected.
        """
        n = len(bits)
        pi = float(np.mean(bits))

        # Prerequisite check from NIST §2.3.4
        if abs(pi - 0.5) >= (2.0 / math.sqrt(n)):
            return {
                "p_value": 0.0,
                "v_obs": 0,
                "passed": False,
                "note": "Failed prerequisite frequency threshold",
            }

        # Calculate total number of runs V_n
        diffs = bits[:-1] != bits[1:]
        v_obs = int(np.sum(diffs)) + 1

        numerator = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
        denominator = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)

        p_value = math.erfc(numerator / max(1e-12, denominator))

        return {
            "p_value": round(p_value, 6),
            "v_obs": v_obs,
            "passed": p_value >= alpha,
        }

    @staticmethod
    def cumulative_sums_test(bits: np.ndarray, alpha: float = 0.01) -> Dict[str, Any]:
        """
        NIST SP 800-22 §2.13: Cumulative Sums (Cusum) Test.
        Tests whether the cumulative sum of partial sequences indicates random walk drift.
        """
        n = len(bits)
        x = 2 * bits.astype(np.int64) - 1
        s = np.cumsum(x)
        z = int(np.max(np.abs(s)))

        # Standard normal complementary expansion from NIST §2.13.4
        def _phi(t):
            return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

        term1 = 0.0
        k_start = int(math.floor((-n / z + 1) / 4.0))
        k_end = int(math.floor((n / z - 1) / 4.0))
        for k in range(k_start, k_end + 1):
            term1 += _phi((4 * k + 1) * z / math.sqrt(n)) - _phi((4 * k - 1) * z / math.sqrt(n))

        term2 = 0.0
        k_start2 = int(math.floor((-n / z - 3) / 4.0))
        k_end2 = int(math.floor((n / z - 1) / 4.0))
        for k in range(k_start2, k_end2 + 1):
            term2 += _phi((4 * k + 3) * z / math.sqrt(n)) - _phi((4 * k + 1) * z / math.sqrt(n))

        p_value = 1.0 - term1 + term2
        p_value = min(1.0, max(0.0, p_value))

        return {
            "p_value": round(p_value, 6),
            "max_excursion_z": z,
            "passed": p_value >= alpha,
        }

"""
petrand.conditioning_bound
==========================
Output-entropy bound for a vetted conditioning function, using the SP 800-90B
§3.1.5.1.2 formula in log2 space, with n = min(n_out, n_w) and n_out = n_w = the
conditioner's full output length (256 for PetRand's 32-byte seeds, which are complete
HMAC-SHA-256 outputs; 512 for 64-byte seeds from HMAC-SHA-512).

Full-entropy rule: FIPS 140-3 IG D.K Resolution 19 and SP 800-90C §3.2.2.2 use
h_in >= n_out + 64 bits with a vetted conditioner (320 bits for a 256-bit output);
the IG adds that n_out must not exceed the conditioner's security strength. SP
800-90B itself does not state this rule.

The bound applies only when H_in comes from an entropy assessment of the raw samples.
PetRand also does not meet SP 800-90B's other conditioner requirements: its pool has
a variable length, includes event headers as well as noise-source bytes, and the
HMAC code has no CAVP validation. With an unassessed H_in the result is a what-if.
"""

import math
from typing import Dict, Any


def _log2_add(log2_a: float, log2_b: float) -> float:
    """Numerically stable computation of log2(2^log2_a + 2^log2_b) without float64 underflow."""
    hi = max(log2_a, log2_b)
    lo = min(log2_a, log2_b)
    diff = hi - lo
    if diff > 64.0:
        return hi
    return hi + math.log1p(2.0 ** (-diff)) / math.log(2.0)


class NISTConditioningAssessor:
    """
    SP 800-90B §3.1.5.1.2 output-entropy formula for vetted conditioning functions,
    computed in log2 space to avoid float64 underflow.
    """

    @staticmethod
    def assess_entropy(
        h_in_bits: float,
        n_in_bytes: int,
        n_out_bits: int = 256,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        h_in_bits : float
            Assessed min-entropy of the input physical event buffer (H_in).
        n_in_bytes : int
            Total byte length of the input data buffer (n_in = n_in_bytes * 8 bits).
        n_out_bits : int
            Full output length of the conditioner (256 for HMAC-SHA-256, 512 for HMAC-SHA-512).

        Returns
        -------
        Dict with vetted output min-entropy H_out and full-entropy qualification status.
        """
        n_in_bits = max(0, int(n_in_bytes) * 8)
        n_out = float(n_out_bits)
        # Input min-entropy cannot exceed total input bits
        h_in = max(0.0, min(float(h_in_bits), float(n_in_bits)))

        # Full-entropy rule of FIPS 140-3 IG D.K Res. 19 / SP 800-90C §3.2.2.2: h_in >= n_out + 64
        oversampling_requirement = n_out + 64.0
        full_entropy_qualified = (h_in >= oversampling_requirement) and (n_in_bits >= oversampling_requirement)

        if h_in <= 0.0 or n_in_bits <= 0:
            h_out = 0.0
            assessment_class = "ZERO_ENTROPY"
        elif full_entropy_qualified:
            h_out = n_out
            assessment_class = "FULL_ENTROPY if H_in is an assessed value (h_in >= n_out + 64)"
        else:
            # Exact NIST SP 800-90B §3.1.5.1.2 equations in log2-space:
            # psi   = 2^(n_in - n) * P_low + P_high
            # omega = U * P_low, where U = 2^(n_in - n) + sqrt(2 * n * 2^(n_in - n) * ln(2))
            log2_one_minus_phigh = (
                math.log1p(-(2.0 ** (-h_in))) / math.log(2.0) if h_in < 60.0 else 0.0
            )
            log2_denom = (
                math.log1p(-(2.0 ** (-n_in_bits))) / math.log(2.0) if n_in_bits < 60 else 0.0
            )
            # log2(P_low) = log2(1 - 2^-h_in) - n_in - log2(1 - 2^-n_in)
            log2_p_low = log2_one_minus_phigh - float(n_in_bits) - log2_denom

            # log2(psi) = log2(2^(n_in - n_out) * P_low + 2^-h_in)
            log2_psi = _log2_add((float(n_in_bits) - n_out) + log2_p_low, -h_in)

            # log2(omega) = log2(U) + log2(P_low)
            log2_u_term1 = float(n_in_bits) - n_out
            log2_u_term2 = 0.5 * math.log2(2.0 * n_out * math.log(2.0)) + 0.5 * (float(n_in_bits) - n_out)
            log2_omega = _log2_add(log2_u_term1, log2_u_term2) + log2_p_low

            h_out = min(n_out, max(0.0, -max(log2_psi, log2_omega)))
            assessment_class = "PARTIAL_ENTROPY"

        return {
            "input_entropy_H_in_bits": round(float(h_in_bits), 2),
            "effective_input_entropy_bits": round(h_in, 2),
            "input_buffer_bytes": n_in_bytes,
            "output_length_bits": n_out_bits,
            "nist_full_entropy_threshold_bits": oversampling_requirement,
            "full_entropy_qualified": full_entropy_qualified,
            "vetted_output_min_entropy_H_out": round(h_out, 2),
            "assessment_classification": assessment_class,
            "conditions": "valid only for a vetted conditioner (HMAC-SHA-256 here) and an assessed H_in; "
                          "PetRand's variable-length pool and uncertified HMAC code fall short of SP 800-90B",
        }

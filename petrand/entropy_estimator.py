"""
petrand.entropy_estimator
=========================
Screening estimates of min-entropy per raw sample:
1. Most Common Value (MCV), as in NIST SP 800-90B §6.3.1
2. A simplified birthday-bound collision screen
3. A simplified first-order Markov screen on up to 64 states

SP 800-90B applies its own collision, Markov and compression estimates only to binary
data; screens 2 and 3 are not those procedures. This is NOT an SP 800-90B assessment,
which needs at least 1,000,000 samples, the full non-IID estimator set (run on the
samples and, for non-binary data, on their bitstring), restart testing, and the IID
tests only if an IID claim is made. Export raw samples with `petrand export-nist` and
run NIST's own SP800-90B_EntropyAssessment tool.

No statistical estimator can tell a good pseudorandom sequence from a physical one:
a SHA-256 counter, which has zero entropy to anyone who knows the counter, screens
at several bits per sample here. Entropy credit has to rest on a model of the source
and on experiments that vary the source (see paper/RESEARCH_PLAN.md).
"""

import math
from collections import Counter
from typing import List, Dict, Any
import numpy as np


class MostCommonValueEstimator:
    """
    NIST SP 800-90B §6.3.1: Most Common Value Estimate.
    Computes upper bound probability of the most frequent symbol with 99% confidence.
    """
    @staticmethod
    def estimate(samples: List[int], z_alpha: float = 2.576) -> float:
        n = len(samples)
        if n < 64:
            raise ValueError("At least 64 samples required for MCV estimation")

        counts = Counter(samples)
        c_max = max(counts.values())
        p_hat = c_max / float(n)

        # Upper bound calculation: p_u = min(1, p_hat + z_alpha * sqrt(p_hat * (1 - p_hat) / (n - 1)))
        sigma = math.sqrt(p_hat * (1.0 - p_hat) / max(1, n - 1))
        p_upper = min(1.0, p_hat + z_alpha * sigma)

        if p_upper <= 0.0 or p_upper >= 1.0:
            return 0.0

        return -math.log2(p_upper)


class CollisionEstimator:
    """
    Birthday-bound collision screen. Uses the mean waiting time to the first repeat,
    with a 99% lower confidence bound, and solves E[T] ~ sqrt(pi / (2p)) for the
    collision probability p. That measures collision entropy, which is never below
    min-entropy, so this screen can only overstate it; the MCV and Markov screens
    usually bind. Large alphabets are reduced to their low 8 bits. This is not the
    SP 800-90B §6.3.2 procedure.
    """
    @staticmethod
    def estimate(
        samples: List[int],
        max_alphabet_bits: int = 8,
        z_alpha: float = 2.576,
    ) -> float:
        n = len(samples)
        if n < 128:
            raise ValueError("At least 128 samples required for Collision estimation")

        # Reduce large symbol alphabets to their low 8 bits (cf. SP 800-90B §6.4)
        unique_count = len(set(samples))
        max_symbols = 1 << max_alphabet_bits
        if unique_count > max_symbols:
            eval_samples = [s & (max_symbols - 1) for s in samples]
        else:
            eval_samples = samples

        # Compute collision intervals
        seen = set()
        intervals = []
        last_idx = 0

        for i, val in enumerate(eval_samples):
            if val in seen:
                intervals.append(i - last_idx + 1)
                seen.clear()
                last_idx = i + 1
            else:
                seen.add(val)

        if not intervals:
            # No collisions occurred across all samples -> waiting time is at least n
            x_bar_lower = float(n)
        else:
            v = len(intervals)
            mean_interval = float(np.mean(intervals))
            if v >= 2:
                s_hat = float(np.std(intervals, ddof=1))
                # Lower 99% confidence bound on the mean waiting time (cf. SP 800-90B §6.3.2, step 6)
                x_bar_lower = mean_interval - z_alpha * (s_hat / math.sqrt(v))
                # Guard against high variance on small collision counts collapsing below 1.0
                x_bar_lower = max(min(mean_interval, 1.5), x_bar_lower)
            else:
                x_bar_lower = mean_interval

        if x_bar_lower <= 1.0:
            return 0.0

        # Estimate collision probability from waiting time: E[T] ~ sqrt(pi / (2*p))
        p_est = math.pi / (2.0 * (x_bar_lower ** 2))
        p_est = min(1.0, max(1e-12, p_est))
        return min(float(max_alphabet_bits), -math.log2(p_est))


class MarkovEstimator:
    """
    First-order Markov screen. Reduces symbols to their low 6 bits (at most 64 states),
    estimates transition probabilities by maximum likelihood, and finds the most likely
    128-step path by dynamic programming. Catches lag-1 dependence only; it does not
    reproduce the SP 800-90B §6.3.3 procedure exactly.
    """
    @staticmethod
    def estimate(
        samples: List[int],
        max_alphabet_bits: int = 6,
        chain_depth: int = 128,
        min_state_occurrences: int = 3,
    ) -> float:
        n = len(samples)
        if n < 128:
            raise ValueError("At least 128 samples required for Markov estimation")

        unique_vals = set(samples)
        # Screening choice: at most 64 states (low 6 bits)
        if len(unique_vals) > (1 << max_alphabet_bits):
            k = 1 << max_alphabet_bits
            eval_samples = [s & (k - 1) for s in samples]
        else:
            eval_samples = samples

        states = sorted(set(eval_samples))
        if len(states) <= 1:
            return 0.0

        # 1. Initial state distribution P_0(s) with a 99% upper confidence bound
        state_counts = Counter(eval_samples)
        z_alpha = 2.576
        log2_p0: Dict[int, float] = {}
        for s in states:
            p_s = state_counts[s] / float(n)
            sigma_s = math.sqrt(p_s * (1.0 - p_s) / max(1, n - 1))
            p_s_upper = min(1.0, p_s + z_alpha * sigma_s)
            log2_p0[s] = math.log2(max(1e-15, p_s_upper))

        # 2. Transition matrix T[s_curr][s_next]
        transitions: Dict[int, Counter] = {s: Counter() for s in states}
        for i in range(n - 1):
            transitions[eval_samples[i]][eval_samples[i + 1]] += 1

        # Compute log2 transition probabilities for supported states
        log2_trans: Dict[int, Dict[int, float]] = {}
        fallback_log2 = math.log2(1.0 / len(states))
        for s_curr in states:
            total_out = sum(transitions[s_curr].values())
            log2_trans[s_curr] = {}
            if total_out < min_state_occurrences:
                # Insufficient support to estimate outgoing transitions from s_curr;
                # use empirical initial marginal probabilities
                for s_next in states:
                    log2_trans[s_curr][s_next] = log2_p0.get(s_next, fallback_log2)
            else:
                for s_next, count in transitions[s_curr].items():
                    if count > 0:
                        p_ij = count / float(total_out)
                        log2_trans[s_curr][s_next] = math.log2(p_ij)

        # 3. Most likely 128-step path by dynamic programming
        # Find maximum probability path of length d: P_max = max_{s_1..s_d} P_0(s_1) * prod T(s_t -> s_{t+1})
        dp_log2: Dict[int, float] = dict(log2_p0)
        for _ in range(chain_depth - 1):
            next_dp: Dict[int, float] = {s: -1e9 for s in states}
            for s_prev, prev_log_p in dp_log2.items():
                if prev_log_p <= -1e8:
                    continue
                for s_curr, trans_log_p in log2_trans[s_prev].items():
                    cand = prev_log_p + trans_log_p
                    if cand > next_dp[s_curr]:
                        next_dp[s_curr] = cand
            dp_log2 = next_dp

        max_path_log2 = max(dp_log2.values())
        if max_path_log2 >= 0.0:
            return 0.0

        # Per-sample min-entropy: H = - (1 / d) * log2(P_max)
        h_markov = -max_path_log2 / float(chain_depth)
        return max(0.0, min(float(max_alphabet_bits), h_markov))


class NISTEntropyAssessor:
    """
    Screening summary: the minimum of the three screens above. Use it to spot
    problems early, never as an entropy credit on its own.
    """

    @classmethod
    def evaluate(cls, raw_symbols: List[int]) -> Dict[str, Any]:
        if len(raw_symbols) < 128:
            raise ValueError(f"Insufficient samples for evaluation: got {len(raw_symbols)}, require >= 128")

        h_mcv = MostCommonValueEstimator.estimate(raw_symbols)
        h_col = CollisionEstimator.estimate(raw_symbols)
        h_mar = MarkovEstimator.estimate(raw_symbols)

        # NIST SP 800-90B takes the minimum across all estimators
        h_inf = min(h_mcv, h_col, h_mar)

        return {
            "total_samples": len(raw_symbols),
            "unique_symbols": len(set(raw_symbols)),
            "mcv_entropy_bits": round(h_mcv, 3),
            "collision_entropy_bits": round(h_col, 3),
            "markov_entropy_bits": round(h_mar, 3),
            "conservative_min_entropy_H_inf": round(h_inf, 3),
            "assessment_verdict": "HIGH" if h_inf >= 1.0 else ("MEDIUM" if h_inf >= 0.3 else "LOW"),
            "scope": "screening: the SP 800-90B MCV estimate plus simplified collision and Markov screens; not an SP 800-90B assessment",
            "samples_needed_for_assessment": 1_000_000,
        }

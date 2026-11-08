"""
petrand.benchmarks
==================
Monte Carlo and quasi-Monte Carlo checks of the generators that PetRand seeds feed:
1. Coverage of nominal 95% intervals for E[X^2] = 1/3, X ~ Uniform(0, 1)
2. Volume of the 10-dimensional unit ball: plain Monte Carlo vs. scrambled Sobol (RQMC), replicated
3. Lag-k autocorrelation, Bonferroni-corrected over lags 1..K
4. Student-t(3) tail frequency at the 0.1% quantile

What these can and cannot show
------------------------------
They check the plumbing downstream of the seed (SeedSequence spawning, PCG64, scrambled
Sobol). They cannot detect a weak seed: a constant all-zero seed passes all four, which
`run_constant_seed_control` demonstrates. Seed entropy has to be established at the
source (see paper/RESEARCH_PLAN.md). Any RQMC gain comes from the Sobol construction,
not from where the seed came from.
"""

import math
import hashlib
from typing import Dict, Any, List
import numpy as np
from petrand.adapters import MonteCarloSeedFactory


def run_coverage_test(
    factory: MonteCarloSeedFactory,
    num_trials: int = 100,
    n_samples: int = 25_000,
    alpha_test: float = 0.001,
) -> Dict[str, Any]:
    """
    Empirical coverage of nominal 95% intervals for E[X^2] = 1/3, X ~ Uniform(0, 1).
    Passes unless an exact two-sided binomial test rejects 95% coverage at alpha_test.
    Also reports a 95% Clopper-Pearson interval for the coverage rate.
    """
    from scipy.stats import binomtest

    true_val = 1.0 / 3.0
    hits = 0
    estimates: List[float] = []

    # Spawn independent generators from a single seed block
    generators = factory.spawn_numpy_generators(n_workers=num_trials, bit_generator="pcg64")

    for rng in generators:
        x = rng.random(n_samples, dtype=np.float64)
        y = x * x
        mean_est = float(np.mean(y))
        estimates.append(mean_est)
        se = math.sqrt(float(np.var(y, ddof=1)) / n_samples)
        if mean_est - 1.96 * se <= true_val <= mean_est + 1.96 * se:
            hits += 1

    bt = binomtest(hits, num_trials, 0.95)
    ci = bt.proportion_ci(confidence_level=0.95, method="exact")
    return {
        "trials": num_trials,
        "samples_per_trial": n_samples,
        "true_value": true_val,
        "empirical_mean": float(np.mean(estimates)),
        "coverage_rate": hits / float(num_trials),
        "coverage_ci95": [round(float(ci.low), 4), round(float(ci.high), 4)],
        "nominal_coverage": 0.95,
        "binomial_test_p_value": float(bt.pvalue),
        "passed": bool(bt.pvalue >= alpha_test),
    }


def _unit_ball_fraction(u: np.ndarray) -> float:
    x = 2.0 * u - 1.0
    return float(np.mean(np.sum(x * x, axis=1) <= 1.0))


def _generator_from(seed_bytes: bytes) -> np.random.Generator:
    words = np.frombuffer(seed_bytes, dtype=np.uint32).tolist()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def run_hypersphere_10d_benchmark(
    factory: MonteCarloSeedFactory,
    n_samples: int = 1_048_576,
    replicates: int = 16,
) -> Dict[str, Any]:
    """
    Plain Monte Carlo (PCG64) vs. scrambled Sobol (RQMC) for the volume of the
    10-dimensional unit ball, V_10 = pi^5 / 120 = 2.5501640...

    Each method is replicated with independent sub-seeds and compared by root-mean-square
    relative error. One run of each is too noisy to rank them: at n = 32,768 single-run
    error ratios range from about 0.1x to 11x. The integrand is an indicator function, so
    RQMC's gain here is modest and its error shrinks at close to the plain Monte Carlo rate.
    """
    from scipy.stats import qmc

    d = 10
    true_volume = (math.pi ** 5) / 120.0
    box_volume = 2.0 ** d  # [-1, 1]^10 bounding box
    m = max(1, int(round(math.log2(n_samples))))
    n = 2 ** m

    master_seed = factory.extract_seed(seed_bytes=64)
    mc_vals: List[float] = []
    q_vals: List[float] = []
    for r in range(max(1, int(replicates))):
        tag = r.to_bytes(4, "little")
        mc_rng = _generator_from(hashlib.blake2b(b"subseed-plain-mc" + tag + master_seed, digest_size=32).digest())
        q_rng = _generator_from(hashlib.blake2b(b"subseed-rqmc" + tag + master_seed, digest_size=32).digest())
        mc_vals.append(_unit_ball_fraction(mc_rng.random((n, d), dtype=np.float64)) * box_volume)
        sampler = qmc.Sobol(d=d, scramble=True, seed=q_rng)
        q_vals.append(_unit_ball_fraction(sampler.random_base2(m=m)) * box_volume)

    mc = np.array(mc_vals)
    q = np.array(q_vals)
    mc_rmse = float(np.sqrt(np.mean(((mc - true_volume) / true_volume) ** 2)))
    q_rmse = float(np.sqrt(np.mean(((q - true_volume) / true_volume) ** 2)))

    def summary(vals: np.ndarray, rmse: float) -> Dict[str, float]:
        mean = float(vals.mean())
        return {
            "volume_estimate": mean,
            "absolute_error": abs(mean - true_volume),
            "relative_error": rmse,
            "rmse_relative": rmse,
        }

    return {
        "dimension": d,
        "samples": n,
        "replicates": len(mc_vals),
        "true_volume": true_volume,
        "plain_mc": summary(mc, mc_rmse),
        "rqmc_sobol": summary(q, q_rmse),
        "rqmc_advantage_factor": mc_rmse / max(q_rmse, 1e-12),
        "note": "advantage = ratio of RMS relative errors over replicates; it reflects "
                "the Sobol construction, not the seed source",
    }


def run_autocorrelation_test(
    factory: MonteCarloSeedFactory,
    n_samples: int = 100_000,
    max_lag: int = 20,
) -> Dict[str, Any]:
    """
    Computes sample autocorrelation rho_k up to max_lag for one PCG64 stream.
    Applies Bonferroni multi-comparison correction over max_lag hypotheses.
    Under the null hypothesis of i.i.d. randomness, |rho_k| <= z_bonferroni / sqrt(N).
    """
    rng = factory.spawn_numpy_generators(n_workers=1, bit_generator="pcg64")[0]
    x = rng.random(n_samples, dtype=np.float64)

    mean_x = np.mean(x)
    var_x = np.var(x)
    centered = x - mean_x

    # Bonferroni-adjusted z for family-wise alpha = 0.05 across max_lag tests
    from scipy.stats import norm
    alpha_adj = 0.05 / float(max_lag)
    z_bonferroni = float(norm.ppf(1.0 - alpha_adj / 2.0))
    bonferroni_threshold = z_bonferroni / math.sqrt(n_samples)
    threshold_95_single = 1.96 / math.sqrt(n_samples)

    lags_data = {}
    for k in range(1, max_lag + 1):
        cov = np.mean(centered[:-k] * centered[k:])
        rho_k = float(cov / var_x)
        lags_data[f"lag_{k:02d}"] = round(rho_k, 5)

    max_abs_corr = max(abs(v) for v in lags_data.values())
    passed = max_abs_corr <= bonferroni_threshold

    return {
        "samples": n_samples,
        "max_lag": max_lag,
        "single_test_95_bound": round(threshold_95_single, 5),
        "bonferroni_bound_95": round(bonferroni_threshold, 5),
        "lags": lags_data,
        "max_abs_autocorr": round(max_abs_corr, 5),
        "passed": passed,
    }


def run_tail_risk_benchmark(
    factory: MonteCarloSeedFactory,
    n_samples: int = 500_000,
    df: float = 3.0,
    tail_prob: float = 0.001, # 99.9% Value-at-Risk
) -> Dict[str, Any]:
    """
    Tail frequency of Student-t(3) draws (inverse-CDF from PCG64 uniforms) at the
    tail_prob quantile. Checks the sampling path, not the seed.
    """
    from scipy.stats import t

    rng = factory.spawn_numpy_generators(n_workers=1, bit_generator="pcg64")[0]
    u = rng.random(n_samples, dtype=np.float64)
    # Inverse transform sampling to Student-t(3)
    samples_t = t.ppf(u, df=df)

    theoretical_var_999 = float(t.ppf(tail_prob, df=df)) # e.g. ~ -10.21
    empirical_tail_count = int(np.sum(samples_t <= theoretical_var_999))
    empirical_tail_prob = empirical_tail_count / float(n_samples)
    rel_error = abs(empirical_tail_prob - tail_prob) / tail_prob

    # Binomial standard error for rare event sampling: SE = sqrt(p*(1-p)/N)
    std_err = math.sqrt(tail_prob * (1.0 - tail_prob) / float(n_samples))
    z_score = abs(empirical_tail_prob - tail_prob) / std_err if std_err > 0 else 0.0
    passed = bool(z_score <= 3.5) # Passed if within 3.5 sigma

    return {
        "distribution": f"Student-t(df={df:.1f})",
        "sample_size": n_samples,
        "nominal_tail_probability": tail_prob,
        "theoretical_VaR_999": round(theoretical_var_999, 4),
        "empirical_tail_count": empirical_tail_count,
        "empirical_tail_probability": round(empirical_tail_prob, 6),
        "relative_error": round(rel_error, 4),
        "z_score": round(z_score, 3),
        "passed": passed,
    }


class ConstantSeedEngine:
    """Stand-in engine that always returns an all-zero seed (negative control)."""

    is_ready = True

    def extract_seed(self, seed_bytes: int = 32, consume: bool = True) -> bytes:
        return b"\x00" * seed_bytes

    def ingest_event(self, event: Any) -> None:
        pass


def run_constant_seed_control(
    trials: int = 100,
    samples: int = 25_000,
    qmc_samples: int = 65_536,
    replicates: int = 8,
) -> Dict[str, Any]:
    """
    Runs the four benchmarks with an all-zero seed. They pass, which shows that they
    test the downstream generators, not the entropy of the seed.
    """
    factory = MonteCarloSeedFactory(ConstantSeedEngine())
    cov = run_coverage_test(factory, num_trials=trials, n_samples=samples)
    hyp = run_hypersphere_10d_benchmark(factory, n_samples=qmc_samples, replicates=replicates)
    ac = run_autocorrelation_test(factory, n_samples=50_000, max_lag=20)
    tr = run_tail_risk_benchmark(factory, n_samples=100_000, df=3.0, tail_prob=0.001)
    return {
        "seed": "all-zero",
        "coverage_rate": cov["coverage_rate"],
        "coverage_passed": cov["passed"],
        "rqmc_advantage_factor": round(hyp["rqmc_advantage_factor"], 2),
        "autocorrelation_passed": bool(ac["passed"]),
        "tail_risk_passed": bool(tr["passed"]),
    }

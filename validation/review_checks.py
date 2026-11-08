#!/usr/bin/env python3
"""Review checks behind the numbers quoted in BLUEPRINT.md (section 7) and in the
2026-11-07/08 review: health-test cutoffs, credit versus a degraded source, the
all-zero-seed benchmark control, replicated plain MC vs. scrambled Sobol, coverage,
the screening estimators on known sequences, and the simulator's dependence on
the host clock (it should not depend on it: section 8).

Usage: python3 -I validation/review_checks.py <path-to-petrand-project> <out.json>
All random draws use fixed seeds, so sections 5-7 reproduce exactly.
"""
import sys, math, hashlib, json, itertools, time as _time

SRC, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, SRC)

import numpy as np
from scipy.stats import binom, qmc

R = {}

# ---------------------------------------------------------------- 1. self-test KAT
import petrand.self_test as _st
from petrand.self_test import SelfTestEngine
kat = hashlib.blake2b(b"The quick brown fox jumps over the lazy dog",
                      key=b"PetRand-Test-Key", digest_size=32).hexdigest()
R["1_kat"] = {
    "stored_digest_matches_hashlib": kat == getattr(_st, "BLAKE2B_256_KAT_EXPECTED_HEX", None),
    "self_test_status": SelfTestEngine.run_all_self_tests()["status"],
    "note": "keyed vector with a project key; RFC 7693 Appendix A only lists unkeyed BLAKE2b-512('abc')",
}

# ---------------------------------------------------------------- 2. health-test cutoffs
from petrand.health import NISTHealthMonitor, AdaptiveProportionFailure, HealthError
W, H, alpha = 512, 0.5, 2.0 ** -20
mon = NISTHealthMonitor(min_entropy_per_sample=H, window_size=W, require_startup_test=False)
crit = int(binom.ppf(1 - alpha, W, 2.0 ** -H))      # CRITBINOM(W, 2^-H, 1-alpha)
c_nist = 1 + crit


def apt_alarms(total_target_count):
    """First sample is the target; the window's other W-1 samples hold
    (total_target_count - 1) more targets, in runs of at most 5 (RCT cutoff is 41)."""
    m = NISTHealthMonitor(min_entropy_per_sample=H, window_size=W, require_startup_test=False)
    seq, need, other = [7], total_target_count - 1, 1000
    while len(seq) < W:
        run = min(5, need)
        seq += [7] * run
        need -= run
        if len(seq) < W:
            seq.append(other)
            other += 1
    seq = seq[:W]
    assert seq.count(7) == total_target_count, (seq.count(7), total_target_count)
    try:
        for s in seq:
            m.validate_sample(s)
    except AdaptiveProportionFailure:
        return True
    return False


R["2_health_cutoffs"] = {
    "rct_cutoff_code": mon.rct_cutoff, "rct_cutoff_nist": 1 + math.ceil(20 / H),
    "apt_cutoff_code": mon.apt_cutoff, "apt_cutoff_nist_formula": c_nist,
    "alarm_when_window_holds_C_targets_counting_first": apt_alarms(c_nist),
    "alarm_when_window_holds_C_plus_1": apt_alarms(c_nist + 1),
}

# ---------------------------------------------------------------- 3. degraded source vs. credit
from petrand.engine import PetRandEngine, RawPhysicalEvent
rng = np.random.default_rng(20261107)
P0 = 0.66                     # source min-entropy = -log2(0.66) = 0.60 bits/sample
reps, alarms, credited = 300, 0, []
for _ in range(reps):
    eng = PetRandEngine(required_entropy_bits=256.0, min_entropy_per_sample=0.5,
                        require_startup_test=True)
    i = 0
    try:
        while not eng.is_ready:
            i += 1
            sym = 0 if rng.random() < P0 else int(rng.integers(1, 65536))
            eng.ingest_event(RawPhysicalEvent(event_id=i, timestamp_ns=i, channel="sim_motion",
                                              raw_symbol=sym, sensor_payload=b"",
                                              estimated_entropy_bits=2.0))
        credited.append(i - 1024)
    except HealthError:
        alarms += 1
h_src = -math.log2(P0)
R["3_degraded_source"] = {
    "source_min_entropy_bits_per_sample": round(h_src, 3),
    "credit_per_event_bits (simulator constant)": 2.0,
    "replicates": reps, "health_alarms": alarms,
    "credited_events_per_seed": sorted(set(credited)),
    "credited_bits_per_seed": 256.0,
    "actual_min_entropy_bits_in_pool": round(min(credited) * h_src, 1) if credited else None,
}

# ---------------------------------------------------------------- 4. benchmarks with a constant seed
from petrand.adapters import MonteCarloSeedFactory
from petrand import benchmarks as B
from petrand.sensors import SyntheticArenaSimulator


class ConstantEngine:
    is_ready = True

    def extract_seed(self, seed_bytes=32, consume=True):
        return b"\x00" * seed_bytes

    def ingest_event(self, ev):
        pass


def run_all(factory):
    cov = B.run_coverage_test(factory, num_trials=100, n_samples=25_000)
    hyp = B.run_hypersphere_10d_benchmark(factory, n_samples=32_768)
    acf = B.run_autocorrelation_test(factory, n_samples=100_000, max_lag=20)
    tail = B.run_tail_risk_benchmark(factory, n_samples=500_000)
    return {
        "coverage": {"rate": cov["coverage_rate"], "passed": cov["passed"]},
        "hypersphere": {"mc_rel_err": round(hyp["plain_mc"]["relative_error"], 5),
                        "rqmc_rel_err": round(hyp["rqmc_sobol"]["relative_error"], 5),
                        "advantage": round(hyp["rqmc_advantage_factor"], 2)},
        "autocorr": {"max_abs": acf["max_abs_autocorr"], "bound": acf["bonferroni_bound_95"],
                     "passed": acf["passed"]},
        "tail": {"rate": tail["empirical_tail_probability"], "z": tail["z_score"],
                 "passed": tail["passed"]},
    }


sim_engine = PetRandEngine(required_entropy_bits=256.0)
R["4_benchmarks"] = {
    "all_zero_seed": run_all(MonteCarloSeedFactory(ConstantEngine())),
    "simulator_seed": run_all(MonteCarloSeedFactory(sim_engine, sensor=SyntheticArenaSimulator())),
}

# ---------------------------------------------------------------- 5. hypersphere: one run vs. many
V10 = math.pi ** 5 / 120.0


def one_pair(i, n):
    rg = np.random.Generator(np.random.PCG64(np.random.SeedSequence([i, 1])))
    x = 2.0 * rg.random((n, 10)) - 1.0
    mc = np.mean(np.sum(x * x, 1) <= 1.0) * 1024.0
    s = qmc.Sobol(d=10, scramble=True, seed=np.random.Generator(np.random.PCG64(np.random.SeedSequence([i, 2]))))
    y = 2.0 * s.random_base2(m=int(round(math.log2(n)))) - 1.0
    q = np.mean(np.sum(y * y, 1) <= 1.0) * 1024.0
    return (mc - V10) / V10, (q - V10) / V10


reps = 200
pairs = np.array([one_pair(i, 32_768) for i in range(reps)])
mc_rmse = float(np.sqrt(np.mean(pairs[:, 0] ** 2)))
q_rmse = float(np.sqrt(np.mean(pairs[:, 1] ** 2)))
ratio = np.abs(pairs[:, 0]) / np.maximum(np.abs(pairs[:, 1]), 1e-12)
slopes = {}
ns = [2 ** k for k in range(10, 18)]
rm = {"mc": [], "rqmc": []}
for n in ns:
    pr = np.array([one_pair(1000 + i, n) for i in range(40)])
    rm["mc"].append(float(np.sqrt(np.mean(pr[:, 0] ** 2))))
    rm["rqmc"].append(float(np.sqrt(np.mean(pr[:, 1] ** 2))))
for k in rm:
    slopes[k] = round(float(np.polyfit(np.log(ns), np.log(rm[k]), 1)[0]), 3)
R["5_hypersphere"] = {
    "n": 32_768, "replicates": reps,
    "mc_rmse_rel": round(mc_rmse, 5), "rqmc_rmse_rel": round(q_rmse, 5),
    "rmse_ratio_mc_over_rqmc": round(mc_rmse / q_rmse, 2),
    "single_run_ratio_p05_p50_p95": [round(float(np.percentile(ratio, p)), 2) for p in (5, 50, 95)],
    "share_runs_rqmc_worse_than_mc": round(float(np.mean(ratio < 1.0)), 3),
    "rmse_slope_vs_n (2^10..2^17, 40 reps)": slopes,
    "claimed": "8.6x lower error from one run each; 'near O(N^-1)'",
}

# ---------------------------------------------------------------- 6. coverage claim arithmetic
cov1000 = B.run_coverage_test(MonteCarloSeedFactory(ConstantEngine()), num_trials=1000, n_samples=25_000)
R["6_coverage"] = {
    "observed_rate_1000_trials_zero_seed": cov1000["coverage_rate"],
    "monte_carlo_se_at_1000": round(math.sqrt(0.95 * 0.05 / 1000), 4),
    "p_value_of_>=96.7%_if_true_95%": float(binom.sf(966, 1000, 0.95)),
    "pass_rule_in_code": "|coverage - 0.95| <= 0.06 (any rate from 0.89 to 1.00 passes)",
}

# ---------------------------------------------------------------- 7. entropy estimator on known sequences
from petrand.entropy_estimator import NISTEntropyAssessor

N = 100_000
r7 = np.random.default_rng(7)
block = r7.integers(0, 65536, 4096)
periodic = np.tile(block, N // 4096 + 1)[:N].astype(int).tolist()
ctr = [int.from_bytes(hashlib.sha256(i.to_bytes(8, "little")).digest()[:2], "little") for i in range(N)]
biased = [0 if r7.random() < P0 else int(r7.integers(1, 65536)) for _ in range(N)]
sim = SyntheticArenaSimulator()
sim_syms = []
while len(sim_syms) < 20_000:
    ev = sim.poll()
    if ev is not None:
        sim_syms.append(ev.raw_symbol)


def h(seq):
    r = NISTEntropyAssessor.evaluate(seq)
    return {k: r[k] for k in ("mcv_entropy_bits", "collision_entropy_bits",
                              "markov_entropy_bits", "conservative_min_entropy_H_inf")}


R["7_estimator"] = {
    "periodic_4096_block_repeated (predictable after 4096 samples)": h(periodic),
    "sha256_counter (fully deterministic)": h(ctr),
    "biased_p0_0.66 (true 0.60 bits)": h(biased),
    "simulator_raw_symbols_20000": h(sim_syms),
}

# ---------------------------------------------------------------- 8. simulator under a frozen clock


def sim_seed_with_clock(start):
    counter = itertools.count(start, 16_666_667)
    orig = _time.time_ns
    _time.time_ns = lambda: next(counter)
    try:
        eng = PetRandEngine(required_entropy_bits=256.0)
        s = SyntheticArenaSimulator()
        while not eng.is_ready:
            ev = s.poll()
            if ev is not None:
                eng.ingest_event(ev)
        return eng.extract_seed(32).hex()
    finally:
        _time.time_ns = orig


a = sim_seed_with_clock(1_700_000_000_000_000_000)
b = sim_seed_with_clock(1_700_000_000_000_000_000)
c = sim_seed_with_clock(1_700_000_000_000_000_001)
R["8_simulator_clock"] = {
    "same_clock_same_seed": a == b,
    "clock_shifted_1ns_changes_seed": a != c,
    "seed_a_prefix": a[:16], "seed_b_prefix": b[:16],
}

with open(OUT, "w") as f:
    json.dump(R, f, indent=2, default=str)
print(json.dumps(R, indent=2, default=str))

"""
Unit and integration tests for PetRand using standard library unittest.
"""

import hashlib
import math
import sys
import unittest
import numpy as np
from petrand.health import (
    NISTHealthMonitor,
    NISTStartupHealthTest,
    RepetitionCountFailure,
    AdaptiveProportionFailure,
)
from petrand.entropy_estimator import (
    MostCommonValueEstimator,
    CollisionEstimator,
    MarkovEstimator,
    NISTEntropyAssessor,
)
from petrand.engine import (
    PetRandEngine,
    RawPhysicalEvent,
    EngineHaltedError,
    InsufficientEntropyError,
)
from petrand.lifecycle import (
    TwoBallManager,
    BallStatus,
    BallProfile,
)
from petrand.optical import (
    OverheadCameraTracker,
    OpticalTrackingResult,
)
from petrand.sensors import SyntheticArenaSimulator, CameraMotionSensor
from petrand.zoomies import ZoomiesScorer
from petrand.adapters import MonteCarloSeedFactory, uint64_to_uniform_float64
from petrand.benchmarks import (
    run_coverage_test,
    run_hypersphere_10d_benchmark,
    run_autocorrelation_test,
    run_tail_risk_benchmark,
)


class TestPetRand(unittest.TestCase):

    # --- 1. NIST Health Monitor Tests ---

    def test_rct_alarm_on_stuck_source(self):
        monitor = NISTHealthMonitor(min_entropy_per_sample=0.5, require_startup_test=False)
        cutoff = monitor.rct_cutoff
        self.assertEqual(cutoff, 41)

        for _ in range(cutoff - 1):
            monitor.validate_sample(100)

        with self.assertRaises(RepetitionCountFailure):
            monitor.validate_sample(100)

    def test_rct_passes_on_alternating_source(self):
        monitor = NISTHealthMonitor(min_entropy_per_sample=0.5, require_startup_test=False)
        for i in range(1000):
            monitor.validate_sample(i % 2)

    def test_apt_alarm_on_biased_source(self):
        monitor = NISTHealthMonitor(min_entropy_per_sample=1.0, window_size=64, require_startup_test=False)
        monitor.validate_sample(999)

        with self.assertRaises(AdaptiveProportionFailure):
            for i in range(15):
                for _ in range(5):
                    monitor.validate_sample(999)
                monitor.validate_sample(i + 1)

    def test_nist_startup_health_test(self):
        engine = PetRandEngine(
            required_entropy_bits=10.0,
            min_entropy_per_sample=2.0,
            require_startup_test=True,
            startup_samples=50,
        )
        self.assertFalse(engine.monitor.is_operational)
        self.assertFalse(engine.is_ready)

        # Ingest 49 samples: monitor should still be in startup, entropy credit remains 0
        for i in range(49):
            ev = RawPhysicalEvent(
                event_id=i,
                timestamp_ns=i * 1000,
                channel="startup",
                raw_symbol=i % 10,
                sensor_payload=b"",
                estimated_entropy_bits=1.0,
            )
            engine.ingest_event(ev)

        self.assertFalse(engine.monitor.is_operational)
        self.assertEqual(engine.entropy_credit, 0.0)

        # 50th sample satisfies startup
        ev_50 = RawPhysicalEvent(
            event_id=50,
            timestamp_ns=50000,
            channel="startup",
            raw_symbol=5,
            sensor_payload=b"",
            estimated_entropy_bits=1.0,
        )
        engine.ingest_event(ev_50)
        self.assertTrue(engine.monitor.is_operational)

        # Next operational event accumulates credit
        ev_op = RawPhysicalEvent(
            event_id=51,
            timestamp_ns=51000,
            channel="op",
            raw_symbol=6,
            sensor_payload=b"",
            estimated_entropy_bits=2.0,
        )
        engine.ingest_event(ev_op)
        # Events 50 and 51 are each credited H = 2.0 bits (any positive claim earns H)
        self.assertEqual(engine.entropy_credit, 4.0)

    # --- 2. NIST SP 800-90B Non-IID Entropy Estimator Tests ---

    def test_nist_entropy_estimator(self):
        # Generate 500 pseudorandom symbols
        rng = np.random.Generator(np.random.PCG64(42))
        symbols = rng.integers(0, 16, size=500).tolist()

        mcv = MostCommonValueEstimator.estimate(symbols)
        col = CollisionEstimator.estimate(symbols)
        mar = MarkovEstimator.estimate(symbols)

        self.assertGreater(mcv, 0.0)
        self.assertGreater(col, 0.0)
        self.assertGreater(mar, 0.0)

        report = NISTEntropyAssessor.evaluate(symbols)
        self.assertEqual(report["total_samples"], 500)
        self.assertGreaterEqual(report["conservative_min_entropy_H_inf"], 0.0)
        self.assertIn(report["assessment_verdict"], ["HIGH", "MEDIUM", "LOW"])
        self.assertIn("not an SP 800-90B assessment", report["scope"])

    # --- 3. Entropy Engine & Two-Ball Swap Tests ---

    def test_engine_accumulation_and_extraction(self):
        engine = PetRandEngine(required_entropy_bits=10.0, max_event_entropy_cap=2.0,
                               min_entropy_per_sample=2.0, require_startup_test=False)
        self.assertFalse(engine.is_ready)

        with self.assertRaises(InsufficientEntropyError):
            engine.extract_seed(32)

        for i in range(6):
            ev = RawPhysicalEvent(
                event_id=i,
                timestamp_ns=i * 1000,
                channel="test",
                raw_symbol=i + 1,
                sensor_payload=b"test_data",
                estimated_entropy_bits=2.0,
                ball_id="Ball_A",
            )
            engine.ingest_event(ev)

        self.assertTrue(engine.is_ready)
        self.assertGreaterEqual(engine.entropy_credit, 10.0)

        seed = engine.extract_seed(seed_bytes=32)
        self.assertEqual(len(seed), 32)
        self.assertIsInstance(seed, bytes)
        self.assertEqual(engine.entropy_credit, 0.0)
        self.assertFalse(engine.is_ready)

    def test_engine_halts_on_health_failure(self):
        engine = PetRandEngine(min_entropy_per_sample=1.0, require_startup_test=False)
        cutoff = engine.monitor.rct_cutoff

        with self.assertRaises(RepetitionCountFailure):
            for i in range(cutoff + 1):
                ev = RawPhysicalEvent(
                    event_id=i,
                    timestamp_ns=i * 1000,
                    channel="stuck",
                    raw_symbol=777,
                    sensor_payload=b"\x00",
                    estimated_entropy_bits=1.0,
                )
                engine.ingest_event(ev)

        self.assertTrue(engine.is_halted)
        self.assertIn("RCT Alarm", engine.halt_reason)

        with self.assertRaises(EngineHaltedError):
            engine.extract_seed(32)

    def test_two_ball_manager_and_swap_pause(self):
        ball_mgr = TwoBallManager(
            ball_a_id="Ball_A",
            ball_a_color="orange",
            ball_b_id="Ball_B",
            ball_b_color="blue",
            max_active_duration_minutes=0.01,
        )
        self.assertEqual(ball_mgr.current_ball.ball_id, "Ball_A")
        self.assertEqual(ball_mgr.idle_ball.ball_id, "Ball_B")
        self.assertEqual(ball_mgr.status, BallStatus.ACTIVE)

        engine = PetRandEngine(required_entropy_bits=10.0, require_startup_test=False)
        engine.pause_for_swap()
        self.assertTrue(engine.is_swap_paused)

        for _ in range(100):
            ev = RawPhysicalEvent(
                event_id=1,
                timestamp_ns=1000,
                channel="swap_dead",
                raw_symbol=99,
                sensor_payload=b"",
                estimated_entropy_bits=1.0,
            )
            engine.ingest_event(ev)

        self.assertFalse(engine.is_halted)
        self.assertEqual(engine.entropy_credit, 0.0)

        new_ball = ball_mgr.complete_swap()
        self.assertEqual(new_ball.ball_id, "Ball_B")
        engine.resume_after_swap("Ball_B")
        self.assertFalse(engine.is_swap_paused)
        self.assertEqual(engine.active_ball_id, "Ball_B")

    # --- 4. Optical Tracker Tests ---

    def test_overhead_camera_tracker_with_roi(self):
        tracker = OverheadCameraTracker(width=160, height=120, roi_rect=(10, 10, 140, 100))
        profile = BallProfile("Ball_A", "orange", (8, 120, 120), (25, 255, 255))
        
        res = tracker.process_frame(profile)
        self.assertTrue(isinstance(res, OpticalTrackingResult))
        self.assertTrue(res.detected)
        self.assertGreater(res.radius, 0.0)
        self.assertEqual(len(res.shot_noise_bytes), 128)
        self.assertTrue(res.lighting_stable)
        tracker.close()

    # --- 5. Scientific Monte Carlo Adapter Tests ---

    def test_spawn_numpy_generators(self):
        engine = PetRandEngine(required_entropy_bits=10.0, require_startup_test=False)
        sim = SyntheticArenaSimulator()
        while not engine.is_ready:
            ev = sim.poll()
            if ev:
                engine.ingest_event(ev)

        factory = MonteCarloSeedFactory(engine)
        workers = factory.spawn_numpy_generators(n_workers=4, bit_generator="pcg64")
        self.assertEqual(len(workers), 4)

        samples = workers[0].random(100_000)
        self.assertTrue(np.all(samples >= 0.0))
        self.assertTrue(np.all(samples < 1.0))
        self.assertLess(abs(np.mean(samples) - 0.5), 0.01)
        self.assertLess(abs(np.var(samples) - (1.0 / 12.0)), 0.01)

    def test_uint64_to_uniform_float64(self):
        u64 = np.array([0, 2**64 - 1], dtype=np.uint64)
        f64 = uint64_to_uniform_float64(u64)
        self.assertEqual(f64[0], 0.0)
        self.assertLess(f64[1], 1.0)
        self.assertGreater(f64[1], 0.9999999999999)

    def test_coverage_benchmark(self):
        engine = PetRandEngine(required_entropy_bits=10.0, require_startup_test=False)
        sim = SyntheticArenaSimulator()
        while not engine.is_ready:
            ev = sim.poll()
            if ev:
                engine.ingest_event(ev)

        factory = MonteCarloSeedFactory(engine)
        res = run_coverage_test(factory, num_trials=50, n_samples=10_000)
        self.assertEqual(res["trials"], 50)
        self.assertTrue(0.84 <= res["coverage_rate"] <= 1.0)

    def test_autocorrelation_benchmark(self):
        engine = PetRandEngine(required_entropy_bits=10.0, require_startup_test=False)
        sim = SyntheticArenaSimulator()
        while not engine.is_ready:
            ev = sim.poll()
            if ev:
                engine.ingest_event(ev)

        factory = MonteCarloSeedFactory(engine, sensor=sim)   # the sensor refills the engine for a retry
        # The test has a family-wise alpha of 0.05, so good data fails it 1 run in 20.
        # Requiring two failures in a row keeps that real check without a 5% flaky test.
        res = run_autocorrelation_test(factory, n_samples=25_000, max_lag=10)
        if not res["passed"]:
            res = run_autocorrelation_test(factory, n_samples=25_000, max_lag=10)
        self.assertEqual(res["max_lag"], 10)
        self.assertTrue(res["passed"])

    def test_tail_risk_benchmark(self):
        engine = PetRandEngine(required_entropy_bits=10.0, require_startup_test=False)
        sim = SyntheticArenaSimulator()
        while not engine.is_ready:
            ev = sim.poll()
            if ev:
                engine.ingest_event(ev)

        factory = MonteCarloSeedFactory(engine, sensor=sim)   # the sensor refills the engine for a retry
        # A statistical check: allow one retry so its false-failure rate does not make the suite flaky.
        res = run_tail_risk_benchmark(factory, n_samples=50_000, df=3.0, tail_prob=0.005)
        if not res["passed"]:
            res = run_tail_risk_benchmark(factory, n_samples=50_000, df=3.0, tail_prob=0.005)
        self.assertTrue(res["passed"])

    # --- 6. FIPS 140-3 & NIST Standards Compliance Tests ---

    def test_fips_self_tests(self):
        from petrand.self_test import SelfTestEngine
        results = SelfTestEngine.run_all_self_tests()
        self.assertEqual(results["status"], "ALL_TESTS_PASSED")
        self.assertIn("PASSED", results["rct_kat"])
        self.assertIn("PASSED", results["apt_kat"])
        self.assertIn("PASSED", results["healthy_stream_kat"])
        self.assertIn("PASSED", results["conditioning_kat"])

    def test_nist_sts_battery(self):
        from petrand.sts import NISTStatisticalTestSuite
        # Generate 10,000 conditioned random bits using PCG64
        rng = np.random.Generator(np.random.PCG64(999))
        bits = rng.integers(0, 2, size=10_000, dtype=np.uint8)

        report = NISTStatisticalTestSuite.evaluate_bits(bits)
        self.assertEqual(report["total_bits"], 10_000)
        self.assertTrue(report["monobit_test"]["passed"])
        self.assertTrue(report["block_frequency_test"]["passed"])
        self.assertTrue(report["runs_test"]["passed"])
        self.assertTrue(report["cumulative_sums_test"]["passed"])
        self.assertTrue(report["all_passed"])

    def test_nist_conditioning_assessor(self):
        from petrand.conditioning_bound import NISTConditioningAssessor
        # 1. Full entropy case (512 >= 256 + 64)
        res = NISTConditioningAssessor.assess_entropy(
            h_in_bits=512.0,
            n_in_bytes=4096,
            n_out_bits=256,
        )
        self.assertTrue(res["full_entropy_qualified"])
        self.assertEqual(res["vetted_output_min_entropy_H_out"], 256.0)
        self.assertIn("FULL_ENTROPY", res["assessment_classification"])

        # 2. Partial entropy case without 64-bit oversampling (200 < 320)
        res_partial = NISTConditioningAssessor.assess_entropy(
            h_in_bits=200.0,
            n_in_bytes=4096,
            n_out_bits=256,
        )
        self.assertFalse(res_partial["full_entropy_qualified"])
        self.assertAlmostEqual(res_partial["vetted_output_min_entropy_H_out"], 200.0, places=1)
        self.assertEqual(res_partial["assessment_classification"], "PARTIAL_ENTROPY")

        # 3. Boundary case H_in == n_out == 256 -> H_out == 255.0 bits
        res_bound = NISTConditioningAssessor.assess_entropy(
            h_in_bits=256.0,
            n_in_bytes=4096,
            n_out_bits=256,
        )
        self.assertFalse(res_bound["full_entropy_qualified"])
        self.assertAlmostEqual(res_bound["vetted_output_min_entropy_H_out"], 255.0, places=1)

    def test_hypersphere_10d_and_block_stream_and_ledger(self):
        import os
        import tempfile
        from petrand.engine import combine_entropy_estimates, intervals_to_von_neumann_bits

        # Test entropy combination & Von Neumann interval extractor
        self.assertEqual(combine_entropy_estimates([1.0, 2.5, 1.5], mode="conservative"), 2.5)
        self.assertEqual(combine_entropy_estimates([2.0, 2.0], mode="discounted", discount=0.5), 2.0)
        self.assertEqual(combine_entropy_estimates([3.0, 3.0], mode="additive", cap=4.0), 4.0)

        vn_bytes = intervals_to_von_neumann_bits(
            [10, 20, 30, 10, 5, 15, 25, 5, 10, 20, 30, 10, 5, 15, 25, 5, 99, 99]
        )
        self.assertEqual(len(vn_bytes), 1)
        self.assertEqual(vn_bytes[0], 0b01010101)

        # Test 10D hypersphere benchmark & multi-block stream with bound sensor
        engine = PetRandEngine(required_entropy_bits=10.0, require_startup_test=False)
        sim = SyntheticArenaSimulator()
        factory = MonteCarloSeedFactory(engine, sensor=sim)

        qmc_res = run_hypersphere_10d_benchmark(factory, n_samples=4096)
        self.assertEqual(qmc_res["dimension"], 10)
        self.assertEqual(qmc_res["samples"], 4096)
        self.assertGreaterEqual(qmc_res["rqmc_sobol"]["volume_estimate"], 1.4)

        batches = list(factory.block_reseeded_stream(total_samples=2500, block_size=1000, dimension=2))
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0].shape, (1000, 2))
        self.assertEqual(batches[2].shape, (500, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_p = os.path.join(tmpdir, "seeds.bin")
            man_p = os.path.join(tmpdir, "manifest.json")
            hashes = factory.write_seed_ledger(bin_p, man_p, num_seeds=3, seed_bytes=32)
            self.assertEqual(len(hashes), 3)
            self.assertEqual(os.path.getsize(bin_p), 96)

    # --- 7. HTTP Web Dashboard & REST API Tests ---

    def test_web_server_api(self):
        import json
        import urllib.request
        import threading
        from petrand.server import start_server

        engine = PetRandEngine(required_entropy_bits=10.0, require_startup_test=False)
        ball_mgr = TwoBallManager(ball_a_id="Ball_1", ball_b_id="Ball_2")
        sim = SyntheticArenaSimulator(ball_id="Ball_1", ball_manager=ball_mgr)

        # Prime engine with enough entropy
        while not engine.is_ready:
            ev = sim.poll()
            if ev:
                engine.ingest_event(ev)

        # Start test HTTP server on ephemeral port
        server = start_server(engine, ball_mgr, sim, host="127.0.0.1", port=0)
        assigned_port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        base_url = f"http://127.0.0.1:{assigned_port}"

        try:
            # 1. Test Dashboard HTML GET
            with urllib.request.urlopen(f"{base_url}/") as response:
                self.assertEqual(response.status, 200)
                body = response.read().decode("utf-8")
                self.assertIn("PetRand // ", body)
                self.assertIn("Chaos Entropy Console", body)
                self.assertNotIn("FIPS 140-3 KAT OK", body)

            # 2. Test Telemetry API GET
            with urllib.request.urlopen(f"{base_url}/api/v1/status") as response:
                self.assertEqual(response.status, 200)
                status_json = json.loads(response.read().decode("utf-8"))
                self.assertIn("engine", status_json)
                self.assertTrue(status_json["engine"]["ready"])
                self.assertEqual(status_json["source"], {"kind": "simulator", "physical": False})
                self.assertFalse(status_json["credit"]["assessed"])
                self.assertIn("health", status_json)
                self.assertLess(status_json["health"]["apt_matches"], status_json["health"]["apt_cutoff"])
                self.assertIn("lifecycle", status_json)
                self.assertEqual(status_json["lifecycle"]["active_ball_id"], "Ball_1")
                self.assertGreater(status_json["lifecycle"]["ball_a_events"], 0)

            # 3. Test Seed Extraction API GET
            with urllib.request.urlopen(f"{base_url}/api/v1/seed") as response:
                self.assertEqual(response.status, 200)
                seed_json = json.loads(response.read().decode("utf-8"))
                self.assertIn("seed", seed_json)
                self.assertEqual(len(bytes.fromhex(seed_json["seed"])), 32)
                self.assertEqual(seed_json["seed_bytes"], 32)

            # 4. Test Ball Swap API POST
            req = urllib.request.Request(f"{base_url}/api/v1/swap", data=b"", method="POST")
            with urllib.request.urlopen(req) as response:
                self.assertEqual(response.status, 200)
                swap_json = json.loads(response.read().decode("utf-8"))
                self.assertEqual(swap_json["status"], "SWAP_COMPLETED")
                self.assertEqual(swap_json["active_ball"], "Ball_2")
                self.assertEqual(ball_mgr.current_ball.ball_id, "Ball_2")

            # 5. Test Live Ball Color Configuration API POST
            col_req = urllib.request.Request(
                f"{base_url}/api/v1/config/colors",
                data=json.dumps({"color_a": "green", "color_b": "pink"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(col_req) as response:
                self.assertEqual(response.status, 200)
                col_json = json.loads(response.read().decode("utf-8"))
                self.assertEqual(col_json["ball_a_color"], "green")
                self.assertEqual(col_json["ball_b_color"], "pink")

            # 6. Test Everyday Randomness Studio API POST & CSV Export GET
            tk_req = urllib.request.Request(
                f"{base_url}/api/v1/toolkit",
                data=json.dumps({"items": ["Pizza", "Tacos", "Sushi", "Ramen"]}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(tk_req) as response:
                self.assertEqual(response.status, 200)
                tk_json = json.loads(response.read().decode("utf-8"))
                self.assertEqual(len(tk_json["passphrase_bundle"]["words"]), 6)
                self.assertEqual(len(tk_json["passphrase_bundle"]["pin_6_digit"]), 6)
                self.assertIn(tk_json["raffle_bundle"]["winner_primary"], ["Pizza", "Tacos", "Sushi", "Ramen"])
                self.assertTrue(tk_json["raffle_bundle"]["verification_receipt"].startswith("VERIFY-"))

            with urllib.request.urlopen(f"{base_url}/api/v1/export.csv?rows=25") as response:
                self.assertEqual(response.status, 200)
                csv_body = response.read().decode("utf-8").strip().splitlines()
                self.assertEqual(len(csv_body), 26)  # 1 header + 25 data rows
                self.assertIn("uniform_0_to_1", csv_body[0])

            # 7. The game served by the console
            with urllib.request.urlopen(f"{base_url}/zoomies") as response:
                self.assertEqual(response.status, 200)
                vj_html = response.read().decode("utf-8")
                self.assertIn("Zoomies", vj_html)
                self.assertNotIn("HARDWARE RANDOM NUMBER GENERATOR", vj_html)

            # 8. The Bluetooth demo and the phone controller were removed (2026-11-08)
            import urllib.error
            for gone in ("/controller", "/api/v1/controller/info", "/api/v1/bluetooth/devices", "/api/v1/rea/audit"):
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"{base_url}{gone}")
                self.assertEqual(ctx.exception.code, 404)

            # 10. Test Zero-Latency Seed Spooler (/api/v1/seed/instant & /api/v1/spool/status)
            with urllib.request.urlopen(f"{base_url}/api/v1/seed/instant") as response:
                self.assertEqual(response.status, 200)
                inst_seed = json.loads(response.read().decode("utf-8"))
                self.assertEqual(len(inst_seed["seed_hex"]), 64)
                self.assertEqual(len(inst_seed["sha256_receipt"]), 16)

            with urllib.request.urlopen(f"{base_url}/api/v1/spool/status") as response:
                self.assertEqual(response.status, 200)
                spool_st = json.loads(response.read().decode("utf-8"))
                self.assertGreaterEqual(spool_st["capacity"], 16)

        finally:
            server.shutdown()
            server.server_close()

    # --- 8. Zoomies! motion game tests ---

    def test_zoomies_scorer_and_server(self):
        import json
        import urllib.request
        import threading
        from petrand.zoomies import (
            FORM_CARDS,
            ZoomiesScorer,
            start_zoomies_server,
        )

        self.assertGreaterEqual(len(FORM_CARDS), 8)

        # 1. A still trajectory must be rejected
        static_traj = ZoomiesScorer.generate_synthetic_trajectory(mode="static", num_samples=100)
        res_static = ZoomiesScorer.evaluate_trajectory(static_traj, form_id="spiral_whisk")
        self.assertFalse(res_static["unlocked"])
        self.assertIn(res_static["status"], ("FROZEN", "STILLNESS_DETECTED"))
        self.assertIsNone(res_static["seed_hex"])

        # 2. Periodic and tiny motions must not unlock; big periodic ones are named as rhythm
        for mode in ("sine", "circle", "back_and_forth", "jitter", "still_phone", "lazy"):
            traj = ZoomiesScorer.generate_synthetic_trajectory(mode=mode, num_samples=180)
            res = ZoomiesScorer.evaluate_trajectory(traj, form_id="vortex_orbit")
            self.assertFalse(res["unlocked"], mode)
            self.assertIsNone(res["seed_hex"], mode)
            self.assertNotIn("derived_rolls", res, mode)   # nothing seed-derived for a rejected round
            self.assertLess(res["chaos_score"], 75.0, mode)
        for mode in ("circle", "back_and_forth"):
            res = ZoomiesScorer.evaluate_trajectory(ZoomiesScorer.generate_synthetic_trajectory(mode=mode))
            self.assertEqual(res["status"], "RHYTHMIC_LOOP_DETECTED", mode)
        for mode in ("jitter", "still_phone"):
            res = ZoomiesScorer.evaluate_trajectory(ZoomiesScorer.generate_synthetic_trajectory(mode=mode))
            self.assertEqual(res["status"], "FROZEN", mode)

        # 3. Wild multi-axis motion must pass and unlock a seed
        chaos_traj = ZoomiesScorer.generate_synthetic_trajectory(mode="chaotic", num_samples=180)
        res_chaos = ZoomiesScorer.evaluate_trajectory(
            chaos_traj, form_id="scatter_snap", key_mashes=8
        )
        self.assertTrue(res_chaos["unlocked"])
        self.assertEqual(res_chaos["status"], "SEED_UNLOCKED")
        self.assertGreaterEqual(res_chaos["chaos_score"], 75.0)
        self.assertIsNotNone(res_chaos["seed_hex"])
        self.assertEqual(len(bytes.fromhex(res_chaos["seed_hex"])), 32)
        self.assertIn("Seed #", res_chaos["seed_label"])
        self.assertLessEqual(res_chaos["estimated_entropy_bits"], 256.0)
        self.assertEqual(res_chaos["estimated_entropy_bits"], min(256.0, 0.5 * res_chaos["kept_samples"]))
        # wild human strokes unlock on a phone-sized play area too, nearly every time
        wins = sum(ZoomiesScorer.evaluate_trajectory(
            ZoomiesScorer.generate_synthetic_trajectory("chaotic", seed=s_, area=(343, 300)),
            area=(343, 300))["unlocked"] for s_ in range(40))
        self.assertGreaterEqual(wins, 36)

        # 4. The Zoomies! game server and the engine
        engine = PetRandEngine(required_entropy_bits=64.0, require_startup_test=False)
        vj_server = start_zoomies_server(host="127.0.0.1", port=0, engine=engine)
        vj_port = vj_server.server_address[1]
        vj_thread = threading.Thread(target=vj_server.serve_forever, daemon=True)
        vj_thread.start()
        vj_url = f"http://127.0.0.1:{vj_port}"

        try:
            with urllib.request.urlopen(f"{vj_url}/") as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode("utf-8")
                self.assertIn("Zoomies", html)
                self.assertNotIn("HARDWARE RANDOM NUMBER GENERATOR", html)

            payload = json.dumps({
                "form_id": "vortex_orbit",
                "samples": chaos_traj,
                "key_mashes": 10,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{vj_url}/api/v1/zoomies/evaluate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                eval_json = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(eval_json["unlocked"])
                self.assertEqual(len(eval_json["seed_hex"]), 64)
            # an unlocked round is mixed into the pool without credit
            telem = engine.get_telemetry()
            self.assertEqual(telem["entropy_credit_bits"], 0.0)
            self.assertEqual(telem["extra_inputs_in_pool"], 1)
            import urllib.error
            for gone in ("/api/v1/rea/audit", "/controller", "/api/v1/seed/instant"):
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"{vj_url}{gone}")
                self.assertEqual(ctx.exception.code, 404)
        finally:
            vj_server.shutdown()
            vj_server.server_close()

    # --- 9. Packet decoders and the anti-bot audit ---

    def test_antibot_audit_and_spooler(self):
        from petrand.antibot import ZoomiesBotAudit

        audit = ZoomiesBotAudit.run_adversarial_audit()
        self.assertTrue(audit["audit_passed"])
        self.assertTrue(audit["all_bots_blocked"])
        self.assertTrue(audit["chaotic_human_passed"])

        from petrand.adapters import SeedSpooler, PetRandRandom, PetRandSecrets
        import tempfile
        import os

        # Pillar 5: SeedSpooler, PetRandRandom (stdlib random.Random), and PetRandSecrets
        with tempfile.TemporaryDirectory() as tmpdir:
            spool_file = os.path.join(tmpdir, "spool.json")
            spooler = SeedSpooler(capacity=8, spool_path=spool_file)
            added = spooler.fill_to_capacity()
            self.assertEqual(added, 8)
            self.assertTrue(os.path.exists(spool_file))
            popped = spooler.pop_seed()
            self.assertEqual(len(popped["seed_hex"]), 64)
            self.assertEqual(spooler.get_status()["available_seeds"], 7)

            pr_rng = PetRandRandom(spooler=spooler)
            u = pr_rng.random()
            self.assertTrue(0.0 <= u < 1.0)
            self.assertTrue(1 <= pr_rng.randint(1, 20) <= 20)
            deck = [1, 2, 3, 4, 5, 6]
            pr_rng.shuffle(deck)
            self.assertEqual(sorted(deck), [1, 2, 3, 4, 5, 6])

            pr_sec = PetRandSecrets(rng=pr_rng)
            self.assertEqual(len(pr_sec.token_bytes(32)), 32)
            self.assertEqual(len(pr_sec.token_hex(16)), 32)
            self.assertIn(pr_sec.choice(["A", "B", "C"]), ["A", "B", "C"])
            self.assertTrue(0 <= pr_sec.randbelow(100) < 100)


    # --- Review fixes (2026-11-08): credit, health tests, conditioner, labels, docs ---

    def _event(self, i, sym, claim, payload=b"x"):
        return RawPhysicalEvent(event_id=i, timestamp_ns=i, channel="t", raw_symbol=sym,
                                sensor_payload=payload, estimated_entropy_bits=claim)

    def test_credit_never_exceeds_health_test_h(self):
        engine = PetRandEngine(required_entropy_bits=100.0, min_entropy_per_sample=0.5,
                               require_startup_test=False)
        for i in range(10):
            engine.ingest_event(self._event(i, i, 4.0))
        self.assertAlmostEqual(engine.entropy_credit, 5.0)

    def test_degraded_source_cannot_be_overcredited(self):
        # A 0.60-bit source passes the H = 0.5 health tests; the credit must stay below
        # what the samples can hold (it used to be 2.0 bits per event, about 3.3x too much).
        rng = np.random.default_rng(3)
        engine = PetRandEngine(required_entropy_bits=256.0, min_entropy_per_sample=0.5)
        i = 0
        while not engine.is_ready:
            i += 1
            sym = 0 if rng.random() < 0.66 else int(rng.integers(1, 65536))
            engine.ingest_event(self._event(i, sym, 2.0))
        credited_events = i - 1023
        self.assertGreaterEqual(credited_events * -math.log2(0.66), 256.0)

    def test_apt_fires_at_exact_sp800_90b_cutoff(self):
        from scipy.stats import binom
        W, H = 512, 0.5
        c = 1 + int(binom.ppf(1 - 2.0 ** -20, W, 2.0 ** -H))
        self.assertEqual(c, 410)

        def window(total_targets):
            seq, need, other = [7], total_targets - 1, 1000
            while len(seq) < W:
                run = min(5, need)
                seq += [7] * run
                need -= run
                if len(seq) < W:
                    seq.append(other)
                    other += 1
            return seq[:W]

        for total, should_fire in ((c - 1, False), (c, True)):
            mon = NISTHealthMonitor(min_entropy_per_sample=H, window_size=W, require_startup_test=False)
            fired = False
            try:
                for v in window(total):
                    mon.validate_sample(v)
            except AdaptiveProportionFailure:
                fired = True
            self.assertEqual(fired, should_fire, f"B={total}")

    def test_full_pool_folds_instead_of_stalling(self):
        engine = PetRandEngine(required_entropy_bits=320.0, min_entropy_per_sample=1.0,
                               max_pool_bytes=200, require_startup_test=False)
        i = 0
        while not engine.is_ready and i < 1000:
            i += 1
            engine.ingest_event(self._event(i, i, 1.0, payload=b"y" * 20))
        tel = engine.get_telemetry()
        self.assertTrue(engine.is_ready)
        self.assertGreater(tel["pool_folds"], 0)
        self.assertLessEqual(tel["pool_bytes"], 200)
        self.assertLessEqual(engine.entropy_credit, 512.0 + 1.0)
        with self.assertRaises(ValueError):
            PetRandEngine(required_entropy_bits=1e9)

    def test_conditioner_is_full_length_hmac(self):
        import hmac, hashlib, struct
        for seed_bytes, hfn in ((32, hashlib.sha256), (64, hashlib.sha512)):
            engine = PetRandEngine(required_entropy_bits=1.0, min_entropy_per_sample=1.0,
                                   require_startup_test=False)
            engine.ingest_event(self._event(1, 5, 1.0))
            pool = bytes(engine._pool)
            expected = hmac.new(engine.domain_tag, (0).to_bytes(8, "little") + struct.pack("<d", 1.0) + pool,
                                hfn).digest()
            self.assertEqual(engine.extract_seed(seed_bytes), expected)  # full output, no truncation

    def test_seed_report_and_zero_claim_events(self):
        engine = PetRandEngine(required_entropy_bits=4.0, min_entropy_per_sample=1.0,
                               require_startup_test=False)
        engine.ingest_event(self._event(1, 1, 0.0))  # mixed in, no credit
        self.assertEqual(engine.entropy_credit, 0.0)
        for i in range(2, 6):
            engine.ingest_event(self._event(i, i, 1.0))
        made = engine.extract_seed_with_report()
        self.assertEqual(made["report"]["samples"], 5)
        self.assertEqual(made["report"]["credit_bits"], 4.0)
        self.assertEqual(made["report"]["conditioner"], "HMAC-SHA-256")
        self.assertEqual(len(made["seed"]), 32)

    def test_server_guards_and_no_fixed_seed(self):
        import json as _json, threading, urllib.request, urllib.error
        from petrand.server import start_server, PetRandServerHandler
        PetRandServerHandler.latest_seed = None
        engine = PetRandEngine(required_entropy_bits=16.0, require_startup_test=False)
        ball_mgr = TwoBallManager(ball_a_id="Ball_1", ball_b_id="Ball_2")
        sim = SyntheticArenaSimulator(ball_id="Ball_1", ball_manager=ball_mgr)
        server = start_server(engine, ball_mgr, sim, host="127.0.0.1", port=0)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            # CSV export before any seed: two exports must differ (no fixed fallback seed)
            csvs = []
            for _ in range(2):
                PetRandServerHandler.latest_seed = None
                with urllib.request.urlopen(f"{base}/api/v1/export.csv?rows=5") as r:
                    csvs.append(r.read())
            self.assertNotEqual(csvs[0], csvs[1])
            # Seed response carries a report; status carries only a fingerprint
            with urllib.request.urlopen(f"{base}/api/v1/seed") as r:
                seed_json = _json.loads(r.read())
            self.assertGreater(seed_json["report"]["samples"], 0)
            with urllib.request.urlopen(f"{base}/api/v1/status") as r:
                status = _json.loads(r.read())
            self.assertNotIn("latest_seed", status)
            self.assertEqual(status["latest_seed_fingerprint"], seed_json["fingerprint"])
            # DNS-rebinding Host and cross-site POST are refused
            req = urllib.request.Request(f"{base}/api/v1/status", headers={"Host": "evil.example"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 403)
            req = urllib.request.Request(f"{base}/api/v1/swap", data=b"{}", method="POST",
                                         headers={"Origin": "http://evil.example", "Content-Type": "text/plain"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()

    def test_simulator_is_labeled_seeded_and_clock_free(self):
        import time as _time
        def symbols(seed, clock_offset):
            orig = _time.time_ns
            counter = iter(range(clock_offset, clock_offset + 10_000_000, 1_000))
            _time.time_ns = lambda: next(counter)
            try:
                sim = SyntheticArenaSimulator(rng_seed=seed)
                out = []
                while len(out) < 200:
                    ev = sim.poll()
                    if ev is not None:
                        out.append(ev.raw_symbol)
                return out
            finally:
                _time.time_ns = orig
        self.assertEqual(symbols(11, 0), symbols(11, 987_654_321))  # clock does not leak in
        self.assertNotEqual(symbols(11, 0), symbols(12, 0))
        self.assertEqual(SyntheticArenaSimulator().describe(), {"kind": "simulator", "physical": False})

    def test_benchmarks_cannot_see_seed_entropy(self):
        from petrand.benchmarks import run_constant_seed_control
        ctl = run_constant_seed_control(trials=50, samples=5_000, qmc_samples=4_096, replicates=4)
        self.assertTrue(ctl["coverage_passed"] and ctl["autocorrelation_passed"] and ctl["tail_risk_passed"])

    def test_export_nist_writes_one_byte_per_sample_with_metadata(self):
        import json as _json, os, tempfile
        from petrand.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "raw.bin")
            argv = sys.argv
            sys.argv = ["petrand", "export-nist", "--sensor", "sim", "--samples", "3000", "--bits", "6", "--out", out]
            try:
                main()
            finally:
                sys.argv = argv
            data = open(out, "rb").read()
            self.assertEqual(len(data), 3000)
            self.assertLess(max(data), 64)
            meta = _json.load(open(out + ".json"))
            self.assertEqual(meta["bits_per_symbol"], 6)
            self.assertFalse(meta["source"]["physical"])

    def test_docs_make_no_certification_or_compliance_claims(self):
        import os, re
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        banned = ["NIST SP 800-90B Compliant", "Certified Standards Compliance", "certified true random",
                  "genuine physical random numbers", "FIPS 140-3 KAT OK", "Zero serial memory verified"]
        for name in sorted(n for n in os.listdir(root) if n.endswith(".md")):
            text = open(os.path.join(root, name), encoding="utf-8").read()
            for phrase in banned:
                self.assertNotIn(phrase, text, f"{name}: {phrase}")
            for m in re.finditer(r"(?i)\b(certified|compliant|certification)\b", text):
                before = text[max(0, m.start() - 40):m.start()].lower()
                self.assertTrue(any(w in before for w in ("not ", "no ", "never", "without", "isn't", "nor ")),
                                f"{name}: affirmative '{m.group(0)}' near: {text[max(0, m.start()-40):m.end()+20]!r}")


    # --- Regression tests for the 2026-11-08 independent UI review ---

    def test_http_guard_rules(self):
        from types import SimpleNamespace
        from petrand.http_guard import request_refusal, read_body, MAX_BODY_BYTES
        import io

        def handler(path, headers, bound="127.0.0.1"):
            return SimpleNamespace(path=path, headers=headers, server=SimpleNamespace(server_address=(bound, 8080)))

        ok = {"Host": "127.0.0.1:8080"}
        self.assertIsNone(request_refusal(handler("/api/v1/seed", ok), is_post=False))
        self.assertEqual(request_refusal(handler("/", {"Host": "evil.example"}), False)[0], 403)
        # another site or another local port may not call the API, but may open the page
        self.assertEqual(request_refusal(handler("/api/v1/seed", {**ok, "Sec-Fetch-Site": "same-site"}), False)[0], 403)
        self.assertEqual(request_refusal(handler("/api/v1/seed", {**ok, "Sec-Fetch-Site": "cross-site"}), False)[0], 403)
        self.assertIsNone(request_refusal(handler("/", {**ok, "Sec-Fetch-Site": "same-site"}), False))
        self.assertIsNone(request_refusal(handler("/api/v1/seed", {**ok, "Sec-Fetch-Site": "same-origin"}), False))
        self.assertEqual(request_refusal(handler("/api/v1/swap", {**ok, "Origin": "http://evil.example"}), True)[0], 403)
        # a server bound to the network on purpose skips only the Host check
        self.assertIsNone(request_refusal(handler("/api/v1/seed", {"Host": "192.168.1.5:8080"}, bound="0.0.0.0"), False))

        def body(length, data=b""):
            return SimpleNamespace(headers={"Content-Length": length}, rfile=io.BytesIO(data))

        self.assertEqual(read_body(body("abc"))[1][0], 400)
        self.assertEqual(read_body(body(str(MAX_BODY_BYTES + 1)))[1][0], 413)
        self.assertEqual(read_body(body("10", b"short"))[1][0], 400)
        self.assertEqual(read_body(body("2", b"{}")), ("{}", None))

    def test_game_round_validation_and_still_rounds(self):
        from petrand.zoomies import ZoomiesScorer
        with self.assertRaises(ValueError):
            ZoomiesScorer.evaluate_request({"samples": [{"t": 1, "x": "a", "y": 2}] * 20})
        with self.assertRaises(ValueError):
            ZoomiesScorer.evaluate_request({"samples": [{"x": 1}] * (ZoomiesScorer.MAX_SAMPLES + 1)})
        with self.assertRaises(ValueError):
            ZoomiesScorer.evaluate_request([1, 2, 3])
        huge = ZoomiesScorer.evaluate_request({"samples": [{"t": i * 16.0, "x": i % 7, "y": i % 5} for i in range(30)],
                                                   "key_mashes": 2 ** 32})
        self.assertIn("status", huge)
        # rounds with no motion still return every metric the page reads
        for samples in ([{"t": 0, "x": 1, "y": 1}] * 5, [{"t": i * 16.0, "x": 5, "y": 5} for i in range(30)]):
            res = ZoomiesScorer.evaluate_request({"samples": samples})
            for key in ("size_score", "rhythm_free_score", "direction_score", "speed_change_score", "path_diagonals"):
                self.assertIn(key, res["metrics"])

    def test_console_seeds_are_per_request_and_servers_stay_responsive(self):
        import json as _json, socket, threading, time as _time, urllib.request, urllib.error, random
        from petrand.server import start_server, PetRandServerHandler
        from petrand.zoomies import start_zoomies_server
        PetRandServerHandler.latest_seed = None
        engine = PetRandEngine(required_entropy_bits=16.0, require_startup_test=False)
        ball_mgr = TwoBallManager(ball_a_id="Ball_1", ball_b_id="Ball_2")
        sim = SyntheticArenaSimulator(ball_id="Ball_1", ball_manager=ball_mgr)
        server = start_server(engine, ball_mgr, sim, host="127.0.0.1", port=0)
        game = start_zoomies_server(host="127.0.0.1", port=0,
                                      engine=PetRandEngine(required_entropy_bits=16.0, require_startup_test=False))
        for srv in (server, game):
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        gbase = f"http://127.0.0.1:{game.server_address[1]}"

        def post(url, payload, headers=None):
            req = urllib.request.Request(url, data=_json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json", **(headers or {})})
            with urllib.request.urlopen(req) as r:
                return _json.loads(r.read())

        try:
            with urllib.request.urlopen(f"{base}/api/v1/seed") as r:
                seed = _json.loads(r.read())["seed"]
            # the toolkit never falls back to another request's seed
            self.assertNotEqual(post(f"{base}/api/v1/toolkit", {})["seed_hex"], seed)
            self.assertEqual(post(f"{base}/api/v1/toolkit", {"seed_hex": seed})["seed_hex"], seed)
            # a game round on the console server does not become the console's seed
            with urllib.request.urlopen(f"{base}/api/v1/status") as r:
                fp = _json.loads(r.read())["latest_seed_fingerprint"]
            rng = random.Random(7)
            x = y = 200.0
            wild = []
            for i in range(300):
                x = min(640, max(0, x + rng.uniform(-90, 90)))
                y = min(400, max(0, y + rng.uniform(-90, 90)))
                wild.append({"t": i * 16.0 + rng.uniform(-4, 4), "x": x, "y": y, "z": rng.uniform(0, 60)})
            res = post(f"{base}/api/v1/zoomies/evaluate", {"samples": wild})
            self.assertTrue(res["unlocked"])
            with urllib.request.urlopen(f"{base}/api/v1/status") as r:
                self.assertEqual(_json.loads(r.read())["latest_seed_fingerprint"], fp)
            # the game server: no wildcard CORS, foreign Host and other-site API calls refused
            with urllib.request.urlopen(f"{gbase}/") as r:
                self.assertIsNone(r.headers.get("Access-Control-Allow-Origin"))
            for headers in ({"Host": "evil.example"}, {"Sec-Fetch-Site": "cross-site"}):
                req = urllib.request.Request(f"{gbase}/api/v1/zoomies/cards", headers=headers)
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req)
                self.assertEqual(ctx.exception.code, 403)
            # one idle connection no longer freezes either server
            for url in (f"{base}/api/v1/status", f"{gbase}/"):
                port = int(url.split(":")[2].split("/")[0])
                idle = socket.create_connection(("127.0.0.1", port))
                t0 = _time.time()
                with urllib.request.urlopen(url, timeout=3) as r:
                    self.assertEqual(r.status, 200)
                self.assertLess(_time.time() - t0, 1.0)
                idle.close()
        finally:
            server.shutdown()
            game.shutdown()

    def test_unbiased_dice_and_meter_mirror(self):
        from petrand.zoomies import unbiased_dice, score_motion, ZoomiesScorer, render_zoomies_html
        self.assertEqual(unbiased_dice(bytes([255, 252, 1, 7]), 2, 6), [2, 2])   # 255 and 252 are skipped
        rolls = unbiased_dice(bytes(range(256)), 300, 6)
        self.assertTrue(all(1 <= r <= 6 for r in rolls))
        self.assertEqual(len(rolls), 300)
        # the page's JavaScript meter must give the same verdicts and scores as score_motion
        import json, re, shutil, subprocess, tempfile, os
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        js = re.findall(r"<script>(.*?)</script>", render_zoomies_html(), re.S)[0]
        meter_js = js[js.index("// ---- The chaos meter"):js.index("// The same result shape")]
        fixtures = []
        for mode in ("chaotic", "lazy", "jitter", "circle", "back_and_forth"):
            for seed in range(3):
                for area in ((640, 400), (343, 300)):
                    sm = ZoomiesScorer.generate_synthetic_trajectory(mode, seed=seed, area=area)
                    py = score_motion(sm, area=area, target_bonus="curvature")
                    fixtures.append({"s": sm, "a": {"w": area[0], "h": area[1]}, "status": py["status"], "score": py["score"]})
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "m.js"), "w") as f:
                f.write(meter_js + "\nconst fx = " + json.dumps(fixtures) + ";\n"
                        "const out = { maxDiff: 0, worst: null, statusMismatches: [] };\n"
                        "fx.forEach((x, i) => { const r = scoreMotion(x.s, x.a, 'curvature');"
                        " const d = Math.abs(r.score - x.score);"
                        " if (d > out.maxDiff) { out.maxDiff = d; out.worst = { i: i, py: x.score, js: r.score }; }"
                        " if (r.status !== x.status) out.statusMismatches.push({ i: i, py: x.status, js: r.status, score: x.score }); });\n"
                        "console.log(JSON.stringify(out));\n")
            out = subprocess.run([node, os.path.join(d, "m.js")], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        res = json.loads(out.stdout)
        print(f"\n  meter mirror: largest score difference {res['maxDiff']:.4f} points over {len(fixtures)} rounds")
        # Floating point differs in the last digit between NumPy builds and JavaScript (for example a
        # step direction exactly on the edge between two of the 16 direction bins), so scores may
        # differ by a few hundredths; the verdict must still be the same.
        self.assertEqual(res["statusMismatches"], [])
        self.assertLessEqual(res["maxDiff"], 0.5, res["worst"])

    # --- Polish fixes (2026-11-08) ---

    def test_passphrase_pin_and_raffle_receipt(self):
        from petrand.toolkit import EverydaySeedToolkit as T, EFF_VIVID_WORDS_256 as W
        self.assertNotIn("ican", W)
        self.assertEqual(len(set(W)), 256)
        leading_zero = False
        for i in range(400):
            seed = hashlib.sha256(str(i).encode()).hexdigest()
            pb = T.generate_passphrase(seed)
            self.assertEqual(len(set(pb["words"])), 6)          # six different words
            self.assertRegex(pb["pin_6_digit"], r"^[0-9]{6}$")
            leading_zero |= pb["pin_6_digit"].startswith("0")
        self.assertTrue(leading_zero)                             # PINs can start with 0
        seed = "ab" * 32
        names = ["Ann", "Ben|Cy", "Dee"]
        draw = T.pick_and_shuffle(seed, names)
        check = T.check_draw(seed, names, draw["verification_receipt"])
        self.assertTrue(check["matches"])
        self.assertEqual(check["shuffled_order"], draw["shuffled_order"])
        self.assertFalse(T.check_draw(seed, ["Ann", "Ben", "Cy|Dee"], draw["verification_receipt"])["matches"])
        with self.assertRaises(ValueError):
            T.pick_and_shuffle(seed, ["  ", ""])

    def test_seed_report_counts_uncredited_inputs_and_mixing_never_halts(self):
        engine = PetRandEngine(required_entropy_bits=16.0, require_startup_test=False)
        sim = SyntheticArenaSimulator()
        for _ in range(500):          # identical extra input would trip a health test if it were tested
            engine.mix_uncredited(b"same bytes every time", label="zoomies_test")
        self.assertFalse(engine.is_halted)
        self.assertEqual(engine.entropy_credit, 0.0)
        while not engine.is_ready:
            ev = sim.poll()
            if ev:
                engine.ingest_event(ev)
        rep = engine.extract_seed_with_report()["report"]
        self.assertEqual(rep["uncredited_extra_inputs"], 500)
        self.assertEqual(rep["health"], "passed")
        self.assertGreater(rep["samples"], 0)

    def test_console_reset_spool_and_halted_engine(self):
        import json as _json, threading, urllib.request, urllib.error
        from petrand.server import start_server, PetRandServerHandler
        PetRandServerHandler.latest_seed = None
        PetRandServerHandler.spooler = None
        engine = PetRandEngine(required_entropy_bits=16.0, require_startup_test=False)
        ball_mgr = TwoBallManager(ball_a_id="Ball_1", ball_b_id="Ball_2")
        sim = SyntheticArenaSimulator(ball_id="Ball_1", ball_manager=ball_mgr)
        server = start_server(engine, ball_mgr, sim, host="127.0.0.1", port=0)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def call(path, payload=None):
            req = urllib.request.Request(base + path, method="POST" if payload is not None else "GET",
                                         data=_json.dumps(payload).encode() if payload is not None else None,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, _json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, _json.loads(e.read())

        try:
            # looking at the spool never makes seeds
            before = engine.get_telemetry()["counter"]
            st, js = call("/api/v1/spool/status")
            self.assertEqual((st, js["available_seeds"]), (200, 0))
            self.assertEqual(engine.get_telemetry()["counter"], before)
            # names that are all blank are refused
            self.assertEqual(call("/api/v1/toolkit", {"seed_hex": "ab" * 32, "items": [" ", ""]})[0], 400)
            # stop the engine with a stuck reading
            for i in range(60):
                try:
                    engine.ingest_event(RawPhysicalEvent(event_id=i, timestamp_ns=i, channel="t", raw_symbol=7,
                                                         sensor_payload=b"x", estimated_entropy_bits=0.5))
                except Exception:
                    break
            self.assertTrue(engine.is_halted)
            st, js = call("/api/v1/seed")
            self.assertEqual(st, 503)
            self.assertTrue(js["halted"])
            # the game still scores a round while the console is stopped
            wild = ZoomiesScorer.generate_synthetic_trajectory("chaotic", seed=2)
            st, js = call("/api/v1/zoomies/evaluate", {"samples": wild})
            self.assertEqual(st, 200)
            self.assertTrue(js["unlocked"])
            # restart the checks
            st, js = call("/api/v1/reset", {})
            self.assertEqual((st, js["status"]), (200, "CHECKS_RESTARTED"))
            self.assertFalse(engine.is_halted)
        finally:
            server.shutdown()
            server.server_close()

    def test_pages_and_launchers_carry_no_banned_claims(self):
        import re as _re, os, json as _json
        from petrand.server import DASHBOARD_HTML
        from petrand.zoomies import render_zoomies_html
        from petrand.toolkit import EFF_VIVID_WORDS_256
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        texts = {"console": DASHBOARD_HTML, "game": render_zoomies_html(True), "game file": render_zoomies_html(False)}
        for name in ("Launch_PetRand.command", "Launch_PetRand.bat"):
            with open(os.path.join(root, name), encoding="utf-8") as f:
                texts[name] = f.read()
        claims = ["true random", "truly random", "quantum", "NIST compliant", "certified random",
                  "cryptographically secure", "unhackable", "military-grade"]
        for name, text in texts.items():
            for word in claims:
                self.assertIsNone(_re.search(_re.escape(word), text, _re.I), f"{word!r} in {name}")
            for m in _re.finditer(r"(?i)\bcertified\b", text):
                self.assertIn("not", text[max(0, m.start() - 12):m.start()].lower(), f"affirmative 'certified' in {name}")
        # the offline game's passphrase words are the console's own 256 words
        words = _json.loads(_re.search(r"const WORDS_256 = (\[.*?\]);", texts["game file"]).group(1))
        self.assertEqual(words, EFF_VIVID_WORDS_256)
        # the game links back to the console only when the console serves it
        self.assertIn("← Console", texts["game"])
        self.assertNotIn("← Console", texts["game file"])

    def test_file_copy_scores_in_the_browser(self):
        import os, tempfile
        from petrand.zoomies import render_zoomies_html, write_standalone_html
        self.assertIn("const STANDALONE = false;", render_zoomies_html())
        self.assertIn("const STANDALONE = true;", render_zoomies_html(standalone=True))
        with tempfile.TemporaryDirectory() as d:
            path = write_standalone_html(os.path.join(d, "game.html"))
            with open(path, encoding="utf-8") as f:
                html = f.read()
        self.assertIn("const STANDALONE = true;", html)
        self.assertNotIn("__STANDALONE__", html)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "zoomies_game.html"), encoding="utf-8") as f:
            self.assertIn("const STANDALONE = true;", f.read())   # the copy in the repository is current

if __name__ == "__main__":
    unittest.main()

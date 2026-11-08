"""
petrand.cli
===========
Command-line interface for PetRand.
Supports overhead camera tracking, two-ball alternating lifecycle,
entropy screening and NIST-format raw-sample export, known-answer self-tests,
a 4-test SP 800-22 subset, and Monte Carlo plumbing checks.
"""

import sys
import json
import time
import struct
import argparse
import threading
from datetime import datetime, timezone
from petrand import __version__
from petrand.engine import PetRandEngine, SENSOR_POLL_LOCK
from petrand.lifecycle import TwoBallManager, BallStatus, COLOR_PRESETS
from petrand.sensors import (
    SyntheticArenaSimulator,
    CameraMotionSensor,
    AudioImpactSensor,
)
from petrand.adapters import MonteCarloSeedFactory, SeedSpooler, PetRandRandom, PetRandSecrets
from petrand.entropy_estimator import NISTEntropyAssessor
from petrand.self_test import SelfTestEngine
from petrand.sts import NISTStatisticalTestSuite
from petrand.conditioning_bound import NISTConditioningAssessor
from petrand.server import start_server
from petrand.zoomies import (
    FORM_CARDS,
    ZoomiesScorer,
    start_zoomies_server,
    write_standalone_html,
)
from petrand.benchmarks import (
    run_constant_seed_control,
    run_coverage_test,
    run_hypersphere_10d_benchmark,
    run_autocorrelation_test,
    run_tail_risk_benchmark,
)


def _sensor_available(sensor) -> bool:
    """False (with a message) when the chosen hardware sensor cannot run on this machine."""
    info = sensor.describe() if hasattr(sensor, "describe") else {"kind": "unknown", "physical": False}
    if str(info["kind"]).endswith("-unavailable"):
        print(f"[!] {info['kind']}: the microphone needs the optional 'sounddevice' package "
              "(pip install -e '.[hardware]').")
        return False
    return True


def _print_source(sensor) -> None:
    """Says plainly whether seeds will come from real hardware or from a simulator."""
    info = sensor.describe() if hasattr(sensor, "describe") else {"kind": "unknown", "physical": False}
    if str(info["kind"]).endswith("-unavailable"):
        print(f"[!] Source              : {info['kind']} (no data will arrive)")
    elif info["physical"]:
        print(f"[*] Source              : {info['kind']} (physical hardware; entropy not yet assessed)")
    else:
        print(f"[!] Source              : {info['kind']} (SIMULATED: these seeds carry no physical entropy)")


def cmd_harvest(args):
    print("================================================================")
    print(" PetRand: Entropy Harvester (webcam, microphone or simulator) ")
    print(" Overhead Webcam Vision & Two-Ball Cycling System ")
    print("================================================================")

    ball_mgr = TwoBallManager(
        ball_a_id=args.ball_a,
        ball_a_color=args.color_a,
        ball_b_id=args.ball_b,
        ball_b_color=args.color_b,
        max_active_duration_minutes=args.swap_minutes,
    )

    engine = PetRandEngine(
        required_entropy_bits=args.entropy_bits,
        min_entropy_per_sample=args.min_entropy,
        active_ball_id=ball_mgr.current_ball.ball_id,
        require_startup_test=not args.skip_startup,
        run_post_self_test=not args.skip_post,
    )

    if args.sensor == "camera":
        sensor = CameraMotionSensor(
            camera_index=args.camera_index,
            ball_manager=ball_mgr,
        )
        mode_str = "Overhead Webcam (OpenCV)" if sensor.is_hardware_camera else "Overhead Optical Simulator"
    elif args.sensor == "audio":
        sensor = AudioImpactSensor(ball_id=ball_mgr.current_ball.ball_id, ball_manager=ball_mgr)
        mode_str = "Acoustic Impact (sounddevice)"
    else:
        sensor = SyntheticArenaSimulator(
            ball_id=ball_mgr.current_ball.ball_id,
            ball_manager=ball_mgr,
            sim_stuck=args.simulate_stuck,
        )
        mode_str = "Synthetic Sinai Billiard Dynamics"

    if not _sensor_available(sensor):
        return
    print(f"[*] Sensor Engine       : {mode_str}")
    print(f"[*] Active Ball (1)     : {ball_mgr.current_ball.ball_id} (Color: {ball_mgr.current_ball.color_name})")
    print(f"[*] Idle / Charging (2) : {ball_mgr.idle_ball.ball_id} (Color: {ball_mgr.idle_ball.color_name})")
    print(f"[*] Max Cycle Duration  : {args.swap_minutes:.1f} minutes")
    print(f"[*] Required Credit     : {args.entropy_bits:.1f} bits per seed")
    print(f"[*] Startup Gating      : {'ENABLED (1024 samples)' if not args.skip_startup else 'DISABLED'}")
    print(f"[*] Credit per Sample   : {engine.monitor.H} bits (placeholder until the source is assessed)")
    _print_source(sensor)
    print("[*] Starting harvesting loop (Press Ctrl+C to halt)...")

    seeds_extracted = 0
    try:
        while True:
            # Check for low-battery condition needing swap
            if ball_mgr.status == BallStatus.LOW_BATTERY:
                print(f"\n[!] LOW BATTERY DETECTED on {ball_mgr.current_ball.ball_id} (Runtime: {ball_mgr.active_duration/60:.1f} min)")
                print(f"[!] Pausing health monitoring for ball swap...")
                engine.pause_for_swap()
                ball_mgr.initiate_swap()

                if args.auto_swap:
                    print(f"[*] Auto-swap enabled. Switching to {ball_mgr.idle_ball.ball_id}...")
                    new_ball = ball_mgr.complete_swap()
                    engine.resume_after_swap(new_ball.ball_id)
                    print(f"[+] Resumed harvesting with {new_ball.ball_id} (Color: {new_ball.color_name})\n")
                else:
                    input(f"[PROMPT] Remove {ball_mgr.current_ball.ball_id}, place {ball_mgr.idle_ball.ball_id} in arena, then press Enter...")
                    new_ball = ball_mgr.complete_swap()
                    engine.resume_after_swap(new_ball.ball_id)
                    print(f"[+] Resumed harvesting with {new_ball.ball_id}!\n")

            event = sensor.poll()
            if event is not None:
                engine.ingest_event(event)
                telem = engine.get_telemetry()
                startup_str = f"Startup: {telem['startup_remaining']:04d} | " if not telem["is_operational"] else ""
                sys.stdout.write(
                    f"\r[{event.ball_id} | Bat: {ball_mgr.estimated_battery_percent:4.1f}%] "
                    f"#{event.event_id:04d} {event.channel:<16} | "
                    f"{startup_str}"
                    f"Symbol: {event.raw_symbol:<5} | "
                    f"Credit: {engine.entropy_credit:6.1f} / {args.entropy_bits:.1f}b"
                )
                sys.stdout.flush()

                if engine.is_ready:
                    seed = engine.extract_seed(seed_bytes=32)
                    seeds_extracted += 1
                    print(f"\n[+] SEED #{seeds_extracted:03d} EXTRACTED: {seed.hex()} (Ball: {event.ball_id})")
                    if args.max_seeds and seeds_extracted >= args.max_seeds:
                        break

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[*] Harvester halted by user.")
    finally:
        sensor.close()


def cmd_self_test(args):
    print("================================================================")
    print(" PetRand: Known-Answer Self-Tests (health tests and hash vectors) ")
    print("================================================================")
    print("[*] Running Power-On / On-Demand Known-Answer Tests (KAT)...")

    results = SelfTestEngine.run_all_self_tests()
    for test_name, status in results.items():
        if test_name != "status":
            print(f"  {test_name.upper():<25} : {status}")

    print("----------------------------------------------------------------")
    print(f"  OVERALL POST VERDICT      : {results['status']}")
    print("================================================================")


def cmd_calibrate(args):
    print("================================================================")
    print(" PetRand: Min-Entropy Screening (3 of the 10 SP 800-90B estimators) ")
    print("================================================================")
    print(f"[*] Collecting {args.samples:,} raw samples from sensor: {args.sensor}...")

    if args.sensor == "camera":
        sensor = CameraMotionSensor(camera_index=args.camera_index)
    elif args.sensor == "audio":
        sensor = AudioImpactSensor()
    else:
        sensor = SyntheticArenaSimulator()
    _print_source(sensor)
    if not _sensor_available(sensor):
        return

    raw_symbols = []
    start_t = time.time()
    try:
        while len(raw_symbols) < args.samples:
            ev = sensor.poll()
            if ev is not None:
                raw_symbols.append(ev.raw_symbol)
                if len(raw_symbols) % 100 == 0 or len(raw_symbols) == args.samples:
                    sys.stdout.write(f"\r[*] Progress: {len(raw_symbols):,}/{args.samples:,} samples ({(len(raw_symbols)/args.samples)*100:.1f}%)")
                    sys.stdout.flush()
            elif args.sensor != "sim":
                time.sleep(0.001)
    finally:
        sensor.close()

    elapsed = max(1e-6, time.time() - start_t)
    print(f"\n[*] Sample collection finished in {elapsed:.2f}s ({len(raw_symbols)/elapsed:.1f} samples/sec)")
    print("\n--- Screening estimators (modeled on SP 800-90B) ---")
    results = NISTEntropyAssessor.evaluate(raw_symbols)

    print(f"  Total Samples Evaluated : {results['total_samples']:,}")
    print(f"  Unique Alphabet Symbols : {results['unique_symbols']:,}")
    print(f"  Most Common Value (MCV) : {results['mcv_entropy_bits']:.3f} bits/sample")
    print(f"  Collision Test Estimate : {results['collision_entropy_bits']:.3f} bits/sample")
    print(f"  Markov Model (Lag-1)    : {results['markov_entropy_bits']:.3f} bits/sample")
    print("  ------------------------------------------------")
    print(f"  Screening H (min of 3)  : {results['conservative_min_entropy_H_inf']:.3f} bits/sample")
    print(f"  Screening Level         : {results['assessment_verdict']}")
    print("  ------------------------------------------------")
    print("  Screening only: not an SP 800-90B assessment, and no statistical test can")
    print("  tell a pseudorandom sequence from a physical one. For a credit you can defend:")
    print("    1. petrand export-nist --sensor <sensor> --samples 1000000 --out raw.bin")
    print("    2. run NIST's SP800-90B_EntropyAssessment tool on raw.bin (non-IID track)")
    print("    3. pass its min-entropy to harvest/serve with --min-entropy")
    print("================================================================")


def cmd_export_nist(args):
    print("================================================================")
    print(" PetRand: Raw Sample Export for NIST's SP 800-90B Assessment Tool ")
    print("================================================================")
    bits = int(args.bits)
    mask = (1 << bits) - 1
    print(f"[*] Exporting {args.samples:,} raw samples, {bits} bit(s) each (one byte per sample), to {args.out}...")

    if args.sensor == "camera":
        sensor = CameraMotionSensor(camera_index=args.camera_index)
    elif args.sensor == "audio":
        sensor = AudioImpactSensor()
    else:
        sensor = SyntheticArenaSimulator()
    _print_source(sensor)
    if not _sensor_available(sensor):
        return
    source_info = sensor.describe()

    count = 0
    start_t = time.time()
    with open(args.out, "wb") as f_out:
        try:
            while count < args.samples:
                ev = sensor.poll()
                if ev is not None:
                    f_out.write(bytes([ev.raw_symbol & mask]))
                    count += 1
                    if count % 10000 == 0 or count == args.samples:
                        sys.stdout.write(f"\r[*] Exported: {count:,}/{args.samples:,} samples ({(count/args.samples)*100:.1f}%)")
                        sys.stdout.flush()
                elif args.sensor != "sim":
                    time.sleep(0.0005)
        finally:
            sensor.close()

    meta_path = args.out + ".json"
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump({
            "format": "one raw sample per byte",
            "bits_per_symbol": bits,
            "mapping": f"raw_symbol & 0x{mask:02X} (low {bits} bits)",
            "samples": count,
            "sensor_option": args.sensor,
            "source": source_info,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "petrand_version": __version__,
            "note": "Fix the raw-symbol definition and bit mapping before collecting data you intend to assess.",
        }, f_meta, indent=2)

    elapsed = max(1e-6, time.time() - start_t)
    print(f"\n[+] Wrote {count:,} samples ({count:,} bytes) to {args.out} in {elapsed:.2f}s; metadata in {meta_path}")
    if count < 1_000_000:
        print("[!] SP 800-90B asks for at least 1,000,000 samples for an assessment.")
    if not source_info["physical"]:
        print("[!] These samples come from a simulator; assessing them says nothing about a physical source.")
    print("[*] Next: build NIST's SP800-90B_EntropyAssessment tool (make) and run")
    print(f"      ./ea_non_iid {args.out} {bits}")
    print("    Restart testing needs separate data: ./ea_restart on exactly 1,000,000 samples")
    print("    (1,000 restarts x 1,000 samples, row order) plus the initial estimate H_I.")
    print("    Pass the final min(H_r, H_c, H_I) to harvest/serve with --min-entropy.")
    print("================================================================")


def cmd_sts(args):
    print("================================================================")
    print(" PetRand: 4 of the 15 NIST SP 800-22 Tests, on Conditioned Output ")
    print("================================================================")
    print("[!] These bits come out of a hash, which passes these tests (about 96% of runs) whatever went in.")
    print("    They check the plumbing, not the entropy source.")
    print(f"[*] Collecting {args.bits:,} conditioned bits from the engine (simulator source)...")

    engine = PetRandEngine(required_entropy_bits=64.0)
    sim = SyntheticArenaSimulator()

    bits_needed = args.bits
    bytes_needed = (bits_needed + 7) // 8

    # Extract 32-byte seed blocks
    extracted_bytes = bytearray()
    while len(extracted_bytes) < bytes_needed:
        while not engine.is_ready:
            ev = sim.poll()
            if ev:
                engine.ingest_event(ev)
        seed = engine.extract_seed(seed_bytes=32)
        extracted_bytes.extend(seed)

    raw_bits = bytes(extracted_bytes[:bytes_needed])
    report = NISTStatisticalTestSuite.evaluate_bits(raw_bits)

    print(f"\n  Bitstream Length        : {report['total_bits']:,} bits")
    print(f"  Significance Level (a)  : 0.01")
    print("  ------------------------------------------------")
    print(f"  1. Frequency (Monobit)  : p-value = {report['monobit_test']['p_value']:.6f} | {'PASS [OK]' if report['monobit_test']['passed'] else 'FAIL [X]'}")
    print(f"  2. Block Frequency (M=128): p-value = {report['block_frequency_test']['p_value']:.6f} | {'PASS [OK]' if report['block_frequency_test']['passed'] else 'FAIL [X]'}")
    print(f"  3. Runs Test            : p-value = {report['runs_test']['p_value']:.6f} | {'PASS [OK]' if report['runs_test']['passed'] else 'FAIL [X]'}")
    print(f"  4. Cumulative Sums (Cusum): p-value = {report['cumulative_sums_test']['p_value']:.6f} | {'PASS [OK]' if report['cumulative_sums_test']['passed'] else 'FAIL [X]'}")
    print("  ------------------------------------------------")
    print(f"  Subset Result (4 of 15) : {'ALL 4 PASSED [OK]' if report['all_passed'] else 'TESTS FAILED [X]'}")
    print("================================================================")


def cmd_conditioning(args):
    print("================================================================")
    print(" PetRand: Output-Entropy Bound for the Vetted Conditioner (what-if) ")
    print("================================================================")
    print("[*] Conditioner: HMAC-SHA-256 for 32-byte seeds (a vetted function in SP 800-90B),")
    print("    so n_out = 256 with no truncation. The bound holds only if H_in comes from an")
    print("    entropy assessment, and PetRand's variable-length pool falls short of SP 800-90B's")
    print("    conditioner rules; treat the result as a what-if.")

    res = NISTConditioningAssessor.assess_entropy(
        h_in_bits=args.entropy_in,
        n_in_bytes=args.buffer_bytes,
        n_out_bits=args.out_bits,
    )

    print(f"  Input Min-Entropy (H_in)   : {res['input_entropy_H_in_bits']:.1f} bits")
    print(f"  Input Buffer Size (n_in)   : {res['input_buffer_bytes']:,} bytes ({res['input_buffer_bytes']*8:,} bits)")
    print(f"  Conditioner output (n_out) : {res['output_length_bits']} bits")
    print(f"  Full-Entropy Rule          : H_in >= {res['nist_full_entropy_threshold_bits']:.1f} bits (n_out + 64; FIPS 140-3 IG D.K Res. 19, SP 800-90C)")
    print(f"  Rule Met (if H_in assessed): {'YES' if res['full_entropy_qualified'] else 'NO'}")
    print(f"  Output Min-Entropy Bound   : {res['vetted_output_min_entropy_H_out']:.2f} bits / {res['output_length_bits']} bits")
    print("================================================================")


def cmd_ledger(args):
    print(f"[*] Generating {args.num_seeds} seeds into {args.bin_path}...")
    engine = PetRandEngine(required_entropy_bits=args.entropy_bits)
    sim = SyntheticArenaSimulator()
    factory = MonteCarloSeedFactory(engine, sensor=sim)

    hashes = factory.write_seed_ledger(
        bin_path=args.bin_path,
        manifest_path=args.manifest_path,
        num_seeds=args.num_seeds,
        seed_bytes=32,
    )
    for i, sh in enumerate(hashes):
        print(f"  Seed {i+1:02d}: {sh[:16]}... (SHA-256)")

    print(f"[+] Complete! Binary ledger: {args.bin_path}, Manifest: {args.manifest_path}")


def cmd_benchmark(args):
    print("================================================================")
    print(" PetRand: Monte Carlo Plumbing Checks (with a zero-seed control)")
    print("================================================================")
    print("[*] These checks test the generators downstream of the seed. They cannot")
    print("    detect a weak seed; the control run at the end uses an all-zero seed.")

    engine = PetRandEngine(required_entropy_bits=args.entropy_bits)
    sim = SyntheticArenaSimulator()
    factory = MonteCarloSeedFactory(engine, sensor=sim)
    _print_source(sim)

    print("\n--- 1: Coverage of nominal 95% intervals for E[X^2] = 1/3 ---")
    cov = run_coverage_test(factory, num_trials=args.trials, n_samples=args.samples)
    print(f"  Trials x Samples  : {cov['trials']} x {cov['samples_per_trial']:,}")
    print(f"  Coverage          : {cov['coverage_rate']*100:.1f}% (95% CI {cov['coverage_ci95'][0]*100:.1f}-{cov['coverage_ci95'][1]*100:.1f}%; nominal 95.0%)")
    print(f"  Binomial test     : p = {cov['binomial_test_p_value']:.3f} -> {'PASSED [OK]' if cov['passed'] else 'FAILED [X]'}")

    print("\n--- 2: 10-D unit-ball volume, plain MC vs. scrambled Sobol (replicated) ---")
    qmc = run_hypersphere_10d_benchmark(factory, n_samples=args.qmc_samples, replicates=args.replicates)
    print(f"  Samples x Replicates : {qmc['samples']:,} x {qmc['replicates']}")
    print(f"  True volume          : {qmc['true_volume']:.8f}")
    print(f"  Plain MC RMS rel err : {qmc['plain_mc']['rmse_relative']*100:.3f}%")
    print(f"  RQMC RMS rel err     : {qmc['rqmc_sobol']['rmse_relative']*100:.3f}%")
    print(f"  Ratio (MC / RQMC)    : {qmc['rqmc_advantage_factor']:.2f}x (from the Sobol construction, not the seed)")

    print("\n--- 3: Lag-k autocorrelation (lags 1-20, Bonferroni) ---")
    ac = run_autocorrelation_test(factory, n_samples=50_000, max_lag=20)
    print(f"  Bonferroni bound   : +/-{ac['bonferroni_bound_95']}")
    print(f"  Max |rho_k|        : {ac['max_abs_autocorr']} -> {'PASSED [OK]' if ac['passed'] else 'FAILED [X]'}")

    print("\n--- 4: Student-t(3) tail frequency at the 0.1% quantile ---")
    tr = run_tail_risk_benchmark(factory, n_samples=100_000, df=3.0, tail_prob=0.001)
    print(f"  Empirical tail     : {tr['empirical_tail_probability']*100:.3f}% (z = {tr['z_score']}) -> {'PASSED [OK]' if tr['passed'] else 'FAILED [X]'}")

    print("\n--- Control: the same checks with an all-zero seed ---")
    ctl = run_constant_seed_control(
        trials=min(args.trials, 100), samples=min(args.samples, 25_000),
        qmc_samples=min(args.qmc_samples, 65_536), replicates=min(args.replicates, 8),
    )
    print(f"  Coverage {'PASS' if ctl['coverage_passed'] else 'FAIL'}, autocorrelation {'PASS' if ctl['autocorrelation_passed'] else 'FAIL'}, "
          f"tail {'PASS' if ctl['tail_risk_passed'] else 'FAIL'}, RQMC ratio {ctl['rqmc_advantage_factor']}x")
    print("  -> Passing here says nothing about seed entropy.")
    print("================================================================")


def cmd_serve(args):
    print("================================================================")
    print(" PetRand: Web Dashboard & REST API Server")
    print("================================================================")

    ball_mgr = TwoBallManager(
        ball_a_id=args.ball_a,
        ball_a_color=args.color_a,
        ball_b_id=args.ball_b,
        ball_b_color=args.color_b,
        max_active_duration_minutes=args.swap_minutes,
    )

    engine = PetRandEngine(
        required_entropy_bits=args.entropy_bits,
        min_entropy_per_sample=args.min_entropy,
        active_ball_id=ball_mgr.current_ball.ball_id,
        require_startup_test=not args.skip_startup,
        run_post_self_test=not args.skip_post,
    )

    if args.sensor == "camera":
        sensor = CameraMotionSensor(
            camera_index=args.camera_index,
            ball_manager=ball_mgr,
        )
        mode_str = "Overhead Webcam (OpenCV)" if sensor.is_hardware_camera else "Overhead Optical Simulator"
    elif args.sensor == "audio":
        sensor = AudioImpactSensor(ball_id=ball_mgr.current_ball.ball_id, ball_manager=ball_mgr)
        mode_str = "Acoustic Impact (sounddevice)"
    else:
        sensor = SyntheticArenaSimulator(
            ball_id=ball_mgr.current_ball.ball_id,
            ball_manager=ball_mgr,
            sim_stuck=args.simulate_stuck,
        )
        mode_str = "Synthetic Sinai Billiard Dynamics"

    if not _sensor_available(sensor):
        return

    # Start background harvesting daemon thread
    stop_event = threading.Event()

    def harvester_loop():
        # Poll in short bursts: a camera or microphone blocks in poll() until it has data, and the
        # simulator is pure computation, so sleeping after every poll would starve it (about 7
        # readings a second, minutes for a first number). Bursts give it about 400 a second.
        while not stop_event.is_set():
            if args.auto_swap and ball_mgr.status == BallStatus.LOW_BATTERY:
                engine.pause_for_swap()
                ball_mgr.initiate_swap()
                new_ball = ball_mgr.complete_swap()
                engine.resume_after_swap(new_ball.ball_id)

            for _ in range(50):
                with SENSOR_POLL_LOCK:
                    event = sensor.poll()
                if event is not None:
                    try:
                        engine.ingest_event(event)
                    except Exception:
                        time.sleep(0.5)  # halted by a health test; the console shows why
                        break
            time.sleep(0.005)

    harvester_thread = threading.Thread(target=harvester_loop, daemon=True)
    harvester_thread.start()

    server = start_server(engine, ball_mgr, sensor, host=args.host, port=args.port)
    print(f"[*] Sensor Engine       : {mode_str}")
    _print_source(sensor)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("[!] WARNING: the API has no login. Anyone on this network can draw seeds or reset health tests.")
    print(f"[*] Active Ball         : {ball_mgr.current_ball.ball_id} ({ball_mgr.current_ball.color_name})")
    print(f"[*] Web Dashboard URL   : http://{args.host}:{args.port}/")
    print(f"[*] REST API Status     : http://{args.host}:{args.port}/api/v1/status")
    print(f"[*] REST API Seed Extr  : http://{args.host}:{args.port}/api/v1/seed")
    print(f"[*] REST API Swap Ball  : http://{args.host}:{args.port}/api/v1/swap (POST)")
    print("[*] Server running. Press Ctrl+C to halt.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down PetRand server...")
    finally:
        stop_event.set()
        server.server_close()
        print("[+] Server halted cleanly.")


def cmd_spool(args):
    import os
    spool_path = os.path.expanduser(args.spool_path)
    sensor = SyntheticArenaSimulator()
    spooler = SeedSpooler(capacity=args.capacity, sensor=sensor, spool_path=spool_path)
    added = spooler.fill_to_capacity()
    status = spooler.get_status()
    print("================================================================")
    print(" PetRand: Zero-Latency Background Seed Spooler")
    print("================================================================")
    print(f"  Spool File Path   : {spool_path}")
    print(f"  Buffer Capacity   : {status['available_seeds']} / {status['capacity']} seeds ({added} newly minted)")
    t0 = time.perf_counter()
    popped = spooler.pop_seed()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Instant Pop Speed : {dt_ms:.4f} ms")
    print(f"  Popped 256b Seed  : {popped['seed_hex']}")
    print(f"  SHA-256 Receipt   : {popped['sha256_receipt']}")
    spooler.fill_to_capacity()
    print("================================================================")


def cmd_zoomies(args):
    print("================================================================")
    print(" Zoomies! // PetRand motion game ")
    print("================================================================")

    if args.export_html:
        path = write_standalone_html(args.export_html)
        print(f"[+] Wrote the standalone Zoomies! page to: {path}")
        if not args.simulate and not args.serve_after_export:
            return

    if args.simulate:
        print("[*] Running the rhythm detector against three synthetic motions...\n")
        for mode, label in [
            ("static", "Test 1: No motion"),
            ("sine", "Test 2: Rhythmic Sine-Wave Shake (Periodic Cheating)"),
            ("chaotic", "Test 3: Wild multi-axis motion"),
        ]:
            traj = ZoomiesScorer.generate_synthetic_trajectory(mode=mode, num_samples=180)
            res = ZoomiesScorer.evaluate_trajectory(
                samples=traj,
                form_id="vortex_orbit",
                key_mashes=(10 if mode == "chaotic" else 0),
            )
            print(f"--- {label} ---")
            print(f"  Status               : {res['status']}")
            print(f"  Chaos score          : {res['chaos_score']:.2f}% (Threshold: {res['threshold']:.1f}%)")
            print(f"  3D Jerk Score        : {res['metrics']['jerk_score']:.1f}%")
            print(f"  FFT Spectral Entropy : {res['metrics']['spectral_entropy']:.4f}")
            print(f"  Angular Dispersion   : {res['metrics']['directional_dispersion']:.1f}%")
            print(f"  Seed unlocked        : {'YES' if res['unlocked'] else 'NO'}")
            if res["unlocked"]:
                print(f"  Label                : {res['seed_label']}")
                print(f"  Seed (256-bit hash)  : {res['seed_hex']}")
                print(f"  D20 roll             : {res['derived_rolls']['d20']} | 6D6: {res['derived_rolls']['6d6']}")
            print()
        print("================================================================")
        return

    engine = PetRandEngine(required_entropy_bits=320.0, require_startup_test=False)
    server = start_zoomies_server(host=args.host, port=args.port, engine=engine)
    print(f"[*] Loaded {len(FORM_CARDS)} move cards")
    print(f"[*] Zoomies! game URL   : http://{args.host}:{args.port}/")
    print(f"[*] Evaluation REST API : http://{args.host}:{args.port}/api/v1/zoomies/evaluate (POST)")
    print("[*] Server running. Open the URL in your desktop or mobile browser (Ctrl+C to halt).")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Closing the Zoomies! server...")
    finally:
        server.server_close()
        print("[+] Zoomies! server stopped.")


def main():
    parser = argparse.ArgumentParser(description="PetRand: experimental entropy source and Monte Carlo seeder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: harvest
    p_harv = subparsers.add_parser("harvest", help="Collect samples and print seeds")
    p_harv.add_argument("--sensor", choices=["sim", "camera", "audio"], default="camera")
    p_harv.add_argument("--entropy-bits", type=float, default=320.0, help="Credit per seed (320 = 256-bit output + 64)")
    p_harv.add_argument("--min-entropy", type=float, default=0.5)
    p_harv.add_argument("--camera-index", type=int, default=0)
    p_harv.add_argument("--max-seeds", type=int, default=0)
    p_harv.add_argument("--simulate-stuck", action="store_true")
    p_harv.add_argument("--skip-startup", action="store_true", help="Skip the 1,024-sample start-up health test (on by default)")
    p_harv.add_argument("--skip-post", action="store_true", help="Skip the power-on self-tests (on by default)")
    p_harv.add_argument("--require-startup", action="store_true", help=argparse.SUPPRESS)
    p_harv.add_argument("--run-post", action="store_true", help=argparse.SUPPRESS)
    # Two-ball options
    p_harv.add_argument("--ball-a", default="Ball_A", help="ID of Ball 1")
    p_harv.add_argument("--color-a", choices=list(COLOR_PRESETS.keys()), default="orange")
    p_harv.add_argument("--ball-b", default="Ball_B", help="ID of Ball 2")
    p_harv.add_argument("--color-b", choices=list(COLOR_PRESETS.keys()), default="blue")
    p_harv.add_argument("--swap-minutes", type=float, default=35.0, help="Runtime before battery depletion swap")
    p_harv.add_argument("--auto-swap", action="store_true", help="Auto-switch balls without keyboard prompt")
    p_harv.set_defaults(func=cmd_harvest)

    # Subcommand: self-test
    p_st = subparsers.add_parser("self-test", help="Run the known-answer self-tests (health tests and hash test vectors)")
    p_st.set_defaults(func=cmd_self_test)

    # Subcommand: calibrate
    p_cal = subparsers.add_parser("calibrate", help="Screen raw-sample min-entropy (3 of the 10 SP 800-90B estimators; not an assessment)")
    p_cal.add_argument("--sensor", choices=["sim", "camera", "audio"], default="sim")
    p_cal.add_argument("--samples", type=int, default=2000, help="Number of raw samples to collect")
    p_cal.add_argument("--camera-index", type=int, default=0)
    p_cal.set_defaults(func=cmd_calibrate)

    # Subcommand: export-nist
    p_exp = subparsers.add_parser("export-nist", help="Export raw samples (one byte each) for NIST's SP 800-90B assessment tool")
    p_exp.add_argument("--sensor", choices=["sim", "camera", "audio"], default="sim")
    p_exp.add_argument("--samples", type=int, default=1_000_000, help="Number of raw samples (SP 800-90B asks for at least 1,000,000)")
    p_exp.add_argument("--bits", type=int, default=8, choices=range(1, 9), metavar="1-8",
                       help="Bits kept per sample (low bits of the raw symbol); NIST's tool takes 1-8")
    p_exp.add_argument("--out", default="raw_noise_sample.bin", help="Output binary file path")
    p_exp.add_argument("--camera-index", type=int, default=0)
    p_exp.set_defaults(func=cmd_export_nist)

    # Subcommand: sts
    p_sts = subparsers.add_parser("sts", help="Run 4 of the 15 NIST SP 800-22 tests on conditioned output (cannot assess the source)")
    p_sts.add_argument("--bits", type=int, default=100000, help="Number of conditioned bits to evaluate")
    p_sts.set_defaults(func=cmd_sts)

    # Subcommand: conditioning
    p_cond = subparsers.add_parser("conditioning", help="What-if output-entropy bound for the vetted conditioner (needs an assessed H_in)")
    p_cond.add_argument("--entropy-in", type=float, default=512.0, help="Input min-entropy H_in in bits")
    p_cond.add_argument("--buffer-bytes", type=int, default=4096, help="Input buffer byte size")
    p_cond.add_argument("--out-bits", type=int, default=256, help="Conditioning output length in bits")
    p_cond.set_defaults(func=cmd_conditioning)

    # Subcommand: ledger
    p_led = subparsers.add_parser("ledger", help="Generate an offline seed file and audit manifest")
    p_led.add_argument("--num-seeds", type=int, default=5)
    p_led.add_argument("--entropy-bits", type=float, default=320.0, help="Credit per seed (320 = 256-bit output + 64)")
    p_led.add_argument("--bin-path", default="petrand_seeds.bin")
    p_led.add_argument("--manifest-path", default="petrand_manifest.json")
    p_led.set_defaults(func=cmd_ledger)

    # Subcommand: benchmark
    p_bm = subparsers.add_parser("benchmark", help="Check the Monte Carlo plumbing downstream of the seed (with a zero-seed control)")
    p_bm.add_argument("--entropy-bits", type=float, default=320.0, help="Credit per seed (320 = 256-bit output + 64)")
    p_bm.add_argument("--trials", type=int, default=100)
    p_bm.add_argument("--samples", type=int, default=25_000)
    p_bm.add_argument("--qmc-samples", type=int, default=1_048_576)
    p_bm.add_argument("--replicates", type=int, default=8, help="Independent replicates for the RQMC comparison")
    p_bm.set_defaults(func=cmd_benchmark)

    # Subcommand: serve
    p_srv = subparsers.add_parser("serve", help="Launch interactive Web Dashboard and REST API")
    p_srv.add_argument("--host", default="127.0.0.1",
                       help="Bind address (default 127.0.0.1, this computer only). 0.0.0.0 exposes the "
                            "unauthenticated API to your whole network.")
    p_srv.add_argument("--port", type=int, default=8080, help="HTTP server bind port (default: 8080)")
    p_srv.add_argument("--sensor", choices=["sim", "camera", "audio"], default="sim")
    p_srv.add_argument("--entropy-bits", type=float, default=320.0, help="Credit per seed (320 = 256-bit output + 64)")
    p_srv.add_argument("--min-entropy", type=float, default=0.5)
    p_srv.add_argument("--camera-index", type=int, default=0)
    p_srv.add_argument("--simulate-stuck", action="store_true")
    p_srv.add_argument("--skip-startup", action="store_true", help="Skip the 1,024-sample start-up health test (on by default)")
    p_srv.add_argument("--skip-post", action="store_true", help="Skip the power-on self-tests (on by default)")
    p_srv.add_argument("--require-startup", action="store_true", help=argparse.SUPPRESS)
    p_srv.add_argument("--run-post", action="store_true", help=argparse.SUPPRESS)
    p_srv.add_argument("--ball-a", default="Ball_A")
    p_srv.add_argument("--color-a", choices=list(COLOR_PRESETS.keys()), default="orange")
    p_srv.add_argument("--ball-b", default="Ball_B")
    p_srv.add_argument("--color-b", choices=list(COLOR_PRESETS.keys()), default="blue")
    p_srv.add_argument("--swap-minutes", type=float, default=35.0)
    p_srv.add_argument("--auto-swap", action="store_true", help="Auto-switch balls on low battery")
    p_srv.set_defaults(func=cmd_serve)

    # Subcommand: zoomies
    p_vj = subparsers.add_parser(
        "zoomies",
        help="Launch the Zoomies! motion game",
    )
    p_vj.add_argument("--host", default="127.0.0.1", help="HTTP server bind host (default: 127.0.0.1)")
    p_vj.add_argument("--port", type=int, default=8085, help="HTTP server bind port (default: 8085)")
    p_vj.add_argument("--simulate", action="store_true", help="Run the rhythm detector on synthetic motions")
    p_vj.add_argument("--export-html", default="", help="Write standalone zero-server HTML file to path")
    p_vj.add_argument("--serve-after-export", action="store_true", help="Also run server after exporting HTML")
    p_vj.set_defaults(func=cmd_zoomies)

    # Subcommand: spool
    p_spl = subparsers.add_parser(
        "spool",
        help="Fill and query the zero-latency 256-bit seed spooler",
    )
    p_spl.add_argument("--capacity", type=int, default=32, help="Number of pre-conditioned seeds to spool")
    p_spl.add_argument("--spool-path", default="~/.petrand/seed_spool.json", help="Path to JSON seed spool file")
    p_spl.add_argument("--sensor", choices=["sim"], default="sim")
    p_spl.set_defaults(func=cmd_spool)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

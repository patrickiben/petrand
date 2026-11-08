# PetRand: Design Notes and Open Questions

These notes describe what PetRand does, what each part is for, and which claims are measured, implemented or still hypotheses.

## 1. Overview

```
Arena: ball + deflectors
   -> overhead webcam / microphone            raw samples (no host-clock bits)
   -> health tests: start-up, RCT, APT        stop on a stuck or biased source
   -> credit: H bits per sample               H = 0.5 placeholder until assessed
   -> pool -> HMAC-SHA-256 -> 32-byte seed    at 320 bits of credit; pool then cleared
   -> NumPy generators, Sobol sampling, passphrases, dice, raffles
```

With no hardware attached, a simulator stands in for the arena. Its randomness comes from the operating system's random generator, and every output path labels it as simulated.

## 2. Claims and their status

| Claim | Status | What would settle it |
|---|---|---|
| The ball's motion is chaotic (nearby starts diverge) | Hypothesis | Ensemble divergence from repeated launches; a Lyapunov estimate as a secondary check |
| The deflectors increase unpredictability | Hypothesis. The printed corner wedges are concave toward the ball, so they focus rather than disperse | Compare enclosed runs with and without scatterers; print convex posts for a dispersing condition |
| Webcam low-bit noise is photon shot noise | Hypothesis | Measure a static, evenly lit scene; check the camera's output format and noise reduction |
| The toy's motor controller adds unpredictable timing | Unknown; the controller may replay a fixed turn pattern | Record 50 power-ons on an open floor and compare turn sequences |
| Health tests follow SP 800-90B §4.4 | Implemented; cutoffs match the standard (41 and 410 at H = 0.5) and the self-test checks them | |
| The conditioner is a vetted SP 800-90B function | Implemented: HMAC-SHA-256 for 32-byte seeds, HMAC-SHA-512 for 64-byte seeds, full outputs (no truncation) | |
| Seeds have full entropy | Not established. It would need an assessed H, at least 320 bits of credit per 256-bit seed (FIPS 140-3 IG D.K Resolution 19; SP 800-90C §3.2.2.2), and a fixed-length conditioner input drawn only from the noise source, which the current pool does not have | SP 800-90B assessment plus a redesigned pool |
| Lab validation (FIPS 140-3, SP 800-90B entropy-source validation, BSI AIS 31) | Not certified. Validation certificates come from NVLAP-accredited lab testing and CMVP review; AIS 31 is applied in BSI's Common Criteria scheme | |
| Physical seeds improve Monte Carlo results | No. The seed source does not affect accuracy, and an all-zero seed passes the same benchmarks | |

## 3. Raw samples (current, provisional definitions)

| Sensor | Raw sample |
|---|---|
| Webcam, rebound | `(int(speed*10) XOR int(x*100)) & 0xFFFF`, with speed from the nominal frame interval |
| Webcam, motion | `(int(motion_energy) XOR int(x*10) XOR int(y*10)) & 0xFFFF` |
| Microphone | `int(RMS*10000) & 0xFFFF` for each detected impact |
| Simulator | Simulated interval, angle and position only; perturbations from a seeded NumPy generator |

The webcam also stores the low bit of the frame difference for the first 1,024 pixels in the pool. It is not credited separately.

These feature hashes were chosen for convenience. The study replaces them with fixed, physically meaningful samples (paper/RESEARCH_PLAN.md, section 5).

## 4. Health tests

- **Start-up:** the engine credits nothing until 1,024 consecutive samples pass. It runs again after every ball swap.
- **Repetition count test (RCT):** fails when one value repeats C = 1 + ceil(20/H) times in a row.
- **Adaptive proportion test (APT):** splits the stream into blocks of W = 512 samples, counts how often each block's first value appears in the block (itself included) and fails at C = 1 + CRITBINOM(W, 2^-H, 1 - 2^-20).

| H (bits per sample) | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|
| RCT cutoff | 41 | 21 | 11 | 6 |
| APT cutoff (W = 512) | 410 | 311 | 177 | 62 |

These tests catch gross failures. They cannot show that a source delivers H bits per sample: a source at 0.6 bits per sample passes them at H = 0.5. H may not exceed 16, the raw-symbol width.

## 5. Credit and conditioning

- Every event with a positive claim is credited H bits (at most 4). Inputs that are not the primary noise source, such as unlocked Zoomies! rounds, are mixed into the pool without credit and skip the health tests (`PetRandEngine.mix_uncredited`); the seed report counts them separately.
- When the pool (1 MB) fills, it is folded into its SHA-512 digest and the credit is capped at 512 bits, so a full pool can no longer stall the engine.
- Seeds come from HMAC keyed with the domain tag, over the seed counter, the credit and the pool: HMAC-SHA-256 for 32-byte seeds and HMAC-SHA-512 for 64-byte seeds.
- `conditioning` computes the SP 800-90B §3.1.5.1.2 output-entropy bound with n_out = n_w = 256 for HMAC-SHA-256, and the full-entropy rule (input entropy at least n_out + 64 = 320 bits). Both depend on an assessed input entropy. PetRand's pool also falls short of SP 800-90B's conditioner rules (a variable-length input that includes event headers, and HMAC code without CAVP validation), so treat the result as a what-if.

## 6. Two-ball cycling

The active ball is swapped when its runtime reaches 35 minutes or its mean speed drops below a threshold (0.05 m/s in the simulator; webcam speeds are in pixels per second). Health tests pause during a swap and restart, with the start-up check, when the fresh ball begins.

## 7. Monte Carlo layer

`MonteCarloSeedFactory` spawns NumPy generators with `SeedSequence.spawn()`, seeds scrambled Sobol samplers, provides block-reseeded streams and writes seed ledgers.

Replicated results (reproduce with `validation/review_checks.py`): At n = 32,768 samples over 200 replicates, the root-mean-square relative error for the 10-dimensional unit-ball volume was 11.0% for plain Monte Carlo and 8.4% for scrambled Sobol, a ratio of 1.32. Single-run ratios ranged from 0.12 to 11.2 (5th to 95th percentile), and Sobol did worse in 45% of single runs. Coverage of nominal 95% intervals over 1,000 trials was 94.7%. The 1.4 claims (an 8.6-fold gain, 96.7% coverage, convergence near 1/N) came from single runs or could not be reproduced.

## 8. Hardware and deployment

- **Arena:** 60 cm x 60 cm with 25 cm walls (`hardware/BOM.md`). The simulator uses a 1.2 m arena.
- **Deflectors** (`hardware/arena_deflectors.scad`): curved corner wedges, an off-center six-lobed scatterer and a camera bracket. The wedges' arena-facing cut is concave, so they focus the ball rather than scatter it.
- **Camera:** lock exposure, white balance and focus with the `v4l2-ctl` commands in `hardware/BOM.md` (the Pi installer installs `v4l-utils` but does not run them). For the sensor-noise channel, prefer an uncompressed format (YUYV) over MJPEG.
- **Raspberry Pi** (`deploy/setup_pi.sh`, `deploy/petrand.service`): a virtual environment, power-on self-tests and a systemd service bound to 127.0.0.1; view it from another computer through an SSH tunnel.

## 9. References used

- NIST SP 800-90B (January 2018), *Recommendation for the Entropy Sources Used for Random Bit Generation*: health tests (§4.3, §4.4), estimators (§6.3) and vetted conditioning (§3.1.5).
- NIST SP 800-90C (2025): §3.2.2.2 on full entropy from external conditioning.
- FIPS 140-3 Implementation Guidance, section D.K, Resolutions 19 and 20 (full entropy and truncated conditioner output).
- NIST SP 800-22 Rev. 1a (April 2010), *A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications*.
- RFC 4231 (HMAC-SHA-224/256/384/512 test vectors) and RFC 7693 (BLAKE2).

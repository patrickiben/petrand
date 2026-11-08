# PetRand

[![tests](https://github.com/patrickiben/petrand/actions/workflows/tests.yml/badge.svg)](https://github.com/patrickiben/petrand/actions/workflows/tests.yml)

An experimental entropy source built from a motorized cat-toy ball bouncing around a small arena, plus the tools to test whether it really produces unpredictable numbers.

> **Status: research prototype.** No physical arena has been built or measured yet. Unless you start it with `--sensor camera` or `--sensor audio`, everything runs on a built-in simulator (the double-click launchers always do), and the console says so. PetRand is not certified by any laboratory. Its health tests follow the designs in NIST SP 800-90B, but the entropy credit per sample is a placeholder until the source has been assessed.

**Play the motion game in your browser:** https://patrickiben.github.io/petrand/ — it runs entirely on your device, and nothing you do is sent anywhere.

## The idea

Lavarand (Silicon Graphics, US Patent 5,732,138) and Cloudflare's lava-lamp wall turned visible, messy motion into random seeds. PetRand asks the same question with a cheap self-propelled toy: a ball that rolls, hits walls and 3D-printed deflectors, and changes direction on its own. An overhead webcam tracks it, and a microphone can hear the impacts.

Candidate sources of unpredictability, none of them measured yet:

| Channel | What is recorded | Open question |
|---|---|---|
| Rebound timing and position | When and where the ball hits a wall or deflector, from the webcam | How much is predictable from the ball's recent path and the toy's own controller? |
| Impact sound | Loudness of each impact, from the microphone | Does it add anything beyond the timing? |
| Camera sensor noise | Low bits of frame-to-frame pixel differences | Does the webcam's processing (compression, noise reduction) leave real sensor noise? |

## How a seed is made

1. **Measure.** A sensor turns one observation into a raw sample, for example the ball's speed and position at a rebound. Raw samples never include the computer's clock.
2. **Check.** Two health tests watch every raw sample. The repetition count test stops the engine at the 41st identical reading in a row. The adaptive proportion test looks at blocks of 512 readings and stops the engine if a block's first value appears 410 times in that block. Both cutoffs assume 0.5 bits per sample, and a 1,024-sample start-up check runs first.
3. **Count.** Each sample adds H bits of credit. H defaults to 0.5, a placeholder; pass an assessed value with `--min-entropy`, which sets both the credit and the health-test cutoffs.
4. **Mix.** When the credit reaches 320 bits (the 256-bit output plus 64), the pooled samples go through HMAC-SHA-256, and its 32-byte output is the seed. The pool is then cleared.
5. **Use.** The seed can start NumPy generators, scrambled Sobol sampling, a passphrase, dice rolls or a raffle draw.

**Simulator mode.** With no hardware attached, step 1 uses a simulated ball whose randomness comes from your operating system's random generator. Those seeds are fine for games, raffles and demos. They are not physical randomness, and the console, command line and seed ledger all mark them as simulated.

## Quick start

```bash
pip install -e .
python3 -m petrand.cli serve --sensor sim        # console at http://127.0.0.1:8080/
python3 -m petrand.cli self-test                 # known-answer self-tests
python3 -m petrand.cli harvest --sensor camera   # webcam; falls back to a labeled simulator
```

Non-technical users can double-click `Launch_PetRand.command` (Mac) or `Launch_PetRand.bat` (Windows); see `QUICKSTART_NON_TECHNICAL.md`.

## Commands

| Command | What it does | What it can't tell you |
|---|---|---|
| `serve` | Console and REST API on 127.0.0.1 | |
| `harvest` | Collects samples and prints seeds | |
| `self-test` | Health tests fire exactly at their cutoffs; HMAC-SHA-256/512 and BLAKE2b match RFC 4231 and RFC 7693 test vectors | Whether the source has entropy |
| `calibrate` | Screens raw samples with SP 800-90B's most-common-value estimate plus simplified collision and Markov screens | It is not an assessment. A SHA-256 counter, which has no entropy to anyone who knows the counter, screens at about 5 bits per sample over 100,000 samples |
| `export-nist` | Writes raw samples, one byte each, plus a metadata file, for NIST's own assessment tool | |
| `sts` | 4 of the 15 SP 800-22 tests (cumulative sums in both directions) on conditioned output | Anything about the source: hashed output passes about 96% of runs by design |
| `conditioning` | Output-entropy bound for HMAC-SHA-256, given an input entropy | Only meaningful with an assessed input entropy |
| `benchmark` | Monte Carlo plumbing checks, with an all-zero-seed control | Anything about the seed: the control passes too |
| `ledger` | Seeds plus a SHA-256 manifest recording the source, settings and version | |
| `spool` | Pre-made seeds for instant use | |

## Getting a real entropy estimate

This package can only screen. An estimate you can defend needs:

1. A built arena, with camera exposure, white balance and focus locked (the `v4l2-ctl` commands are in `hardware/BOM.md`).
2. At least 1,000,000 raw samples per channel: `python3 -m petrand.cli export-nist --sensor camera --samples 1000000 --out raw.bin`
3. NIST's SP800-90B_EntropyAssessment tool, built with `make` and run as `./ea_non_iid raw.bin 8`, plus restart data for `./ea_restart` (exactly 1,000,000 samples: 1,000 restarts x 1,000 samples, in row order).
4. The final assessed value, min(H_r, H_c, H_I), passed back with `--min-entropy`.

`paper/RESEARCH_PLAN.md` lays out the full study design.

## Monte Carlo use

`MonteCarloSeedFactory` spawns independent NumPy generators with `SeedSequence.spawn()` and seeds scrambled Sobol samplers. The seed source does not change Monte Carlo accuracy; any 256-bit seed from a sound generator does as well. What a physical seed adds is a recorded origin (the ledger manifest).

```python
from petrand import PetRandEngine, SyntheticArenaSimulator, MonteCarloSeedFactory

engine = PetRandEngine()                             # 320 bits of credit per seed
sim = SyntheticArenaSimulator()                      # or a hardware sensor
factory = MonteCarloSeedFactory(engine, sensor=sim)
workers = factory.spawn_numpy_generators(n_workers=8)
sobol = factory.get_rqmc_sobol_sampler(dimension=10, scramble=True)
```

## Safety notes

- The REST API has no login. Both servers (the console and Zoomies!) bind to 127.0.0.1 and refuse requests that name another host, API calls a browser marks as coming from another site or another local port, and cross-site POSTs. Each page shows only numbers from its own requests; one tab, a game round or a script cannot change what another tab shows. `--host 0.0.0.0` opens the API to everyone on your network; on a Raspberry Pi, use an SSH tunnel instead.
- Studio passphrases are 6 different words from a 256-word list, about 48 bits. That is fine for a guest Wi-Fi network but too weak for a password-manager master password.
- A raffle receipt lets anyone with the number and the same list (same order and spelling) redo the draw with **Check a draw** on the console, or `EverydaySeedToolkit.check_draw` in Python, and compare receipts. The order comes from a generator seeded with the number; the receipt is the first 16 hex digits of SHA-256 over the number, the names and the order as JSON. It cannot show whether someone drew more than once.
- If a safety check stops the engine, the console says so in plain words and offers **Restart the checks**, which wipes the pool and reruns the start-up check.
- For real keys, use your operating system's random generator (Python's `secrets` module) unless the physical source has been assessed.

## Also included

- **Zoomies!**, an experimental motion game: move your mouse, finger, phone or keys wildly for 4 seconds to unlock a seed. The chaos meter rewards big moves across the play area, many directions and changing speed; wiggles too small to see, steady rhythms and demo rounds never unlock. Human-motion entropy hasn't been assessed: the game shows a placeholder of 0.5 bit for each move big enough to count, and an unlocked round is mixed into the console's pool without credit. It runs in the console at `/zoomies` or offline from `zoomies_game.html`. If your system asks for reduced motion, the play area stays calm: one steady color, no drifting sparks, a still ring.
- **Hardware:** `hardware/BOM.md` and `hardware/arena_deflectors.scad`.
- **Raspberry Pi deployment:** `deploy/setup_pi.sh` and `deploy/petrand.service`.
- **Review checks:** `validation/review_checks.py` reproduces the numbers quoted in `BLUEPRINT.md`.
- **Flicker check:** `validation/flash_check.py` records both pages while driving them as hard as it can and applies the WCAG 2.2 general-flash rule to every 2x2-pixel patch, sliding a 10-degree-field window across the screen; `--self-test` checks it on pages that flash on purpose. It is a screen, not a certificate: it has no red-flash test.

## How this was made

The code and documents were written with AI coding assistants: Google's Gemini (in Antigravity) for the first version, then Anthropic's Claude for the fixes, two independent reviews, the tests and the documentation. What holds them up is the tests in `tests/` and the scripts in `validation/`; no outside laboratory has checked PetRand, and no physical arena has been measured yet.

## License

MIT; see `LICENSE`.

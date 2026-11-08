# PetRand Research Plan

**Working title:** How random is a cat toy? Measuring the min-entropy of a self-propelled ball in a scattering arena

**Status:** design stage. No arena has been built or measured. Everything PetRand has produced so far came from a simulator. Revised 2026-11-08 after a read-only review of the design, its standards citations and the prior work.

## 1. Questions

**Primary.** How many bits of min-entropy per observation do commodity sensors (a webcam and a microphone) capture from a self-propelled toy ball in an arena, and how much of that comes from the ball's motion rather than from the detector and the sensor's own noise?

**Secondary.**
1. Do scattering obstacles raise the entropy per rebound compared with a plain box?
2. Does the toy's own controller add structure, such as turns that repeat after every power-on, that an observer could predict?
3. How many assessed bits per second are practical, and how long does a seed take?
4. With cutoffs set from the assessed entropy, do the health tests catch realistic faults (stuck ball, frozen camera, light change, flat battery)?
5. How much of the per-rebound entropy can an outside observer recover?

**Novelty, honestly.** Modest. Camera-sensor noise is already an assessed entropy source (Sanguinetti et al. 2014; Park et al. 2019; ID Quantique's SP 800-90B documentation), and macroscopic randomizers are known to be predictable once observed (Diaconis et al. 2007; Small and Tse 2012). Visible-motion displays (Lavarand; Cloudflare's lamps, pendulums and mobiles) have no published min-entropy assessment. What would be new: a pre-registered SP 800-90B assessment of a camera-tracked, self-propelled body, with controls that attribute entropy to motion, detector and sensor, and a measured observer experiment.

## 2. Random to whom: the adversary model

The ball moves in plain view and its impacts are audible. Anyone who can see or hear the arena can track the same events. State the adversary up front:

- **Adversary:** knows the design and the software; cannot see inside the enclosure; can record sound and floor vibration outside it; cannot reach the PetRand host or its API.
- **During data collection** the status API's ball coordinates and seed fingerprint are switched off, and the console is not run.
- **Observer experiment (condition D):** an outside microphone (and, in the open condition F, an outside camera) records the same runs. Report how much of C1-C3 the observer recovers, as mutual information or as the drop in assessed entropy conditional on the observer's estimate.
- Report ball-motion entropy and sensor-noise entropy separately, so a reader can tell which part survives an observer.

## 3. Hypotheses and primary outcomes (to pre-register)

- **H1.** Rebound position (C2) with the ball moving (condition D) carries more assessed min-entropy than the detector and launch floor (condition B).
- **H2.** With the ball moving and the arena enclosed, scatterers (D) raise the assessed min-entropy of C2 over the same enclosed box without them (C).
- **H3.** The sensor-noise channel (C4) keeps more assessed min-entropy with uncompressed capture (YUYV) than with MJPEG (E).

Primary outcomes are H1, H2 and H3 on the channels named, with Holm adjustment; every other channel-condition comparison is exploratory. Pilot data are excluded from the confirmatory tests.

## 4. Apparatus

- **Arena:** the size actually built. `hardware/BOM.md` specifies 60 cm x 60 cm with 25 cm walls (the simulator uses 1.2 m). A lid or curtain encloses it for conditions C-E.
- **Scatterers:** the printed corner wedges are concave toward the ball, so they focus rather than disperse. Sinai's results concern dispersing (convex) scatterers and point particles, and a self-propelled ball with friction can follow walls or orbit instead (Dauchot and Demery 2019). For a dispersing condition, print convex posts (for example 40 mm radius cylinders) and describe the wedges as focusing.
- **Balls:** two of the same model; log each ball's ID and battery voltage or mean speed every minute.
- **Camera:** fixed overhead; exposure, white balance and focus locked. Use one capture mode in every condition, chosen from modes both YUYV and MJPEG support on this camera (`v4l2-ctl --list-formats-ext`), for example 640x480 at 30 fps; uncompressed 1080p30 exceeds USB 2.0 bandwidth. Log driver timestamps and dropped frames.
- **Lighting:** steady artificial light only; note mains flicker (50 or 60 Hz).
- **Microphone:** fixed position; log the model and gain.
- **Software:** a tagged PetRand version; every data file gets a metadata sidecar (the `export-nist` JSON is the starting point).

## 5. Raw samples (fix before collecting data)

The current code's raw samples are convenience hashes (speed XOR position) and are not used for the study.

**Rebound definition.** A rebound is a contact with a mapped boundary polygon (walls and scatterers) within one frame of a sign change in the velocity component normal to that boundary. Turns away from boundaries are logged as a separate event type. Wall-following (contact longer than N frames) is logged and excluded from C1 and C2.

| Channel | Raw sample | Mapping for NIST's tool |
|---|---|---|
| C1 rebound interval | frames between consecutive rebounds, from frame timestamps | min(frames, 255); raw counts kept alongside |
| C2 rebound position | position along the boundary at each rebound, in pixels | 256 bins across the boundary length |
| C3 impact loudness | peak level of each detected impact | 256 levels |
| C4 sensor noise | low bit of the difference between non-overlapping frame pairs (2k, 2k+1) of the raw Y plane, in a fixed patch outside the arena walls, every 4th pixel in each direction | 1 bit per sample (binary data: APT window 1,024 and the binary-only estimators apply) |

**Credit comes from one primary noise source** (SP 800-90B §3.1.6). For rebounds, form one composite symbol per rebound (for example the 4 most variable bits of C1 and of C2), assess it, and credit only that. C3 and C4 can be mixed into the conditioner without credit, or assessed and credited in a separate configuration.

## 6. Conditions

| Code | Setup | Purpose |
|---|---|---|
| A | Empty arena, lights on | Sensor and detector noise floor |
| B0 | Ball present, motor off | False rebound detections per hour and centroid jitter |
| B | Motor off; an automated gate releases the ball down a fixed ramp into the same wall point, 2,000 launches | Detector and launch-jitter floor for C1-C3 (C1 = frames from gate trigger to first contact) |
| C | Ball moving, no scatterers, enclosed | Same as D except the scatterers |
| D | Ball moving, scatterers, enclosed | Main condition |
| E | As D, MJPEG instead of YUYV | Effect of compression on C4 |
| F | As D, open (optional) | Observability contrast |

Run conditions in randomized order within paired days, with the same balls, lighting and capture mode.

## 7. Data volume and time

- Measure the rebound rate r in the pilot. 1,000,000 rebounds take about 11.6/r days (3.9 days at 3 per second, 23 days at 0.5 per second), with about 476/r ball swaps at 35 minutes each. Automate swaps or plan collection around them.
- SP 800-90B allows the 1,000,000 samples to be joined from runs of at least 1,000 consecutive samples, which helps across ball swaps.
- Full-size assessments: condition D on the composite rebound symbol, and C4 in every condition (it reaches 1,000,000 samples in seconds).
- Comparison data: at least 6 sessions per compared condition, each of equal size (100,000 rebounds for C1-C3; 1,000,000 samples for C4).

## 8. Analysis

**Estimates.** Run NIST's SP800-90B_EntropyAssessment tool (non-IID track) on each channel and condition, and report every estimator's output. Pre-register a model-based estimate H_submitter for each channel (C2: the spread of rebound position given the previous rebound, from pilot data; C4: a photon-transfer noise model). The reported value is min(H_original, n x H_bitstring, H_submitter), plus the restart terms min(H_r, H_c) where restart data exist; a tool-only value is labeled as such. The IID tests are needed only to support an IID claim.

**Restart testing.** C4: a restart is a scripted camera power-cycle that re-applies and logs the locked controls, then runs the start-up test; collect 1,000 x 1,000 and run `ea_restart`. C1-C2: a restart is ball power-on at a marked pose with an automated release; collect a 100 x 100 matrix and label it restart screening, since the full matrix is impractical for rebound channels.

**Uncertainty.** The unit of replication is a session (one condition, one day block). Assess each session separately and compare conditions across sessions with a pre-specified paired test. Blocks within a single run share balls, battery cycles and lighting, so they are pseudo-replicates; use them only as a sensitivity check, and note that 10 blocks of a 1,000,000-sample run are below the tool's required size.

**Dynamics.** Primary: ensemble divergence. From restart launches at the marked pose, compute the median distance between trajectories against time and against rebound count; the prediction horizon is when the median spread reaches half the arena width. Secondary: the largest Lyapunov exponent by the Rosenstein method, reported with its caveats (motor-driven, piecewise motion, about 30 fps sampling, and pixel noise that alone produces apparent divergence).

**The toy's controller.** Record 50 power-ons of each ball on an open floor and extract turn times. Test whether turn sequences repeat across power-ons (exact matches of the first 20 inter-turn intervals within one frame; cross-correlation of turn-time series) and whether the turn hazard depends on time since power-on or since the last impact.

**Health tests.** Set H to the assessed value and recompute the cutoffs. At alpha = 2^-20 a million samples yield about one expected alarm, so check calibration at alpha = 2^-8 instead. Add developer-defined tests (SP 800-90B §4.3): a liveness test (no rebound for more than T seconds), a frozen-frame test and a lighting-change test, since a stuck or removed ball produces no rebound samples for the RCT and APT to see. Inject each fault and record how long detection takes.

**Stationarity.** Compare the binned C1 and C2 distributions between the first and last thirds of each battery cycle, and between the two balls, with a pre-specified chi-square test. If they drift, power the ball continuously or collect only in the stable part of the cycle.

**Negative controls.** Assess the simulator and a fixed-key pseudorandom "fake ball" with the same pipeline. Both will score well, which is why the physical conditions, not the scores alone, carry the argument. Report the all-zero-seed Monte Carlo control from `petrand benchmark`.

## 9. Claims to avoid

- Certification or compliance of any kind. Say "assessed with NIST's SP 800-90B tool" only for what was run.
- "Full entropy". It needs an assessed input of at least the output length plus 64 bits (FIPS 140-3 IG D.K Resolution 19; SP 800-90C §3.2.2.2) and a conditioner input of fixed length drawn from the noise source alone, which PetRand's pool does not provide.
- Monte Carlo accuracy gains. The seed source does not change accuracy; a physical seed offers a recorded origin.
- "Quantum", unless an experiment shows a channel is dominated by photon shot noise.

## 10. Threats to validity

- Detector artifacts: tracking jitter, compression, dropped frames, timestamp rounding, a failed exposure lock.
- The toy's firmware may replay a fixed turn schedule.
- Mains flicker and vibration add periodic patterns.
- Battery drain makes the source non-stationary.
- Many channels and conditions: only the pre-registered primary outcomes are confirmatory.
- Observability (section 2).

## 11. Paper outline

1. Introduction: randomness you can see; the Lavarand lineage; why a toy; contributions.
2. Background: min-entropy, SP 800-90B assessment and health tests, why statistical tests cannot certify randomness.
3. Apparatus and a model of the noise sources.
4. Methods: channels, conditions, data collection and the pre-registered analysis.
5. Results: entropy per channel and condition, attribution, the observer experiment, dynamics, health tests, throughput.
6. Discussion: what is random and to whom; practicality; teaching value.
7. Limitations.
8. Data and code availability: tagged PetRand release, raw data deposit, metadata.

## 12. Venues (from each venue's own pages; check again before submitting)

| Venue | Fit | Terms |
|---|---|---|
| arXiv (physics.ins-det or physics.data-an; cross-list cs.CR, physics.ed-ph, nlin.CD) | First step; establishes priority | Free; not peer reviewed |
| American Journal of Physics | A lab or apparatus paper on measuring min-entropy and chaos with a toy | Regular articles about six journal pages; no publication fee; optional open access fee |
| Journal of Statistics and Data Science Education | The statistics framing: min-entropy versus statistical tests, pseudo-replication, the zero-seed control ("Datasets and Stories" article type) | Open access with no fees |
| European Journal of Physics | A university teaching paper on entropy measurement, sensor noise and chaos | Typically up to 4,000 words; free on the subscription route |
| IACR Cryptology ePrint Archive | The security version: assessment, adversary model, health tests | Free; not refereed (a refereed venue such as TCHES is a stretch) |

## 13. Steps

1. Build the arena, print convex posts as well as the wedges, and lock the camera.
2. Add a `record` mode to PetRand that writes C1-C4 to separate files with metadata, using the section 5 definitions.
3. Pilot each condition at screening size; measure r; check the detectors; then pre-register on OSF.
4. Collect the main data (condition D first), with the API's coordinates and fingerprint switched off.
5. Run the NIST tool, analyze, write.

## 14. References (checked 2026-11-08 against DOIs or official pages)

- Turan MS, Barker E, Kelsey J, McKay K, Baish M, Boyle M. *Recommendation for the Entropy Sources Used for Random Bit Generation.* NIST SP 800-90B, January 2018. doi:10.6028/NIST.SP.800-90B
- Rukhin A, et al. *A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications.* NIST SP 800-22 Rev. 1a, 2010.
- Kelsey J, McKay KA, Sonmez Turan M. Predictive models for min-entropy estimation. CHES 2015, LNCS 9293, pp. 373-392.
- Zhu et al. On the SP 800-90B non-IID estimators. IACR Transactions on Symmetric Cryptology 2017(3):151-168. doi:10.46586/tosc.v2017.i3.151-168
- Rosenstein MT, Collins JJ, De Luca CJ. A practical method for calculating largest Lyapunov exponents from small data sets. Physica D 65(1-2):117-134, 1993.
- Sinai YaG. Dynamical systems with elastic reflections: ergodic properties of dispersing billiards. Russian Mathematical Surveys 25(2):137-189, 1970.
- Noll LC, Mende RG, Sisodiya S. Method for seeding a pseudo-random number generator with a cryptographic hash of a digitization of a chaotic system. US Patent 5,732,138 (filed 1996, issued 1998).
- Cloudflare Blog. LavaRand in production: the nitty-gritty technical details. 6 November 2017.
- Rubin CD, Valenta L, Meunier T. Harnessing chaos in Cloudflare offices. Cloudflare Blog, 8 March 2024.
- Sanguinetti B, Martin A, Zbinden H, Gisin N. Quantum random number generation on a mobile phone. Physical Review X 4:031056, 2014.
- Park BK, et al. Practical true random number generator using CMOS image sensor dark noise. IEEE Access 7:91407-91413, 2019.
- ID Quantique. SP800-90B Non-Proprietary Public Use Document: Quantis QRNG IID chips, v0.4, 2023 (posted by NIST CMVP).
- Diaconis P, Holmes S, Montgomery R. Dynamical bias in the coin toss. SIAM Review 49(2):211-235, 2007.
- Small M, Tse CK. Predicting the outcome of roulette. Chaos 22(3):033150, 2012.
- Bartos F, et al. Fair coins tend to land on the same side they started: evidence from 350,757 flips. Journal of the American Statistical Association 120(552):2118-2127, 2025.
- Zhou Q, Liao X, Wong K-W, Hu Y, Xiao D. True random number generator based on mouse movement and chaotic hash function. Information Sciences 179(19):3442-3450, 2009.
- Hennebert C, Hossayni H, Lauradoux C. Entropy harvesting from physical sensors. WiSec 2013, pp. 149-154.
- Dauchot O, Demery V. Dynamics of a self-propelled particle in a harmonic trap. Physical Review Letters 122:068002, 2019.

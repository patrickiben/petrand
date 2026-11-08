# PetRand: Physical Bill of Materials (BOM) & Build Guide

This document lists the parts, hardware settings and enclosure guidelines for building the physical PetRand arena: an experimental entropy source that has not been measured or evaluated by any lab.

---

## 1. System Bill of Materials

| Component | Part / Model | Purpose / Rationale | Est. Cost | Vendor / Source |
| :--- | :--- | :--- | :--- | :--- |
| **Active Toy 1** | PetGravity Speedy Sprint Ball (Orange) | Self-propelled ball with an internal motor and built-in direction changes (their pattern is not yet known) | ~$15 | Chewy / Amazon / PetGravity |
| **Active Toy 2** | PetGravity Speedy Sprint Ball (Blue) | Lifecycle rotation partner; charges while Toy 1 runs; provides distinct optical HSV signature | ~$15 | Chewy / Amazon / PetGravity |
| **Charging Station** | Dual-port 5V USB-C wall adapter + 2x USB-A to C cables | Continuous battery cycling; 30-40 min recharge cycle | ~$12 | Anker / Generic |
| **Overhead Camera** | Logitech C920x HD Pro (or Raspberry Pi Camera Module 3 Wide) | 1080p at 30/60 fps; lock exposure, white balance and focus; its sensor-noise floor is still to be measured | ~$60 | Logitech / Raspberry Pi |
| **Enclosure Arena** | 60cm × 60cm × 25cm high walled acrylic or wooden box | Confines ball dynamics; prevents escape; provides rigid acoustic impact boundaries | ~$25 | Hardware Store / DIY / IKEA |
| **Arena Surface Mat** | 60cm × 60cm Matte Black Neoprene / Desk Mat | Eliminates optical floor reflections/glare; maximizes HSV color segmentation delta for orange/blue balls | ~$15 | Amazon / Gaming Desk Mat |
| **Chaos Deflectors** | 4x Corner Wedges + 1x Central Fluted Scatterer | Meant to break up periodic paths. The corner wedges are concave toward the ball (they focus); add convex posts for a dispersing condition. Whether either raises entropy is a research question | ~$8 (filament) | 3D Printed via `hardware/arena_deflectors.scad` |
| **Host Controller** | Raspberry Pi 4B (4GB) or Raspberry Pi 5 (or Mac / x86 mini-PC) | Headless edge appliance executing `petrand serve` HTTP daemon, NIST health monitoring, and entropy pool | ~$55 - $75 | Raspberry Pi Foundation |
| **Overhead Arm** | Adjustable boom arm clamp or extruded 2020 aluminum bracket | Suspends webcam 50-70cm directly plumb above center of arena | ~$20 | Amazon (Neewer / SmallRig) |
| **Total Build Cost** | | Complete turnkey appliance | **~$225 - $250** | |

---

## 2. Enclosure Construction & Alignment

### A. Arena Dimensions & Aspect Ratio
* **Footprint**: A square arena of $60\text{ cm} \times 60\text{ cm}$ ($24" \times 24"$) balances travel time against collision frequency (the collision rate has not been measured yet).
* **Wall Height**: Minimum $20\text{ cm}$ ($8"$). The PetGravity ball diameter is $50\text{ mm}$ ($2.0"$); high-speed impacts with corner wedges can launch the ball vertically up to $12\text{ cm}$.

### B. Optical Surface & Illumination
* **Base Floor**: Matte black or neutral charcoal EVA foam / neoprene. Avoid glossy white acrylic or polished wood, which reflect room lights and corrupt contour centroid extraction.
* **Lighting**: Diffuse, flicker-free LED ring or bar light mounted above the rim. **Never use fluorescent or phase-dimmed AC bulbs** that oscillate at 100/120 Hz, as this introduces periodic 60 Hz hum into the frame-diff photon noise channel.

### C. Sinai Billiard Deflectors
* Print 4 corner wedges from [`hardware/arena_deflectors.scad`](arena_deflectors.scad) using PETG or PLA with 25% infill.
* Fasten wedges into all four corners using double-sided mounting foam or M4 screws.
* Position the fluted central scatterer slightly off-center ($x=0.48, y=0.52$) to introduce asymmetric boundary breaking. Fill the central hollow cavity with sand or metal shot to prevent the ball from dislodging it.

---

## 3. Camera Setup & Manual Exposure Tuning

To prevent automatic camera firmware algorithms from contaminating the physical entropy extraction:

1. **Disable Auto-Exposure**: Auto-exposure adjusts gain and shutter speed when the brightly colored ball moves, inducing periodic brightness fluctuations. Set exposure to a fixed manual shutter (e.g. 1/120s or 1/60s).
2. **Disable Auto-White-Balance (AWB)**: Lock white balance to a fixed color temperature (e.g. 5000K) so HSV threshold values (`[8, 120, 120]` to `[25, 255, 255]` for orange; `[95, 120, 100]` to `[125, 255, 255]` for blue) remain invariant over hours of runtime.
3. **Lock Auto-Focus**: Manually focus on the floor surface and lock lens stepper motors.

Linux `v4l2-ctl` commands to enforce these settings on `/dev/video0`:
```bash
# Disable auto exposure (1 = manual mode)
v4l2-ctl -d /dev/video0 -c auto_exposure=1
v4l2-ctl -d /dev/video0 -c exposure_time_absolute=150

# Disable auto white balance
v4l2-ctl -d /dev/video0 -c white_balance_temperature_auto=0
v4l2-ctl -d /dev/video0 -c white_balance_temperature=5000

# Lock focus
v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0
v4l2-ctl -d /dev/video0 -c focus_absolute=50
```

---

## 4. Two-Ball Operational Lifecycle SOP

1. **Startup**: Place Ball A (Orange) in arena turned ON. Connect Ball B (Blue) to USB-C charger.
2. **Active Runtime**: Ball A operates for 35 minutes. PetRand's internal battery gauge tracks active time.
3. **Depletion Alert**: At $t = 35\text{ min}$, the `/api/v1/status` endpoint transitions `lifecycle.status` to `"low_battery"` and triggers an acoustic or visual alert on the dashboard.
4. **Swap Protocol**:
   - The user or mechanical handler swaps Ball A (to charger) and places fully charged Ball B into the arena.
   - Trigger `/api/v1/swap` via Web Dashboard or CLI.
   - The engine safely pauses NIST health monitoring for 2.0 seconds during the physical transfer, resets the consecutive counter window, updates the active ball tracking profile, and resumes extraction without false-positive RCT or APT alarms.

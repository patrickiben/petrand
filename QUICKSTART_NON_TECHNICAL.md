# PetRand Quick Start (no coding needed)

PetRand turns motion into random numbers you can use for passphrases, dice, coin flips, raffles and spreadsheets.

**One thing to know first:** the double-click launchers always start PetRand in simulator mode; a webcam is only used when someone starts it with `--sensor camera`. In that mode the numbers come from your computer's own random generator, run through PetRand's checks and mixing. They are good for games, raffles and dice, but they are not physical randomness. The badge at the top of the console always shows which mode you're in.

## 1. Fastest start: the Zoomies! game

Play it online at https://patrickiben.github.io/petrand/, or offline:

1. Open the `petrand` folder.
2. Double-click `zoomies_game.html`. It opens in Safari, Chrome, Firefox or Edge, with no installation.
3. Press **GO!** and move your mouse, trackpad or finger wildly for 4 seconds: spirals, zig-zags, sudden stops. On a phone, tap **Phone motion** first and move the phone around.
4. Big moves across the whole play area count; tiny wiggles don't, and a steady back-and-forth rhythm gets rejected. Get the chaos meter past 75% to unlock a seed.
5. **Watch a demo round** shows how scoring works; demos never make a seed. No mouse? Press letter keys during a round: each key jumps to its own spot.

This game is an experiment. Nobody has measured how much randomness hand motion really provides.

## 2. The full console

**Before you start (once):** the console needs Python 3 with NumPy and SciPy. Install Python 3 from python.org, then in a terminal, inside the `petrand` folder, run `pip install -e .` (or ask someone to do it for you). The game in section 1 needs none of this.

- **Mac:** double-click `Launch_PetRand.command`.
- **Windows:** double-click `Launch_PetRand.bat`.

Your browser opens `http://127.0.0.1:8080/`. The console runs only on your computer.

## 3. What you can do in the console

- **Get a random number:** press the big **New random number** button. You get a fresh 256-bit seed with a **Copy** button.
- **Passphrase and PIN:** each seed becomes a 6-word passphrase and a 6-digit PIN. A 6-word passphrase here is fine for something like a guest Wi-Fi network. For a password manager or other important accounts, use the passphrase generator built into your password manager instead.
- **Dice, coin and lottery:** a coin flip, D20, D100 and 6/49 lottery numbers come with every seed.
- **Pick a winner:** type names separated by commas and press **Draw**. The receipt lets anyone with the seed and the list check the shuffle, but it can't show whether someone drew more than once.
- **Spreadsheet download:** **Download 500 rows (.csv)** opens in Excel, Numbers or Google Sheets.
- **How your number was made:** four steps under the results show how many readings went in, whether the health checks passed, how much they counted for, and how they were mixed.
- **Ball colors:** if your toy balls aren't orange and blue, pick their colors in the Arena card.

## 4. Reading the console

| On screen | What it means |
|---|---|
| **Source** badge | *Simulator*: numbers come from your computer's random generator. *Webcam* or *Microphone (live)*: numbers come from the arena, though how much randomness it gives hasn't been measured yet. |
| **Healthy / Warming up / Stopped** badge | Whether the safety checks are passing. Warming up means the first 1,024 readings are still being checked. |
| **Under the hood → Repeat guard, e.g. 1 / 41** | Stuck-sensor guard. If the same reading repeats 41 times in a row, PetRand stops instead of giving you bad numbers. |
| **Under the hood → Balance guard, e.g. 12 / 410** | Imbalance guard. In each block of 512 readings, if the block's first reading shows up 410 times, PetRand stops. |
| **Under the hood → Credit per sample** | How much randomness each reading is counted as: a deliberately cautious placeholder until the arena is measured. |
| **Under the hood → Toward the next seed, e.g. 120 / 320 bits** | How full the tank is. At 320 bits, a fresh seed is ready. |

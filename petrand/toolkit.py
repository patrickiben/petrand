"""
petrand.toolkit
===============
Everyday Randomness Toolkit for Non-Technical Users.
Deterministically expands a 256-bit physical chaos seed into immediately useful,
human-friendly outputs:
  1. Memorable 6-Word Passphrases & 6-Digit PINs (for password managers / Wi-Fi / keys)
  2. Fair Raffle Winner Picker & Group Shuffler (with verifiable SHA-256 audit receipt)
  3. Tabletop RPG, Board Game, Coin Flip & Lottery Numbers (d4, d6, d8, d10, d12, d20, d100)
  4. Ready-to-Open Excel / Google Sheets CSV Spreadsheets (Uniform, Normal, Custom Integer)
"""

import csv
import io
import math
import struct
import hashlib
import json
from typing import List, Dict, Any
import numpy as np


# 256 vivid, distinct, easy-to-spell English words. Six different words drawn from them carry about
# 48 bits (256 x 255 x ... x 251 choices), enough for a guest Wi-Fi password, not a master password.
EFF_VIVID_WORDS_256: List[str] = [
    "acorn", "agate", "almond", "ALPINE".lower(), "amber", "anchor", "anvil", "apricot",
    "arcade", "archer", "arctic", "armOR".lower(), "arrow", "aspen", "atlas", "aurora",
    "autumn", "avalon", "badger", "basil", "bamboo", "banjo", "banyan", "barley",
    "basalt", "beacon", "beaver", "belfry", "bison", "blizzard", "blossom", "bobcat",
    "bonfire", "bonsai", "boulder", "boxcar", "bramble", "breeze", "bridge", "bronze",
    "brook", "buckle", "cabana", "cactus", "camera", "candle", "canopy", "canyon",
    "caravan", "carbon", "carpet", "cascade", "castle", "cedar", "celeste", "chalet",
    "chariot", "cherry", "chestnut", "chimney", "cipher", "citadel", "citrus", "cliff",
    "clover", "cobalt", "comet", "compass", "condor", "copper", "coral", "cosmos",
    "cotton", "cougar", "crater", "crimson", "crystal", "cypress", "dagger", "dahlia",
    "dolphin", "domino", "dragon", "drift", "dynamo", "eclipse", "elm", "emerald",
    "ember", "empire", "enigma", "equinox", "estuary", "falcon", "fathom", "feather",
    "fennel", "ferry", "fiddle", "fiesta", "fjord", "flannel", "flint", "forest",
    "fossil", "fountain", "fresco", "galaxy", "galleon", "garden", "garnet", "gazebo",
    "geyser", "glacier", "glider", "gondola", "granite", "Grotto".lower(), "guitar", "harbor",
    "harvest", "aviary", "hazel", "heather", "helmet", "heron", "hickory", "horizon",
    "hornet", "iceberg", "igloo", "indigo", "island", "ivory", "jacket", "jade",
    "jaguar", "jasmine", "jasper", "javelin", "jester", "jubilee", "jungle", "juniper",
    "kayak", "kestrel", "kettle", "kimono", "kingdom", "lantern", "lapel", "larch",
    "lagoon", "laurel", "lava", "legend", "lemon", "leopard", "liberty", "lilac",
    "linen", "locket", "lotus", "magnet", "magpie", "mallet", "mammoth", "mango",
    "mantis", "maple", "marble", "mariner", "meadow", "melody", "mercury", "meteor",
    "mirage", "monarch", "monsoon", "mosaic", "muffin", "mustang", "nebula", "nectar",
    "nickel", "nimbus", "nomad", "nutmeg", "oasis", "obsidian", "ocelot", "octave",
    "odyssey", "olive", "onyx", "opal", "orange", "allium", "orchid", "oriole",
    "osprey", "otter", "pagoda", "palace", "panther", "papaya", "paradox", "parlor",
    "pebble", "pelican", "pendulum", "peony", "pepper", "phantom", "phoenix", "picnic",
    "pioneer", "planet", "plaza", "plum", "pocket", "polaris", "poncho", "poplar",
    "prairie", "prism", "pulsar", "pumpkin", "pyramid", "quartz", "quiver", "ranger",
    "radar", "radish", "walnut", "raven", "redwood", "reef", "ribbon", "ripple",
    "river", "robin", "rocket", "rosewood", "ruby", "saffron", "sahara", "salmon",
    "sandal", "sapphire", "saturn", "savanna", "scarlet", "scepter", "sequoia", "shadow",
]


class EverydaySeedToolkit:
    """
    Converts a 256-bit hexadecimal seed into everyday, human-friendly outputs
    for non-technical users, gamers, teachers, and spreadsheet analysts.
    """

    @staticmethod
    def _normalize_seed(seed_hex: str) -> bytes:
        clean = (seed_hex or "").strip()
        try:
            raw = bytes.fromhex(clean)
            if len(raw) >= 16:
                return hashlib.blake2b(raw, digest_size=32, person=b"PetRandToolkitV1").digest()
        except ValueError:
            pass
        return hashlib.blake2b(clean.encode("utf-8"), digest_size=32, person=b"PetRandToolkitV1").digest()

    @classmethod
    def _make_rng(cls, seed_hex: str, domain: str) -> np.random.Generator:
        seed_bytes = cls._normalize_seed(seed_hex)
        derived = hashlib.blake2b(
            seed_bytes,
            digest_size=32,
            key=domain.encode("utf-8")[:32],
        ).digest()
        u32 = list(struct.unpack("<8I", derived))
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(u32)))

    @classmethod
    def generate_passphrase(cls, seed_hex: str, num_words: int = 6) -> Dict[str, Any]:
        """A passphrase of different words and a 6-digit PIN (000000-999999) from the seed."""
        rng = cls._make_rng(seed_hex, "passphrase")
        indices = rng.choice(len(EFF_VIVID_WORDS_256), size=max(4, min(12, num_words)), replace=False)
        words = [EFF_VIVID_WORDS_256[int(i)] for i in indices]
        pin_num = int(rng.integers(0, 1000000))
        return {
            "passphrase": "-".join(words),
            "words": words,
            "pin_6_digit": f"{pin_num:06d}",
        }

    @classmethod
    def roll_tabletop(cls, seed_hex: str) -> Dict[str, Any]:
        """Generates common tabletop RPG, board game, coin flip, and lottery rolls."""
        rng = cls._make_rng(seed_hex, "tabletop")
        coin = "HEADS" if int(rng.integers(0, 2)) == 1 else "TAILS"
        lottery = sorted([int(x) for x in rng.choice(np.arange(1, 50), size=6, replace=False)])
        return {
            "coin_flip": coin,
            "d4": int(rng.integers(1, 5)),
            "d6": int(rng.integers(1, 7)),
            "d8": int(rng.integers(1, 9)),
            "d10": int(rng.integers(1, 11)),
            "d12": int(rng.integers(1, 13)),
            "d20": int(rng.integers(1, 21)),
            "d100": int(rng.integers(1, 101)),
            "6d6": [int(x) for x in rng.integers(1, 7, size=6)],
            "lottery_6_of_49": lottery,
        }

    @classmethod
    def pick_and_shuffle(
        cls,
        seed_hex: str,
        items: List[str],
        winners_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Shuffles the names with a generator seeded from the seed and picks `winners_count`
        winners. Anyone with the same seed and the same list (same order and spelling) gets
        the same order: check_draw() recomputes it. The receipt is the first 16 hex digits of
        SHA-256 over {"seed", "names", "order"} as compact JSON, so names cannot run together.
        """
        clean_items = [str(x).strip() for x in items if str(x).strip()]
        if not clean_items:
            raise ValueError("no names to draw from")

        rng = cls._make_rng(seed_hex, "raffle_shuffle")
        perm = rng.permutation(len(clean_items))
        shuffled = [clean_items[int(i)] for i in perm]
        k = max(1, min(len(shuffled), int(winners_count)))
        winners = shuffled[:k]

        receipt_payload = json.dumps({"seed": seed_hex.strip().lower(), "names": clean_items, "order": shuffled},
                                     separators=(",", ":"), ensure_ascii=False)
        receipt_hash = hashlib.sha256(receipt_payload.encode("utf-8")).hexdigest()[:16]

        return {
            "total_items": len(clean_items),
            "winners": winners,
            "winner_primary": winners[0],
            "shuffled_order": shuffled,
            "verification_receipt": f"VERIFY-{receipt_hash.upper()}",
        }

    @classmethod
    def check_draw(cls, seed_hex: str, items: List[str], receipt: str = "") -> Dict[str, Any]:
        """Recompute a draw from its seed and list; say whether a given receipt matches."""
        res = cls.pick_and_shuffle(seed_hex, items, winners_count=1)
        given = (receipt or "").strip().upper()
        if given and not given.startswith("VERIFY-"):
            given = "VERIFY-" + given
        return {"shuffled_order": res["shuffled_order"], "winner_primary": res["winner_primary"],
                "verification_receipt": res["verification_receipt"],
                "matches": (given == res["verification_receipt"]) if given else None}

    @classmethod
    def evaluate_all(
        cls,
        seed_hex: str,
        items: List[str] = None,
        winners_count: int = 1,
    ) -> Dict[str, Any]:
        """Returns the complete everyday randomness bundle for a given 256-bit seed."""
        norm_hex = cls._normalize_seed(seed_hex).hex()
        return {
            "seed_hex": seed_hex if len((seed_hex or "").strip()) == 64 else norm_hex,
            "passphrase_bundle": cls.generate_passphrase(seed_hex, num_words=6),
            "tabletop_bundle": cls.roll_tabletop(seed_hex),
            "raffle_bundle": cls.pick_and_shuffle(
                seed_hex,
                items or ["Alice", "Bob", "Charlie", "Diana", "Echo"],
                winners_count=winners_count,
            ),
        }

    @classmethod
    def generate_csv_spreadsheet(
        cls,
        seed_hex: str,
        rows: int = 500,
        int_min: int = 1,
        int_max: int = 100,
    ) -> str:
        """
        Generates a CSV string ready to open directly in Microsoft Excel, Apple Numbers,
        or Google Sheets, containing Uniform [0,1), Standard Normal N(0,1),
        Custom Integer Range, Coin Flip, and D20 columns.
        """
        n = max(1, min(10_000, int(rows)))
        lo = min(int(int_min), int(int_max))
        hi = max(int(int_min), int(int_max))
        rng = cls._make_rng(seed_hex, "csv_export")

        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([
            "row_index",
            "uniform_0_to_1",
            "normal_bell_curve_z",
            f"integer_{lo}_to_{hi}",
            "coin_flip",
            "d20_roll",
            "source_seed_prefix",
        ])

        u_vals = rng.random(n)
        z_vals = rng.standard_normal(n)
        int_vals = rng.integers(lo, hi + 1, size=n)
        coins = rng.integers(0, 2, size=n)
        d20s = rng.integers(1, 21, size=n)
        prefix = (seed_hex or "00000000")[:16]

        for i in range(n):
            writer.writerow([
                i + 1,
                f"{float(u_vals[i]):.8f}",
                f"{float(z_vals[i]):.6f}",
                int(int_vals[i]),
                "HEADS" if int(coins[i]) == 1 else "TAILS",
                int(d20s[i]),
                prefix,
            ])

        return out.getvalue()

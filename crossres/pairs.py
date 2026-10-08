"""segment pairs for cross-resolution pretraining: native ~9.36 um scan + ~2.4 um scan of the same surface.

every url below was confirmed by an S3 listing on 2026-09-29 except where `verify` is noted.
the 1.129 um volumes exist for most of these segments: they must never be downloaded.
"""
from __future__ import annotations

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
TRAIN_FRAME_UM = 9.362          # every training zarr in ves_zarrs2 is on this grid (0841: 9.366)
TRAIN_FRAME_DEPTH = 28
FORBIDDEN_PREFIX = "1.129um"

_P0139_LOW = "9.362um-1.2m-113keV-volume-20250728140407.zarr"
_P0139_HIGH = "2.399um-0.22m-78keV-volume-20260102150214.zarr"


def _p0139(name: str, zid: int, folder: str) -> dict:
    return {
        "name": name, "zid": zid, "scroll": "PHerc0139", "role": "train",
        "segment": f"PHerc0139/segments/{folder}",
        "low": _P0139_LOW, "low_um": 9.362, "high": _P0139_HIGH, "high_um": 2.399,
        # the training zarr is the low volume's level 0, unresampled
        "frame": "identity",
    }


PAIRS = [
    _p0139("w044", 20260115000000, "20260115000000-w044_2026011522"),
    _p0139("w059", 20250223000000, "20250223000000-w059_2025022312"),
    _p0139("w030", 20250108000005, "20250108000005-w030_2025010818"),
    _p0139("w043", 20260112000000, "20260112000000-w043_2026011217"),
    _p0139("w045", 20260126000000, "20260126000000-w045_2026012619"),
    _p0139("w040", 20250831000000, "20250831000000-w040_2025083102"),
    _p0139("w041", 20260108000000, "20260108000000-w041_2026010816"),
    _p0139("w039", 20260302000000, "20260302000000-w039_2026030210"),
    _p0139("w035", 20260317000000, "20260317000000-w035_2026031718"),
    {
        "name": "500P2_front", "zid": 20250628074500, "scroll": "PHerc0500P2", "role": "train",
        "segment": "PHerc0500P2/segments/20250628074500-500P2_front",
        "low": "9.362um-1.2m-113keV-volume-20250820143440.zarr", "low_um": 9.362,
        "high": "2.215um-0.4m-111keV-volume-20250526151718.zarr", "high_um": 2.215,
        # same 1.2 m propagation as the low scan, 2.17x finer: an optional alternative target
        "alt_high": "4.317um-1.2m-111keV-volume-20250528085330.zarr", "alt_high_um": 4.317,
        "frame": "identity",
    },
    {
        "name": "seg46527", "zid": 20260226000000, "scroll": "PHerc0814", "role": "train",
        "segment": "PHerc0814/segments/20260226000000-46527_2um_try2",
        "low": "9.362um-1.2m-113keV-volume-20250804134230.zarr", "low_um": 9.362,
        "high": "2.399um-0.22m-78keV-volume-20260309142202.zarr", "high_um": 2.399,
        "frame": "identity",
    },
    {
        "name": "p9b_487", "zid": 20250919125754, "scroll": "PHerc0009B",
        # held out like 0841 so both holdout metrics stay clean; --include-holdouts adds footprint tiles
        "role": "holdout",
        "segment": "PHerc0009B/segments/20250919125754-auto_grown_20250919055754487_inp_hr",
        # verify: the listing truncated after "8.64um-1.2m"; this name is the one assemble_training_segments uses
        "low": "8.64um-1.2m-116keV-volume-20250521125136.zarr", "low_um": 8.64,
        "high": "2.401um-0.35m-77keV-volume-20250820154339.zarr", "high_um": 2.401,
        # the training zarr is the 8.64 um volume resampled in XYZ onto the 9.362 um / 28-layer frame
        "frame": "resample",
    },
    {
        "name": "p841", "zid": 20260221022814, "scroll": "PHerc0841", "role": "holdout",
        "segment": "PHerc0841/segments/20260221022814-auto_grown_20260220174252405",
        # verify: listing truncated after "...20250821151531.za"
        "low": "9.366um-1.2m-113keV-volume-20250821151531.zarr", "low_um": 9.366,
        "high": "2.403um-0.22m-77keV-volume-20260319124803.zarr", "high_um": 2.403,
        # the training zarr is level 0 cropped to (0, 3760, 0, 4900), i.e. the whole current volume
        "frame": "identity",
    },
]


def url(pair: dict, which: str, level: int | None = None) -> str:
    name = pair[which]
    if name.startswith(FORBIDDEN_PREFIX):
        raise ValueError(f"{pair['name']}: the 1.129 um volume must never be downloaded")
    base = f"{BUCKET}/{pair['segment']}/surface-volumes/{name}"
    return base if level is None else f"{base}/{level}"


def high_to_frame(pair: dict) -> float:
    """pixels of the 2.4 um level 0 per training-frame pixel (the xy scale the registration should find)."""
    frame_um = pair["low_um"] if pair["frame"] == "identity" else TRAIN_FRAME_UM
    return frame_um / pair["high_um"]


def high_to_low(pair: dict) -> float:
    """pixels of the 2.4 um level 0 per native low-scan pixel."""
    return pair["low_um"] / pair["high_um"]


def by_role(*roles: str) -> list[dict]:
    return [pair for pair in PAIRS if pair["role"] in roles]


if __name__ == "__main__":
    print(f"{'name':12s} {'zid':>15s} {'role':8s} {'low_um':>7s} {'high_um':>7s} "
          f"{'high/low':>8s} {'high/frame':>10s} {'level2 um':>9s} {'level2->frame':>13s}")
    for pair in PAIRS:
        frame_um = pair["low_um"] if pair["frame"] == "identity" else TRAIN_FRAME_UM
        level2 = 4 * pair["high_um"]
        print(f"{pair['name']:12s} {pair['zid']:>15d} {pair['role']:8s} {pair['low_um']:7.3f} {pair['high_um']:7.3f} "
              f"{high_to_low(pair):8.4f} {high_to_frame(pair):10.4f} {level2:9.3f} {level2 / frame_um:13.4f}")

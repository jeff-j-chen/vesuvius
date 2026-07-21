"""_discover_segments.py -- list S3 contents for the 10 new PHerc0139 training segments.

for each segment prints the exact filenames + shapes we need:
  - 9.362um-...-volume-20250728140407.zarr   (the training surface volume)
  - 1.129um-...-L1.zarr                        (high-res surface, for overlap midslice)
  - ink-detection/*.tif                        (the ink label)
no downloads of bulk data -- only tiny .zarray metadata + listing XML.
"""
import subprocess, json, re, sys

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"

# (name, segment_prefix, zarr_id)  -- zarr_id = segment timestamp
SEGMENTS = [
    ("w058", "PHerc0139/segments/20260210000000-w058_2026021020", "20260210000000"),
    ("w052", "PHerc0139/segments/20260227000000-w052_2026022705", "20260227000000"),
    ("w049", "PHerc0139/segments/20260318000000-w049_20260318",   "20260318000000"),
    ("w046", "PHerc0139/segments/20260325000000-w046_20260325",   "20260325000000"),
    ("w041", "PHerc0139/segments/20260108000000-w041_2026010816", "20260108000000"),
    ("w040", "PHerc0139/segments/20250831000000-w040_2025083102", "20250831000000"),
    ("w039", "PHerc0139/segments/20260302000000-w039_2026030210", "20260302000000"),
    ("w038", "PHerc0139/segments/20260306000000-w038_2026030608", "20260306000000"),
    ("w037", "PHerc0139/segments/20260310000000-w037_2026031015", "20260310000000"),
    ("w034", "PHerc0139/segments/20260303000000-w034_2026030317", "20260303000000"),
]


def curl(url):
    r = subprocess.run(["curl.exe", "-s", "--fail", "--max-time", "60", url],
                       capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", "replace")


def list_prefix(prefix):
    """return list of keys (Contents) and common-prefixes (folders) under prefix."""
    url = f"{BUCKET}/?prefix={prefix}/&delimiter=/"
    xml = curl(url)
    if xml is None:
        return [], []
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    prefixes = re.findall(r"<Prefix>([^<]+)</Prefix>", xml)
    # first <Prefix> is the query prefix itself; drop it
    folders = [p for p in prefixes if p != prefix + "/"]
    return keys, folders


def zarray_shape(zarr_base):
    j = curl(f"{zarr_base}/0/.zarray")
    if j is None:
        # maybe not a pyramid; try root .zarray
        j = curl(f"{zarr_base}/.zarray")
    if j is None:
        return None
    try:
        za = json.loads(j)
        return tuple(za["shape"]), za.get("dtype")
    except Exception:
        return None


for name, seg, zid in SEGMENTS:
    print(f"\n{'='*70}\n{name}  seg={seg}  zarr_id={zid}\n{'='*70}")
    sv_keys, sv_folders = list_prefix(f"{seg}/surface-volumes")
    vol9 = vol1 = None
    for f in sv_folders:
        base = f.rstrip("/")
        fn = base.split("/")[-1]
        if fn.startswith("9.362um") and "20250728140407" in fn:
            vol9 = base
        elif fn.startswith("1.129um") and fn.endswith("-L1.zarr"):
            vol1 = base
    print(f"  9.362um vol: {vol9.split('/')[-1] if vol9 else 'NOT FOUND'}")
    if vol9:
        s = zarray_shape(f"{BUCKET}/{vol9}")
        print(f"     shape/dtype: {s}")
    print(f"  1.129um L1 : {vol1.split('/')[-1] if vol1 else 'NOT FOUND'}")
    if vol1:
        s = zarray_shape(f"{BUCKET}/{vol1}")
        print(f"     shape/dtype: {s}")
    ink_keys, _ = list_prefix(f"{seg}/ink-detection")
    ink_tifs = [k for k in ink_keys if k.endswith(".tif")]
    print(f"  ink tif(s) : {len(ink_tifs)}")
    for k in ink_tifs:
        print(f"     {k.split('/')[-1]}")

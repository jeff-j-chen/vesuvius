import tifffile, json
segs = [
    ('auto_grown_20260717193517520', '20260717193517', (28, 10821, 10821)),
    ('auto_grown_20260719202304218', '20260719202304', (28, 6741, 6741)),
    ('auto_grown_20260720090842117', '20260720090842', (28, 13201, 13201)),
]
base_root = r'C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths'
for uid, zid, cur in segs:
    base = base_root + '\\' + uid
    x = tifffile.imread(base + r'\x.tif')
    m = json.load(open(base + r'\meta.json'))
    H = (x.shape[0] - 1) * 20 + 1
    print(f'{zid}: mesh_grid={x.shape} expected_HW={H} area={m["area_cm2"]:.2f}cm2 max_gen={m["max_gen"]} current_zarr={cur}')

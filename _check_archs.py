from utils.model import _ARCH_MAP

expected = [
    'v16_arch_ctx_fovea', 'v16_dual_stream_early', 'v16_dual_stream_late', 
    'v16_dual_stream_gated', 'v16_dual_stream_asym', 'v16_hybrid_depth_per_window', 
    'v16_hybrid_depth_global', 'v16_hybrid_depth_triple', 'v16_hybrid_depth_gated', 
    'v16_multiscale_pyramid', 'v16_depth_se', 'v16_depthwise_sep', 
    'v16_mixed_depth_windows', 'v16_octave_conv', 'v16_efficientnet_scale', 
    'v16_nonlocal_depth', 'v16_coord_attention', 'v16_deformable_conv', 
    'v16_progressive_depth', 'v16_dual_attention', 'v16_axial_attention', 
    'v16_fpn', 'v16_bifpn', 'v16_ghost_conv', 'v16_inverted_residual', 
    'v16_resnext_groups', 'v16_depth_shift'
]

missing = [arch for arch in expected if arch not in _ARCH_MAP]

print(f'Total architectures in _ARCH_MAP: {len(_ARCH_MAP)}')
print(f'Expected 27 new archs, missing: {len(missing)}')
if missing:
    print('Missing architectures:')
    for arch in missing:
        print(f'  - {arch}')
else:
    print('✅ All 27 architectures registered successfully!')

print('\nAll registered architectures:')
for arch in sorted(_ARCH_MAP.keys()):
    print(f'  {arch}')

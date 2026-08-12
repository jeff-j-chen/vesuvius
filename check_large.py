from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

# Check the largest runs
large_runs = [
    'cmp_archs5_depsc_high_depth_supcon_lam03_10_21-05-30',
    'cmp_archs5_baseline_baseline_10_16-06-54',
    'cmp_archs5_depsc_surf_depth_supcon_learned_surf_11_01-08-07',
    'cmp_archs5_focal_loss_focal_gamma15_11_05-06-33',
    'cmp_archs5_asym_smooth_asym_label_smooth_11_09-11-08',
]

for run_name in large_runs:
    run_dir = Path(f'runs_archs2/{run_name}')
    if not run_dir.exists():
        print(f"Skipping {run_name} (not found)")
        continue
        
    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    
    scalars = ea.Tags().get('scalars', [])
    tid = run_name.split('_')[2]
    
    print(f"\n{tid} ({run_name}):")
    if not scalars:
        print("  No scalar tags found")
    else:
        print(f"  {len(scalars)} scalar tags found")
        # Look for key metrics
        for metric in ['valid/pr_auc', 'valid/f1', 'valid/readability_composite', 'train/loss']:
            if metric in scalars:
                events = ea.Scalars(metric)
                if events:
                    last = events[-1]
                    print(f"  {metric}: step {last.step}, value {last.value:.5f}")

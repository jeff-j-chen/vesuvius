from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

run_name = 'cmp_archs5_depsc_high_depth_supcon_lam03_10_21-05-30'
run_dir = Path(f'runs_archs2/{run_name}')

ea = event_accumulator.EventAccumulator(str(run_dir))
ea.Reload()

scalars = ea.Tags().get('scalars', [])

print(f"All {len(scalars)} scalar tags:")
for tag in sorted(scalars):
    events = ea.Scalars(tag)
    if events:
        last = events[-1]
        print(f"  {tag:<50} step={last.step:>3} value={last.value:>10.5f}")

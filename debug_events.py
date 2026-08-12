from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

# Pick one of the larger event files
run_dir = Path('runs_archs2/cmp_archs5_baseline_baseline_10_14-49-25')

ea = event_accumulator.EventAccumulator(str(run_dir))
ea.Reload()

print("Available tags:")
print("Scalars:", ea.Tags().get('scalars', []))
print("Images:", ea.Tags().get('images', []))
print("Histograms:", ea.Tags().get('histograms', []))

# Check what we can read
scalars = ea.Tags().get('scalars', [])
if scalars:
    print(f"\nFound {len(scalars)} scalar tags")
    for tag in sorted(scalars)[:20]:  # Show first 20
        events = ea.Scalars(tag)
        if events:
            print(f"  {tag}: {len(events)} events, last step={events[-1].step}, last value={events[-1].value:.4f}")

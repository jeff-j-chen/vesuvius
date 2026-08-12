"""Simple event file reader without tensorboard dependency"""
import struct
import glob
from pathlib import Path

def parse_events_simple(event_file):
    """Read event file and extract scalar summaries"""
    try:
        with open(event_file, 'rb') as f:
            data = f.read()
        
        # Look for common metric names in the raw bytes
        metrics = {
            'pr_auc': None,
            'f1': None,
            'readability': None,
            'max_epoch': 0
        }
        
        # Simple string search for metric names
        content = data.decode('latin1', errors='ignore')
        
        # Count how many times we see "epoch" to estimate progress
        epoch_count = content.count('epoch')
        if epoch_count > metrics['max_epoch']:
            metrics['max_epoch'] = epoch_count
            
        # Try to find last values by searching backwards
        if 'valid/pr_auc' in content:
            # Find all occurrences and try to extract nearby floats
            idx = content.rfind('valid/pr_auc')
            if idx > 0:
                # Look for float values near this string (within next 200 bytes)
                chunk = content[idx:idx+200]
                # Very crude: look for sequences that might be float representations
                metrics['pr_auc'] = 'found'
        
        return metrics
    except Exception as e:
        return {'error': str(e), 'max_epoch': 0}

# Check all runs
runs_dir = Path('runs_archs2')
results = []

for run_dir in sorted(runs_dir.glob('cmp_archs5_*')):
    tid = run_dir.name.split('_')[2]
    event_files = list(run_dir.glob('events.out.tfevents*'))
    
    if event_files:
        event_file = event_files[0]
        size_mb = event_file.stat().st_size / (1024 * 1024)
        
        # Estimate completion based on file size
        # Typical complete run: 3-4 MB
        # Failed/incomplete: < 1 MB
        estimated_complete = size_mb > 2.5
        
        results.append({
            'tid': tid,
            'size_mb': size_mb,
            'complete': estimated_complete,
            'dir': run_dir.name
        })

# Sort by size (larger = more training done)
results.sort(key=lambda x: -x['size_mb'])

print(f"{'TID':<14} {'Size (MB)':<12} {'Status':<15}")
print('=' * 45)
for r in results:
    status = 'COMPLETE' if r['complete'] else 'INCOMPLETE'
    print(f"{r['tid']:<14} {r['size_mb']:>8.2f}     {status}")

complete = [r for r in results if r['complete']]
incomplete = [r for r in results if not r['complete']]

print()
print(f"SUMMARY: {len(complete)} likely complete, {len(incomplete)} incomplete")
print()
print("Complete runs (by size):")
for r in complete:
    print(f"  - {r['tid']:<14} ({r['size_mb']:.1f} MB)")

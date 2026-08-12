"""Read tensorboard events manually using the tfrecord format"""
import struct
from pathlib import Path

def read_varint(f):
    """Read a varint from file"""
    shift = 0
    result = 0
    while True:
        byte = f.read(1)
        if not byte:
            return None
        b = byte[0]
        result |= (b & 0x7f) << shift
        if not (b & 0x80):
            return result
        shift += 7

def read_record(f):
    """Read one TFRecord"""
    # Read length (8 bytes)
    length_bytes = f.read(8)
    if len(length_bytes) < 8:
        return None
    length = struct.unpack('<Q', length_bytes)[0]
    
    # Skip masked CRC (4 bytes)
    f.read(4)
    
    # Read data
    data = f.read(length)
    if len(data) < length:
        return None
    
    # Skip data CRC (4 bytes)  
    f.read(4)
    
    return data

def extract_scalars(event_file):
    """Extract scalar values from event file"""
    scalars = {}
    max_step = 0
    
    try:
        with open(event_file, 'rb') as f:
            while True:
                record = read_record(f)
                if record is None:
                    break
                
                # Decode as string to search for metric names
                try:
                    text = record.decode('latin1', errors='ignore')
                    
                    # Look for step numbers
                    if 'step' in text.lower():
                        # Try to find step value in nearby bytes
                        for i in range(len(record)-8):
                            try:
                                val = struct.unpack('<Q', record[i:i+8])[0]
                                if 0 < val < 100:  # Reasonable epoch range
                                    max_step = max(max_step, val)
                            except:
                                pass
                    
                    # Look for metric names and nearby float values
                    for metric in ['valid/pr_auc', 'valid/f1', 'valid/readability_composite']:
                        if metric in text:
                            # Search for float values near this string
                            idx = text.find(metric)
                            # Look in window around the metric name
                            window_start = max(0, idx - 50)
                            window_end = min(len(record), idx + 200)
                            window = record[window_start:window_end]
                            
                            # Try to find float values (4 bytes)
                            for i in range(len(window) - 4):
                                try:
                                    val = struct.unpack('<f', window[i:i+4])[0]
                                    # Valid metric range: 0 to 1
                                    if 0 <= val <= 1.0:
                                        short_name = metric.split('/')[-1]
                                        if short_name not in scalars or scalars[short_name]['step'] < max_step:
                                            scalars[short_name] = {'value': val, 'step': max_step}
                                except:
                                    pass
                except:
                    pass
    except Exception as e:
        pass
    
    return scalars, max_step

# Analyze all runs
runs_dir = Path('runs_archs2')
results = []

for run_dir in sorted(runs_dir.glob('cmp_archs5_*')):
    tid = run_dir.name.split('_')[2]
    event_files = list(run_dir.glob('events.out.tfevents*'))
    
    if event_files:
        event_file = event_files[0]
        size_mb = event_file.stat().st_size / (1024 * 1024)
        
        scalars, max_step = extract_scalars(event_file)
        
        pr_auc = scalars.get('pr_auc', {}).get('value')
        f1 = scalars.get('f1', {}).get('value')
        read = scalars.get('readability_composite', {}).get('value')
        
        results.append({
            'tid': tid,
            'epochs': max_step,
            'pr_auc': pr_auc,
            'f1': f1,
            'read': read,
            'size_mb': size_mb,
            'dir': run_dir.name
        })

# Sort by epochs, then pr_auc
results.sort(key=lambda x: (-x['epochs'], -(x['pr_auc'] or 0)))

print(f"{'TID':<14} {'Epochs':<8} {'PR-AUC':<10} {'F1':<10} {'Read':<10} Status")
print('=' * 70)

for r in results:
    pr = f"{r['pr_auc']:.5f}" if r['pr_auc'] else 'N/A'
    f1 = f"{r['f1']:.5f}" if r['f1'] else 'N/A'
    rd = f"{r['read']:.5f}" if r['read'] else 'N/A'
    status = 'COMPLETE' if r['epochs'] >= 14 else f"EP{r['epochs']}"
    print(f"{r['tid']:<14} {r['epochs']:<8} {pr:<10} {f1:<10} {rd:<10} {status}")

complete = [r for r in results if r['epochs'] >= 14 and r['pr_auc']]
print()
print(f"SUMMARY: {len(complete)} complete runs with metrics")

if complete:
    print()
    print("BEST PERFORMERS (complete runs only):")
    best_pr = max(complete, key=lambda x: x['pr_auc'])
    best_f1 = max(complete, key=lambda x: x['f1'] or 0)
    best_read = max(complete, key=lambda x: x['read'] or 0)
    
    print(f"  Best PR-AUC: {best_pr['tid']:<14} {best_pr['pr_auc']:.5f}")
    if best_f1['f1']:
        print(f"  Best F1:     {best_f1['tid']:<14} {best_f1['f1']:.5f}")
    if best_read['read']:
        print(f"  Best Read:   {best_read['tid']:<14} {best_read['read']:.5f}")

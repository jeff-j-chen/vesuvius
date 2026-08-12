from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

runs_dir = Path('runs_archs2')
results = []

for run_dir in sorted(runs_dir.glob('cmp_archs5_*')):
    tid = run_dir.name.split('_')[2]
    
    try:
        ea = event_accumulator.EventAccumulator(str(run_dir))
        ea.Reload()
        
        scalars = ea.Tags().get('scalars', [])
        
        if not scalars:
            continue
            
        # Extract metrics using correct tag names
        pr_auc = None
        f1 = None
        read = None
        max_epoch = 0
        
        if 'AUC/PR_AUC/Valid' in scalars:
            events = ea.Scalars('AUC/PR_AUC/Valid')
            if events:
                pr_auc = events[-1].value
                max_epoch = max(max_epoch, events[-1].step)
                
        if 'P_M/F1_Score/Valid' in scalars:
            events = ea.Scalars('P_M/F1_Score/Valid')
            if events:
                f1 = events[-1].value
                max_epoch = max(max_epoch, events[-1].step)
                
        if 'R_M/Probe/ALL/ReadabilityComposite' in scalars:
            events = ea.Scalars('R_M/Probe/ALL/ReadabilityComposite')
            if events:
                read = events[-1].value
                max_epoch = max(max_epoch, events[-1].step)
        
        results.append({
            'tid': tid,
            'epochs': max_epoch,
            'pr_auc': pr_auc,
            'f1': f1,
            'read': read,
            'dir': run_dir.name
        })
    except Exception as e:
        print(f"Error processing {run_dir.name}: {e}")

# Sort by epochs (descending), then pr_auc (descending)
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
print(f"SUMMARY: {len(complete)} complete runs (14+ epochs)")

if complete:
    print()
    print("BEST PERFORMERS:")
    best_pr = max(complete, key=lambda x: x['pr_auc'])
    best_f1 = max(complete, key=lambda x: x['f1'] or 0)
    best_read = max(complete, key=lambda x: x['read'] or 0)
    
    print(f"  Best PR-AUC:        {best_pr['tid']:<14} {best_pr['pr_auc']:.5f}")
    print(f"  Best F1:            {best_f1['tid']:<14} {best_f1['f1']:.5f}")
    print(f"  Best Readability:   {best_read['tid']:<14} {best_read['read']:.5f}")
    
    print()
    print("ALL COMPLETE RUNS (ranked by PR-AUC):")
    complete_sorted = sorted(complete, key=lambda x: -x['pr_auc'])
    for i, r in enumerate(complete_sorted, 1):
        print(f"  {i}. {r['tid']:<14} PR-AUC={r['pr_auc']:.5f}  F1={r['f1']:.5f}  Read={r['read']:.5f}")

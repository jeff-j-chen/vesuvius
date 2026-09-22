from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.stats import pearsonr, spearmanr
import json,re,math,statistics
ROOT=Path('/vesuvius/runs_archs29'); OUT=Path('/data/extra/tmp/campaign29_30_tb_analysis.json'); TXT=Path('/data/extra/tmp/campaign29_30_tb_report.txt')
def sane(x):
 x=float(x); return x if math.isfinite(x) else None
def corr(a,b,kind='p'):
 pairs=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
 if len(pairs)<3 or len(set(x for x,y in pairs))<2 or len(set(y for x,y in pairs))<2:return None
 x,y=zip(*pairs); return sane(pearsonr(x,y).statistic if kind=='p' else spearmanr(x,y).statistic)
def armof(s):
 m=re.match(r'cmp_archs(\d+)_(?:\d+_)?(.+?)_\d+_\d\d-\d\d-\d\d$',s); return (int(m.group(1)),m.group(2)) if m else (None,s)
def val_at(arr,step):
 z=[x['value'] for x in arr if x['step']==step]; return z[-1] if z else None
def flat(x,p=''):
 d={}
 if isinstance(x,dict):
  for k,v in x.items():d.update(flat(v,f'{p}.{k}' if p else k))
 else:d[p]=x
 return d
runs={}; alltags=set()
for ef in sorted(ROOT.glob('*/events.out.tfevents*')):
 if 'f1_bounds_0_to_1' in str(ef):continue
 name=ef.parent.name; camp,arm=armof(name)
 ea=EventAccumulator(str(ef),size_guidance={'scalars':0});ea.Reload(); tags=sorted(ea.Tags()['scalars']);alltags.update(tags)
 scalars={t:[{'step':int(v.step),'wall_time':float(v.wall_time),'value':sane(v.value)} for v in ea.Scalars(t)] for t in tags}
 core=scalars.get('Character/F1Macro/Valid',[]); steps=sorted(set(x['step'] for x in core)); frag='fragments' in arm
 cfg={}; cp=ef.parent/'config.json'
 if cp.exists():
  try:cfg=json.loads(cp.read_text())
  except Exception as ex:cfg={'_error':str(ex)}
 time_tags=[t for t in tags if 'time_elapsed' in t.lower() or t.lower()=='time/elapsed']
 times={t:scalars[t] for t in time_tags}
 tv=[x['value'] for t in time_tags for x in scalars[t] if x['value'] is not None]
 runs[arm]={'directory':name,'campaign':camp,'arm':arm,'event_file':str(ef),'tags':tags,'scalar_count':len(tags),'scalars':scalars,'config':cfg,
  'inventory':{'core_points':len(core),'core_steps':steps,'max_epoch':max(steps) if steps else None,'epoch9_present':9 in steps,'complete':9 in steps and len(steps)>=10,'class':'fragments_only' if frag else ('completed_standard' if 9 in steps and len(steps)>=10 else 'partial'),'time_tags':time_tags,'time_elapsed':times,'total_elapsed':sum(tv) if tv else None,'per_epoch_elapsed':tv}}
patch_re=re.compile(r'^Character/Patch/(.+)/F1/Valid$')
patches=sorted({patch_re.match(t).group(1) for r in runs.values() for t in r['tags'] if patch_re.match(t)})
standard=[a for a,r in runs.items() if r['inventory']['class']=='completed_standard']
fragments=[a for a,r in runs.items() if r['inventory']['class']=='fragments_only']
partial=[a for a,r in runs.items() if r['inventory']['class']=='partial']
def endpoint(a):
 r=runs[a]; s=r['scalars']; pf={p:val_at(s.get(f'Character/Patch/{p}/F1/Valid',[]),9) for p in patches}; vals=[x for x in pf.values() if x is not None]
 tags={'f1':'Character/F1Macro/Valid','ap':'Character/APMacro/Valid','recall':'Character/RecallMacro/Valid','ring_fpr':'Character/RingFPRMacro/Valid','success':'Character/SuccessFraction/Valid','count':'Character/Count/Valid'}
 for short,pat in [('pixel_f1','P_M/F1_Score'),('pr_auc','AUC/PR_AUC'),('roc_auc','AUC/ROC_AUC'),('gm_loss','G_M/Loss'),('gm_balacc','G_M/BalAcc')]:
  cand=[t for t in s if pat in t and ('Valid' in t or '/Validation' in t)]; tags[short]=cand[0] if cand else None
 z={k:(val_at(s.get(t,[]),9) if t else None) for k,t in tags.items()}; ftraj=s.get(tags['f1'],[]); vv=[x for x in vals]
 z.update({'patch_values':pf,'patch_n':len(vv),'patch_mean':sane(statistics.mean(vv)) if vv else None,'patch_median':sane(statistics.median(vv)) if vv else None,'patch_min':min(vv) if vv else None,'patch_std':sane(statistics.pstdev(vv)) if vv else None,
 'best_f1':max((x['value'] for x in ftraj if x['value'] is not None),default=None),'best_f1_epoch':max((x for x in ftraj if x['value'] is not None),key=lambda x:x['value'],default={}).get('step'),'f1_mean_epochs_7_9':sane(statistics.mean([x['value'] for x in ftraj if x['step'] in (7,8,9)])) if any(x['step'] in (7,8,9) for x in ftraj) else None,'total_elapsed':r['inventory']['total_elapsed']})
 return z
end={a:endpoint(a) for a in runs}; rank=sorted(standard,key=lambda a:end[a]['f1'],reverse=True)
rankings=[{'rank':i+1,'arm':a,**{k:end[a][k] for k in ['f1','ap','success','patch_mean','patch_median','patch_min','patch_std','pixel_f1','pr_auc','roc_auc','best_f1','best_f1_epoch','f1_mean_epochs_7_9','total_elapsed']}} for i,a in enumerate(rank)]
patch_ranks={}; wins={a:0 for a in standard};top3={a:0 for a in standard}
for p in patches:
 rr=sorted([(a,end[a]['patch_values'].get(p)) for a in standard if end[a]['patch_values'].get(p) is not None],key=lambda x:x[1],reverse=True)
 rows=[]
 means={a:statistics.mean([v for q,v in end[a]['patch_values'].items() if q!=p and v is not None]) for a,v in rr}
 mrank={a:i+1 for i,(a,v) in enumerate(sorted(means.items(),key=lambda x:x[1],reverse=True))}
 for i,(a,v) in enumerate(rr[:5]):rows.append({'rank':i+1,'arm':a,'value':v,'global_f1_rank':rank.index(a)+1,'mean_other_patches':means[a],'mean_other_rank':mrank[a],'specialization':(i==0 and (rank.index(a)+1>5 or mrank[a]>5))})
 if rr:wins[rr[0][0]]+=1
 for a,v in rr[:3]:top3[a]+=1
 patch_ranks[p]=rows
# correlations
within={}
for a in standard:
 mat={}; vals=[];pos=0;n=0
 for i,p in enumerate(patches):
  mat[p]={}
  x=[val_at(runs[a]['scalars'].get(f'Character/Patch/{p}/F1/Valid',[]),e) for e in range(10)]
  for q in patches:
   y=[val_at(runs[a]['scalars'].get(f'Character/Patch/{q}/F1/Valid',[]),e) for e in range(10)]
   pc=corr(x,y);sp=corr(x,y,'s');mat[p][q]={'pearson':pc,'spearman':sp}
  for q in patches[i+1:]:
   c=mat[p][q]['pearson'];
   if c is not None:vals.append(c);n+=1;pos+=c>0
 within[a]={'median_offdiag_pearson':statistics.median(vals) if vals else None,'positive_pair_fraction':pos/n if n else None,'matrix':mat}
def matrix_vectors(labels,vectors):
 return {a:{b:{'pearson':corr(vectors[a],vectors[b]),'spearman':corr(vectors[a],vectors[b],'s')} for b in labels} for a in labels}
patchvec={p:[end[a]['patch_values'].get(p) for a in standard] for p in patches}; patch_cross=matrix_vectors(patches,patchvec)
armvec={a:[end[a]['patch_values'].get(p) for p in patches] for a in standard}; arm_sim=matrix_vectors(standard,armvec)
patch_global={}
for p in patches:
 pv=patchvec[p]; gv=[end[a]['f1'] for a in standard]; ov=[statistics.mean([v for q,v in end[a]['patch_values'].items() if q!=p and v is not None]) for a in standard]
 patch_global[p]={'vs_global_pearson':corr(pv,gv),'vs_global_spearman':corr(pv,gv,'s'),'vs_mean_other_pearson':corr(pv,ov),'vs_mean_other_spearman':corr(pv,ov,'s')}
clusters=[];conflicts=[]
for i,p in enumerate(patches):
 for q in patches[i+1:]:
  c=patch_cross[p][q]['pearson']
  if c is not None and c>=.7:clusters.append([p,q,c])
  if c is not None and c<=-.4:conflicts.append([p,q,c])
# comparisons
pairs=[]
def add(base,other,label=None):
 if base not in runs or other not in runs:return
 b,o=end[base],end[other]; common=[p for p in patches if b['patch_values'].get(p) is not None and o['patch_values'].get(p) is not None]
 keys=['f1','ap','patch_mean','patch_min']; delta={k:{'base':b[k],'other':o[k],'absolute':(o[k]-b[k] if b[k] is not None and o[k] is not None else None),'percent':((o[k]/b[k]-1)*100 if b[k] not in (None,0) and o[k] is not None else None)} for k in keys}
 fb,fo=flat(runs[base]['config']),flat(runs[other]['config']); dif={k:{'base':fb.get(k),'other':fo.get(k)} for k in sorted(set(fb)|set(fo)) if fb.get(k)!=fo.get(k)}
 pairs.append({'label':label or f'{base} vs {other}','base':base,'other':other,'metrics':delta,'patches_improved':sum(o['patch_values'][p]>b['patch_values'][p] for p in common),'patches_compared':len(common),'runtime_ratio':(o['total_elapsed']/b['total_elapsed'] if b['total_elapsed'] and o['total_elapsed'] else None),'config_differences':dif})
for a in standard:
 if runs[a]['campaign']==29 and a!='baseline':add('baseline',a)
for a in standard:
 if runs[a]['campaign']==30 and a!='current_mid_control':add('current_mid_control',a)
add('baseline','current_mid_control','nominal replicate')
for b,o,l in [('early_gated','early_wide15_gated','early width'),('current_mid_control','mid_wide15_gated','mid width'),('early_gated','early_residual2d','early residual'),('current_mid_control','mid_residual2d','mid residual'),('mid_residual2d','mid_deep_residual2d','deep residual vs residual'),('mid_deep2d','mid_deep_residual2d','deep residual vs deep'),('early_residual2d','early_deep_residual2d','early deep residual')]:add(b,o,l)
# dedup
seen=set();pairs=[x for x in pairs if not ((x['base'],x['other'],x['label']) in seen or seen.add((x['base'],x['other'],x['label'])))]
res={'metadata':{'source':str(ROOT),'excluded':'f1_bounds_0_to_1','all_scalar_tags':sorted(alltags),'patches':patches,'caveat':'Correlations use 10 epochs and intervention endpoints from one seed; they are descriptive, not causal.'},'inventory':{a:r['inventory']|{'directory':r['directory'],'campaign':r['campaign'],'event_file':r['event_file'],'scalar_tags':r['tags']} for a,r in runs.items()},'raw_runs':runs,'endpoints':end,'groups':{'completed_standard':standard,'fragments_only':fragments,'partial':partial},'primary_ranking':rankings,'patch_rankings_top5':patch_ranks,'patch_win_counts':wins,'patch_top3_counts':top3,'correlations':{'within_runs':within,'across_arms_patch_matrix':patch_cross,'strong_clusters':clusters,'conflicts':conflicts,'patch_vs_global_and_others':patch_global,'run_vector_similarity':arm_sim},'comparisons':pairs}
OUT.write_text(json.dumps(res,indent=2,allow_nan=False))
lines=[];lines+=['INVENTORY']
for a,r in runs.items():lines.append(f"{r['campaign']} {a:32s} {r['inventory']['class']:18s} n={r['inventory']['core_points']:2d} max={r['inventory']['max_epoch']} time={r['inventory']['total_elapsed']}")
lines+=['','PRIMARY EPOCH-9 RANKING']
for x in rankings:lines.append(f"{x['rank']:2d} {x['arm']:30s} F1={x['f1']:.4f} AP={x['ap']:.4f} succ={x['success']:.4f} patch mean/med/min/std={x['patch_mean']:.4f}/{x['patch_median']:.4f}/{x['patch_min']:.4f}/{x['patch_std']:.4f} best={x['best_f1']:.4f}@{x['best_f1_epoch']} stable={x['f1_mean_epochs_7_9']:.4f}")
lines+=['','PATCH TOP 5']
for p,rr in patch_ranks.items():lines.append(p+' '+', '.join(f"{x['arm']}={x['value']:.4f}(G{x['global_f1_rank']},O{x['mean_other_rank']}{'*SPEC*' if x['specialization'] else ''})" for x in rr))
lines+=['','WINS/TOP3']+[f'{a}: {wins[a]}/{top3[a]}' for a in sorted(standard,key=lambda a:(-wins[a],-top3[a]))]
lines+=['','COMPARISONS']
for x in pairs:
 d=x['metrics'];fmt=lambda k:f"{d[k]['base']:.4f}->{d[k]['other']:.4f} ({d[k]['absolute']:+.4f},{d[k]['percent']:+.1f}%)" if d[k]['base'] is not None and d[k]['other'] is not None else 'NA'
 lines.append(f"{x['label']}: F1 {fmt('f1')} AP {fmt('ap')} mean {fmt('patch_mean')} min {fmt('patch_min')} improved {x['patches_improved']}/{x['patches_compared']} runtime={x['runtime_ratio']:.2f}x" if x['runtime_ratio'] is not None else f"{x['label']}: F1 {fmt('f1')} AP {fmt('ap')} mean {fmt('patch_mean')} min {fmt('patch_min')} improved {x['patches_improved']}/{x['patches_compared']} runtime=NA")
lines+=['','WITHIN-RUN CORRELATION']+[f"{a}: median={v['median_offdiag_pearson']:.3f} positive={v['positive_pair_fraction']:.3f}" for a,v in within.items()]
lines+=['','STRONG PATCH CLUSTERS']+[f'{p} ~ {q}: {c:.3f}' for p,q,c in sorted(clusters,key=lambda x:-x[2])]
lines+=['','PATCH CONFLICTS']+[f'{p} vs {q}: {c:.3f}' for p,q,c in sorted(conflicts,key=lambda x:x[2])]
TXT.write_text('\n'.join(lines)+'\n')
print(f'WROTE {OUT} {OUT.stat().st_size} bytes');print(f'WROTE {TXT} {TXT.stat().st_size} bytes');print('\n'.join(lines[:75]))

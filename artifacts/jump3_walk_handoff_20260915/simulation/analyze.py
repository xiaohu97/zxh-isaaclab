from pathlib import Path
import json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
results=[]
for p in sorted(root.glob('*.csv')):
    a=np.genfromtxt(p,delimiter=',',names=True)
    if not a.size: continue
    ii=np.where((a['state'][1:]==3) & (a['state'][:-1]==112))[0]+1
    if not len(ii) and not np.any(a['state']==112): ii=np.where(a['state']==3)[0]
    row={'case':p.stem,'duration_s':float(a['time'][-1]),'passive':bool(np.any(a['state']==1)), 'jump_peak_root_height_m':float(a['z'].max())}
    if len(ii):
        k=ii[0]; b=a[k:];t=a['time'][k];targets=np.column_stack([b[f'target{i}'] for i in range(29)])
        window=b[b['time']<t+.5]
        row.update(switch_s=float(t),switch_root_height_m=float(b['z'][0]),switch_velocity_world_m_s=[float(b[c][0]) for c in ['vx','vy','vz']],switch_pelvis_tilt_deg=float(np.degrees(b['pelvis_tilt'][0])),switch_torso_tilt_deg=float(np.degrees(b['torso_tilt'][0])),after_switch_min_root_height_m=float(b['z'].min()),after_switch_max_pelvis_tilt_deg=float(np.degrees(b['pelvis_tilt'].max())),after_switch_max_torso_tilt_deg=float(np.degrees(b['torso_tilt'].max())),after_switch_max_target_step_rad=float(np.abs(np.diff(targets,axis=0)).max()),first_half_second_max_target_step_rad=float(np.abs(np.diff(targets[:len(window)],axis=0)).max()),final_root_height_m=float(b['z'][-1]),final_torso_tilt_deg=float(np.degrees(b['torso_tilt'][-1])))
    log=p.with_suffix('.log')
    if log.exists():
        match=re.search(r'max_wall_lag=(\S+)',log.read_text())
        if match: row['max_wall_lag_s']=float(match[1])
    results.append(row)
(root/'metrics.json').write_text(json.dumps(results,indent=2))
fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True)
for name,label in [('matched_b30_nominal','0.30 s, nominal'),('matched_b30_forward','0.30 s, velocity impulse'),('matched_b10_nominal','0.10 s, nominal')]:
    a=np.genfromtxt(root/f'{name}.csv',delimiter=',',names=True);t=a['time'];sw=t[np.where(a['state']==3)[0][0]];x=t-sw
    axs[0,0].plot(x,a['z'],label=label)
    axs[0,1].plot(x,np.degrees(a['torso_tilt']),label=label)
    axs[1,0].plot(x,np.degrees(a['target14']),label=label)
    q=np.column_stack([a[f'target{i}'] for i in range(12)])
    axs[1,1].plot(x[1:],np.max(abs(np.diff(q,axis=0)),axis=1),label=label,alpha=.8)
for ax,title,y in zip(axs.flat,['Root height','Torso tilt from vertical','Waist pitch target','Largest leg target change per 1 ms'],['m','deg','deg','rad']):
    ax.set_title(title);ax.set_ylabel(y);ax.axvline(0,color='k',ls='--',lw=1);ax.axvspan(0,.3,color='gray',alpha=.12);ax.set_xlim(-.4,1.6);ax.grid(alpha=.25)
for ax in axs[1]:ax.set_xlabel('Time relative to Jump -> Walk (s)')
axs[0,0].legend(fontsize=8)
fig.suptitle('G1 jump3 -> walk: continuous MuJoCo physics, actual C++ states / ONNX\nMimic matrix lifetime corrected; no physical-state reset at handoff')
fig.tight_layout();fig.savefig(root/'handoff_diagnostics.png',dpi=160)
print(json.dumps([r for r in results if re.match('(sdk|matched)_b',r['case'])],indent=2))

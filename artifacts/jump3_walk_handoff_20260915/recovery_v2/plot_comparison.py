from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('trace_directory',type=Path);a=p.parse_args()
fig,axs=plt.subplots(2,2,figsize=(10,6),sharex=True)
for i,plant in enumerate(['sdk','matched']):
 for profile,color in [('legacy','tab:red'),('recovery','tab:blue')]:
  d=np.genfromtxt(a.trace_directory/f'fresh_paired_{plant}_{profile}_4.csv',delimiter=',',names=True)
  axs[i,0].plot(d['time'],d['z'],label=profile,color=color)
  axs[i,1].plot(d['time'],np.degrees(d['tilt']),label=profile,color=color)
 axs[i,0].set_ylabel(f'{plant}\nRoot height (m)');axs[i,1].set_ylabel('Pelvis tilt (deg)')
 axs[i,1].axhline(np.degrees(1),ls='--',color='gray',label='protection threshold')
 for ax in axs[i]:ax.grid(alpha=.3);ax.set_xlim(0,2)
for ax in axs[1]:ax.set_xlabel('Time after handoff (s)')
axs[0,0].legend();axs[0,1].legend()
fig.suptitle('Same initial state/history; synthetic gyro + waist tracking error\nRecovery candidate still fails in the matched plant; not a real-robot replay')
fig.tight_layout();fig.savefig(Path(__file__).parent/'comparison.png',dpi=160)

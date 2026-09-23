"""Regenerate Fig. 3 at 89 mm column width using the original recorded data."""
from pathlib import Path
import json
import hashlib
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
import numpy as np

ROOT=Path('/home/rclab/minipb_project/results/fig3_representative_seed42_20260824')
OUT=Path(__file__).resolve().parent
TIMES=(0.5,2.5,6.5)
STAGES=("Initial", "IMU fault", "Post-fault")
HEIGHT=2.85
COLORS=('#1764ab','#d95f02')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,
 'axes.labelweight':'semibold','axes.linewidth':0.8,'xtick.labelsize':7,
 'ytick.labelsize':7,'xtick.major.width':0.7,'ytick.major.width':0.7,
 'xtick.major.size':2.5,'ytick.major.size':2.5,'pdf.fonttype':42,
 'savefig.facecolor':'white'})
fig=plt.figure(figsize=(3.5,HEIGHT))
fig.legend(handles=[Line2D([],[],color='black',lw=1,ls='--',label='Command'),
 Patch(facecolor='#dedede',label='IMU fault (2–4 s)')],loc='upper center',
 bbox_to_anchor=(.54,.999),ncol=2,frameon=False,fontsize=7,
 handlelength=2,columnspacing=1.5,borderaxespad=0.2)
manifest={'source_pdf':'/home/rclab/Downloads/fig3_combined (1).pdf',
 'size_inches':[3.5,HEIGHT],'snapshot_times_s':TIMES,'snapshot_crop_xyxy':[360,320,920,700], 'snapshot_stages':STAGES,
 'line_width_pt':1.2,'font_size_pt':7,'datasets':{}}
for i,(method,title,color) in enumerate(zip(('hybrid','mlp_only'),('(a) LocoDyad','(b) MLP-E2E (Aux)'),COLORS)):
 left=.23+.41*i
 photo_left=.03+.51*i
 path=ROOT/method/'timeseries.npz'
 data=np.load(path,allow_pickle=False)
 t=data['time_s']
 manifest['datasets'][method]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samples':len(t)}
 fig.text(photo_left,.885,title,color=color,size=8,weight='bold',va='bottom')
 for j,(ts,stage) in enumerate(zip(TIMES,STAGES)):
  candidates=list((ROOT/'figure/snapshots'/method).glob(f'*_t{ts:g}s.png'))
  assert len(candidates)==1, (method,ts,candidates)
  snapshot=candidates[0]
  ax=fig.add_axes([photo_left+.149*j,.725,.142,.13])
  arr=np.asarray(Image.open(snapshot))
  ax.imshow(arr[320:700,360:920],interpolation='antialiased')
  ax.set_xticks([]);ax.set_yticks([])
  for spine in ax.spines.values():spine.set_color(color);spine.set_linewidth(.9)
  fig.text(photo_left+.149*j+.142/2,.715,f'{stage}\n{ts:g} s',ha='center',va='top',size=6.5,weight='semibold')
 velocity=fig.add_axes([left,.472,.345,.15])
 yaw=fig.add_axes([left,.300,.345,.14])
 contact=fig.add_axes([left,.145,.345,.12])
 for ax in (velocity,yaw,contact):
  ax.set_xlim(0,7)
  ax.set_xticks([0,2,4,6])
  ax.tick_params(pad=2)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
 for ax,values,ylim,yticks,ylabel in [
  (velocity,data['linear_velocity_b'][:,0],(-.2,1.05),(0,.5,1),'Forward\nvelocity\n(m/s)'),
  (yaw,data['yaw_rate_error'],(-2.7,2.7),(-2,0,2),'Yaw-vel\n(rad/s)')]:
  ax.axvspan(2,4,color='#dedede',alpha=.7,lw=0,zorder=0)
  ax.plot(t,values,color=color,lw=1.2,zorder=3)
  ax.set_ylim(*ylim);ax.set_yticks(yticks)
  if i == 0:
   ax.set_ylabel(ylabel,rotation=0,ha='right',va='center',labelpad=0)
   ax.yaxis.set_label_coords(-.27,.5)
  else:
   ax.tick_params(labelleft=False)
  ax.grid(axis='y',color='#d9d9d9',lw=.4,zorder=0)
  ax.tick_params(labelbottom=False)
  for ts in TIMES:
   idx=int(np.argmin(abs(t-ts)))
   ax.plot(t[idx],values[idx],'o',ms=2.7,mec=color,mfc='white',mew=.7,zorder=4)
 velocity.plot(t,data['command'][:,0],color='black',ls='--',lw=.85,zorder=2)
 yaw.axhline(0,color='black',ls='--',lw=.85,zorder=2)
 contact.set_facecolor('#f3f3f3')
 contacts=data['foot_contact'].astype(bool)
 dt=float(np.median(np.diff(t)))
 for row in range(4):
  edges=np.diff(np.r_[False,contacts[:,row],False].astype(int))
  starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
  segments=[(float(t[a]),float((t[b] if b<len(t) else t[-1]+dt)-t[a])) for a,b in zip(starts,ends)]
  contact.broken_barh(segments,(row-.45,.9),facecolors=color,edgecolors='none')
 contact.axvspan(2,4,color='#bcbcbc',alpha=.25,lw=0,zorder=3)
 contact.set_ylim(3.5,-.5)
 contact.set_yticks(range(4),('FL','FR','RL','RR'))
 contact.tick_params(axis='y',length=0,labelsize=7)
 if i == 0:
  contact.set_ylabel('Contact',rotation=0,ha='right',va='center',labelpad=0)
  contact.yaxis.set_label_coords(-.27,.5)
 else:
  contact.tick_params(labelleft=False)
fig.text(.60,.02,'Time (s)',ha='center',fontsize=7,weight='semibold')
# Save at the intended publication width, without a bounding-box resize.
fig.savefig(OUT/'fig3_combined_side_by_side.pdf',dpi=600,metadata={'Creator':None,'CreationDate':None})
fig.savefig(OUT/'fig3_combined_side_by_side_600dpi.png',dpi=600)
fig.savefig(OUT/'fig3_combined_side_by_side_preview.png',dpi=180)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved 3.5 x 2.85 inch figure with original 700-sample traces and 6 snapshots.')

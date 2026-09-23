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
from matplotlib.patches import Patch, ConnectionPatch
from PIL import Image
import numpy as np

ROOT=Path('/home/rclab/minipb_project/results/fig3_representative_seed42_20260824')
OUT=Path(__file__).resolve().parent
TIMES=(0.5,2.5,6.5)
STAGES=("Initial", "IMU fault", "Post-fault")
HEIGHT=3.0
COLORS=('#1764ab','#d95f02')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,
 'axes.labelweight':'semibold','axes.linewidth':0.8,'xtick.labelsize':7,
 'ytick.labelsize':7,'xtick.major.width':0.7,'ytick.major.width':0.7,
 'xtick.major.size':2.5,'ytick.major.size':2.5,'pdf.fonttype':42,
 'savefig.facecolor':'white'})
fig=plt.figure(figsize=(3.5,HEIGHT))
fig.legend(handles=[Line2D([],[],color=COLORS[0],lw=1.2,label='LocoDyad'),
 Line2D([],[],color=COLORS[1],lw=1.2,label='MLP-E2E (Aux)'),
 Line2D([],[],color='black',lw=.85,ls='--',label='Command'),
 Patch(facecolor='#dedede',label='IMU fault')],loc='upper center',
 bbox_to_anchor=(.54,1.0),ncol=2,frameon=False,fontsize=7,
 handlelength=1.8,columnspacing=1.4,borderaxespad=0.2)
manifest={'source_pdf':'/home/rclab/Downloads/fig3_combined (2).pdf',
 'size_inches':[3.5,HEIGHT],'snapshot_times_s':TIMES,'snapshot_crop_xyxy':[430,300,850,700], 'snapshot_stages':STAGES,
 'line_width_pt':1.0,'font_size_pt':7,'datasets':{}}
for i,(method,title,color) in enumerate(zip(('hybrid','mlp_only'),('(a) LocoDyad','(b) MLP-E2E (Aux)'),COLORS)):
 left=.22+.41*i
 photo_left=left
 snapshot_axes=[]
 path=ROOT/method/'timeseries.npz'
 data=np.load(path,allow_pickle=False)
 t=data['time_s']
 manifest['datasets'][method]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samples':len(t)}
 fig.text(photo_left,.852,title,color=color,size=7,weight='bold',va='bottom')
 for j,(ts,stage) in enumerate(zip(TIMES,STAGES)):
  candidates=list((ROOT/'figure/snapshots'/method).glob(f'*_t{ts:g}s.png'))
  assert len(candidates)==1, (method,ts,candidates)
  snapshot=candidates[0]
  ax=fig.add_axes([photo_left+.121*j,.680,.113,.153])
  arr=np.asarray(Image.open(snapshot))
  ax.imshow(arr[300:700,430:850],interpolation='antialiased')
  snapshot_axes.append(ax)
  ax.set_xticks([]);ax.set_yticks([])
  for spine in ax.spines.values():spine.set_color(color);spine.set_linewidth(.9)
  fig.text(photo_left+.121*j+.113/2,.672,f'{stage}\n{ts:g} s',ha='center',va='top',size=6,weight='normal',bbox=dict(facecolor='white',edgecolor='none',pad=.2))
 velocity=fig.add_axes([left,.435,.355,.155])
 yaw=fig.add_axes([left,.265,.355,.145])
 contact=fig.add_axes([left,.105,.355,.13])
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
  ax.plot(t,values,color=color,lw=1.0,zorder=3)
  ax.set_ylim(*ylim);ax.set_yticks(yticks)
  if i == 0:
   ax.set_ylabel(ylabel,rotation=0,ha='right',va='center',labelpad=0)
   ax.yaxis.set_label_coords(-.235,.5)
  else:
   ax.tick_params(labelleft=False)
  ax.grid(axis='y',color='#d9d9d9',lw=.4,zorder=0)
  ax.tick_params(labelbottom=False)
  for ts in TIMES:
   idx=int(np.argmin(abs(t-ts)))
   ax.plot(t[idx],values[idx],'o',ms=2.7,mec=color,mfc='white',mew=.7,zorder=4)
 for photo_axis,ts in zip(snapshot_axes,TIMES):
  for plot_axis in (velocity,yaw):
   plot_axis.axvline(ts,color=color,lw=.4,alpha=.4,zorder=1)
  fig.add_artist(ConnectionPatch(xyA=(.5,0),coordsA=photo_axis.transAxes,
   xyB=(ts,1.05),coordsB=velocity.transData,color=color,lw=.45,alpha=.5,zorder=0))
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
  contact.yaxis.set_label_coords(-.235,.5)
 else:
  contact.tick_params(labelleft=False)
fig.text(.60,.02,'Time (s)',ha='center',fontsize=7,weight='semibold')
# Save at the intended publication width, without a bounding-box resize.
fig.savefig(OUT/'fig3_combined_v2_readable.pdf',dpi=600,metadata={'Creator':None,'CreationDate':None})
fig.savefig(OUT/'fig3_combined_v2_readable_600dpi.png',dpi=600)
fig.savefig(OUT/'fig3_combined_v2_readable_preview.png',dpi=180)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved 3.5 x 3.0 inch figure with original 700-sample traces and 6 snapshots.')

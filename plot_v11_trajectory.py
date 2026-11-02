#!/usr/bin/env python3
"""Plot V11 command-integrated estimated trajectory, not measured ground truth."""
import argparse
import csv
import math
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('trajectory', nargs='?', default='tag_navigation_v11_trajectory.csv')
p.add_argument('--landmark-map', default='')
p.add_argument('--output', default='tag_navigation_v11_trajectory.png')
a=p.parse_args()
with open(a.trajectory, newline='') as f:
    rows=list(csv.DictReader(f))
if not rows:
    raise SystemExit('No trajectory data')
x=[float(r['x_m']) for r in rows]
y=[float(r['y_m']) for r in rows]
path=float(rows[-1]['distance_traveled_m'])
turn=float(rows[-1]['absolute_turn_rad'])
fig,ax=plt.subplots(figsize=(7,6))
ax.plot(x,y,linewidth=1,alpha=.55,label='Estimated path')
ax.scatter(x,y,s=12,label='Estimated pose samples (0.5s)')
ax.scatter([x[0]],[y[0]],marker='o',s=80,label='Start')
ax.scatter([x[-1]],[y[-1]],marker='x',s=90,label='End')
if a.landmark_map:
    with open(a.landmark_map,newline='') as f:
        for lm in csv.DictReader(f):
            tx,ty=float(lm['x_m']),float(lm['y_m'])
            ax.scatter([tx],[ty],marker='s',s=65)
            ax.annotate('Tag '+lm['id'],(tx,ty),xytext=(4,4),textcoords='offset points')
ax.set_aspect('equal',adjustable='datalim')
ax.set_xlabel('Estimated world x (m)')
ax.set_ylabel('Estimated world y (m)')
ax.set_title('V11 estimated trajectory (command dead reckoning; not ground truth)')
ax.grid(alpha=.25)
ax.legend(loc='best')
fig.tight_layout()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
fig.savefig(a.output,dpi=180)
print(f'Saved {a.output}')
print(f'Estimated translation path length: {path:.3f} m')
print(f'Integrated absolute heading change: {math.degrees(turn):.1f} deg')
print('Rotation-inclusive metric must be explicitly defined and calibrated; not computed here.')

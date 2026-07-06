"""
Radar (spider) chart for episode-length results across 7 environments and 8 methods.

Geometry convention:
  ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
  -> angle theta is measured CLOCKWISE from north (top).
  -> a point (theta, r) maps to cartesian (r * sin(theta), r * cos(theta)).
  -> theta increases monotonically from 0 to 2*pi as we walk the categories
     clockwise, so the closing segment from angles[N-1] back to 2*pi covers
     exactly one wedge (no wraparound bug).

Tangent / text rotation:
  d/d(theta) (sin theta, cos theta) = (cos theta, -sin theta) (clockwise tangent)
  -> on the RIGHT half (0 < theta < pi), this points generally downward, so
     text following the arc reads top-to-bottom with rotation = -theta_deg.
  -> on the LEFT half (pi < theta < 2pi), we instead want the COUNTER-clockwise
     tangent (-cos theta, sin theta) so that text still reads top-to-bottom,
     giving rotation = 180 - theta_deg.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{mathptmx}\usepackage{amsmath}',
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.unicode_minus': False,
})

categories = ['Original', 'Light', 'Long', 'Soft Gravity',
              'Heavy', 'Short', 'Strong Gravity']

methods_data = [
    ('ERM',      [186.4, 188.7, 191.4, 217.3, 178.6, 155.6, 157.6]),
    ('PA',       [342.4, 349.9, 298.0, 545.2, 278.8, 236.4, 238.4]),
    ('RO',       [345.0, 351.6, 300.6, 551.2, 286.3, 240.0, 243.7]),
    ('WFR',      [435.0, 459.6, 387.4, 703.8, 337.5, 276.2, 287.3]),
    ('SDRO',     [246.4, 248.7, 244.6, 378.1, 215.9, 176.7, 183.3]),
    ('NN-DRO',   [228.0, 231.4, 227.0, 293.3, 214.8, 193.4, 192.6]),
    ('MPA',      [565.2, 578.7, 418.8, 766.0, 420.9, 332.9, 328.3]),
    ('ICNN-DRO', [833.1, 844.5, 800.8, 996.9, 732.1, 568.8, 477.7]),
]

styles = {
    'ERM':      dict(color='#7f7f7f', lw=1.2, ls=(0,(1,1.2)),         marker='o', ms=4.0, fill=0.0,  z=3),
    'PA':       dict(color='#1f77b4', lw=1.3, ls='-',                 marker='s', ms=3.8, fill=0.0,  z=4),
    'RO':       dict(color='#17becf', lw=1.3, ls=(0,(4,2)),           marker='^', ms=4.2, fill=0.0,  z=4),
    'WFR':      dict(color='#2ca02c', lw=1.3, ls='-',                 marker='D', ms=3.5, fill=0.0,  z=4),
    'SDRO':     dict(color='#bcbd22', lw=1.3, ls=(0,(5,1.5,1,1.5)),   marker='v', ms=4.2, fill=0.0,  z=4),
    'NN-DRO':   dict(color='#9467bd', lw=1.3, ls=(0,(4,2)),           marker='P', ms=4.0, fill=0.0,  z=4),
    'MPA':      dict(color='#ff7f0e', lw=2.1, ls='-',                 marker='*', ms=8.5, fill=0.10, z=5),
    'ICNN-DRO': dict(color='#d62728', lw=2.6, ls='-',                 marker='*', ms=10.5, fill=0.20, z=6),
}

LEGEND_NAMES = {
    'ERM':      r'ERM',
    'PA':       r'PA',
    'RO':       r'RO',
    'WFR':      r'WFR',
    'SDRO':     r'SDRO',
    'NN-DRO':   r'NN-DRO',
    'MPA':      r'MPA \,\textbf{(Ours)}',
    'ICNN-DRO': r'ICNN-DRO \,\textbf{(Ours)}',
}

N = len(categories)
angles = np.linspace(0.0, 2*np.pi, N, endpoint=False)
angles_closed = np.concatenate([angles, [2*np.pi]])

RMAX = 1050.0

fig = plt.figure(figsize=(9.0, 9.6))
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)


for name, values in methods_data:
    s = styles[name]
    v = list(values) + [values[0]]
    ax.plot(angles_closed, v,
            color=s['color'], linewidth=s['lw'], linestyle=s['ls'],
            marker=s['marker'], markersize=s['ms'],
            markerfacecolor=s['color'],
            markeredgecolor='white' if name in ('MPA', 'ICNN-DRO') else s['color'],
            markeredgewidth=0.6 if name in ('MPA', 'ICNN-DRO') else 0.0,
            label=LEGEND_NAMES[name], zorder=s['z'])
    if s['fill'] > 0:
        ax.fill(angles_closed, v, color=s['color'], alpha=s['fill'], zorder=s['z']-1)


ax.set_ylim(0, RMAX)
ax.set_yticks([200, 400, 600, 800, 1000])
ax.set_yticklabels([])  
ax.set_rlabel_position(180)
ax.tick_params(axis='y', pad=1)


for r in [200, 400, 600, 800, 1000]:
    ax.text(np.pi, r, rf'${r}$', ha='center', va='center',
            fontsize=8.5, color='#444444', zorder=10,
            bbox=dict(facecolor='white', edgecolor='none',
                      boxstyle='round,pad=0.12', alpha=0.85))


ax.set_xticks(angles)
ax.set_xticklabels([])
ax.grid(True, color='#888888', alpha=0.45, linestyle='--', linewidth=0.55)
ax.spines['polar'].set_color('#555555')
ax.spines['polar'].set_linewidth(0.9)


label_radius = RMAX * 1.085
for ang, cat in zip(angles, categories):
    if cat in ('Soft Gravity', 'Strong Gravity'):
        text_str = r'\begin{tabular}{c}' + cat.split(' ')[0] + r'\\' + cat.split(' ')[1] + r'\end{tabular}'
    else:
        text_str = cat
    weight = 'bold'
    color  = '#222222'
    if cat == 'Original':
        color = '#000000'
    ax.text(ang, label_radius, text_str, ha='center', va='center',
            fontsize=12.5, fontweight=weight, color=color)


def arc_text_rotation(theta_rad):

    deg = (np.degrees(theta_rad)) % 360.0
    if deg < 180.0:                      
        return -deg
    else:                               
        return 180.0 - deg

def draw_group_arc(ax, theta_a, theta_b, radius, label, *,
                   fontsize=12.5, label_offset=85, arrow_size=10):
    n = 160
    th = np.linspace(theta_a, theta_b, n)
    ax.plot(th, [radius]*n, color='black', lw=1.0,
            clip_on=False, zorder=11, solid_capstyle='round')
    tick_size = 22
    for t in (theta_a, theta_b):
        ax.plot([t, t], [radius - tick_size, radius],
                color='black', lw=1.0, clip_on=False, zorder=11)
    mid = 0.5 * (theta_a + theta_b)
    rot = arc_text_rotation(mid)
    ax.text(mid, radius + label_offset, r'\textit{' + label + r'}',
            ha='center', va='center', fontsize=fontsize,
            rotation=rot, rotation_mode='anchor',
            clip_on=False, zorder=11)

arc_radius = RMAX * 1.235
half_axis  = (2*np.pi / N) / 2

draw_group_arc(ax,
               angles[1] - half_axis*0.30, angles[3] + half_axis*0.30,
               arc_radius, 'Easier Environments')


draw_group_arc(ax,
               angles[4] - half_axis*0.30, angles[6] + half_axis*0.30,
               arc_radius, 'Harder Environments')


leg = ax.legend(
    loc='lower center', bbox_to_anchor=(0.5, 1.10),
    ncol=4, frameon=True, fontsize=10.5,
    handlelength=2.6, handletextpad=0.6,
    columnspacing=1.6, labelspacing=0.45, borderpad=0.7,
    title=r'\textbf{Method}', title_fontsize=11.5,
)
leg.get_frame().set_edgecolor('#666666')
leg.get_frame().set_linewidth(0.7)
leg.get_frame().set_alpha(0.96)

plt.subplots_adjust(left=0.06, right=0.94, top=0.82, bottom=0.04)

out_png = '/home/Alireza/radar_chart.png'
out_pdf = '/home/Alireza/radar_chart.pdf'
plt.savefig(out_png, dpi=240, bbox_inches='tight', facecolor='white', pad_inches=0.10)
plt.savefig(out_pdf,             bbox_inches='tight', facecolor='white', pad_inches=0.10)
print('saved', out_png, out_pdf)
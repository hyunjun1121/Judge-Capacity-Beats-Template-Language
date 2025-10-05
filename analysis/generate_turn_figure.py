#!/usr/bin/env python3
"""
Generate turn-level analysis figure for appendix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# Set publication-quality parameters
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 11
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Load data
df = pd.read_csv('analysis_turn_breakdown.csv')

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
fig.subplots_adjust(bottom=0.12, top=0.93, left=0.09, right=0.97)

# Plot data for each condition
conditions = ['Apertus-Eng', 'Apertus-Trans', 'GPT-Eng', 'GPT-Trans']
colors = ['#e74c3c', '#e67e22', '#27ae60', '#3498db']
markers = ['o', 's', 'D', '^']

for condition, color, marker in zip(conditions, colors, markers):
    cond_df = df[df['Condition'] == condition].sort_values('Turn')

    turns = cond_df['Turn'].values
    means = cond_df['Mean'].values
    stds = cond_df['Std'].values

    # Plot mean with error bars (std)
    ax.errorbar(turns, means, yerr=stds,
                marker=marker, color=color, linewidth=2,
                markersize=8, alpha=0.85, label=condition,
                markeredgecolor='black', markeredgewidth=0.5,
                capsize=4, capthick=1.5, elinewidth=1.5)

ax.set_xlabel('Turn Number', fontsize=11, fontweight='bold')
ax.set_ylabel('StrongREJECT Score', fontsize=11, fontweight='bold')
ax.set_title('Multi-Turn Jailbreak: Score by Turn Number', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_xlim(0.5, 6.5)
ax.set_ylim(-0.05, 1.05)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.grid(axis='x', alpha=0.15, linestyle=':', linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=True, edgecolor='black',
          fancybox=False, framealpha=0.95, ncol=2)

# Save figure
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

plt.savefig(figures_dir / 'turn_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'turn_analysis.png', dpi=300, bbox_inches='tight')

print("Turn analysis figure saved to figures/turn_analysis.pdf/png")

# Print statistics summary
print("\n" + "="*70)
print("TURN-LEVEL SUMMARY")
print("="*70)

for condition in conditions:
    cond_df = df[df['Condition'] == condition].sort_values('Turn')

    mean_scores = cond_df['Mean'].values
    first_turn = mean_scores[0]
    last_turn = mean_scores[-1]
    change = last_turn - first_turn
    pct_change = (change / first_turn * 100) if first_turn > 0 else 0

    print(f"{condition:15s}: Turn 1={first_turn:.3f}, Turn 6={last_turn:.3f}, "
          f"Change={change:+.3f} ({pct_change:+.1f}%)")

plt.show()

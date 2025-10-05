#!/usr/bin/env python3
"""
Generate all figures for the paper - IMPROVED VERSION
Better layout, no overlapping text
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import seaborn as sns

# Set publication-quality parameters
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['figure.titlesize'] = 11
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Create figures directory
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

def load_evaluation_results(eval_dir, eval_key):
    """Load evaluation results from a directory"""
    results = {}
    eval_path = Path(eval_dir)

    if not eval_path.exists():
        print(f"Directory not found: {eval_dir}")
        return results

    for file in sorted(eval_path.glob('*_eval.json')):
        filename_base = file.stem
        if '_gpt_' in filename_base:
            language = filename_base.split('_gpt_')[0]
        elif '_apertus_' in filename_base:
            language = filename_base.split('_apertus_')[0]
        else:
            language = filename_base.split('_')[0]

        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            scores = []
            for entry in data:
                if eval_key in entry:
                    eval_data = entry[eval_key]
                    score = eval_data.get('strongreject_score')
                    if score is not None and 0 <= score <= 1:
                        scores.append(score)

            if scores:
                results[language] = {'scores': scores}
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return results


# Load all results
print("Loading evaluation results...")
gpt_english = load_evaluation_results('final_results/gpt_english_eval', 'gpt_english_evaluation')
gpt_translated = load_evaluation_results('final_results/gpt_translated_eval', 'gpt_translated_evaluation')
apertus_translated = load_evaluation_results('final_results/apertus_translated_eval', 'apertus_translated_evaluation')

# Original Apertus evaluation
original_apertus = {}
for file in sorted(Path('final_results').glob('*_complete.json')):
    language = file.stem.replace('_complete', '')
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        scores = []
        for entry in data:
            if 'evaluation' in entry and 'strongreject_score' in entry['evaluation']:
                score = entry['evaluation']['strongreject_score']
                if score is not None:
                    scores.append(score)
        if scores:
            original_apertus[language] = {'scores': scores}
    except Exception as e:
        print(f"Error loading {file}: {e}")


# ============================================================================
# FIGURE 1: Score Distribution Comparison (2x2 grid) - IMPROVED
# ============================================================================
print("\nGenerating Figure 1: Score Distribution (Improved)...")

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
fig.subplots_adjust(hspace=0.4, wspace=0.35, top=0.90, bottom=0.08, left=0.08, right=0.98)

methods = [
    ('Apertus-Eng', original_apertus, axes[0, 0]),
    ('Apertus-Trans', apertus_translated, axes[0, 1]),
    ('GPT-Trans', gpt_translated, axes[1, 0]),
    ('GPT-Eng', gpt_english, axes[1, 1])
]

for method_name, results, ax in methods:
    all_scores = []
    for lang_data in results.values():
        all_scores.extend(lang_data['scores'])

    # Create histogram
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    counts, _ = np.histogram(all_scores, bins=bins)
    percentages = counts / len(all_scores) * 100

    # Plot bars
    x_pos = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ['#27ae60', '#95a5a6', '#f39c12', '#e74c3c', '#c0392b']

    bars = ax.bar(x_pos, percentages, width=0.16, color=colors, alpha=0.9, edgecolor='black', linewidth=0.7)

    # Add percentage labels on bars - IMPROVED positioning
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        if pct > 5:  # Only show label if bar is high enough
            # Position label inside bar if tall enough
            y_pos = bar.get_height() - 3 if bar.get_height() > 12 else bar.get_height() + 2
            text_color = 'white' if bar.get_height() > 12 else 'black'
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{pct:.1f}%', ha='center', va='top' if bar.get_height() > 12 else 'bottom',
                   fontsize=8, fontweight='bold', color=text_color)

    ax.set_xlim(-0.05, 1.05)
    max_pct = max(percentages)
    ax.set_ylim(0, max_pct * 1.20)  # More space at top
    ax.set_xlabel('StrongREJECT Score', fontsize=10)
    ax.set_ylabel('Percentage (%)', fontsize=10)
    ax.set_title(method_name, fontsize=11, fontweight='bold', pad=10)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xticklabels(['[0-0.2]', '[0.2-0.4]', '[0.4-0.6]', '[0.6-0.8]', '[0.8-1.0]'], fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add mean score annotation - IMPROVED positioning
    mean_score = np.mean(all_scores)
    ax.text(0.98, 0.96, f'Mean: {mean_score:.2f}',
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9, linewidth=1),
           fontsize=8.5, fontweight='bold')

# Add legend - IMPROVED
legend_elements = [
    mpatches.Patch(color='#27ae60', label='Safe (0-0.2)', alpha=0.9, edgecolor='black', linewidth=0.5),
    mpatches.Patch(color='#95a5a6', label='Low risk (0.2-0.4)', alpha=0.9, edgecolor='black', linewidth=0.5),
    mpatches.Patch(color='#f39c12', label='Medium (0.4-0.6)', alpha=0.9, edgecolor='black', linewidth=0.5),
    mpatches.Patch(color='#e74c3c', label='High risk (0.6-0.8)', alpha=0.9, edgecolor='black', linewidth=0.5),
    mpatches.Patch(color='#c0392b', label='Very harmful (0.8-1.0)', alpha=0.9, edgecolor='black', linewidth=0.5)
]
fig.legend(handles=legend_elements, loc='upper center', ncol=5,
          bbox_to_anchor=(0.5, 0.98), frameon=True, fontsize=8, edgecolor='black', fancybox=False)

plt.savefig(figures_dir / 'score_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved: {figures_dir / 'score_distribution.pdf'}")
plt.close()


# ============================================================================
# FIGURE 2: Template Translation Effect per Language - IMPROVED
# ============================================================================
print("\nGenerating Figure 2: Template Effect (Improved)...")

# Load language-level data
lang_data = pd.read_csv('analysis_2_language_breakdown.csv')

# Calculate deltas
apertus_deltas = []
gpt_deltas = []
languages = []

for _, row in lang_data.iterrows():
    lang = row['Language']
    languages.append(lang)

    # Apertus delta
    try:
        a_eng = float(row['Apertus-Eng'])
        a_trans = float(row['Apertus-Trans'])
        apertus_deltas.append(a_trans - a_eng)
    except:
        apertus_deltas.append(0)

    # GPT delta
    try:
        g_eng = float(row['GPT-Eng'])
        g_trans = float(row['GPT-Trans'])
        gpt_deltas.append(g_trans - g_eng)
    except:
        gpt_deltas.append(0)

# Sort by Apertus delta
sorted_indices = np.argsort(apertus_deltas)
languages_sorted = [languages[i] for i in sorted_indices]
apertus_sorted = [apertus_deltas[i] for i in sorted_indices]
gpt_sorted = [gpt_deltas[i] for i in sorted_indices]

# Create figure - IMPROVED layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(wspace=0.30, top=0.90, bottom=0.10, left=0.10, right=0.97)

# Apertus subplot - IMPROVED
y_pos = np.arange(len(languages_sorted))
colors_a = ['#27ae60' if x > 0 else '#e74c3c' for x in apertus_sorted]
bars1 = ax1.barh(y_pos, apertus_sorted, color=colors_a, alpha=0.85, edgecolor='black', linewidth=0.6)

ax1.axvline(x=0, color='black', linewidth=1.5, linestyle='-', alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(languages_sorted, fontsize=9)
ax1.set_xlabel('Change in Score (Native - English)', fontsize=10, fontweight='bold')
ax1.set_title('Apertus-70B Judge', fontsize=11, fontweight='bold', pad=10)
ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_axisbelow(True)

# Add value labels for extreme cases - IMPROVED positioning
for i, (lang, val) in enumerate(zip(languages_sorted, apertus_sorted)):
    if abs(val) > 0.5:  # Show labels for very large effects only
        # Position text away from bar end to avoid overlap
        x_offset = 0.04 if val > 0 else -0.04
        ha = 'left' if val > 0 else 'right'
        ax1.text(val + x_offset, i, f'{val:+.2f}',
                va='center', ha=ha, fontsize=7.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

# Set x-axis limits to ensure labels don't get cut off
x_min, x_max = min(apertus_sorted), max(apertus_sorted)
ax1.set_xlim(x_min - 0.1, x_max + 0.1)

# GPT subplot - IMPROVED
colors_g = ['#27ae60' if x > 0 else '#e74c3c' for x in gpt_sorted]
bars2 = ax2.barh(y_pos, gpt_sorted, color=colors_g, alpha=0.85, edgecolor='black', linewidth=0.6)

ax2.axvline(x=0, color='black', linewidth=1.5, linestyle='-', alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(languages_sorted, fontsize=9)
ax2.set_xlabel('Change in Score (Native - English)', fontsize=10, fontweight='bold')
ax2.set_title('GPT-4.1 Judge', fontsize=11, fontweight='bold', pad=10)
ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_axisbelow(True)

# Add value labels for extreme cases - IMPROVED positioning
for i, (lang, val) in enumerate(zip(languages_sorted, gpt_sorted)):
    if abs(val) > 0.4:  # Show labels for large negative effects
        x_offset = 0.04 if val > 0 else -0.04
        ha = 'left' if val > 0 else 'right'
        ax2.text(val + x_offset, i, f'{val:+.2f}',
                va='center', ha=ha, fontsize=7.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

# Set x-axis limits
x_min_g, x_max_g = min(gpt_sorted), max(gpt_sorted)
ax2.set_xlim(x_min_g - 0.06, x_max_g + 0.06)

# Add legend - IMPROVED
legend_elements = [
    mpatches.Patch(color='#27ae60', label='Positive (helped)', alpha=0.85, edgecolor='black', linewidth=0.5),
    mpatches.Patch(color='#e74c3c', label='Negative (hurt)', alpha=0.85, edgecolor='black', linewidth=0.5)
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2,
          bbox_to_anchor=(0.5, 0.98), frameon=True, fontsize=9, edgecolor='black', fancybox=False)

plt.savefig(figures_dir / 'template_effect.pdf', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'template_effect.png', dpi=300, bbox_inches='tight')
print(f"Saved: {figures_dir / 'template_effect.pdf'}")
plt.close()


# ============================================================================
# FIGURE 3 (Appendix): Language-level scatter plot - IMPROVED
# ============================================================================
print("\nGenerating Figure 3 (Appendix): Language Scatter (Improved)...")

fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
fig.subplots_adjust(bottom=0.15, top=0.93, left=0.08, right=0.98)

languages_display = [lang.split('.')[0] for lang in languages]
x_pos = np.arange(len(languages))

# Plot lines for each method - IMPROVED markers and lines
methods_data = [
    ('Apertus-Eng', [float(lang_data[lang_data['Language'] == lang]['Apertus-Eng'].values[0]) for lang in languages], '#e74c3c', 'o', 1.8),
    ('Apertus-Trans', [float(lang_data[lang_data['Language'] == lang]['Apertus-Trans'].values[0]) for lang in languages], '#e67e22', 's', 1.8),
    ('GPT-Trans', [float(lang_data[lang_data['Language'] == lang]['GPT-Trans'].values[0]) for lang in languages], '#3498db', '^', 1.8),
    ('GPT-Eng', [float(lang_data[lang_data['Language'] == lang]['GPT-Eng'].values[0]) for lang in languages], '#27ae60', 'D', 1.8)
]

for label, scores, color, marker, lw in methods_data:
    ax.plot(x_pos, scores, marker=marker, color=color, linewidth=lw,
           markersize=7, alpha=0.85, label=label, markeredgecolor='black', markeredgewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(languages_display, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('StrongREJECT Score', fontsize=11, fontweight='bold')
ax.set_xlabel('Language', fontsize=11, fontweight='bold')
ax.set_title('Judge Performance Across Languages', fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(-0.05, 1.05)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.grid(axis='x', alpha=0.15, linestyle=':', linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc='upper left', frameon=True, fontsize=9, edgecolor='black', fancybox=False, framealpha=0.95)

plt.tight_layout()
plt.savefig(figures_dir / 'language_scatter.pdf', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'language_scatter.png', dpi=300, bbox_inches='tight')
print(f"Saved: {figures_dir / 'language_scatter.pdf'}")
plt.close()

print("\n" + "="*80)
print("ALL FIGURES GENERATED SUCCESSFULLY (IMPROVED)")
print("="*80)
print("\nGenerated files:")
print("  - figures/score_distribution.pdf (no overlapping text)")
print("  - figures/score_distribution.png")
print("  - figures/template_effect.pdf (better label positioning)")
print("  - figures/template_effect.png")
print("  - figures/language_scatter.pdf (clearer markers)")
print("  - figures/language_scatter.png")

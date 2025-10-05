#!/usr/bin/env python3
"""
Generate Language-Judge Agreement Heatmap (Figure 4)
Shows StrongReject scores across 16 languages and 4 judge conditions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_all_results():
    """Load all evaluation results"""

    languages = [
        'arb.Arab', 'ces.Latn', 'cmn.Hani', 'deu.Latn', 'fra.Latn',
        'ind.Latn', 'ita.Latn', 'jpn.Jpan', 'kor.Hang', 'nld.Latn',
        'pol.Latn', 'por.Latn', 'ron.Latn', 'rus.Cyrl', 'spa.Latn', 'tur.Latn'
    ]

    language_names = {
        'arb.Arab': 'Arabic', 'ces.Latn': 'Czech', 'cmn.Hani': 'Chinese',
        'deu.Latn': 'German', 'fra.Latn': 'French', 'ind.Latn': 'Indonesian',
        'ita.Latn': 'Italian', 'jpn.Jpan': 'Japanese', 'kor.Hang': 'Korean',
        'nld.Latn': 'Dutch', 'pol.Latn': 'Polish', 'por.Latn': 'Portuguese',
        'ron.Latn': 'Romanian', 'rus.Cyrl': 'Russian', 'spa.Latn': 'Spanish',
        'tur.Latn': 'Turkish'
    }

    conditions = [
        ('final_results', 'complete.json', 'evaluation', 'Apertus-Eng'),
        ('final_results/apertus_translated_eval', 'apertus_translated_eval.json', 'apertus_translated_evaluation', 'Apertus-Trans'),
        ('final_results/gpt_english_eval', 'gpt_english_eval.json', 'gpt_english_evaluation', 'GPT-Eng'),
        ('final_results/gpt_translated_eval', 'gpt_translated_eval.json', 'gpt_translated_evaluation', 'GPT-Trans')
    ]

    # Matrix: 16 languages x 4 conditions
    scores = np.zeros((len(languages), len(conditions)))

    for lang_idx, lang in enumerate(languages):
        for cond_idx, (folder, suffix, eval_key, _) in enumerate(conditions):
            file_path = Path(folder) / f'{lang}_{suffix}'

            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                strongreject_scores = []
                for entry in data:
                    eval_data = entry.get(eval_key, {})
                    score = eval_data.get('strongreject_score')
                    if score is not None:
                        strongreject_scores.append(score)

                if strongreject_scores:
                    scores[lang_idx, cond_idx] = np.mean(strongreject_scores)

    return scores, [language_names[lang] for lang in languages], [c[3] for c in conditions]


def create_heatmap():
    """Create language-judge agreement heatmap"""

    scores, languages, conditions = load_all_results()

    # Transpose: now 4 conditions x 16 languages
    scores_T = scores.T

    # Create figure - wider, shorter
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Create heatmap
    im = ax.imshow(scores_T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    # Set ticks - swapped
    ax.set_xticks(np.arange(len(languages)))
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_xticklabels(languages, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(conditions, fontsize=10, fontweight='bold')

    # Add score annotations - swapped indices
    for i in range(len(conditions)):
        for j in range(len(languages)):
            score = scores_T[i, j]
            text_color = 'white' if score > 0.5 else 'black'
            ax.text(j, i, f'{score:.2f}',
                   ha="center", va="center", color=text_color,
                   fontsize=7.5, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('StrongReject Score', rotation=270, labelpad=20, fontsize=10)

    # Add grid - swapped
    ax.set_xticks(np.arange(len(languages))-0.5, minor=True)
    ax.set_yticks(np.arange(len(conditions))-0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    # Labels - swapped
    ax.set_xlabel('Language', fontsize=11, fontweight='bold')
    ax.set_ylabel('Judge Condition', fontsize=11, fontweight='bold')
    ax.set_title('Language-Judge Agreement Heatmap', fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()

    # Save
    plt.savefig('figures/language_judge_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/language_judge_heatmap.png', dpi=300, bbox_inches='tight')

    print("Heatmap saved to figures/language_judge_heatmap.pdf/png")

    # Print statistics
    print("\nScore Matrix:")
    print(f"{'Language':<12} {'Apt-Eng':>10} {'Apt-Trans':>10} {'GPT-Eng':>10} {'GPT-Trans':>10}")
    print("-" * 56)
    for i, lang in enumerate(languages):
        print(f"{lang:<12} {scores[i,0]:>10.2f} {scores[i,1]:>10.2f} {scores[i,2]:>10.2f} {scores[i,3]:>10.2f}")

    return scores


if __name__ == "__main__":
    create_heatmap()

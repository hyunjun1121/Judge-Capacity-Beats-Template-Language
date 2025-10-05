#!/usr/bin/env python3
"""
Comprehensive analysis for paper: All key metrics and findings
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

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
            refused_counts = {'refused': 0, 'not_refused': 0}
            convincing_scores = []
            specific_scores = []

            for entry in data:
                if eval_key in entry:
                    eval_data = entry[eval_key]
                    score = eval_data.get('strongreject_score')

                    if score is not None and 0 <= score <= 5:
                        scores.append(score)

                        # Collect component scores
                        if 'refused' in eval_data:
                            if eval_data['refused'] == 1:
                                refused_counts['refused'] += 1
                            else:
                                refused_counts['not_refused'] += 1

                        if 'convincing_score' in eval_data:
                            convincing_scores.append(eval_data['convincing_score'])

                        if 'specific_score' in eval_data:
                            specific_scores.append(eval_data['specific_score'])

            if scores:
                results[language] = {
                    'avg_score': np.mean(scores),
                    'median_score': np.median(scores),
                    'std_score': np.std(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'count': len(scores),
                    'refused_count': refused_counts['refused'],
                    'not_refused_count': refused_counts['not_refused'],
                    'refused_ratio': refused_counts['refused'] / len(scores) if len(scores) > 0 else 0,
                    'avg_convincing': np.mean(convincing_scores) if convincing_scores else None,
                    'avg_specific': np.mean(specific_scores) if specific_scores else None,
                    'scores': scores  # Keep all scores for distribution analysis
                }
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return results


def analyze_score_distribution(results, method_name):
    """Analyze score distribution"""
    print(f"\n{'='*80}")
    print(f"SCORE DISTRIBUTION: {method_name}")
    print(f"{'='*80}")

    all_scores = []
    for lang, data in results.items():
        all_scores.extend(data['scores'])

    if not all_scores:
        print("No scores found")
        return

    bins = [0, 1, 2, 3, 4, 5]
    hist, _ = np.histogram(all_scores, bins=bins)

    print(f"\nScore range distribution:")
    print(f"  0.0-1.0 (Safe):        {hist[0]:4d} ({hist[0]/len(all_scores)*100:5.1f}%)")
    print(f"  1.0-2.0 (Low risk):    {hist[1]:4d} ({hist[1]/len(all_scores)*100:5.1f}%)")
    print(f"  2.0-3.0 (Medium):      {hist[2]:4d} ({hist[2]/len(all_scores)*100:5.1f}%)")
    print(f"  3.0-4.0 (High risk):   {hist[3]:4d} ({hist[3]/len(all_scores)*100:5.1f}%)")
    print(f"  4.0-5.0 (Very harmful):{hist[4]:4d} ({hist[4]/len(all_scores)*100:5.1f}%)")
    print(f"\nTotal samples: {len(all_scores)}")


def main():
    print("="*80)
    print("COMPREHENSIVE ANALYSIS FOR PAPER")
    print("="*80)

    # Load all results
    print("\nLoading evaluation results...")

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
            refused_counts = {'refused': 0, 'not_refused': 0}
            convincing_scores = []
            specific_scores = []

            for entry in data:
                if 'evaluation' in entry and 'strongreject_score' in entry['evaluation']:
                    eval_data = entry['evaluation']
                    score = eval_data['strongreject_score']

                    if score is not None:
                        scores.append(score)

                        if 'refused' in eval_data:
                            if eval_data['refused'] == 1:
                                refused_counts['refused'] += 1
                            else:
                                refused_counts['not_refused'] += 1

                        if 'convincing_score' in eval_data:
                            convincing_scores.append(eval_data['convincing_score'])

                        if 'specific_score' in eval_data:
                            specific_scores.append(eval_data['specific_score'])

            if scores:
                original_apertus[language] = {
                    'avg_score': np.mean(scores),
                    'median_score': np.median(scores),
                    'std_score': np.std(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'count': len(scores),
                    'refused_count': refused_counts['refused'],
                    'not_refused_count': refused_counts['not_refused'],
                    'refused_ratio': refused_counts['refused'] / len(scores) if len(scores) > 0 else 0,
                    'avg_convincing': np.mean(convincing_scores) if convincing_scores else None,
                    'avg_specific': np.mean(specific_scores) if specific_scores else None,
                    'scores': scores
                }
        except Exception as e:
            print(f"Error loading {file}: {e}")

    # ========================================================================
    # TABLE 1: Overall Average Scores
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 1: OVERALL AVERAGE SCORES (Primary Result)")
    print("="*80)

    methods = {
        'Apertus-Eng': original_apertus,
        'Apertus-Trans': apertus_translated,
        'GPT-Trans': gpt_translated,
        'GPT-Eng': gpt_english
    }

    overall_stats = []
    for method_name, results in methods.items():
        if results:
            avg = np.mean([v['avg_score'] for v in results.values()])
            std = np.std([v['avg_score'] for v in results.values()])
            min_val = np.min([v['avg_score'] for v in results.values()])
            max_val = np.max([v['avg_score'] for v in results.values()])

            overall_stats.append({
                'Method': method_name,
                'Mean': f"{avg:.3f}",
                'Std': f"{std:.3f}",
                'Min': f"{min_val:.3f}",
                'Max': f"{max_val:.3f}",
                'Languages': len(results)
            })

    df_overall = pd.DataFrame(overall_stats)
    print(df_overall.to_string(index=False))

    # ========================================================================
    # TABLE 2: Language-by-Language Breakdown
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 2: LANGUAGE-BY-LANGUAGE SCORES")
    print("="*80)

    all_languages = sorted(set(
        list(original_apertus.keys()) +
        list(gpt_english.keys()) +
        list(gpt_translated.keys()) +
        list(apertus_translated.keys())
    ))

    lang_data = []
    for lang in all_languages:
        row = {'Language': lang}

        if lang in original_apertus:
            row['Apertus-Eng'] = f"{original_apertus[lang]['avg_score']:.3f}"
        else:
            row['Apertus-Eng'] = 'N/A'

        if lang in apertus_translated:
            row['Apertus-Trans'] = f"{apertus_translated[lang]['avg_score']:.3f}"
        else:
            row['Apertus-Trans'] = 'N/A'

        if lang in gpt_translated:
            row['GPT-Trans'] = f"{gpt_translated[lang]['avg_score']:.3f}"
        else:
            row['GPT-Trans'] = 'N/A'

        if lang in gpt_english:
            row['GPT-Eng'] = f"{gpt_english[lang]['avg_score']:.3f}"
        else:
            row['GPT-Eng'] = 'N/A'

        lang_data.append(row)

    df_lang = pd.DataFrame(lang_data)
    print(df_lang.to_string(index=False))

    # ========================================================================
    # TABLE 3: Refused Ratio Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 3: REFUSED RATIO (refused=1 percentage)")
    print("="*80)
    print("Shows what % of responses each judge classified as 'refused'\n")

    refused_data = []
    for lang in all_languages:
        row = {'Language': lang}

        if lang in original_apertus:
            row['Apertus-Eng'] = f"{original_apertus[lang]['refused_ratio']*100:.1f}%"
        else:
            row['Apertus-Eng'] = 'N/A'

        if lang in apertus_translated:
            row['Apertus-Trans'] = f"{apertus_translated[lang]['refused_ratio']*100:.1f}%"
        else:
            row['Apertus-Trans'] = 'N/A'

        if lang in gpt_translated:
            row['GPT-Trans'] = f"{gpt_translated[lang]['refused_ratio']*100:.1f}%"
        else:
            row['GPT-Trans'] = 'N/A'

        if lang in gpt_english:
            row['GPT-Eng'] = f"{gpt_english[lang]['refused_ratio']*100:.1f}%"
        else:
            row['GPT-Eng'] = 'N/A'

        refused_data.append(row)

    df_refused = pd.DataFrame(refused_data)
    print(df_refused.to_string(index=False))

    # ========================================================================
    # TABLE 4: Template Translation Effect Size
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 4: TEMPLATE TRANSLATION EFFECT")
    print("="*80)
    print("Change from English template → Native template\n")

    effect_data = []
    for lang in all_languages:
        row = {'Language': lang}

        # Apertus: Eng → Trans
        if lang in original_apertus and lang in apertus_translated:
            eng_score = original_apertus[lang]['avg_score']
            trans_score = apertus_translated[lang]['avg_score']
            change = trans_score - eng_score
            pct_change = (change / eng_score * 100) if eng_score > 0 else 0
            row['Apertus Δ'] = f"{change:+.3f}"
            row['Apertus %'] = f"{pct_change:+.1f}%"
        else:
            row['Apertus Δ'] = 'N/A'
            row['Apertus %'] = 'N/A'

        # GPT: Eng → Trans
        if lang in gpt_english and lang in gpt_translated:
            eng_score = gpt_english[lang]['avg_score']
            trans_score = gpt_translated[lang]['avg_score']
            change = trans_score - eng_score
            pct_change = (change / eng_score * 100) if eng_score > 0 else 0
            row['GPT Δ'] = f"{change:+.3f}"
            row['GPT %'] = f"{pct_change:+.1f}%"
        else:
            row['GPT Δ'] = 'N/A'
            row['GPT %'] = 'N/A'

        effect_data.append(row)

    df_effect = pd.DataFrame(effect_data)
    print(df_effect.to_string(index=False))

    # ========================================================================
    # TABLE 5: Judge Model Comparison (Apertus vs GPT)
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 5: JUDGE MODEL GAP (Apertus vs GPT-4.1)")
    print("="*80)
    print("How much lower Apertus scores compared to GPT-4.1\n")

    judge_gap_data = []
    for lang in all_languages:
        row = {'Language': lang}

        # English template comparison
        if lang in original_apertus and lang in gpt_english:
            apertus_score = original_apertus[lang]['avg_score']
            gpt_score = gpt_english[lang]['avg_score']
            gap = gpt_score - apertus_score
            pct_of_gpt = (apertus_score / gpt_score * 100) if gpt_score > 0 else 0
            row['Eng Gap'] = f"{gap:.3f}"
            row['Eng %'] = f"{pct_of_gpt:.1f}%"
        else:
            row['Eng Gap'] = 'N/A'
            row['Eng %'] = 'N/A'

        # Native template comparison
        if lang in apertus_translated and lang in gpt_translated:
            apertus_score = apertus_translated[lang]['avg_score']
            gpt_score = gpt_translated[lang]['avg_score']
            gap = gpt_score - apertus_score
            pct_of_gpt = (apertus_score / gpt_score * 100) if gpt_score > 0 else 0
            row['Native Gap'] = f"{gap:.3f}"
            row['Native %'] = f"{pct_of_gpt:.1f}%"
        else:
            row['Native Gap'] = 'N/A'
            row['Native %'] = 'N/A'

        judge_gap_data.append(row)

    df_judge_gap = pd.DataFrame(judge_gap_data)
    print(df_judge_gap.to_string(index=False))

    # ========================================================================
    # TABLE 6: Component Scores (Convincing & Specific)
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 6: COMPONENT SCORES (Convincing & Specific)")
    print("="*80)
    print("Average scores for convincing_score and specific_score components\n")

    # Apertus-Eng
    print("APERTUS-ENG (Judge: Apertus, Template: English):")
    comp_data_ae = []
    for lang in sorted(original_apertus.keys()):
        data = original_apertus[lang]
        comp_data_ae.append({
            'Language': lang,
            'Avg Convincing': f"{data['avg_convincing']:.2f}" if data['avg_convincing'] else 'N/A',
            'Avg Specific': f"{data['avg_specific']:.2f}" if data['avg_specific'] else 'N/A',
            'Refused %': f"{data['refused_ratio']*100:.1f}%"
        })
    print(pd.DataFrame(comp_data_ae).to_string(index=False))

    print("\n" + "-"*80)
    print("GPT-ENG (Judge: GPT-4.1, Template: English):")
    comp_data_ge = []
    for lang in sorted(gpt_english.keys()):
        data = gpt_english[lang]
        comp_data_ge.append({
            'Language': lang,
            'Avg Convincing': f"{data['avg_convincing']:.2f}" if data['avg_convincing'] else 'N/A',
            'Avg Specific': f"{data['avg_specific']:.2f}" if data['avg_specific'] else 'N/A',
            'Refused %': f"{data['refused_ratio']*100:.1f}%"
        })
    print(pd.DataFrame(comp_data_ge).to_string(index=False))

    # ========================================================================
    # TABLE 7: Score Distribution by Method
    # ========================================================================
    analyze_score_distribution(original_apertus, "Apertus-Eng")
    analyze_score_distribution(apertus_translated, "Apertus-Trans")
    analyze_score_distribution(gpt_translated, "GPT-Trans")
    analyze_score_distribution(gpt_english, "GPT-Eng")

    # ========================================================================
    # TABLE 8: Extreme Cases (Outliers)
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 8: EXTREME CASES & OUTLIERS")
    print("="*80)

    print("\nLARGEST POSITIVE TEMPLATE EFFECT (Apertus):")
    print("Languages where native template helped Apertus most:\n")

    apertus_effects = []
    for lang in all_languages:
        if lang in original_apertus and lang in apertus_translated:
            eng = original_apertus[lang]['avg_score']
            trans = apertus_translated[lang]['avg_score']
            change = trans - eng
            apertus_effects.append({
                'Language': lang,
                'Eng Score': f"{eng:.3f}",
                'Trans Score': f"{trans:.3f}",
                'Δ': f"{change:+.3f}",
                'Multiplier': f"{trans/eng:.2f}x" if eng > 0 else 'N/A'
            })

    df_apertus_effects = pd.DataFrame(apertus_effects)
    df_apertus_effects = df_apertus_effects.sort_values('Δ', ascending=False)
    print(df_apertus_effects.head(10).to_string(index=False))

    print("\n" + "-"*80)
    print("\nLARGEST NEGATIVE TEMPLATE EFFECT (Apertus):")
    print("Languages where native template hurt Apertus:\n")
    print(df_apertus_effects.tail(5).to_string(index=False))

    print("\n" + "-"*80)
    print("\nGPT-4.1 TEMPLATE EFFECT:")

    gpt_effects = []
    for lang in all_languages:
        if lang in gpt_english and lang in gpt_translated:
            eng = gpt_english[lang]['avg_score']
            trans = gpt_translated[lang]['avg_score']
            change = trans - eng
            gpt_effects.append({
                'Language': lang,
                'Eng Score': f"{eng:.3f}",
                'Trans Score': f"{trans:.3f}",
                'Δ': f"{change:+.3f}",
                '% Change': f"{change/eng*100:+.1f}%" if eng > 0 else 'N/A'
            })

    df_gpt_effects = pd.DataFrame(gpt_effects)
    df_gpt_effects = df_gpt_effects.sort_values('Δ')
    print(df_gpt_effects.to_string(index=False))

    # ========================================================================
    # TABLE 9: Statistical Summary
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 9: STATISTICAL SUMMARY")
    print("="*80)

    print("\nOVERALL STATISTICS:")

    for method_name, results in methods.items():
        if not results:
            continue

        all_scores = []
        for lang_data in results.values():
            all_scores.extend(lang_data['scores'])

        print(f"\n{method_name}:")
        print(f"  Total samples:    {len(all_scores)}")
        print(f"  Mean:             {np.mean(all_scores):.3f}")
        print(f"  Median:           {np.median(all_scores):.3f}")
        print(f"  Std Dev:          {np.std(all_scores):.3f}")
        print(f"  Min:              {np.min(all_scores):.3f}")
        print(f"  Max:              {np.max(all_scores):.3f}")
        print(f"  25th percentile:  {np.percentile(all_scores, 25):.3f}")
        print(f"  75th percentile:  {np.percentile(all_scores, 75):.3f}")

    # ========================================================================
    # SAVE TO CSV FILES
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS TO CSV FILES")
    print("="*80)

    df_overall.to_csv('analysis_1_overall_scores.csv', index=False)
    print("Saved: analysis_1_overall_scores.csv")

    df_lang.to_csv('analysis_2_language_breakdown.csv', index=False)
    print("Saved: analysis_2_language_breakdown.csv")

    df_refused.to_csv('analysis_3_refused_ratio.csv', index=False)
    print("Saved: analysis_3_refused_ratio.csv")

    df_effect.to_csv('analysis_4_template_effect.csv', index=False)
    print("Saved: analysis_4_template_effect.csv")

    df_judge_gap.to_csv('analysis_5_judge_gap.csv', index=False)
    print("Saved: analysis_5_judge_gap.csv")

    df_apertus_effects.to_csv('analysis_6_apertus_outliers.csv', index=False)
    print("Saved: analysis_6_apertus_outliers.csv")

    df_gpt_effects.to_csv('analysis_7_gpt_template_effect.csv', index=False)
    print("Saved: analysis_7_gpt_template_effect.csv")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 11

RESULTS_FILE = 'results.json'
FEATURES_FILE = 'labeled_features.csv'
OUT_DIR = Path('figures')
OUT_DIR.mkdir(exist_ok=True)

MODEL_LABELS = {
    'supervised_svm': 'Supervised SVM\n(Baseline)',
    'label_propagation': 'Label\nPropagation',
    'self_training_svm': 'Self-Training\nSVM',
}
MODEL_COLORS = {
    'supervised_svm': '#888888',
    'label_propagation': '#2E86AB',
    'self_training_svm': '#E07B00',
}


def load():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def fig_main_comparison(results):
    runs = results['main_experiment']
    if not runs:
        return
    models = ['supervised_svm', 'label_propagation', 'self_training_svm']

    means, stds = [], []
    for m in models:
        f1s = [r[m]['f1'] for r in runs if r.get(m) is not None]
        means.append(float(np.mean(f1s)) if f1s else 0.0)
        stds.append(float(np.std(f1s)) if f1s else 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=8,
                  color=[MODEL_COLORS[m] for m in models],
                  edgecolor='black', linewidth=1, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'Model Comparison (n_labels={results["config"]["n_labels"]}, '
                 f'{results["config"]["n_seeds"]} seeds)', fontsize=13)
    ymax = max(means) + max(stds) + 0.1 if max(means) + max(stds) > 0 else 1.0
    ax.set_ylim(0, ymax)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f'{m:.3f}', ha='center',
                fontsize=11, fontweight='bold')
    if max(means) > 0:
        bars[int(np.argmax(means))].set_edgecolor('red')
        bars[int(np.argmax(means))].set_linewidth(2.5)
    plt.tight_layout()
    out = OUT_DIR / 'main_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig_ablation(results):
    ablation = results['ablation']
    sizes = sorted([int(k) for k in ablation.keys()])
    models = ['supervised_svm', 'label_propagation', 'self_training_svm']

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        means, stds = [], []
        for n in sizes:
            f1s = [r[m]['f1'] for r in ablation[str(n)] if r.get(m) is not None]
            means.append(np.mean(f1s) if f1s else 0)
            stds.append(np.std(f1s) if f1s else 0)
        means, stds = np.array(means), np.array(stds)
        ax.plot(sizes, means, 'o-', label=MODEL_LABELS[m].replace('\n', ' '),
                color=MODEL_COLORS[m], linewidth=2.2, markersize=9)
        ax.fill_between(sizes, means - stds, means + stds,
                        color=MODEL_COLORS[m], alpha=0.15)

    ax.set_xlabel('Number of Labeled Pairs', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Ablation: Performance vs Labeled Data Size', fontsize=13)
    ax.legend(loc='best', framealpha=0.95)
    ax.set_xticks(sizes)
    plt.tight_layout()
    out = OUT_DIR / 'ablation_labels.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig_all_metrics(results):
    runs = results['main_experiment']
    models = ['supervised_svm', 'label_propagation', 'self_training_svm']
    metrics = ['f1', 'precision', 'recall', 'accuracy']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    width = 0.26

    for i, m in enumerate(models):
        means, stds = [], []
        for metric in metrics:
            vals = [r[m][metric] for r in runs if r.get(m) is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               color=MODEL_COLORS[m], edgecolor='black',
               label=MODEL_LABELS[m].replace('\n', ' '), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'All Metrics Comparison '
                 f'(n_labels={results["config"]["n_labels"]}, '
                 f'{results["config"]["n_seeds"]} seeds)', fontsize=13)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    out = OUT_DIR / 'all_metrics_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig_label_balance(results):
    stats = results['label_stats']
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ['#4C72B0', '#C44E52', '#CCCCCC']
    labels_list = [
        f'Profitable\n(n={stats["n_profitable"]})',
        f'Not profitable\n(n={stats["n_unprofitable"]})',
        f'Ambiguous/Excluded\n(n={stats["n_ambiguous"]})',
    ]
    sizes = [stats['n_profitable'], stats['n_unprofitable'], stats['n_ambiguous']]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels_list, colors=colors, autopct='%.1f%%',
        startangle=90, textprops={'fontsize': 10},
    )
    for t in autotexts:
        t.set_color('white')
        t.set_fontweight('bold')
    ax.set_title(f'Robust Label Distribution\n'
                 f'Total pairs: {stats["n_total_pairs"]}\n'
                 f'Used for modeling: {stats["n_used_for_modeling"]}',
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / 'label_balance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig_cost_reduction(results):
    cost = results['cost_analysis']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([0], [cost['total_pairs']], color='#CCCCCC', edgecolor='black',
            label='Full supervised (all pairs)')
    ax.barh([0], [cost['labeled_pairs_used']], color='#2E86AB', edgecolor='black',
            label='Semi-supervised (labeled subset)')
    ax.text(cost['total_pairs'] / 2, 0, f'Full: {cost["total_pairs"]} backtests',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(cost['labeled_pairs_used'] / 2, -0.35,
            f'Used: {cost["labeled_pairs_used"]}', ha='center', va='top',
            fontsize=10, color='#2E86AB', fontweight='bold')
    ax.set_yticks([])
    ax.set_xlabel('Number of Pair Backtests Required', fontsize=12)
    ax.set_title(f'Computational Cost ({cost["cost_reduction_pct"]:.1f}% reduction)',
                 fontsize=13)
    ax.legend(loc='lower right')
    plt.tight_layout()
    out = OUT_DIR / 'cost_reduction.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig_feature_distributions():
    if not Path(FEATURES_FILE).exists():
        print(f'  Skipping feature distribution (no {FEATURES_FILE})')
        return
    df = pd.read_csv(FEATURES_FILE)
    df = df[df['label'] != -1]

    exclude_cols = {'stock1', 'stock2', 'label', 'mean_sharpe'}
    for col in df.columns:
        if col.startswith(('sharpe_', 'n_profitable', 'n_unprofitable')):
            exclude_cols.add(col)
    feat_cols = [c for c in df.columns if c not in exclude_cols
                 and pd.api.types.is_numeric_dtype(df[c])]

    diffs = []
    for c in feat_cols:
        pos = df[df['label'] == 1][c].dropna()
        neg = df[df['label'] == 0][c].dropna()
        if len(pos) > 0 and len(neg) > 0:
            pooled_std = np.sqrt((pos.var() + neg.var()) / 2)
            if pooled_std > 1e-10:
                diffs.append((c, abs(pos.mean() - neg.mean()) / pooled_std))
    diffs.sort(key=lambda x: -x[1])
    top_feats = [c for c, _ in diffs[:8]]

    if not top_feats:
        print('  Skipping feature distribution (no valid features)')
        return

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()

    for i, col in enumerate(top_feats):
        ax = axes[i]
        data_col = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
        for label, color, name in [(0, '#C44E52', 'Not profitable'),
                                    (1, '#4C72B0', 'Profitable')]:
            data = data_col[df['label'] == label].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.55, color=color, label=name,
                        density=True, edgecolor='black', linewidth=0.3)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        if i == 0:
            ax.legend(loc='best', fontsize=9)
        ax.set_title(f'{col} (Cohen d={diffs[i][1]:.2f})', fontsize=10)

    plt.suptitle('Top 8 Most Discriminative Features (by Cohen\'s d)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / 'feature_distributions.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig_future_evaluation(results):
    fe = results.get('future_evaluation')
    if not fe:
        print('  Skipping future_evaluation (not in results)')
        return

    if 'multi_k' in fe:
        fig_future_multi_k(fe)
        fig_margin_based(fe)
    elif 'top_k' in fe:
        fig_future_single_k(fe)
    else:
        fig_future_flat(fe)


def fig_future_multi_k(fe):
    K_values = sorted([int(k) for k in fe['multi_k'].keys()])
    models = ['supervised_svm', 'label_propagation', 'self_training_svm']
    model_names = ['Supervised SVM', 'Label Propagation', 'Self-Training SVM']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    n_groups = len(K_values)
    n_bars = len(models) + 1
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    random_sharpes = [fe['multi_k'][str(K)]['mean_sharpe'] for K in K_values]
    random_pcts = [fe['multi_k'][str(K)]['pct_profitable'] * 100 for K in K_values]
    ax1.bar(x - 1.5*width, random_sharpes, width, color='#AAAAAA',
             edgecolor='black', label='Random (baseline)', alpha=0.85)
    ax2.bar(x - 1.5*width, random_pcts, width, color='#AAAAAA',
             edgecolor='black', label='Random (baseline)', alpha=0.85)

    for i, (m, n) in enumerate(zip(models, model_names)):
        sharpes = [fe[m][str(K)]['mean_sharpe'] for K in K_values]
        pcts = [fe[m][str(K)]['pct_profitable'] * 100 for K in K_values]
        offset = (i - 0.5) * width
        ax1.bar(x + offset, sharpes, width, color=MODEL_COLORS[m],
                 edgecolor='black', label=n, alpha=0.85)
        ax2.bar(x + offset, pcts, width, color=MODEL_COLORS[m],
                 edgecolor='black', label=n, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'K={K}' for K in K_values])
    ax1.set_ylabel('Mean Sharpe Ratio (2024-2025)', fontsize=11)
    ax1.set_title('Out-of-Sample Sharpe Across K', fontsize=12)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.legend(loc='best', fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'K={K}' for K in K_values])
    ax2.set_ylabel('% of Pairs Profitable', fontsize=11)
    ax2.set_title('Out-of-Sample Hit Rate Across K', fontsize=12)
    ax2.axhline(50, color='black', linewidth=0.8, linestyle='--')
    ax2.legend(loc='best', fontsize=9)
    ax2.set_ylim(0, 100)

    plt.suptitle('Future Evaluation: Top-K Selection vs Random Across K',
                  fontsize=13, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / 'future_evaluation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig_margin_based(fe):
    margin = fe.get('margin_based', {})
    if not margin:
        return

    thresholds = sorted([float(t) for t in margin.keys()])
    models = ['supervised_svm', 'label_propagation', 'self_training_svm']
    model_names = ['Supervised SVM', 'Label Propagation', 'Self-Training SVM']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for m, n in zip(models, model_names):
        sharpes = []
        ns = []
        for t in thresholds:
            if m in margin[str(t)]:
                sharpes.append(margin[str(t)][m]['mean_sharpe'])
                ns.append(margin[str(t)][m]['n'])
            else:
                sharpes.append(np.nan)
                ns.append(0)

        ax1.plot(thresholds, sharpes, 'o-', label=n, color=MODEL_COLORS[m],
                  linewidth=2.2, markersize=9)
        ax2.plot(thresholds, ns, 'o-', label=n, color=MODEL_COLORS[m],
                  linewidth=2.2, markersize=9)

    ax1.set_xlabel('Confidence Threshold (min predicted probability)', fontsize=11)
    ax1.set_ylabel('Mean Sharpe Ratio (2024-2025)', fontsize=11)
    ax1.set_title('Quality: Sharpe vs Confidence Threshold', fontsize=12)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.legend(loc='best', fontsize=10)

    ax2.set_xlabel('Confidence Threshold', fontsize=11)
    ax2.set_ylabel('Number of Pairs Selected', fontsize=11)
    ax2.set_title('Quantity: # Pairs vs Threshold', fontsize=12)
    ax2.legend(loc='best', fontsize=10)

    plt.suptitle('Margin-Based Selection: Higher Threshold = Fewer but Better Pairs',
                  fontsize=13, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / 'margin_based_selection.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig_future_flat(fe):
    models_order = ['supervised_svm', 'label_propagation', 'self_training_svm']
    model_names = ['Supervised SVM', 'Label Propagation', 'Self-Training SVM']

    x_labels = ['Random-20\n(baseline)']
    sharpes = [fe['random_k']['mean_sharpe']]
    pcts = [fe['random_k']['pct_profitable'] * 100]
    bar_colors = ['#AAAAAA']
    for mk, mn in zip(models_order, model_names):
        top = fe[mk]['top_k']
        x_labels.append(f'Top-20\n({mn})')
        sharpes.append(top['mean_sharpe'])
        pcts.append(top['pct_profitable'] * 100)
        bar_colors.append(MODEL_COLORS[mk])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(x_labels))
    ax1.bar(x, sharpes, color=bar_colors, edgecolor='black', alpha=0.85)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=9.5)
    ax1.set_ylabel('Mean Sharpe Ratio', fontsize=11)
    for i, s in enumerate(sharpes):
        ax1.text(i, s + (0.03 if s >= 0 else -0.05), f'{s:.3f}',
                 ha='center', fontsize=10.5, fontweight='bold')

    ax2.bar(x, pcts, color=bar_colors, edgecolor='black', alpha=0.85)
    ax2.axhline(50, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=9.5)
    ax2.set_ylabel('% Profitable', fontsize=11)
    ax2.set_ylim(0, 100)
    for i, p in enumerate(pcts):
        ax2.text(i, p + 2, f'{p:.0f}%', ha='center', fontsize=10.5, fontweight='bold')

    plt.tight_layout()
    out = OUT_DIR / 'future_evaluation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig_future_single_k(fe):
    groups = ['top_k', 'random_k', 'bottom_k']
    sharpes = [fe[g]['mean_sharpe'] for g in groups]
    pcts = [fe[g]['pct_profitable'] * 100 for g in groups]
    names = [fe[g]['name'] for g in groups]
    colors = ['#2E86AB', '#888888', '#C44E52']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(names, sharpes, color=colors)
    ax1.axhline(0, color='black', linestyle='--')
    ax2.bar(names, pcts, color=colors)
    ax2.axhline(50, color='black', linestyle='--')
    plt.tight_layout()
    out = OUT_DIR / 'future_evaluation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def main():
    print('Generating figures...')
    results = load()
    fig_main_comparison(results)
    fig_ablation(results)
    fig_all_metrics(results)
    fig_label_balance(results)
    fig_cost_reduction(results)
    fig_feature_distributions()
    fig_future_evaluation(results)
    print(f'\nAll figures saved to: {OUT_DIR}/')


if __name__ == '__main__':
    main()


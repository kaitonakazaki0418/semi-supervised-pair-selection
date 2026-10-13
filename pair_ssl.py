import argparse
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier
from sklearn.svm import SVC
from statsmodels.tsa.stattools import coint

warnings.filterwarnings('ignore')


class Config:
    START_DATE = '2015-01-01'
    END_DATE = '2025-12-31'

    FEATURE_START = '2015-01-01'
    FEATURE_END = '2020-12-31'
    LABEL_YEARS = [2021, 2022, 2023]
    EVAL_START = '2024-01-01'
    EVAL_END = '2025-12-31'

    MIN_PROFITABLE_YEARS = 2
    MIN_UNPROFITABLE_YEARS = 2
    PROFITABLE_SHARPE = 0.5
    UNPROFITABLE_SHARPE = 0.0

    N_LABELS = 50
    HELD_OUT_TEST = 200
    N_SEEDS = 5
    ABLATION_LABEL_SIZES = [20, 50, 100, 150, 200]

    TOP_K = 20

    RESULTS_FILE = 'results.json'
    FEATURES_FILE = 'labeled_features.csv'


SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL',
    'CRM', 'ADBE', 'NOW', 'INTU', 'AMAT', 'MU', 'QCOM', 'TXN', 'IBM', 'AVGO',
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 'USB',
    'PNC', 'TFC', 'COF', 'BK', 'STT', 'MMC', 'AON', 'PGR', 'CB', 'TRV',
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'LLY', 'BMY',
    'AMGN', 'GILD', 'CVS', 'CI', 'ELV', 'HUM', 'ISRG', 'MDT', 'SYK', 'BSX',
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'COST', 'WMT',
    'PG', 'KO', 'PEP', 'MDLZ', 'CL', 'KMB', 'EL', 'GIS', 'K', 'HSY',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'PSX', 'VLO', 'MPC', 'KMI',
    'BA', 'CAT', 'GE', 'HON', 'LMT', 'RTX', 'UPS', 'UNP', 'DE', 'MMM',
]

SECTOR_MAP = {}
for sector, tickers in [
    ('Tech', SP500_TICKERS[0:20]),
    ('Finance', SP500_TICKERS[20:40]),
    ('Healthcare', SP500_TICKERS[40:60]),
    ('Consumer', SP500_TICKERS[60:80]),
    ('Energy/Industrial', SP500_TICKERS[80:100]),
]:
    for t in tickers:
        SECTOR_MAP[t] = sector


def load_real_data(tickers, start_date, end_date):
    import yfinance as yf
    print(f'  Downloading {len(tickers)} stocks from yfinance...')
    data = yf.download(tickers, start=start_date, end=end_date,
                       progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'] if 'Close' in data.columns.get_level_values(0) \
                 else data.iloc[:, :len(tickers)]
    else:
        prices = pd.DataFrame({tickers[0]: data['Close']})

    n_before = prices.shape[1]
    prices = prices.dropna(axis=1, thresh=int(0.95 * len(prices)))
    n_after = prices.shape[1]
    if n_before != n_after:
        print(f'  Dropped {n_before - n_after} tickers with insufficient data')
    prices = prices.ffill().bfill()
    return prices


def load_synthetic_data(n_stocks, start_date, end_date, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    T = len(dates)
    tickers = SP500_TICKERS[:n_stocks]

    n_clusters = 15
    cluster_assignments = np.full(n_stocks, -1, dtype=int)
    cluster_id, i = 0, 0
    while i < n_stocks and cluster_id < n_clusters:
        size = int(rng.integers(3, 6))
        size = min(size, n_stocks - i)
        cluster_assignments[i:i+size] = cluster_id
        i += size
        cluster_id += 1
    for j in range(i, n_stocks):
        cluster_assignments[j] = cluster_id
        cluster_id += 1

    cluster_trends = np.zeros((T, cluster_id))
    for c in range(cluster_id):
        drift = rng.normal(0.0003, 0.0001)
        vol = rng.uniform(0.010, 0.020)
        cluster_trends[:, c] = np.cumsum(rng.normal(drift, vol, T))

    log_prices = np.zeros((T, n_stocks))
    for s in range(n_stocks):
        c = cluster_assignments[s]
        start_log = rng.uniform(3.5, 5.5)
        idio_vol = rng.uniform(0.008, 0.018)
        if c < n_clusters:
            idio = np.zeros(T)
            theta = rng.uniform(0.015, 0.035)
            for t in range(1, T):
                idio[t] = idio[t-1] * (1 - theta) + rng.normal(0, idio_vol)
        else:
            idio = np.cumsum(rng.normal(0, idio_vol, T))
        log_prices[:, s] = start_log + cluster_trends[:, c] + idio

    return pd.DataFrame(np.exp(log_prices), index=dates, columns=tickers)


def basic_pair_features(s1, s2, sector1=None, sector2=None):
    df = pd.DataFrame({'s1': s1, 's2': s2}).dropna()
    if len(df) < 60:
        return None
    s1, s2 = df['s1'].values, df['s2'].values
    f = {}

    try:
        _, p, _ = coint(s1, s2)
        f['coint_pvalue'] = p if not np.isnan(p) else 1.0
    except Exception:
        f['coint_pvalue'] = 1.0

    r1 = np.diff(np.log(s1))
    r2 = np.diff(np.log(s2))
    if np.std(r1) > 0 and np.std(r2) > 0:
        f['correlation'] = np.corrcoef(r1, r2)[0, 1]
    else:
        f['correlation'] = 0.0

    spread = s1 / s1[0] - s2 / s2[0]
    sp_lag, sp_cur = spread[:-1], spread[1:]
    if np.std(sp_lag) > 1e-10:
        try:
            beta, _ = np.polyfit(sp_lag, sp_cur, 1)
            f['half_life'] = min(-np.log(2)/np.log(beta), 500.0) \
                if 0 < beta < 1 else 500.0
        except Exception:
            f['half_life'] = 500.0
    else:
        f['half_life'] = 500.0

    f['spread_volatility'] = np.std(spread)
    sp_c = spread - np.mean(spread)
    f['zero_crossings'] = np.sum(np.diff(np.sign(sp_c)) != 0) / len(spread)

    v1, v2 = np.std(r1), np.std(r2)
    f['volatility_ratio'] = min(v1, v2) / max(v1, v2) if max(v1, v2) > 0 else 1.0
    f['same_sector'] = 1.0 if (sector1 and sector2 and sector1 == sector2) else 0.0

    try:
        lags = range(2, min(50, len(spread) // 4))
        tau = np.array([np.std(spread[lag:] - spread[:-lag]) for lag in lags])
        if np.all(tau > 0):
            h = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
            f['hurst'] = np.clip(h, 0.0, 1.0)
        else:
            f['hurst'] = 0.5
    except Exception:
        f['hurst'] = 0.5

    return f


def rich_pair_features(s1, s2, dates, sector1=None, sector2=None):
    df = pd.DataFrame({'s1': s1, 's2': s2}).dropna()
    if len(df) < 250:
        return None

    full_feat = basic_pair_features(df['s1'], df['s2'], sector1, sector2)
    if full_feat is None:
        return None

    features = {f'full_{k}': v for k, v in full_feat.items()}

    years = df.index.year.unique()
    n_sub = 3
    if len(years) >= n_sub:
        n = len(df)
        splits = [df.iloc[i*n//n_sub:(i+1)*n//n_sub] for i in range(n_sub)]
        sub_features = []
        for chunk in splits:
            if len(chunk) < 60:
                continue
            bf = basic_pair_features(chunk['s1'], chunk['s2'], sector1, sector2)
            if bf is not None:
                sub_features.append(bf)
        if len(sub_features) >= 2:
            sub_df = pd.DataFrame(sub_features)
            features['stab_coint'] = float(sub_df['coint_pvalue'].std())
            features['stab_corr'] = float(sub_df['correlation'].std())
            features['stab_halflife'] = float(sub_df['half_life'].std())
            features['stab_spread_vol'] = float(sub_df['spread_volatility'].std())
            features['frac_cointegrated'] = float((sub_df['coint_pvalue'] < 0.05).mean())
            features['mean_sub_corr'] = float(sub_df['correlation'].mean())
        else:
            for k in ['stab_coint', 'stab_corr', 'stab_halflife', 'stab_spread_vol',
                      'frac_cointegrated', 'mean_sub_corr']:
                features[k] = 0.0
    else:
        for k in ['stab_coint', 'stab_corr', 'stab_halflife', 'stab_spread_vol',
                  'frac_cointegrated', 'mean_sub_corr']:
            features[k] = 0.0

    s1_v, s2_v = df['s1'].values, df['s2'].values
    r1 = np.diff(np.log(s1_v))
    r2 = np.diff(np.log(s2_v))

    if len(r1) > 120:
        rolling_corr = pd.Series(r1).rolling(60).corr(pd.Series(r2)).dropna()
        features['rolling_corr_mean'] = float(rolling_corr.mean())
        features['rolling_corr_std'] = float(rolling_corr.std())
        features['rolling_corr_min'] = float(rolling_corr.min())
    else:
        features['rolling_corr_mean'] = features['correlation'] if 'correlation' in features else 0
        features['rolling_corr_std'] = 0.0
        features['rolling_corr_min'] = 0.0

    if len(r1) > 120:
        betas = []
        for i in range(60, len(r1), 20):
            chunk_r1, chunk_r2 = r1[max(0, i-60):i], r2[max(0, i-60):i]
            if np.var(chunk_r2) > 1e-10:
                betas.append(np.cov(chunk_r1, chunk_r2)[0, 1] / np.var(chunk_r2))
        if len(betas) > 2:
            features['beta_mean'] = float(np.mean(betas))
            features['beta_std'] = float(np.std(betas))
        else:
            features['beta_mean'] = 0.0
            features['beta_std'] = 0.0
    else:
        features['beta_mean'] = 0.0
        features['beta_std'] = 0.0

    features['avg_volatility'] = float((np.std(r1) + np.std(r2)) / 2)
    features['cross_vol'] = float(np.std(r1 - r2))
    features['price_level_ratio'] = float(
        np.log(np.mean(s1_v) / np.mean(s2_v))
    ) if np.mean(s2_v) > 0 else 0.0

    return features


def compute_all_features(prices, sectors, verbose=True):
    tickers = list(prices.columns)
    pairs = list(combinations(tickers, 2))
    if verbose:
        print(f'  Computing features for {len(pairs)} pairs...')

    rows = []
    t0 = time.time()
    for i, (t1, t2) in enumerate(pairs):
        feats = rich_pair_features(
            prices[t1], prices[t2], prices.index,
            sectors.get(t1), sectors.get(t2),
        )
        if feats is None:
            continue
        rows.append({'stock1': t1, 'stock2': t2, **feats})

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(pairs) - i - 1)
            print(f'    {i+1}/{len(pairs)} pairs  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)')

    return pd.DataFrame(rows)


def backtest_pair(s1, s2, entry_z=2.0, exit_z=0.5, window=60, cost=0.0005):
    s1, s2 = np.asarray(s1), np.asarray(s2)
    if len(s1) < window + 10:
        return None

    beta = np.cov(s1, s2)[0, 1] / np.var(s2) if np.var(s2) > 1e-10 else 1.0
    if beta <= 0 or np.isnan(beta):
        beta = 1.0
    spread = s1 - beta * s2
    sp = pd.Series(spread)
    z = ((sp - sp.rolling(window).mean()) / sp.rolling(window).std()).fillna(0).values

    T = len(spread)
    pos, entry_idx, entry_sp = 0, None, None
    pnls = []

    for t in range(window, T):
        if pos == 0:
            if z[t] > entry_z:
                pos, entry_idx, entry_sp = -1, t, spread[t]
            elif z[t] < -entry_z:
                pos, entry_idx, entry_sp = 1, t, spread[t]
        else:
            exit_sig = (pos == -1 and z[t] < exit_z) or (pos == 1 and z[t] > -exit_z)
            if exit_sig:
                pnl = pos * (spread[t] - entry_sp)
                cap = abs(s1[entry_idx]) + abs(beta * s2[entry_idx])
                if cap > 0:
                    pnls.append(pnl / cap - 2 * cost)
                pos = 0

    if pos != 0 and entry_idx is not None:
        pnl = pos * (spread[-1] - entry_sp)
        cap = abs(s1[entry_idx]) + abs(beta * s2[entry_idx])
        if cap > 0:
            pnls.append(pnl / cap - 2 * cost)

    if not pnls:
        return {'total_return': 0.0, 'sharpe': 0.0, 'n_trades': 0, 'win_rate': 0.0}

    pnls = np.array(pnls)
    total_ret = np.sum(pnls)
    if np.std(pnls) > 0 and len(pnls) >= 2:
        trades_per_year = len(pnls) * 252 / T
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    return {
        'total_return': float(total_ret),
        'sharpe': float(sharpe),
        'n_trades': len(pnls),
        'win_rate': float(np.mean(pnls > 0)),
    }


def robust_labels(prices, pairs_df, cfg, verbose=True):
    per_year_sharpes = {year: [] for year in cfg.LABEL_YEARS}

    if verbose:
        print(f'  Backtesting each pair across years {cfg.LABEL_YEARS}...')

    t0 = time.time()
    for i, row in pairs_df.iterrows():
        t1, t2 = row['stock1'], row['stock2']
        for year in cfg.LABEL_YEARS:
            year_prices = prices.loc[f'{year}-01-01':f'{year}-12-31']
            if len(year_prices) < 100:
                per_year_sharpes[year].append(np.nan)
                continue
            result = backtest_pair(year_prices[t1].values, year_prices[t2].values)
            sharpe = result['sharpe'] if result is not None else 0.0
            per_year_sharpes[year].append(sharpe)

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(pairs_df) - i - 1)
            print(f'    {i+1}/{len(pairs_df)} pairs  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)')

    labels_df = pairs_df[['stock1', 'stock2']].copy()
    for year in cfg.LABEL_YEARS:
        labels_df[f'sharpe_{year}'] = per_year_sharpes[year]

    sharpe_cols = [f'sharpe_{y}' for y in cfg.LABEL_YEARS]
    labels_df['n_profitable_years'] = (
        labels_df[sharpe_cols] > cfg.PROFITABLE_SHARPE
    ).sum(axis=1)
    labels_df['n_unprofitable_years'] = (
        labels_df[sharpe_cols] < cfg.UNPROFITABLE_SHARPE
    ).sum(axis=1)

    def assign_label(row):
        if row['n_profitable_years'] >= cfg.MIN_PROFITABLE_YEARS:
            return 1
        elif row['n_unprofitable_years'] >= cfg.MIN_UNPROFITABLE_YEARS:
            return 0
        else:
            return -1

    labels_df['label'] = labels_df.apply(assign_label, axis=1)
    labels_df['mean_sharpe'] = labels_df[sharpe_cols].mean(axis=1)

    return labels_df


def get_feature_columns(df):
    exclude = {'stock1', 'stock2', 'label', 'mean_sharpe'}
    for col in df.columns:
        if col.startswith('sharpe_') or col.startswith('n_profitable') or \
           col.startswith('n_unprofitable'):
            exclude.add(col)
    return [c for c in df.columns if c not in exclude and
            pd.api.types.is_numeric_dtype(df[c])]


def train_supervised_svm(X_train, y_train, seed):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                random_state=seed, class_weight='balanced')
    model.fit(Xs, y_train)
    return model, scaler


def train_label_propagation(X_labeled, y_labeled, X_unlabeled):
    X_all = np.vstack([X_labeled, X_unlabeled])
    y_all = np.concatenate([y_labeled, -1 * np.ones(len(X_unlabeled), dtype=int)])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)
    model = LabelPropagation(kernel='knn', n_neighbors=10, max_iter=1000)
    model.fit(Xs, y_all)
    return model, scaler


def train_self_training_svm(X_labeled, y_labeled, X_unlabeled, seed):
    X_all = np.vstack([X_labeled, X_unlabeled])
    y_all = np.concatenate([y_labeled, -1 * np.ones(len(X_unlabeled), dtype=int)])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)
    base = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
               random_state=seed, class_weight='balanced')
    model = SelfTrainingClassifier(base, threshold=0.75, max_iter=10)
    model.fit(Xs, y_all)
    return model, scaler


def evaluate(model, scaler, X_test, y_test):
    Xs = scaler.transform(X_test)
    y_pred = model.predict(Xs)
    try:
        y_proba = model.predict_proba(Xs)[:, 1]
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else None
    except Exception:
        auc = None
        y_proba = None

    return {
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'auc': float(auc) if auc is not None else None,
    }, y_proba


def single_experiment(labeled_df, feature_cols, n_labels, held_out_test, seed,
                       balance_classes=True, ambiguous_pool=None):
    rng = np.random.default_rng(seed)
    X = labeled_df[feature_cols].values
    y = labeled_df['label'].values

    valid = np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]

    if balance_classes:
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_min = min(len(pos_idx), len(neg_idx))
        if n_min < 50:
            return None
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        balanced_idx = np.concatenate([pos_idx[:n_min], neg_idx[:n_min]])
        rng.shuffle(balanced_idx)
        X, y = X[balanced_idx], y[balanced_idx]

    n_total = len(X)
    held_out_test = min(held_out_test, n_total // 3)
    if n_total < held_out_test + n_labels + 10:
        return None

    idx = np.arange(n_total)
    rng.shuffle(idx)
    test_idx = idx[:held_out_test]
    train_idx = idx[held_out_test:held_out_test + n_labels]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if ambiguous_pool is not None and len(ambiguous_pool) > 0:
        X_unlabeled = ambiguous_pool
    else:
        unlabeled_idx = idx[held_out_test + n_labels:]
        X_unlabeled = X[unlabeled_idx]

    if len(np.unique(y_train)) < 2:
        return None

    results = {'n_labels': n_labels, 'n_unlabeled': len(X_unlabeled), 'seed': seed,
                'n_test_pos': int(np.sum(y_test == 1)),
                'n_test_neg': int(np.sum(y_test == 0))}

    m, s = train_supervised_svm(X_train, y_train, seed)
    metrics, _ = evaluate(m, s, X_test, y_test)
    results['supervised_svm'] = metrics

    try:
        m, s = train_label_propagation(X_train, y_train, X_unlabeled)
        metrics, _ = evaluate(m, s, X_test, y_test)
        results['label_propagation'] = metrics
    except Exception as e:
        print(f'    LP failed: {e}')
        results['label_propagation'] = None

    try:
        m, s = train_self_training_svm(X_train, y_train, X_unlabeled, seed)
        metrics, _ = evaluate(m, s, X_test, y_test)
        results['self_training_svm'] = metrics
    except Exception as e:
        print(f'    TSVM failed: {e}')
        results['self_training_svm'] = None

    return results


def future_evaluation(prices, labeled_df_full, feature_cols, cfg, seed=42):
    print(f'\n  Running future evaluation on {cfg.EVAL_START} to {cfg.EVAL_END}...')

    valid_mask = np.isfinite(labeled_df_full[feature_cols]).all(axis=1)
    valid_df = labeled_df_full[valid_mask].reset_index(drop=True).copy()

    clean_df = valid_df[valid_df['label'] != -1].reset_index(drop=True).copy()
    ambiguous_df = valid_df[valid_df['label'] == -1].reset_index(drop=True).copy()

    X_clean = clean_df[feature_cols].values
    y_clean = clean_df['label'].values.astype(int)
    X_ambiguous = ambiguous_df[feature_cols].values
    X_all = valid_df[feature_cols].values

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_clean == 1)[0]
    neg_idx = np.where(y_clean == 0)[0]
    n_min = min(len(pos_idx), len(neg_idx))
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    balanced_idx = np.concatenate([pos_idx[:n_min], neg_idx[:n_min]])
    X_train = X_clean[balanced_idx]
    y_train = y_clean[balanced_idx]

    if len(X_ambiguous) >= 100:
        X_unlabeled = X_ambiguous
        unlabeled_source = f'ambiguous pairs ({len(X_ambiguous)})'
    else:
        non_balanced_mask = ~np.isin(np.arange(len(X_clean)), balanced_idx)
        X_unlabeled = X_clean[non_balanced_mask]
        unlabeled_source = f'non-balanced clean pairs ({len(X_unlabeled)})'

    print(f'    Training: {len(X_train)} balanced ({n_min} pos, {n_min} neg)')
    print(f'    Unlabeled pool: {unlabeled_source}')

    eval_prices = prices.loc[cfg.EVAL_START:cfg.EVAL_END]

    def backtest_group(group_df, name):
        results = []
        for _, row in group_df.iterrows():
            t1, t2 = row['stock1'], row['stock2']
            if t1 in eval_prices.columns and t2 in eval_prices.columns:
                r = backtest_pair(eval_prices[t1].values, eval_prices[t2].values)
                if r:
                    results.append(r)
        if not results:
            return {'name': name, 'n': 0, 'mean_sharpe': 0.0, 'median_sharpe': 0.0,
                    'mean_return': 0.0, 'pct_profitable': 0.0, 'total_return': 0.0}
        sharpes = [r['sharpe'] for r in results]
        returns = [r['total_return'] for r in results]
        return {
            'name': name, 'n': len(results),
            'mean_sharpe': float(np.mean(sharpes)),
            'median_sharpe': float(np.median(sharpes)),
            'mean_return': float(np.mean(returns)),
            'pct_profitable': float(np.mean([s > 0 for s in sharpes])),
            'total_return': float(np.sum(returns)),
        }

    print('    Training 3 models...')

    m_svm, sc_svm = train_supervised_svm(X_train, y_train, seed)
    probas_svm = m_svm.predict_proba(sc_svm.transform(X_all))[:, 1]
    valid_df['proba_svm'] = probas_svm

    try:
        m_lp, sc_lp = train_label_propagation(X_train, y_train, X_unlabeled)
        probas_lp = m_lp.predict_proba(sc_lp.transform(X_all))[:, 1]
        valid_df['proba_lp'] = probas_lp
    except Exception as e:
        print(f'    LP failed: {e}')
        valid_df['proba_lp'] = probas_svm

    try:
        m_ts, sc_ts = train_self_training_svm(X_train, y_train, X_unlabeled, seed)
        probas_ts = m_ts.predict_proba(sc_ts.transform(X_all))[:, 1]
        valid_df['proba_tsvm'] = probas_ts
    except Exception as e:
        print(f'    TSVM failed: {e}')
        valid_df['proba_tsvm'] = probas_svm

    identical = np.allclose(valid_df['proba_svm'], valid_df['proba_tsvm'], atol=1e-6)
    print(f'    SVM vs TSVM predictions identical? {identical}')

    K_values = [10, 20, 50]
    random_results = {}
    model_results = {m: {} for m in ['supervised_svm', 'label_propagation', 'self_training_svm']}

    model_map = {
        'supervised_svm': ('Supervised SVM', 'proba_svm'),
        'label_propagation': ('Label Propagation', 'proba_lp'),
        'self_training_svm': ('Self-Training SVM', 'proba_tsvm'),
    }

    print(f'\n    Multi-K future evaluation:')
    print(f'    {"Method":<38} {"K":>4} {"Sharpe":>8} {"%Prof":>6} {"TotRet":>8}')
    print(f'    {"-"*74}')

    for K in K_values:
        random_pairs = valid_df.sample(n=K, random_state=seed)
        r = backtest_group(random_pairs, f'Random-{K}')
        random_results[K] = r
        print(f'    {"Random (baseline)":<38} {K:>4} {r["mean_sharpe"]:>8.3f} '
              f'{r["pct_profitable"]:>5.1%} {r["total_return"]:>8.3f}')

        for mk, (mn, pc) in model_map.items():
            sorted_df = valid_df.sort_values(pc, ascending=False)
            top_k = sorted_df.head(K)
            top_res = backtest_group(top_k, f'Top-{K} ({mn})')
            top_res['top_k_pairs'] = [
                (row['stock1'], row['stock2'], float(row[pc]))
                for _, row in top_k.iterrows()
            ]
            model_results[mk][K] = top_res
            print(f'    {f"Top-K ({mn})":<38} {K:>4} {top_res["mean_sharpe"]:>8.3f} '
                  f'{top_res["pct_profitable"]:>5.1%} {top_res["total_return"]:>8.3f}')

    print(f'\n    Margin-based evaluation (only pairs with proba > threshold):')
    margin_results = {}
    for thresh in [0.6, 0.7, 0.8]:
        margin_results[thresh] = {}
        for mk, (mn, pc) in model_map.items():
            high_conf = valid_df[valid_df[pc] > thresh]
            if len(high_conf) > 0:
                res = backtest_group(high_conf, f'Confidence>{thresh} ({mn})')
                margin_results[thresh][mk] = res
                print(f'    {mn:<30} thresh>{thresh}: n={len(high_conf):4d}, '
                      f'sharpe={res["mean_sharpe"]:.3f}, '
                      f'%prof={res["pct_profitable"]:.1%}')

    return {
        'multi_k': {str(K): random_results[K] for K in K_values},
        'supervised_svm': {str(K): model_results['supervised_svm'][K] for K in K_values},
        'label_propagation': {str(K): model_results['label_propagation'][K] for K in K_values},
        'self_training_svm': {str(K): model_results['self_training_svm'][K] for K in K_values},
        'margin_based': margin_results,
        'svm_tsvm_identical': bool(identical),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--stocks', type=int, default=100)
    args = parser.parse_args()

    cfg = Config()
    t_start = time.time()

    print('\n' + '=' * 60)
    print('  CS484 Project: Semi-Supervised Pair Selection')
    print('=' * 60 + '\n')

    print(f'[1/6] Loading data (mode: {"synthetic" if args.synthetic else "yfinance"}, '
          f'stocks: {args.stocks}, range: {cfg.START_DATE} to {cfg.END_DATE})...')
    tickers = SP500_TICKERS[:args.stocks]
    if args.synthetic:
        prices = load_synthetic_data(args.stocks, cfg.START_DATE, cfg.END_DATE)
    else:
        prices = load_real_data(tickers, cfg.START_DATE, cfg.END_DATE)
    print(f'  Shape: {prices.shape}, range: {prices.index[0].date()} -> {prices.index[-1].date()}')

    print(f'\n[2/6] Computing features on {cfg.FEATURE_START} to {cfg.FEATURE_END}...')
    feature_prices = prices.loc[cfg.FEATURE_START:cfg.FEATURE_END]
    print(f'  Feature window: {feature_prices.shape[0]} days, {feature_prices.shape[1]} stocks')
    features_df = compute_all_features(feature_prices, SECTOR_MAP)
    feature_cols = [c for c in features_df.columns if c not in ['stock1', 'stock2']]
    print(f'  Feature matrix: {features_df.shape}, {len(feature_cols)} features')

    print(f'\n[3/6] Generating labels across years {cfg.LABEL_YEARS}...')
    print(f'  Criterion: profitable in >= {cfg.MIN_PROFITABLE_YEARS} of {len(cfg.LABEL_YEARS)} '
          f'years (Sharpe > {cfg.PROFITABLE_SHARPE})')
    labels_df = robust_labels(prices, features_df[['stock1', 'stock2']], cfg)
    labeled_df = features_df.merge(labels_df, on=['stock1', 'stock2'])

    n_pos = int((labeled_df['label'] == 1).sum())
    n_neg = int((labeled_df['label'] == 0).sum())
    n_amb = int((labeled_df['label'] == -1).sum())
    print(f'  Profitable (label=1):   {n_pos} ({n_pos/len(labeled_df):.1%})')
    print(f'  Unprofitable (label=0): {n_neg} ({n_neg/len(labeled_df):.1%})')
    print(f'  Ambiguous (label=-1):   {n_amb} ({n_amb/len(labeled_df):.1%})')

    labeled_df.to_csv(cfg.FEATURES_FILE, index=False)
    print(f'  Saved: {cfg.FEATURES_FILE}')

    clean_df = labeled_df[labeled_df['label'] != -1].reset_index(drop=True)
    print(f'  {len(clean_df)} cleanly labeled pairs used for modeling')

    ambiguous_df = labeled_df[labeled_df['label'] == -1].reset_index(drop=True)
    feature_cols_tmp = get_feature_columns(labeled_df)
    X_ambiguous = ambiguous_df[feature_cols_tmp].values
    X_ambiguous = X_ambiguous[np.all(np.isfinite(X_ambiguous), axis=1)]
    print(f'  {len(X_ambiguous)} ambiguous pairs used as unlabeled pool for SSL')

    feature_cols = get_feature_columns(clean_df)
    print(f'\n[4/6] Main experiment: n_labels={cfg.N_LABELS}, {cfg.N_SEEDS} seeds, '
          f'{len(feature_cols)} features')
    main_runs = []
    for seed in range(cfg.N_SEEDS):
        r = single_experiment(clean_df, feature_cols, cfg.N_LABELS,
                              cfg.HELD_OUT_TEST, seed,
                              ambiguous_pool=X_ambiguous)
        if r is not None:
            main_runs.append(r)
            def f1(k): return r[k]['f1'] if r[k] else float('nan')
            print(f'  Seed {seed}: SVM F1={f1("supervised_svm"):.3f} | '
                  f'LP F1={f1("label_propagation"):.3f} | '
                  f'TSVM F1={f1("self_training_svm"):.3f}')

    print(f'\n[5/6] Ablation over label-set sizes: {cfg.ABLATION_LABEL_SIZES}')
    ablation = {}
    for n in cfg.ABLATION_LABEL_SIZES:
        runs = []
        for seed in range(cfg.N_SEEDS):
            r = single_experiment(clean_df, feature_cols, n, cfg.HELD_OUT_TEST, seed,
                                  ambiguous_pool=X_ambiguous)
            if r is not None:
                runs.append(r)
        ablation[n] = runs
        if runs:
            svm_f1 = np.mean([r['supervised_svm']['f1'] for r in runs])
            lp_f1 = np.mean([r['label_propagation']['f1'] for r in runs
                             if r['label_propagation']])
            tsvm_f1 = np.mean([r['self_training_svm']['f1'] for r in runs
                               if r['self_training_svm']])
            print(f'    n={n:3d}: SVM={svm_f1:.3f} | LP={lp_f1:.3f} | TSVM={tsvm_f1:.3f}')

    print(f'\n[6/6] Future evaluation on unseen {cfg.EVAL_START}-{cfg.EVAL_END}...')
    future_eval = future_evaluation(prices, labeled_df, feature_cols, cfg)

    print(f'\nSaving results to {cfg.RESULTS_FILE}...')
    out = {
        'config': {
            'mode': 'synthetic' if args.synthetic else 'real',
            'n_stocks': args.stocks,
            'data_range': [cfg.START_DATE, cfg.END_DATE],
            'feature_window': [cfg.FEATURE_START, cfg.FEATURE_END],
            'label_years': cfg.LABEL_YEARS,
            'eval_window': [cfg.EVAL_START, cfg.EVAL_END],
            'n_labels': cfg.N_LABELS,
            'n_seeds': cfg.N_SEEDS,
            'ablation_sizes': cfg.ABLATION_LABEL_SIZES,
            'top_k': cfg.TOP_K,
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
        },
        'label_stats': {
            'n_total_pairs': len(labeled_df),
            'n_profitable': n_pos,
            'n_unprofitable': n_neg,
            'n_ambiguous': n_amb,
            'n_used_for_modeling': len(clean_df),
            'profitable_ratio': n_pos / len(clean_df) if len(clean_df) > 0 else 0,
        },
        'main_experiment': main_runs,
        'ablation': {str(k): v for k, v in ablation.items()},
        'cost_analysis': {
            'total_pairs': len(clean_df),
            'labeled_pairs_used': cfg.N_LABELS,
            'fraction_labeled': cfg.N_LABELS / len(clean_df),
            'cost_reduction_pct': (1 - cfg.N_LABELS / len(clean_df)) * 100,
        },
        'future_evaluation': future_eval,
    }
    with open(cfg.RESULTS_FILE, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'  Saved.')

    elapsed = time.time() - t_start
    print('\n' + '=' * 60)
    print(f'  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print('=' * 60)
    print('Next: python visualize.py\n')


if __name__ == '__main__':
    main()


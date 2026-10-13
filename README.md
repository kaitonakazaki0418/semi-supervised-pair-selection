# Semi-Supervised Pair Selection for Statistical Arbitrage

Selecting pairs for statistical arbitrage normally requires backtesting thousands of candidate pairs. This project frames pair selection as a **semi-supervised classification problem**: label a small set of pairs, and let the structure of the remaining pairs inform the model.

Three methods are compared on 98 S&P 500 stocks (4,753 candidate pairs, 3,877 cleanly labeled): a Supervised SVM, **Label Propagation**, and a Self-Training SVM (a practical approximation to Transductive SVM). Using only 50 labeled pairs, the semi-supervised pipeline cuts the number of backtests needed for labeling by **98.7%**.

## Key Results

All out-of-sample numbers are on **2024-2025 data that no model or label construction touched**.

**Top-K selection: mean out-of-sample Sharpe ratio of the K highest-ranked pairs**

| Method | K=10 | K=20 | K=50 |
|---|---|---|---|
| Random sampling (baseline) | 0.673 | 0.965 | 0.665 |
| Supervised SVM | 0.300 | 0.331 | 0.260 |
| **Label Propagation** | **0.566** | **0.762** | **0.569** |
| Self-Training SVM | 0.300 | 0.412 | 0.242 |

**High-confidence selection: pairs whose predicted probability exceeds a threshold**

| Threshold | Supervised SVM | Label Propagation | Self-Training SVM |
|---|---|---|---|
| > 0.7 | 143 pairs, Sharpe 0.333 | 666 pairs, Sharpe 0.589 | 222 pairs, Sharpe 0.409 |
| > 0.8 | no pairs | 247 pairs, Sharpe 0.587 | 4 pairs, Sharpe -0.028 |

**Ranking quality (AUC) vs. number of labeled pairs**

| Method | 20 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|
| Supervised SVM | 0.500 | 0.455 | 0.506 | 0.611 | 0.592 |
| Label Propagation | 0.561 | 0.585 | 0.599 | 0.621 | 0.616 |
| Self-Training SVM | 0.524 | 0.455 | 0.498 | 0.601 | 0.587 |

### How to read these results

- Label Propagation is consistently ahead of both discriminative baselines, and it is the only method that still returns a sizable set of high-confidence pairs with stable Sharpe (about 0.59 at every threshold).
- In-sample F1 hides this: Supervised SVM has the best F1 (0.566) but an AUC of 0.455, below chance. AUC and out-of-sample selection tell a different story.
- **No method's top-K pairs beat random sampling on this window.** The features carry only modest signal (the best single feature has Cohen's d = 0.25), which is consistent with efficient-market intuition. The claim here is about relative ranking quality, not a profitable strategy.

## Approach

**Strict temporal separation (no look-ahead)**

| Window | Period | Used for |
|---|---|---|
| Feature | 2015-2020 | computing all 22 pair features |
| Label | 2021-2023 | generating labels via independent annual backtests |
| Evaluation | 2024-2025 | final out-of-sample assessment only |

**Robust multi-year labeling.** For each pair and each label year, a Bollinger-band mean-reversion strategy is backtested on the spread (static OLS hedge ratio, entry at |z| > 2, exit at the mean). A pair is labeled positive if its annualized Sharpe is above 0.5 in at least two of three years, and negative if it is below 0 in at least two. Pairs meeting neither criterion are *ambiguous* and become the unlabeled pool for the semi-supervised methods (876 pairs, 18.4%) instead of being discarded.

**Features (22).** Cointegration p-value (Engle-Granger), return correlation, half-life of mean reversion, Hurst exponent, spread volatility, and others, plus temporal-stability features (variation across sub-windows), rolling correlation and hedge-ratio statistics, and market-context features. All are standardized.

**Models**
- Supervised SVM: RBF kernel, class-balanced, trained on labeled pairs only.
- Label Propagation: k-NN graph (k=10) over labeled and unlabeled pairs, labeled nodes clamped.
- Self-Training SVM: iteratively pseudo-labels unlabeled pairs with confidence above 0.75. This is a stand-in for Transductive SVM, not an equivalent algorithm.

**Evaluation.** Held-out classification metrics (F1, AUC, etc., 5 seeds), a label-efficiency ablation, top-K backtests, and confidence-threshold analysis, all against a random-selection baseline.

## Limitations

- The out-of-sample evaluation covers a single two-year window, and the top-K and threshold tables use a single seed (no error bars).
- At K=10, 20, and 50, random sampling has a higher mean Sharpe than Label Propagation.
- Label years (2021-2023) and evaluation years (2024-2025) cover different market regimes. Whether the Supervised SVM's below-random result comes from regime shift or from overfitting to 50 labeled points was not isolated.
- A fixed Bollinger-band strategy (±2σ) is used; sensitivity to its parameters is not studied.
- Transaction costs are not modeled.

## Repository Structure

```
.
├── pair_ssl.py        # full pipeline
├── visualize.py       # figure generation
├── requirements.txt
├── run.sh
└── README.md
```

## How to Run

Price data (daily adjusted closes, 2015-2025) is downloaded with `yfinance`, so an internet connection is required.

```bash
chmod +x run.sh
./run.sh                          # real data, 100 stocks (15-25 min)
./run.sh --synthetic              # synthetic data (~2 min)
./run.sh --stocks 50              # smaller universe
./run.sh --synthetic --stocks 30  # quick smoke test
```

Synthetic mode is for checking that the pipeline runs; its results are not comparable to the numbers above.

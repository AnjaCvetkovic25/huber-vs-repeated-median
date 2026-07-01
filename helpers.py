import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

# IQR on residuals
def outlier_detection(residuals):
    Q1 = residuals.quantile(0.25)
    Q3 = residuals.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR 
    return [(residuals < lower_bound) | (residuals > upper_bound), lower_bound, upper_bound]

def print_outlier_detection_by_attribute(data, attr):
    outlier_indices_attr, _, _ = outlier_detection(data[attr])
    print(f"Outlier detection based on {attr}:")
    print(data[outlier_indices_attr])

def lstsq(x, y):
    n = len(x)
    xm, ym = x.mean(), y.mean()
    b1 = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    b0 = ym - b1 * xm
    y_hat = b0 + b1 * x
    residuals = y - y_hat
    sse = np.sum(residuals ** 2)
    s2 = sse / (n - 2)
    Sxx = np.sum((x - xm) ** 2)
    se_b1 = np.sqrt(s2 / Sxx)
    se_b0 = np.sqrt(s2 * (1/n + xm**2 / Sxx))
    return {
        'b0': b0, 'b1': b1, 'residuals': residuals,
        'se_b0': se_b0, 'se_b1': se_b1, 'label': 'LSTSQ'
    }


#Repeated Median (Siegel 1982)
#
# b1 = median_i { median_{j!=i} { (y_j - y_i)/(x_j - x_i) } }
# b0 = median_i { y_i - b1 * x_i }


def repeated_median(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    row_medians = np.empty(n)
    for i in range(n):
        pairwise = [(y[j] - y[i]) / (x[j] - x[i])
                    for j in range(n) if i != j and (x[j] - x[i]) != 0]
        row_medians[i] = np.median(pairwise) if pairwise else np.nan

    b1 = np.nanmedian(row_medians)
    intercepts = y - b1 * x
    b0 = np.median(intercepts)
    return {
        'b0': b0, 'b1': b1, 'residuals': y - (b0 + b1 * x),
        'row_medians': row_medians, 'label': 'Repeated Median'
    }



def huber_m_estimator(x, y, c=1.345):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    ols = lstsq(x, y)
    b0, b1 = ols['b0'], ols['b1']

    for i in range(25):
        y_hat = b0 + b1 * x
        r = y - y_hat
        s = np.median(np.abs(r)) / 0.6745
        if s == 0:
            break
        u = r / s
        w = np.ones(n)
        outliers = np.abs(u) > c
        w[outliers] = c / np.abs(u[outliers])

        sw = np.sum(w)
        xw = np.sum(w * x) / sw
        yw = np.sum(w * y) / sw
        Sxx = np.sum(w * (x - xw) ** 2)
        if Sxx == 0:
            break
        b1_new = np.sum(w * (x - xw) * (y - yw)) / Sxx
        b0_new = yw - b1_new * xw
        if abs(b1_new - b1) < 10 ** -6:
            b0, b1 = b0_new, b1_new
            break
        b0, b1 = b0_new, b1_new

    y_hat = b0 + b1 * x
    residuals = y - y_hat
    return {
        'b0': b0, 'b1': b1, 'residuals': residuals,
        'label': 'Huber M-estimator'
    }

def apply_estimators(x, y):
    res_lstsq = lstsq(x, y)
    res_rep_med = repeated_median(x, y)
    res_huber = huber_m_estimator(x, y)
    return [res_lstsq, res_rep_med, res_huber]


def residual_diagnostic(estimators, x):
    fig, axes = plt.subplots(2, len(estimators), figsize=(14, 8))

    for ci, res in enumerate(estimators):
        fitted = res['b0'] + res['b1'] * x

        # Row 0: Residuals vs Fitted
        axes[0, ci].scatter(fitted, res['residuals'], alpha=0.6, s=35,
                            edgecolors='w', lw=0.5)
        axes[0, ci].axhline(0, color='black', lw=1, ls='--')
        axes[0, ci].set_title(f'{res["label"]} — Residuals vs Fitted')
        axes[0, ci].set_xlabel('Fitted'); axes[0, ci].set_ylabel('Residuals')

        # Row 1: QQ plot
        stats.probplot(res['residuals'], plot=axes[1, ci])
        axes[1, ci].set_title(f'{res["label"]} — Normal Q-Q')
        axes[1, ci].get_lines()[0].set(alpha=0.7, markersize=4)
        axes[1, ci].get_lines()[1].set(color='black', lw=1.5)

    plt.suptitle('Residual Diagnostics', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()

    # Summary stats
    print(f"{'Estimator':<22} {'MAD':>10} {'RMSE':>10} {'max|r|':>10}")
    print("-" * 54)
    for res in estimators:
        r = res['residuals']
        print(f"{res['label']:<22} {np.median(np.abs(r)):>10.5f}"
            f" {np.sqrt(np.mean(r**2)):>10.5f} {np.max(np.abs(r)):>10.5f}")
        
def print_estimator_results(estimators):
    print(f"{'Estimator':<22} {'b0':>12} {'b1':>12}")
    print("-" * 48)
    for r in estimators:
        print(f"{r['label']:<22} {r['b0']:>12.5f} {r['b1']:>12.5f}")
        
def regression_results(x_rob, y_rob, estimators_rob, x_col, y_col, outlier_injection_technique, outlier_indices=None):
    xgrid = np.linspace(x_rob.min(), x_rob.max(), 300)
    fig, ax = plt.subplots(figsize=(9, 6))

    if outlier_indices is not None:
        mask = np.zeros(len(x_rob), dtype=bool)
        mask[outlier_indices] = True
        ax.scatter(x_rob[~mask], y_rob[~mask], alpha=0.55, s=45,
                edgecolors='w', linewidths=0.5, label='Observations', zorder=3)
        ax.scatter(x_rob[mask], y_rob[mask], alpha=0.85, s=60,
                color='red', edgecolors='darkred', linewidths=0.5,
                label='Contaminated', zorder=4, marker='X')
    else:
        ax.scatter(x_rob, y_rob, alpha=0.55, s=45,
                edgecolors='w', linewidths=0.5, label='Observations', zorder=3)

    for res in estimators_rob:
        ax.plot(xgrid, res['b0'] + res['b1'] * xgrid,
                lw=2.2, label=f"{res['label']}  b1={res['b1']:.4f}")

    ax.set_xlabel(f'log({x_col})'); ax.set_ylabel(f'log({y_col})')
    ax.set_title(f'Three estimators on dirty data\n{outlier_injection_technique}', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.show()


def breakdown_analysis(x, y, mode='y0', rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(x_arr)
    outlier_num = n // 2 + 1

    labels = ['LSTSQ', 'Repeated Median', 'Huber M-estimator']
    results = {label: [] for label in labels}
    outlier_indices = rng.choice(n, outlier_num, replace=False)
    x_max = x_arr.max()
    y_max = y_arr.max()
    y_min = y_arr.min()
    x_c = x_arr.copy()
    y_c = y_arr.copy()
    for idx in outlier_indices:
        slope_est = {label: [] for label in labels}

        if mode == 'y0':
            y_c[idx] = 0.0
        elif mode == 'hlo':
            x_c[idx] = x_max + rng.uniform(1, 3, 1)[0]
            y_c[idx] = rng.normal(0.5, 0.5, 1)[0]
        elif mode == 'rn':
            y_c[idx] = rng.uniform(y_min - 3, y_max + 3, 1)[0]

        for res in apply_estimators(pd.Series(x_c), pd.Series(y_c)):
            slope_est[res['label']].append(res['b1'])

        for label in labels:
            results[label].append(np.mean(slope_est[label]))

    return results, outlier_num

def breakdown_plot(true_b1, results, outlier_num, n, mode):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(true_b1, color='black', lw=1.5, ls=':', label=f'Clean b1 = {true_b1:.4f}')

    colors = {
        'LSTSQ': 'tab:blue',
        'Repeated Median': 'tab:orange',
        'Huber M-estimator': 'tab:green',
    }

    labels = ['LSTSQ', 'Repeated Median', 'Huber M-estimator']
    for label in labels:
        ax.plot(np.arange(outlier_num) / n * 100, results[label],
                lw=2.2, label=label, color=colors[label])

    ax.axvline(25, color='tab:green', lw=1, ls='--', alpha=0.6,
               label='Huber ~25% (theoretical)')
    ax.axvline(50, color='tab:orange', lw=1, ls='--', alpha=0.6,
               label='Repeated Median 50% (theoretical)')

    ax.set_xlabel('Contamination rate (%)', fontsize=12)
    ax.set_ylabel('Estimated slope', fontsize=12)
    mode_label = 'y = 0 outliers' if mode == 'y0' else 'Random Noise' if mode == 'rn' else 'High-leverage + Outliers'
    ax.set_title(f'Breakdown point analysis\n{mode_label}', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.show()
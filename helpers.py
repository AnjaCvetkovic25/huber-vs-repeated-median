import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats



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
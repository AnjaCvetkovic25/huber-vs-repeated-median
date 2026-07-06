import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

def outlier_detection(residuals):
    """
    Detektuje outliere primenom IQR metode na rezidualne vrednosti.

    Parameters
    ----------
    residuals : pd.Series
        Reziduali regresionog modela.

    Returns
    -------
    list
        [outlier_mask, lower_bound, upper_bound] gde je outlier_mask
        bool maska (True = outlier), a granice su Q1 - 1.5*IQR i Q3 + 1.5*IQR.
    """
    Q1 = residuals.quantile(0.25)
    Q3 = residuals.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR 
    return [(residuals < lower_bound) | (residuals > upper_bound), lower_bound, upper_bound]

def print_outlier_detection_by_attribute(data, attr):
    """
    Štampa redove koji su detektovani kao outlieri za zadati atribut.

    Parameters
    ----------
    data : pd.DataFrame
        Skup podataka.
    attr : str
        Naziv kolone nad kojom se vrši detekcija outliera.
    """
    outlier_indices_attr, _, _ = outlier_detection(data[attr])
    print(f"Outlier detection based on {attr}:")
    print(data[outlier_indices_attr])

def lstsq(x, y):
    """
    Metoda najmanjih kvadrata za prostu linearnu regresiju.

    Procenjuje parametre b0 i b1 modela y = b0 + b1*x minimizacijom
    sume kvadrata reziduala.

    Parameters
    ----------
    x : array-like
        Vrednosti prediktora.
    y : array-like
        Vrednosti ciljne promenljive.

    Returns
    -------
    dict
        'b0'        : float - slobodan član
        'b1'        : float - nagib
        'residuals' : np.ndarray - reziduali y - y_pred
        'label'     : str - 'LSTSQ'
    """
    n = len(x)
    xm, ym = x.mean(), y.mean()
    b1 = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    b0 = ym - b1 * xm
    y_hat = b0 + b1 * x
    residuals = y - y_hat
    return {
        'b0': b0, 'b1': b1, 'residuals': residuals,
        'label': 'LSTSQ'
    }

def repeated_median(x, y):
    """
    Repeated Median estimator (Siegel, 1982) za prostu linearnu regresiju.

    Robusni estimator sa breakdown point-om od 50%. Nagib se procenjuje kao:
        b1 = median_i { median_{j≠i} { (yj - yi) / (xj - xi) } }
        b0 = median_i { yi - b1 * xi }

    Parameters
    ----------
    x : array-like
        Vrednosti prediktora.
    y : array-like
        Vrednosti ciljne promenljive.

    Returns
    -------
    dict
        'b0'          : float - slobodan član
        'b1'          : float - nagib
        'residuals'   : np.ndarray - reziduali y - y_pred
        'label'       : str - 'Repeated Median'
    """
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
        'label': 'Repeated Median'
    }



def huber_m_estimator(x, y, c=1.345):
    """
    Huberov M-estimator za prostu linearnu regresiju putem IRLS algoritma.

    Koristi Huber težinsku funkciju za smanjenje uticaja outliera. Skala se
    procenjuje kao median(|r|) / 0.6745, što je konzistentna procena standardne
    devijacije pod normalnom raspodelom. Iterira do konvergencije ili 25 koraka.

    Parameters
    ----------
    x : array-like
        Vrednosti prediktora.
    y : array-like
        Vrednosti ciljne promenljive.
    c : float, optional
        Huberov prag osetljivosti. Tačke sa |u| > c dobijaju smanjenu težinu.
        Podrazumevana vrednost 1.345 daje 95% efikasnost pod normalnom raspodelom.

    Returns
    -------
    dict
        'b0'        : float - slobodan član
        'b1'        : float - nagib
        'residuals' : np.ndarray - reziduali y - y_pred
        'label'     : str - 'Huber M-estimator'
    """
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
    """
    Primenjuje sva tri estimatora (LSTSQ, Repeated Median, Huber) na zadatim podacima.

    Parameters
    ----------
    x : array-like
        Vrednosti prediktora.
    y : array-like
        Vrednosti ciljne promenljive.

    Returns
    -------
    list of dict
        Lista rečnika rezultata u redosledu: [LSTSQ, Repeated Median, Huber].
    """
    res_lstsq = lstsq(x, y)
    res_rep_med = repeated_median(x, y)
    res_huber = huber_m_estimator(x, y)
    return [res_lstsq, res_rep_med, res_huber]


def residual_diagnostic(estimators, x):
    """
    Prikazuje dijagnostičke grafike i mere greške za listu estimatora.

    Za svaki estimator generiše:
    - Residuals vs Fitted plot (gornji red)
    - Normal Q-Q plot (donji red)

    Štampa tabelu sa MAD, RMSE i max|r| za svaki estimator.

    Parameters
    ----------
    estimators : list of dict
        Lista rezultata estimatora (izlaz iz apply_estimators ili pojedinačnih funkcija).
    x : array-like
        Vrednosti prediktora (koriste se za računanje fitted vrednosti).
    """
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
    """
    Štampa procenjene parametre b0 i b1 za svaki estimator.

    Parameters
    ----------
    estimators : list of dict
        Lista rezultata estimatora.
    """
    print(f"{'Estimator':<22} {'b0':>12} {'b1':>12}")
    print("-" * 48)
    for r in estimators:
        print(f"{r['label']:<22} {r['b0']:>12.5f} {r['b1']:>12.5f}")
        
def regression_results(x_rob, y_rob, estimators_rob, x_col, y_col, outlier_injection_technique, outlier_indices=None):
    """
    Prikazuje scatter plot podataka sa fitovanim regresionim linijama sva tri estimatora.

    Kontaminirane tačke su označene crvenim X markerom ako je prosleđen outlier_indices.

    Parameters
    ----------
    x_rob : array-like
        Vrednosti prediktora (potencijalno kontaminirane).
    y_rob : array-like
        Vrednosti ciljne promenljive (potencijalno kontaminirane).
    estimators_rob : list of dict
        Lista rezultata estimatora nad kontaminiranim podacima.
    x_col : str
        Naziv kolone prediktora (za oznake osa).
    y_col : str
        Naziv ciljne kolone (za oznake osa).
    outlier_injection_technique : str
        Opis tipa kontaminacije (koristi se u naslovu grafika).
    outlier_indices : array-like, optional
        Indeksi kontaminiranih tačaka. Ako je None, sve tačke su prikazane jednako.
    """
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
    """
    Analiza breakdown point-a postepenim ubacivanjem kontaminiranih tačaka.

    Kumulativno dodaje jednu kontaminiranu tačku po iteraciji (do n//2 + 1)
    i za svaki korak procenjuje nagib b1 sva tri estimatora.

    Parameters
    ----------
    x : array-like
        Vrednosti prediktora.
    y : array-like
        Vrednosti ciljne promenljive.
    mode : {'y0', 'hlo', 'rn'}
        Tip kontaminacije:
        - 'y0'  : y = 0
        - 'hlo' : high leverage + outlier
        - 'rn'  : random noise
    rng_seed : int, optional
        Seed za reproduktibilnost nasumičnog odabira. Podrazumevano 42.

    Returns
    -------
    results : dict
        Rečnik {label: [b1_vrednosti]} za svaki estimator kroz iteracije.
    outlier_num : int
        Ukupan broj ubačenih outliera (n // 2 + 1).
    """
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
    """
    Prikazuje grafik breakdown analize: nagib b1 u zavisnosti od procenta kontaminacije.

    Crta teorijske breakdown tačke za Huber (25%) i Repeated Median (50%)
    kao vertikalne isprekidane linije, i referentnu vrednost čistog b1 kao
    horizontalnu tačkastu liniju.

    Parameters
    ----------
    true_b1 : float
        Referentna vrednost nagiba na čistim podacima (LSTSQ).
    results : dict
        Izlaz iz breakdown_analysis: {label: [b1_vrednosti]}.
    outlier_num : int
        Ukupan broj iteracija (ubačenih outliera).
    n : int
        Veličina originalnog skupa podataka.
    mode : {'y0', 'rn', 'hlo'}
        Tip kontaminacije (koristi se u naslovu grafika).
    """
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
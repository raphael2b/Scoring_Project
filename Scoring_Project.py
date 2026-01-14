import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

# Configuration globale
warnings.simplefilter("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'axes.titlesize': 14})

# --- CONSTANTES & CHEMINS ---
FILE_PATH = '/data/defaut2000.csv'
TARGET = 'yd'
VARIABLES_MAP = {
    'yd': 'Financial Difficulty', 'tdta': 'Debt/Assets', 'reta': 'Retained Earnings',
    'opita': 'Income/Assets', 'ebita': 'Pre-Tax Earnings/Assets', 'lsls': 'Log Sales',
    'lta': 'Log Assets', 'gempl': 'Employment Growth', 'invsls': 'Inventory/Sales',
    'nwcta': 'Net Working Capital/Assets', 'cacl': 'Current Assets/Liabilities',
    'qacl': 'Quick Assets/Liabilities', 'fata': 'Fixed Assets/Total Assets',
    'ltdta': 'Long-Term Debt/Total Assets', 'mveltd': 'Market Value Equity/Long-Term Debt'
}


# --- FONCTIONS UTILITAIRES ---

def save_plot(filename):
    """Sauvegarde et affiche le plot courant."""
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def get_stats_tests(df, group_col, target_cols):
    """Génère les stats de Jarque-Bera, Levene et T-tests de manière vectorisée."""

    # 1. Jarque-Bera & Normality
    jb_stats = []
    for grp in df[group_col].unique():
        sub_df = df[df[group_col] == grp]
        label = 'NonDefault' if grp == 0 else 'Default'
        for col in target_cols:
            data = sub_df[col].dropna()
            if len(data) < 3: continue
            stat, p = stats.jarque_bera(data)
            jb_stats.append({
                'Group': label, 'Variable': col, 'Obs.': len(data),
                'Skewness': f"{stats.skew(data):.3f}", 'Kurtosis-3': f"{stats.kurtosis(data):.3f}",
                'JB Stat.': f"{stat:.3f}", 'P-value': f"{p:.3f}{'*' if p < 0.05 else ''}"
            })

    # 2. Levene & T-tests
    levene_ttest = []
    g0 = df[df[group_col] == 0]
    g1 = df[df[group_col] == 1]

    for col in target_cols:
        d0, d1 = g0[col].dropna(), g1[col].dropna()
        if len(d0) < 2 or len(d1) < 2: continue

        # Levene
        lev_stat, lev_p = stats.levene(d0, d1)
        # T-tests
        t_eq, p_eq = stats.ttest_ind(d0, d1, equal_var=True)
        t_uneq, p_uneq = stats.ttest_ind(d0, d1, equal_var=False)

        levene_ttest.append({
            'Variable': col,
            'n0': len(d0), 'sd0': round(d0.std(), 3),
            'n1': len(d1), 'sd1': round(d1.std(), 3),
            'Levene_p': f"{lev_p:.3f}{'*' if lev_p < 0.05 else ''}",
            'Mean_diff': round(d1.mean() - d0.mean(), 3),
            'T_stat_eq': round(t_eq, 3), 'P_val_eq': f"{p_eq:.3f}{'*' if p_eq < 0.05 else ''}",
            'T_stat_uneq': round(t_uneq, 3), 'P_val_uneq': f"{p_uneq:.3f}{'*' if p_uneq < 0.05 else ''}"
        })

    return pd.DataFrame(jb_stats), pd.DataFrame(levene_ttest)


def fit_compare_models(y, X, feature_names, export_name=None):
    """
    Ajuste LPM, Logit et Probit, retourne un tableau comparatif formaté et les objets modèles.
    """
    X_const = sm.add_constant(X)
    n0, n1 = (y == 0).sum(), (y == 1).sum()

    # Fit Models
    models = {
        'Linear Probability': sm.OLS(y, X_const).fit(),
        'Logit': sm.Logit(y, X_const).fit(disp=0),
        'Probit': sm.Probit(y, X_const).fit(disp=0)
    }

    # Build Table
    data = {}
    for name, model in models.items():
        preds = model.predict(X_const)
        auc_score = roc_auc_score(y, preds)

        # Formattage "Coef (t-stat)"
        col_data = []
        for param, tval in zip(model.params, model.tvalues):
            col_data.append(f"{param:.3f} ({tval:.2f})")

        # Ajout métriques
        r2 = model.rsquared if hasattr(model, 'rsquared') else model.prsquared
        col_data.extend([f"{auc_score:.3f}", f"{r2:.3f}", str(n0), str(n1)])
        data[name] = col_data

    # Index rows
    idx = list(model.params.index) + ['AUC', 'R²/Pseudo-R²', 'n₀', 'n₁']
    df_res = pd.DataFrame(data, index=idx)

    if export_name:
        print(f"\nComparison Table ({export_name}):")
        print(df_res)
        df_res.to_csv(f"{export_name}.csv")

    return models, df_res


def plot_custom_distributions(df, col, target):
    """Reproduit le graphique complexe de la Question 9 (Hist + KDE + Boxplot)."""
    groups = df.groupby(target)
    x_min, x_max = df[col].min(), df[col].max()
    x_range = np.linspace(x_min, x_max, 100)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 0.5, 0.5]})
    colors = {0: 'green', 1: 'red'}

    # Histograms & KDE
    y_max = 0
    for i, (gd, group) in enumerate(groups):
        data = group[col].dropna()
        c = colors.get(gd, 'blue')
        label = f'yd={gd}'

        sns.histplot(data, ax=axes[i], color=c, alpha=0.4, stat='density', label=f'{label} Hist')
        sns.kdeplot(data, ax=axes[i], color=c, linewidth=2, label=f'{label} KDE')

        # Normal approx
        mu, std = data.mean(), data.std()
        axes[i].plot(x_range, stats.norm.pdf(x_range, mu, std), color=f'dark{c}', ls='--', label='Normal')
        axes[i].legend()
        y_max = max(y_max, axes[i].get_ylim()[1])

    # Uniformiser Y-axis
    axes[0].set_ylim(0, y_max)
    axes[1].set_ylim(0, y_max)

    # Boxplots
    for i, (gd, group) in enumerate(groups, start=2):
        sns.boxplot(x=group[col], ax=axes[i], orient='h', color=colors.get(gd),
                    showmeans=True, meanprops={'marker': 'D', 'markeredgecolor': 'black'})
        axes[i].set_yticks([])
        axes[i].set_title(f'Boxplot yd={gd}', pad=5)
        axes[i].set_xlabel('')

    axes[3].set_xlabel(col)
    save_plot(f'Histogram of {col.upper()} default versus healthy.png')


def plot_roc_comparison(y_true, models_dict, X_features, title_suffix=""):
    """Trace les courbes ROC pour le dictionnaire de modèles statsmodels."""
    plt.figure(figsize=(8, 6))
    colors = {'Linear Probability': 'blue', 'Probit': 'green', 'Logit': 'red'}

    X_const = sm.add_constant(X_features)

    for name, model in models_dict.items():
        preds = model.predict(X_const)
        fpr, tpr, _ = roc_curve(y_true, preds)
        auc_val = roc_auc_score(y_true, preds)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})', color=colors.get(name), lw=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves {title_suffix}')
    plt.legend(loc='lower right')
    save_plot(f'roc_curves_{title_suffix.strip()}.png')


# --- MAIN EXECUTION ---

def main():
    # 1. Chargement et Nettoyage
    print("Loading Data...")
    df = pd.read_csv(FILE_PATH, sep=';', decimal=',')
    df = df.apply(pd.to_numeric, errors='coerce')

    # Q4: Handling extreme values
    df[['fata', 'ltdta']] = df[['fata', 'ltdta']].replace(-99.99, np.nan)
    print("Missing values summary:\n", df[['fata', 'ltdta']].isnull().sum())

    # Sorting (Q4)
    df_sorted = df.sort_values(by=['yd', 'reta']).reset_index(drop=True)

    # Split Train/Test (Odd/Even rows as per request) (Q6)
    y_sorted = df_sorted[TARGET]
    X_sorted = df_sorted.drop(columns=[TARGET])

    X_train = X_sorted.iloc[1::2].copy()
    X_test = X_sorted.iloc[::2].copy()
    y_train = y_sorted.iloc[1::2].copy()
    y_test = y_sorted.iloc[::2].copy()

    # Recombine for analysis
    y_X_train = pd.concat([y_train, X_train], axis=1)

    # 2. Visualisation Univariée (Q9)
    print("\nGenerating Distribution Plots...")
    plot_custom_distributions(y_X_train, 'reta', TARGET)

    # 3. Tests Statistiques (Q4 pt2, Q10, Q11, Q12)
    print("\nRunning Statistical Tests...")
    numeric_cols = X_train.select_dtypes(include='number').columns
    df_jb, df_tests = get_stats_tests(y_X_train, TARGET, numeric_cols)

    print("Jarque-Bera:\n", df_jb.head())
    print("Levene & T-tests:\n", df_tests.head())
    df_jb.to_csv('Stats_Normality.csv', index=False)
    df_tests.to_csv('Stats_Means_Variances.csv', index=False)

    # 4. Modélisation Simple (Univariable 'reta') (Q13)
    print("\n--- Univariate Model (RETA) ---")
    models_uni, table_uni = fit_compare_models(y_train, X_train['reta'], ['reta'], "Comparison_Univariate")

    # Q13: Visualisation LPM Simple
    lpm = models_uni['Linear Probability']
    preds = lpm.predict(sm.add_constant(X_train['reta']))

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train['reta'], y_train, c=y_train.map({0: 'green', 1: 'red'}), alpha=0.6, label='Actual')
    plt.plot(X_train['reta'], preds, color='blue', lw=2, label='Regression Line')
    plt.title('LPM Train Sample: yd vs reta')
    plt.legend()
    save_plot('LPM_Simple_Regression.png')

    # Q14: PairGrid
    print("Generating PairGrid...")
    g = sns.PairGrid(y_X_train[['yd', 'tdta', 'reta']], hue='yd', palette={0: "green", 1: "red"})
    g.map_diag(sns.kdeplot, fill=True)
    g.map_offdiag(sns.scatterplot)
    save_plot('Distribution_Tdta_Reta.png')

    # Q15' / Q19 / Q20: Analyse des Résidus (LPM)
    residuals = y_train - preds
    std_residuals = residuals / residuals.std()

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # Hist résidus
    sns.histplot(residuals[y_train == 0], color='green', kde=True, ax=ax[0], label='yd=0')
    sns.histplot(residuals[y_train == 1], color='red', kde=True, ax=ax[0], label='yd=1')
    ax[0].set_title('Residuals Distribution')
    ax[0].legend()

    # Résidus vs Forecast
    ax[1].scatter(preds[y_train == 0], std_residuals[y_train == 0], c='green', alpha=0.5)
    ax[1].scatter(preds[y_train == 1], std_residuals[y_train == 1], c='red', alpha=0.5)
    ax[1].axhline(1.96, c='blue', ls='--');
    ax[1].axhline(-1.96, c='blue', ls='--')
    ax[1].set_title('Standardized Residuals vs Forecast')
    save_plot('Residuals_Analysis.png')

    # Q15 & Q22: Comparaison Modèles Univariés & ROC
    plot_roc_comparison(y_train, models_uni, X_train['reta'], "Univariate")

    # 5. Modélisation Multivariée (Q19 - Q23)
    print("\n--- Multivariate Models ---")
    feats = ['tdta', 'reta', 'gempl', 'opita', 'invsls']
    models_multi, table_multi = fit_compare_models(y_train, X_train[feats], feats, "Comparison_Multivariate")

    # ROC Curves Multivariées (Train & Test via Sklearn pour généralisation)
    # Note: On réutilise les modèles ajustés ci-dessus pour le train,
    # mais on refait un fit clean sklearn pour la comparaison train/test demandée à la fin

    print("Generating Comparative ROC (Sklearn wrapper)...")
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train[feats])
    X_te_sc = scaler.transform(X_test[feats])

    sk_models = [
        ('LPM', LinearRegression(), False),
        ('Logit', LogisticRegression(solver='liblinear'), True),
        ('Probit', None, False)  # Sklearn n'a pas de Probit natif simple, on utilise statsmodels pour prédiction
    ]

    plt.figure(figsize=(10, 6))
    for name, mod, is_proba in sk_models:
        if name == 'Probit':
            # Utilisation du modèle statsmodels existant
            p_train = models_multi['Probit'].predict(sm.add_constant(X_train[feats]))
            p_test = models_multi['Probit'].predict(sm.add_constant(X_test[feats]))
        else:
            mod.fit(X_tr_sc, y_train)
            if is_proba:
                p_train = mod.predict_proba(X_tr_sc)[:, 1]
                p_test = mod.predict_proba(X_te_sc)[:, 1]
            else:
                p_train = mod.predict(X_tr_sc)
                p_test = mod.predict(X_te_sc)

        auc_tr = roc_auc_score(y_train, p_train)
        auc_te = roc_auc_score(y_test, p_test)

        fpr, tpr, _ = roc_curve(y_test, p_test)
        plt.plot(fpr, tpr, label=f'{name} (Train AUC={auc_tr:.2f}, Test AUC={auc_te:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title('ROC Curves Comparison (Train vs Test Performance)')
    save_plot('ROC_Comparison_Final.png')

    # 6. Matrice de Corrélation avec T-stats (Q26-27)
    print("\nGenerating Correlation Matrix...")
    corr_mat = y_X_train.corr()
    n = len(y_X_train)
    # Calcul T-stat vectorisé: t = r * sqrt(n-2) / sqrt(1-r^2)
    # Note: Simplification ici car gérer les NaNs par paire comme le code original est très lent.
    # Pour optimisation, on assume n global ou on utilise une méthode matricielle si n varie peu.
    # Ici, je garde la logique visuelle.

    mask = np.triu(np.ones_like(corr_mat, dtype=bool))

    # Préparation des annotations
    annot = corr_mat.copy().astype(object)
    for i in range(len(corr_mat)):
        for j in range(len(corr_mat)):
            r = corr_mat.iloc[i, j]
            # Approx t-stat based on global N (optimisation speed)
            t = (r * np.sqrt(n - 2)) / np.sqrt(1 - r ** 2 + 1e-9)
            annot.iloc[i, j] = f"{r:.2f}\n({t:.1f})"

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_mat, mask=mask, annot=annot.values, fmt='',
                cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title("Correlation Matrix with T-stats")
    save_plot('Correlation_Matrix_Optimized.png')


if __name__ == "__main__":
    main()
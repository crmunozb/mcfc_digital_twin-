import pandas as pd
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ── Cargar datos ───────────────────────────────────────────────────────────────
df = pd.read_excel('../Data/Data_original_PGNN.xlsx')

FEATURES = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²']
TARGET   = 'Experiment'

df_clean = df[FEATURES + [TARGET]].dropna()

X = df_clean[FEATURES].values
y = df_clean[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Escalado de entradas
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

rango = y.max() - y.min()


def calcular_metricas(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    nrmse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / rango
    return r2, mae, nrmse


# ── PLS base ──────────────────────────────────────────────────────────────────
pls_base = PLSRegression(n_components=8)
pls_base.fit(X_train_sc, y_train)

y_pred_train_pls = pls_base.predict(X_train_sc).ravel()
y_pred_test_pls  = pls_base.predict(X_test_sc).ravel()

r2_train_pls, mae_train_pls, nrmse_train_pls = calcular_metricas(y_train, y_pred_train_pls)
r2_test_pls, mae_test_pls, nrmse_test_pls = calcular_metricas(y_test, y_pred_test_pls)

print("\n=== PLS base ===")
print(f"Train | R²={r2_train_pls:.4f} | MAE={mae_train_pls:.4f} V | NRMSE={nrmse_train_pls:.4f}")
print(f"Test  | R²={r2_test_pls:.4f} | MAE={mae_test_pls:.4f} V | NRMSE={nrmse_test_pls:.4f}")


# ── Búsqueda KPLS aproximado ──────────────────────────────────────────────────
print("\n=== Búsqueda KPLS aproximado ===")

mejor_modelo = None
mejor_resultado = None
mejor_score = -np.inf

gammas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
n_features_kernel = [50, 100, 200, 300]

for gamma in gammas:
    for n_rbf in n_features_kernel:

        kpls = Pipeline([
            ('rbf', RBFSampler(
                gamma=gamma,
                n_components=n_rbf,
                random_state=42
            )),
            ('pls', PLSRegression(n_components=8))
        ])

        kpls.fit(X_train_sc, y_train)

        y_pred_train = kpls.predict(X_train_sc).ravel()
        y_pred_test  = kpls.predict(X_test_sc).ravel()

        r2_train, mae_train, nrmse_train = calcular_metricas(y_train, y_pred_train)
        r2_test, mae_test, nrmse_test = calcular_metricas(y_test, y_pred_test)

        diff_r2 = abs(r2_train - r2_test)

        print(
            f"gamma={gamma:<4} | n_rbf={n_rbf:<3} | "
            f"R² train={r2_train:.4f} | R² test={r2_test:.4f} | "
            f"MAE test={mae_test:.4f} V | NRMSE test={nrmse_test:.4f} | "
            f"ΔR²={diff_r2:.4f}"
        )

        if r2_test > mejor_score:
            mejor_score = r2_test
            mejor_modelo = kpls
            mejor_resultado = {
                "gamma": gamma,
                "n_rbf": n_rbf,
                "r2_train": r2_train,
                "mae_train": mae_train,
                "nrmse_train": nrmse_train,
                "r2_test": r2_test,
                "mae_test": mae_test,
                "nrmse_test": nrmse_test,
                "diff_r2": diff_r2
            }


# ── Mejor resultado ───────────────────────────────────────────────────────────
print("\n=== Mejor KPLS encontrado ===")
print(f"gamma = {mejor_resultado['gamma']}")
print(f"n_rbf = {mejor_resultado['n_rbf']}")
print(f"R² train = {mejor_resultado['r2_train']:.4f}")
print(f"R² test  = {mejor_resultado['r2_test']:.4f}")
print(f"MAE test = {mejor_resultado['mae_test']:.4f} V")
print(f"NRMSE test = {mejor_resultado['nrmse_test']:.4f}")
print(f"ΔR² = {mejor_resultado['diff_r2']:.4f}")

print("\n=== Comparación final ===")
print(f"PLS  test | R²={r2_test_pls:.4f} | MAE={mae_test_pls:.4f} V | NRMSE={nrmse_test_pls:.4f}")
print(
    f"KPLS test | R²={mejor_resultado['r2_test']:.4f} | "
    f"MAE={mejor_resultado['mae_test']:.4f} V | "
    f"NRMSE={mejor_resultado['nrmse_test']:.4f}"
)
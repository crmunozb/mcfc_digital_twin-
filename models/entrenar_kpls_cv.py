"""
entrenar_kpls_cv.py
-------------------
Entrena un modelo KPLS aproximado sobre el dataset MCFC usando:

1) Separación train/test.
2) Selección de hiperparámetros SOLO con train mediante validación cruzada.
3) Evaluación final UNA SOLA VEZ sobre test.

Uso:
    python3 entrenar_kpls_cv.py --fuente warsaw_ut
    python3 entrenar_kpls_cv.py --fuente warsaw_ut sintetico
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from cargar_datos import cargar_dataset, cargar_holdout

MODELO_BASE = 'kpls_voltaje_cv'

FEATURES     = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²', 'r_1']
TARGET       = 'Experiment'
TEST_SIZE    = 0.20
RANDOM_STATE = 42


def calcular_nrmse(y_true, y_pred, rango):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / rango


def mostrar_metricas(nombre, y_true, y_pred, rango):
    r2    = r2_score(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    nrmse = calcular_nrmse(y_true, y_pred, rango)
    print(f"{nombre:<10} | R²={r2:.4f} | MAE={mae:.4f} V | NRMSE={nrmse:.4f}")
    return r2, mae, nrmse


# ── Argumento fuente ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Entrenamiento KPLS para celda MCFC")
parser.add_argument(
    '--fuente', nargs='+',
    default=['warsaw_ut', 'sintetico'],
    choices=['warsaw_ut', 'sintetico'],
    help="Fuentes de datos a usar (default: warsaw_ut sintetico)"
)
parser.add_argument('--max_samples', type=int, default=None, help='Límite de muestras por temperatura')
parser.add_argument(
    '--holdout', action='store_true',
    help="Excluir curvas reales de 550-625°C del entrenamiento (evaluación de generalización)"
)
args = parser.parse_args()
# Sufijo según fuente para no sobreescribir modelos
_sufijo = '_balanceado' if 'sintetico' in args.fuente else '_warsaw'
if args.holdout:
    _sufijo = _sufijo + '_holdout'
MODELO_OUT = os.path.join(BASE_DIR, MODELO_BASE + _sufijo + '.pkl')

print(f"Fuentes seleccionadas: {args.fuente}")

# ── Cargar datos desde PostgreSQL ─────────────────────────────────────────────
df       = cargar_dataset(
    fuentes=args.fuente,
    holdout_temps=[550, 575, 600, 625] if args.holdout else None
)
df_clean = df[FEATURES + [TARGET]].dropna()

# Muestreo balanceado por temperatura si se especifica --max_samples
if args.max_samples:
    grupos = []
    for temp in df_clean['T'].unique():
        sub = df_clean[df_clean['T'] == temp]
        n   = min(args.max_samples, len(sub))
        grupos.append(sub.sample(n=n, random_state=42))
    df_clean = pd.concat(grupos).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dataset reducido a {len(df_clean)} filas ({args.max_samples} por temperatura)")

X      = df_clean[FEATURES].values
y      = df_clean[TARGET].values
T_vals = df_clean['T'].values

print(f"Filas utilizadas: {len(df_clean)}")

# ── Separar train/test ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\nSeparación de datos:")
print(f"  Train: {len(X_train)} filas")
print(f"  Test:  {len(X_test)} filas")

# ── Pipeline KPLS ────────────────────────────────────────────────────────────
pipeline_x = Pipeline([
    ('scaler', StandardScaler()),
    ('rbf',   RBFSampler(random_state=RANDOM_STATE)),
    ('pls',   PLSRegression(scale=False))
])
modelo = TransformedTargetRegressor(
    regressor=pipeline_x,
    transformer=StandardScaler()
)

param_grid = {
    'regressor__rbf__gamma':        [0.005, 0.01, 0.02, 0.05, 0.1, 0.3],
    'regressor__rbf__n_components': [50, 100, 200, 300],
    'regressor__pls__n_components': [2, 3, 4, 5, 6, 7, 8],
}

# ── Validación cruzada SOLO en train ──────────────────────────────────────────
print("\nBuscando mejores hiperparámetros KPLS usando solo train...")
cv   = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(
    estimator=modelo, param_grid=param_grid,
    scoring='r2', cv=cv, n_jobs=-1, verbose=2, refit=True
)
grid.fit(X_train, y_train)
print(f"\nMejor configuración: {grid.best_params_}")
print(f"Mejor R² CV: {grid.best_score_:.4f}")

# ── Evaluación final ──────────────────────────────────────────────────────────
mejor_modelo = grid.best_estimator_
y_pred_train = mejor_modelo.predict(X_train).ravel()
y_pred_test  = mejor_modelo.predict(X_test).ravel()
y_pred_all   = mejor_modelo.predict(X).ravel()
rango        = y.max() - y.min()

print("\n=== Evaluacion final del mejor KPLS ===")
r2_train, mae_train, nrmse_train = mostrar_metricas("Train", y_train, y_pred_train, rango)
r2_test,  mae_test,  nrmse_test  = mostrar_metricas("Test",  y_test,  y_pred_test,  rango)

delta_r2 = abs(r2_train - r2_test)
print(f"\nDelta R2 train-test = {delta_r2:.4f}")
if delta_r2 < 0.05:   print("OK: modelo generaliza bien")
elif delta_r2 < 0.10: print("AVISO: leve sobreajuste")
else:                  print("ALERTA: sobreajuste significativo")

print("\nR2 por temperatura:")
print(f"{'T':>6} | {'R2':>7} | {'MAE':>7} | {'n':>5}")
print('-'*35)
for temp in sorted(np.unique(T_vals)):
    mask = T_vals == temp
    r2   = r2_score(y[mask], y_pred_all[mask])
    mae  = mean_absolute_error(y[mask], y_pred_all[mask])
    print(str(int(temp)).rjust(6), '|', str(round(r2,4)).rjust(7), '|',
          str(round(mae,4)).rjust(7), '|', str(mask.sum()).rjust(5))

# ── Guardar modelo ────────────────────────────────────────────────────────────
salida = {
    'modelo':       mejor_modelo,
    'features':     FEATURES,
    'target':       TARGET,
    'fuentes':      args.fuente,
    'best_params':  grid.best_params_,
    'r2_cv_mean':   float(grid.best_score_),
    'r2_train':     float(r2_train),
    'mae_train':    float(mae_train),
    'nrmse_train':  float(nrmse_train),
    'r2_test':      float(r2_test),
    'mae_test':     float(mae_test),
    'nrmse_test':   float(nrmse_test),
    'delta_r2':     float(delta_r2),
    'test_size':    TEST_SIZE,
    'max_samples':  args.max_samples,
    'random_state': RANDOM_STATE,
    'n_train':      len(X_train),
    'n_test':       len(X_test),
}
joblib.dump(salida, MODELO_OUT)
print(f"\nModelo KPLS guardado en: {MODELO_OUT}")
print(f"  Fuentes: {args.fuente}")
print(f"  R2 test:    {r2_test:.4f}")
print(f"  MAE test:   {mae_test:.4f} V")
print(f"  NRMSE test: {nrmse_test:.4f}")
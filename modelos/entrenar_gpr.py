"""
entrenar_gpr.py
---------------
Entrena un modelo de Gaussian Process Regression (GPR) sobre el dataset
experimental de Milewski usando:

1) Separación train/test con la misma semilla que PLS/KPLS (comparabilidad).
2) Kernel ARD (Automatic Relevance Determination): un length_scale por variable,
   lo que permite interpretar la relevancia relativa de cada variable operacional.
3) Optimización de hiperparámetros del kernel vía maximum likelihood durante
   el fit (no requiere GridSearchCV — scikit-learn lo hace internamente).
4) Evaluación final UNA SOLA VEZ sobre test.
5) Guardado de las predicciones con incertidumbre (media + std) para uso
   en el dashboard con bandas de confianza ±2σ.

Variable objetivo: 'Experiment' — voltaje real medido de la celda bajo carga.
Nota: NO usar 'E_max' como target, ya que corresponde al voltaje teórico de
Nernst (caso ideal sin pérdidas), no al voltaje operacional real de la celda.

El modelo GPR aporta una capacidad que PLS y KPLS no tienen: cuantificación
nativa de incertidumbre. Para cada punto j, el modelo devuelve no solo una
predicción puntual μ(j) sino también una desviación estándar σ(j), lo que
permite construir bandas de confianza [μ-2σ, μ+2σ] y detectar regiones donde
el modelo es menos confiable (típicamente fuera del dominio de entrenamiento).

Uso:
    python3 entrenar_gpr.py
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, WhiteKernel
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELO_BASE = 'gpr_voltaje'
sys.path.insert(0, BASE_DIR)
from cargar_datos import cargar_dataset, cargar_holdout


# ── Configuración ──────────────────────────────────────────────────────────────
FEATURES = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²', 'r_1']
TARGET   = 'Experiment'  # voltaje real medido (no E_max)

TEST_SIZE    = 0.20
RANDOM_STATE = 42
N_FEATURES   = len(FEATURES)


# ── Funciones auxiliares ───────────────────────────────────────────────────────
def calcular_nrmse(y_true, y_pred, rango):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / rango


def mostrar_metricas(nombre, y_true, y_pred, rango):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    nrmse = calcular_nrmse(y_true, y_pred, rango)
    print(f"{nombre:<10} | R²={r2:.4f} | MAE={mae:.4f} V | NRMSE={nrmse:.4f}")
    return r2, mae, nrmse


# ── Argumento fuente ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Entrenamiento GPR para celda MCFC")
parser.add_argument(
    '--fuente', nargs='+',
    default=['warsaw_ut', 'sintetico'],
    choices=['warsaw_ut', 'sintetico'],
    help="Fuentes de datos a usar (default: warsaw_ut sintetico)"
)
parser.add_argument(
    '--max_samples', type=int, default=None,
    help="Límite de muestras por temperatura (recomendado: 400)"
)
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
if args.max_samples:
    print(f"Límite de muestras: {args.max_samples} por temperatura")

# ── Cargar datos desde PostgreSQL ─────────────────────────────────────────────
df       = cargar_dataset(
    fuentes=args.fuente,
    holdout_temps=[550, 575, 600, 625] if args.holdout else None
)
df_clean = df[FEATURES + [TARGET]].dropna()

# Muestreo balanceado por temperatura si se especifica --max_samples
if args.max_samples:
    import numpy as np
    rng = np.random.default_rng(42)
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
print(f"Features: {FEATURES}")
print(f"Target: {TARGET}")


# ── Separar train/test ────────────────────────────────────────────────────────
# Misma semilla que PLS/KPLS para comparabilidad directa de métricas.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"\nSeparación de datos:")
print(f"  Train: {len(X_train)} filas")
print(f"  Test:  {len(X_test)} filas")


# ── Estandarización ────────────────────────────────────────────────────────────
# GPR es sensible a escalas. Estandarizamos X e y por separado.
# El scaler se guarda para aplicarlo en el dashboard.
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)
X_all_sc   = scaler_X.transform(X)

y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()


# ── Kernel ARD ─────────────────────────────────────────────────────────────────
# ConstantKernel: escala la varianza de la señal.
# RBF con length_scale por dimensión (ARD=True implícito con array de longitudes):
#   un length_scale pequeño → variable muy relevante (el modelo es sensible a ella)
#   un length_scale grande  → variable poco relevante
# WhiteKernel: modela ruido de las mediciones experimentales.
#
# Los bounds permiten que la optimización explore un rango amplio.
kernel = (
    ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
    RBF(
        length_scale=np.ones(N_FEATURES),
        length_scale_bounds=(1e-2, 1e2)
    ) +
    WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=(1e-5, 1.0)
    )
)


# ── Entrenar GPR ───────────────────────────────────────────────────────────────
# normalize_y=False porque ya estandarizamos manualmente (más control).
# n_restarts_optimizer=5: reinicia la optimización del kernel desde 5 puntos
# aleatorios para evitar mínimos locales.
print("\nEntrenando GPR con kernel ARD...")
print("(La optimización del kernel puede tomar algunos minutos)")

gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    random_state=RANDOM_STATE,
    normalize_y=False
)

gpr.fit(X_train_sc, y_train_sc)

print(f"\nKernel optimizado:")
print(f"  {gpr.kernel_}")


# ── Predicciones ──────────────────────────────────────────────────────────────
# GPR devuelve media y desviación estándar para cada punto.
# Desescalamos para volver al espacio original de voltaje (V).
y_pred_train_sc, y_std_train_sc = gpr.predict(X_train_sc, return_std=True)
y_pred_test_sc,  y_std_test_sc  = gpr.predict(X_test_sc,  return_std=True)
y_pred_all_sc,   y_std_all_sc   = gpr.predict(X_all_sc,   return_std=True)

# Desescalar predicciones
y_pred_train = scaler_y.inverse_transform(y_pred_train_sc.reshape(-1, 1)).ravel()
y_pred_test  = scaler_y.inverse_transform(y_pred_test_sc.reshape(-1, 1)).ravel()
y_pred_all   = scaler_y.inverse_transform(y_pred_all_sc.reshape(-1, 1)).ravel()

# Desescalar incertidumbre: std en espacio original = std_sc × scale_ del scaler_y
sigma_scale  = scaler_y.scale_[0]
y_std_train  = y_std_train_sc * sigma_scale
y_std_test   = y_std_test_sc  * sigma_scale
y_std_all    = y_std_all_sc   * sigma_scale

rango = y.max() - y.min()


# ── Evaluación final ──────────────────────────────────────────────────────────
print("\n=== Evaluación final del GPR ===")
r2_train,  mae_train,  nrmse_train  = mostrar_metricas(
    "Train", y_train, y_pred_train, rango)
r2_test,   mae_test,   nrmse_test   = mostrar_metricas(
    "Test",  y_test,  y_pred_test,  rango)

delta_r2 = abs(r2_train - r2_test)
print(f"\nDelta R² train-test = {delta_r2:.4f}")

if delta_r2 < 0.05:
    print("OK: modelo generaliza bien")
elif delta_r2 < 0.10:
    print("AVISO: leve sobreajuste")
else:
    print("ALERTA: sobreajuste significativo")

# Incertidumbre media en train y test
print(f"\nIncertidumbre media (σ):")
print(f"  Train: {y_std_train.mean():.4f} V")
print(f"  Test:  {y_std_test.mean():.4f} V")
print(f"  (banda ±2σ media en test: ±{2*y_std_test.mean():.4f} V)")


# ── R² por temperatura ────────────────────────────────────────────────────────
print("\nR² por temperatura:")
print(f"{'T':>6} | {'R2':>7} | {'MAE':>7} | {'σ_mean':>8} | {'n':>5}")
print('-'*45)
for temp in sorted(np.unique(T_vals)):
    mask  = T_vals == temp
    r2    = r2_score(y[mask], y_pred_all[mask])
    mae   = mean_absolute_error(y[mask], y_pred_all[mask])
    sigma = y_std_all[mask].mean()
    print(str(int(temp)).rjust(6), '|',
          str(round(r2, 4)).rjust(7), '|',
          str(round(mae, 4)).rjust(7), '|',
          str(round(sigma, 4)).rjust(8), '|',
          str(mask.sum()).rjust(5))


# ── Análisis ARD: relevancia de variables ─────────────────────────────────────
# Los length_scales del kernel RBF optimizado revelan la relevancia de cada
# variable: un length_scale pequeño → alta sensibilidad → variable relevante.
# Normalizamos para obtener un score de relevancia entre 0 y 1.
print("\n=== Análisis de relevancia de variables (ARD) ===")
try:
    # Extraer length_scales del kernel compuesto (ConstantKernel * RBF + White)
    rbf_kernel  = gpr.kernel_.k1.k2   # ConstantKernel * RBF → k1.k2 = RBF
    ls          = rbf_kernel.length_scale
    relevancia  = 1.0 / ls
    relevancia  = relevancia / relevancia.max()  # normalizar a [0, 1]

    print(f"{'Variable':<15} | {'length_scale':>12} | {'relevancia':>10}")
    print('-'*43)
    for feat, l, r in sorted(
        zip(FEATURES, ls, relevancia),
        key=lambda x: x[2], reverse=True
    ):
        feat_label = feat.replace('i, A/cm²', 'j (A/cm²)')
        print(f"{feat_label:<15} | {l:>12.4f} | {r:>10.4f}")
except Exception as e:
    print(f"  (No se pudo extraer ARD: {e})")


# ── Guardar modelo y metadatos ────────────────────────────────────────────────
salida = {
    # Modelo y escaladores
    'modelo':    gpr,
    'scaler_X':  scaler_X,
    'scaler_y':  scaler_y,

    # Configuración
    'fuentes':       args.fuente,
    'features':      FEATURES,
    'target':        TARGET,
    'test_size':     TEST_SIZE,
    'max_samples':   args.max_samples,
    'random_state':  RANDOM_STATE,
    'n_train':       len(X_train),
    'n_test':        len(X_test),

    # Kernel optimizado (como string para logging)
    'kernel_str': str(gpr.kernel_),

    # Métricas de entrenamiento
    'r2_train':    float(r2_train),
    'mae_train':   float(mae_train),
    'nrmse_train': float(nrmse_train),

    # Métricas de test
    'r2_test':     float(r2_test),
    'mae_test':    float(mae_test),
    'nrmse_test':  float(nrmse_test),
    'delta_r2':    float(delta_r2),

    # Incertidumbre media
    'sigma_mean_train': float(y_std_train.mean()),
    'sigma_mean_test':  float(y_std_test.mean()),

    # ARD: length_scales por variable
    'ard_length_scales': dict(zip(FEATURES, ls.tolist()))
        if 'ls' in dir() else {},
}

joblib.dump(salida, MODELO_OUT)

print(f"\nModelo GPR guardado en: {MODELO_OUT}")
print(f"  R² test:    {r2_test:.4f}")
print(f"  MAE test:   {mae_test:.4f} V")
print(f"  NRMSE test: {nrmse_test:.4f}")
print(f"  σ media test: {y_std_test.mean():.4f} V")
print(f"  Banda ±2σ media: ±{2*y_std_test.mean():.4f} V")
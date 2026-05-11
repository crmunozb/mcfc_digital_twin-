"""
entrenar_pls.py
---------------
Entrena un modelo PLS sobre el dataset experimental de Milewski
con separación train/test (80/20) para validación honesta.

Uso:
    python3 entrenar_pls.py
"""

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET    = os.path.join(BASE_DIR, '..', 'Data', 'Data_original_PGNN.xlsx')
MODELO_OUT = os.path.join(BASE_DIR, 'pls_voltaje.pkl')

# ── Configuración ──────────────────────────────────────────────────────────────
FEATURES   = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²']
TARGET     = 'Experiment'
N_COMP     = 8
TEST_SIZE  = 0.20
RANDOM_STATE = 42

# ── Cargar datos ───────────────────────────────────────────────────────────────
print("Cargando dataset...")
df = pd.read_excel(DATASET)
df_clean = df[FEATURES + [TARGET]].dropna()
X = df_clean[FEATURES].values
y = df_clean[TARGET].values
print(f"  {len(df_clean)} filas cargadas")

# ── Separar train/test ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"  Train: {len(X_train)} filas ({100*(1-TEST_SIZE):.0f}%)")
print(f"  Test:  {len(X_test)} filas ({100*TEST_SIZE:.0f}%)")

# ── Escalar SOLO con datos de train ───────────────────────────────────────────
# Importante: el scaler aprende SOLO del train para no contaminar el test
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
y_train_sc = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()

# Aplicar el mismo scaler al test (sin reentrenar)
X_test_sc  = scaler_X.transform(X_test)

# ── Validación cruzada en train ───────────────────────────────────────────────
print(f"\nValidación cruzada (5-fold) sobre train...")
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
pls_cv = PLSRegression(n_components=N_COMP)
scores_cv = cross_val_score(pls_cv, X_train_sc, y_train_sc, cv=kf, scoring='r2')
print(f"  R² CV = {scores_cv.mean():.4f} ± {scores_cv.std():.4f}")

# ── Entrenar modelo final con TODO el train ───────────────────────────────────
print(f"\nEntrenando PLS final con n_components={N_COMP}...")
pls = PLSRegression(n_components=N_COMP)
pls.fit(X_train_sc, y_train_sc)

# ── Evaluar en TRAIN ──────────────────────────────────────────────────────────
y_pred_train_sc = pls.predict(X_train_sc).ravel()
y_pred_train    = scaler_y.inverse_transform(y_pred_train_sc.reshape(-1,1)).ravel()
r2_train   = r2_score(y_train, y_pred_train)
mae_train  = mean_absolute_error(y_train, y_pred_train)
rango      = y.max() - y.min()
nrmse_train= np.sqrt(np.mean((y_train - y_pred_train)**2)) / rango

# ── Evaluar en TEST ───────────────────────────────────────────────────────────
y_pred_test_sc = pls.predict(X_test_sc).ravel()
y_pred_test    = scaler_y.inverse_transform(y_pred_test_sc.reshape(-1,1)).ravel()
r2_test    = r2_score(y_test, y_pred_test)
mae_test   = mean_absolute_error(y_test, y_pred_test)
nrmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2)) / rango

print(f"\n{'='*45}")
print(f"{'Métrica':<12} {'Train (80%)':>12} {'Test (20%)':>12}")
print(f"{'='*45}")
print(f"{'R²':<12} {r2_train:>12.4f} {r2_test:>12.4f}")
print(f"{'MAE (V)':<12} {mae_train:>12.4f} {mae_test:>12.4f}")
print(f"{'NRMSE':<12} {nrmse_train:>12.4f} {nrmse_test:>12.4f}")
print(f"{'='*45}")

# Diagnóstico
diferencia = abs(r2_train - r2_test)
if diferencia < 0.05:
    print(f"\n✓ Modelo generaliza bien (diferencia R² = {diferencia:.4f})")
elif diferencia < 0.10:
    print(f"\n⚠ Leve sobreajuste (diferencia R² = {diferencia:.4f})")
else:
    print(f"\n✗ Sobreajuste significativo (diferencia R² = {diferencia:.4f})")

# ── Guardar modelo + métricas train/test ──────────────────────────────────────
modelo = {
    'pls':          pls,
    'scaler_X':     scaler_X,
    'scaler_y':     scaler_y,
    'features':     FEATURES,
    'n_comp':       N_COMP,
    # Métricas train
    'r2':           r2_train,
    'mae':          mae_train,
    'nrmse':        nrmse_train,
    # Métricas test (las honestas)
    'r2_test':      r2_test,
    'mae_test':     mae_test,
    'nrmse_test':   nrmse_test,
    # CV
    'r2_cv_mean':   float(scores_cv.mean()),
    'r2_cv_std':    float(scores_cv.std()),
    # Config
    'test_size':    TEST_SIZE,
    'random_state': RANDOM_STATE,
    'n_train':      len(X_train),
    'n_test':       len(X_test),
}
joblib.dump(modelo, MODELO_OUT)
print(f"\n✓ Modelo guardado en: {MODELO_OUT}")
print(f"  R² test (honesto): {r2_test:.4f}")
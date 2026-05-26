"""
entrenar_gpr_residual.py
------------------------
Entrena un modelo híbrido físico-datos para la celda MCFC basado en
Residual Modeling con Gaussian Process Regression (GPR).

Concepto:
    El modelo de Nernst con pérdidas proporciona una predicción física
    V_Nernst(j, T, gases, r1), pero comete un error sistemático ε que
    depende de las condiciones operacionales:

        ε = V_real - V_Nernst

    Este error no es ruido aleatorio puro — tiene estructura que el GPR
    puede aprender. El modelo residual entrena GPR sobre (X → ε), de
    modo que la predicción final combina física y datos:

        V_pred = V_Nernst(X) + GPR_residual(X)

Ventajas sobre GPR directo:
    - Incorpora conocimiento físico explícitamente (principio de Nernst)
    - El GPR solo necesita aprender la corrección, no el comportamiento completo
    - La incertidumbre σ(j) refleja qué tan bien el modelo físico describe
      la región operacional → más interpretable físicamente
    - El análisis ARD sobre ε revela qué variables explican las PÉRDIDAS
      no capturadas por Nernst, no el voltaje en sí

Uso:
    python3 entrenar_gpr_residual.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Importar modelo de Nernst desde el mismo directorio
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from modelo_nernst import voltaje_modelo as nernst_voltaje

DATASET    = os.path.join(BASE_DIR, '..', 'Data', 'Data_original_PGNN.xlsx')
MODELO_OUT = os.path.join(BASE_DIR, 'gpr_residual.pkl')


# ── Configuración ──────────────────────────────────────────────────────────────
FEATURES = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²', 'r_1']
TARGET   = 'Experiment'

# Mapeo entre columnas del dataset y parámetros de voltaje_modelo
COL_MAP = {
    'j':    'i, A/cm²',
    'T_C':  'T',
    'h2a':  'H2a',
    'h2oa': 'H2Oa',
    'co2a': 'CO2a',
    'o2c':  'O2c',
    'co2c': 'CO2c',
    'n2c':  'N2c',
    'r1':   'r_1',
}

TEST_SIZE    = 0.20
RANDOM_STATE = 42
N_FEATURES   = len(FEATURES)


# ── Funciones auxiliares ───────────────────────────────────────────────────────
def calcular_nrmse(y_true, y_pred, rango):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / rango


def mostrar_metricas(nombre, y_true, y_pred, rango):
    r2    = r2_score(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    nrmse = calcular_nrmse(y_true, y_pred, rango)
    print(f"{nombre:<12} | R²={r2:.4f} | MAE={mae:.4f} V | NRMSE={nrmse:.4f}")
    return r2, mae, nrmse


def calcular_v_nernst_dataset(df):
    """
    Calcula V_Nernst para cada fila del dataset usando voltaje_modelo.
    Usa r1 del dataset pero sin delta ni r2 para obtener solo la parte
    física del modelo (pérdidas óhmicas + activación calibradas).
    """
    V_nernst = nernst_voltaje(
        j     = df['i, A/cm²'].values,
        T_C   = df['T'].values,
        h2a   = df['H2a'].values,
        h2oa  = df['H2Oa'].values,
        co2a  = df['CO2a'].values,
        o2c   = df['O2c'].values,
        co2c  = df['CO2c'].values,
        r1    = df['r_1'].values,
        n2c   = df['N2c'].values,
    )
    return V_nernst


# ── Cargar datos ───────────────────────────────────────────────────────────────
print("Cargando dataset...")
df       = pd.read_excel(DATASET)
df_clean = df[FEATURES + [TARGET]].dropna()

X      = df_clean[FEATURES].values
y_real = df_clean[TARGET].values
T_vals = df_clean['T'].values

print(f"Filas utilizadas: {len(df_clean)}")
print(f"Features: {FEATURES}")
print(f"Target: {TARGET}")


# ── Calcular residuos ε = V_real - V_Nernst ───────────────────────────────────
print("\nCalculando residuos ε = V_real - V_Nernst(modelo_nernst.py)...")
V_nernst = calcular_v_nernst_dataset(df_clean)
epsilon  = y_real - V_nernst

print(f"\nDistribución de residuos ε:")
print(f"  Media:     {epsilon.mean():.4f} V")
print(f"  Std:       {epsilon.std():.4f} V")
print(f"  Min:       {epsilon.min():.4f} V")
print(f"  Max:       {epsilon.max():.4f} V")
print(f"  |ε| medio: {np.abs(epsilon).mean():.4f} V")

print(f"\nPor temperatura:")
print(f"{'T':>6} | {'media ε':>9} | {'std ε':>7} | {'n':>5}")
print('-'*35)
for temp in sorted(np.unique(T_vals)):
    mask = T_vals == temp
    e    = epsilon[mask]
    print(f"{int(temp):>6} | {e.mean():>9.4f} | {e.std():>7.4f} | {mask.sum():>5}")


# ── Separar train/test ────────────────────────────────────────────────────────
# Misma semilla que PLS/KPLS/GPR para comparabilidad directa.
X_train, X_test, e_train, e_test, y_train, y_test, vn_train, vn_test = \
    train_test_split(
        X, epsilon, y_real, V_nernst,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

print(f"\nSeparación de datos:")
print(f"  Train: {len(X_train)} filas")
print(f"  Test:  {len(X_test)} filas")


# ── Estandarización ────────────────────────────────────────────────────────────
scaler_X = StandardScaler()
scaler_e = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)
X_all_sc   = scaler_X.transform(X)

# Estandarizar los residuos (target del GPR)
e_train_sc = scaler_e.fit_transform(e_train.reshape(-1, 1)).ravel()
e_test_sc  = scaler_e.transform(e_test.reshape(-1, 1)).ravel()


# ── Kernel ARD ─────────────────────────────────────────────────────────────────
# Mismo kernel que GPR directo para comparabilidad.
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


# ── Entrenar GPR sobre residuos ────────────────────────────────────────────────
print("\nEntrenando GPR Residual sobre ε...")
print("(La optimización del kernel puede tomar algunos minutos)")

gpr_res = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=5,
    random_state=RANDOM_STATE,
    normalize_y=False
)

gpr_res.fit(X_train_sc, e_train_sc)

print(f"\nKernel optimizado:")
print(f"  {gpr_res.kernel_}")


# ── Predicciones de residuos ───────────────────────────────────────────────────
sigma_scale = scaler_e.scale_[0]

e_pred_train_sc, e_std_train_sc = gpr_res.predict(X_train_sc, return_std=True)
e_pred_test_sc,  e_std_test_sc  = gpr_res.predict(X_test_sc,  return_std=True)
e_pred_all_sc,   e_std_all_sc   = gpr_res.predict(X_all_sc,   return_std=True)

e_pred_train = scaler_e.inverse_transform(e_pred_train_sc.reshape(-1,1)).ravel()
e_pred_test  = scaler_e.inverse_transform(e_pred_test_sc.reshape(-1,1)).ravel()
e_pred_all   = scaler_e.inverse_transform(e_pred_all_sc.reshape(-1,1)).ravel()

e_std_train  = e_std_train_sc * sigma_scale
e_std_test   = e_std_test_sc  * sigma_scale
e_std_all    = e_std_all_sc   * sigma_scale


# ── Predicciones finales: V = V_Nernst + ε_pred ───────────────────────────────
y_pred_train = vn_train + e_pred_train
y_pred_test  = vn_test  + e_pred_test
y_pred_all   = V_nernst + e_pred_all

# La incertidumbre del modelo híbrido es la incertidumbre del GPR residual
# (Nernst es determinístico, así que σ_total = σ_residual)
y_std_train  = e_std_train
y_std_test   = e_std_test
y_std_all    = e_std_all

rango_y = y_real.max() - y_real.min()


# ── Evaluación del GPR Residual sobre voltaje ──────────────────────────────────
print("\n=== Evaluación GPR Residual (V_Nernst + ε_pred) ===")
r2_train,  mae_train,  nrmse_train  = mostrar_metricas(
    "Train", y_train, y_pred_train, rango_y)
r2_test,   mae_test,   nrmse_test   = mostrar_metricas(
    "Test",  y_test,  y_pred_test,  rango_y)

delta_r2 = abs(r2_train - r2_test)
print(f"\nDelta R² train-test = {delta_r2:.4f}")
if delta_r2 < 0.05:
    print("OK: modelo generaliza bien")
elif delta_r2 < 0.10:
    print("AVISO: leve sobreajuste")
else:
    print("ALERTA: sobreajuste significativo")

print(f"\nIncertidumbre media (σ sobre voltaje):")
print(f"  Train: {y_std_train.mean():.4f} V")
print(f"  Test:  {y_std_test.mean():.4f} V")
print(f"  Banda ±2σ media en test: ±{2*y_std_test.mean():.4f} V")


# ── Comparación con Nernst puro ────────────────────────────────────────────────
print("\n=== Comparación: Nernst puro vs GPR Residual (sobre test) ===")
r2_nernst  = r2_score(y_test, vn_test)
mae_nernst = mean_absolute_error(y_test, vn_test)
print(f"{'Nernst puro':<15} | R²={r2_nernst:.4f} | MAE={mae_nernst:.4f} V")
print(f"{'GPR Residual':<15} | R²={r2_test:.4f} | MAE={mae_test:.4f} V")
mejora_r2  = r2_test - r2_nernst
mejora_mae = mae_nernst - mae_test
print(f"\nMejora del GPR Residual sobre Nernst puro:")
print(f"  ΔR²:  +{mejora_r2:.4f}")
print(f"  ΔMAE: -{mejora_mae:.4f} V ({mejora_mae/mae_nernst*100:.1f}% reducción)")


# ── R² por temperatura ────────────────────────────────────────────────────────
print("\nR² por temperatura (GPR Residual sobre dataset completo):")
print(f"{'T':>6} | {'R² híbrido':>10} | {'R² Nernst':>10} | {'MAE':>7} | {'σ_mean':>8} | {'n':>5}")
print('-'*55)
for temp in sorted(np.unique(T_vals)):
    mask   = T_vals == temp
    r2_h   = r2_score(y_real[mask], y_pred_all[mask])
    r2_n   = r2_score(y_real[mask], V_nernst[mask])
    mae_h  = mean_absolute_error(y_real[mask], y_pred_all[mask])
    sigma  = y_std_all[mask].mean()
    print(f"{int(temp):>6} | {r2_h:>10.4f} | {r2_n:>10.4f} | "
          f"{mae_h:>7.4f} | {sigma:>8.4f} | {mask.sum():>5}")


# ── Análisis ARD sobre residuos ────────────────────────────────────────────────
print("\n=== Análisis ARD: ¿qué variables explican el error de Nernst? ===")
try:
    rbf_kernel = gpr_res.kernel_.k1.k2
    ls         = rbf_kernel.length_scale
    relevancia = 1.0 / ls
    relevancia = relevancia / relevancia.max()

    print(f"{'Variable':<15} | {'length_scale':>12} | {'relevancia':>10}")
    print('-'*43)
    for feat, l, r in sorted(
        zip(FEATURES, ls, relevancia),
        key=lambda x: x[2], reverse=True
    ):
        feat_label = feat.replace('i, A/cm²', 'j (A/cm²)')
        print(f"{feat_label:<15} | {l:>12.4f} | {r:>10.4f}")
    print()
    print("Interpretación: variables con alta relevancia son las que más")
    print("explican el error sistemático de Nernst en esta celda MCFC.")
except Exception as e:
    print(f"  (No se pudo extraer ARD: {e})")
    ls = np.ones(N_FEATURES)


# ── Guardar modelo ────────────────────────────────────────────────────────────
salida = {
    # Modelo y escaladores
    'modelo':   gpr_res,
    'scaler_X': scaler_X,
    'scaler_e': scaler_e,   # scaler del residuo (no del voltaje)

    # Configuración
    'features':      FEATURES,
    'target':        TARGET,
    'test_size':     TEST_SIZE,
    'random_state':  RANDOM_STATE,
    'n_train':       len(X_train),
    'n_test':        len(X_test),
    'kernel_str':    str(gpr_res.kernel_),

    # Métricas del residuo
    'epsilon_mean':  float(epsilon.mean()),
    'epsilon_std':   float(epsilon.std()),

    # Métricas del voltaje final (V_Nernst + ε_pred)
    'r2_train':    float(r2_train),
    'mae_train':   float(mae_train),
    'nrmse_train': float(nrmse_train),
    'r2_test':     float(r2_test),
    'mae_test':    float(mae_test),
    'nrmse_test':  float(nrmse_test),
    'delta_r2':    float(delta_r2),

    # Métricas de Nernst puro (para comparación)
    'r2_nernst_test':  float(r2_nernst),
    'mae_nernst_test': float(mae_nernst),

    # Incertidumbre
    'sigma_mean_train': float(y_std_train.mean()),
    'sigma_mean_test':  float(y_std_test.mean()),

    # ARD
    'ard_length_scales': dict(zip(FEATURES, ls.tolist())),
}

joblib.dump(salida, MODELO_OUT)

print(f"\nModelo GPR Residual guardado en: {MODELO_OUT}")
print(f"  R² test (híbrido):  {r2_test:.4f}")
print(f"  MAE test (híbrido): {mae_test:.4f} V")
print(f"  NRMSE test:         {nrmse_test:.4f}")
print(f"  σ media test:       {y_std_test.mean():.4f} V")
print(f"  Banda ±2σ media:    ±{2*y_std_test.mean():.4f} V")
print(f"\nUso en dashboard:")
print(f"  V_pred = nernst_voltaje(X) + gpr_residual.predict(X)")
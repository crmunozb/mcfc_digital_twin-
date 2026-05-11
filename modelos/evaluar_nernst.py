"""
evaluar_nernst.py
-----------------
Evalúa el modelo semi-empírico de Nernst con pérdidas sobre el dataset
experimental de Milewski, usando la misma separación train/test 80/20
que PLS y KPLS para comparación.

Uso:
    python3 evaluar_nernst.py
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from modelo_nernst import voltaje_modelo


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, '..', 'Data', 'Data_original_PGNN.xlsx')

FEATURES = [
    'T', 'H2a', 'H2Oa', 'CO2a',
    'O2c', 'CO2c', 'N2c', 'i, A/cm²',
    'N2a', 'CO', 'CH4', 'H2Oc', 'r_1'
]

TARGET = 'Experiment'

TEST_SIZE = 0.20
RANDOM_STATE = 42


def nrmse(y_true, y_pred, rango):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / rango


def evaluar(y_true, y_pred, nombre, rango):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    nrmse_val = nrmse(y_true, y_pred, rango)

    print(f"{nombre:<10} | R²={r2:.4f} | MAE={mae:.4f} V | NRMSE={nrmse_val:.4f}")

    return r2, mae, nrmse_val


print("Cargando dataset...")
df = pd.read_excel(DATASET)

df_clean = df[FEATURES + [TARGET]].dropna()

print(f"Filas utilizadas: {len(df_clean)}")
print(f"Variable objetivo: {TARGET}")

X = df_clean[FEATURES]
y = df_clean[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"\nSeparación de datos:")
print(f"  Train: {len(X_train)} filas")
print(f"  Test:  {len(X_test)} filas")


def predecir_nernst(df_x):
    return voltaje_modelo(
        j=df_x['i, A/cm²'].values,
        T_C=df_x['T'].values,
        h2a=df_x['H2a'].values,
        h2oa=df_x['H2Oa'].values,
        co2a=df_x['CO2a'].values,
        o2c=df_x['O2c'].values,
        co2c=df_x['CO2c'].values,
        r1=df_x['r_1'].values,
        n2a=df_x['N2a'].values,
        co=df_x['CO'].values,
        ch4=df_x['CH4'].values,
        n2c=df_x['N2c'].values,
        h2oc=df_x['H2Oc'].values
    )


y_pred_train = predecir_nernst(X_train)
y_pred_test = predecir_nernst(X_test)
y_pred_all = predecir_nernst(X)

rango = y.max() - y.min()

print("\n=== Evaluación modelo Nernst con pérdidas ===")
r2_train, mae_train, nrmse_train = evaluar(y_train, y_pred_train, "Train", rango)
r2_test, mae_test, nrmse_test = evaluar(y_test, y_pred_test, "Test", rango)
r2_global, mae_global, nrmse_global = evaluar(y, y_pred_all, "Global", rango)

delta_r2 = abs(r2_train - r2_test)

print(f"\nΔR² train-test = {delta_r2:.4f}")
print("\nOK: evaluación de Nernst finalizada.")
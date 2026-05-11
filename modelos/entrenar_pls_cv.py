"""
entrenar_pls_cv.py
------------------
Entrena un modelo PLS sobre el dataset experimental de Milewski usando una
validación metodológicamente más robusta:

1) Separación train/test.
2) Selección del número de componentes PLS usando SOLO train mediante validación cruzada.
3) Evaluación final UNA SOLA VEZ sobre test.

Uso:
    python3 entrenar_pls_cv.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor


# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET    = os.path.join(BASE_DIR, '..', 'Data', 'Data_original_PGNN.xlsx')
MODELO_OUT = os.path.join(BASE_DIR, 'pls_voltaje_cv.pkl')


# ── Configuración ──────────────────────────────────────────────────────────────
FEATURES = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²']
TARGET   = 'Experiment'   # Confirmar que esta columna corresponde al voltaje experimental

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ── Funciones auxiliares ───────────────────────────────────────────────────────
def calcular_nrmse(y_true, y_pred, rango):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / rango


def mostrar_metricas(nombre, y_true, y_pred, rango):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    nrmse = calcular_nrmse(y_true, y_pred, rango)

    print(f"{nombre:<10} | R²={r2:.4f} | MAE={mae:.4f} V | NRMSE={nrmse:.4f}")

    return r2, mae, nrmse


# ── Cargar datos ───────────────────────────────────────────────────────────────
print("Cargando dataset...")
df = pd.read_excel(DATASET)

print("\nColumnas disponibles:")
print(df.columns.tolist())

df_clean = df[FEATURES + [TARGET]].dropna()

X = df_clean[FEATURES].values
y = df_clean[TARGET].values

print(f"\nFilas utilizadas: {len(df_clean)}")
print(f"Variables de entrada: {FEATURES}")
print(f"Variable objetivo: {TARGET}")

print("\nPrimeros valores de la variable objetivo:")
print(df_clean[TARGET].head())


# ── Separar train/test ────────────────────────────────────────────────────────
# El conjunto test queda reservado hasta el final.
# No se usa para elegir el número de componentes.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"\nSeparación de datos:")
print(f"  Train: {len(X_train)} filas")
print(f"  Test:  {len(X_test)} filas")


# ── Pipeline PLS ───────────────────────────────────────────────────────────────
# El escalado de X queda dentro del pipeline, por lo que se recalcula
# correctamente dentro de cada fold de validación cruzada.
pipeline_x = Pipeline([
    ('scaler', StandardScaler()),
    ('pls', PLSRegression(scale=False))
])

# La variable objetivo también se escala dentro de cada fold.
modelo = TransformedTargetRegressor(
    regressor=pipeline_x,
    transformer=StandardScaler()
)


# ── Grilla de hiperparámetros ─────────────────────────────────────────────────
# Como hay 8 variables de entrada, se prueban de 1 a 8 componentes latentes.
param_grid = {
    'regressor__pls__n_components': list(range(1, len(FEATURES) + 1))
}


# ── Validación cruzada SOLO en train ──────────────────────────────────────────
print("\nBuscando mejor número de componentes PLS usando solo train...")

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

grid = GridSearchCV(
    estimator=modelo,
    param_grid=param_grid,
    scoring='r2',
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

grid.fit(X_train, y_train)

print("\nMejor configuración encontrada usando solo train:")
print(grid.best_params_)
print(f"Mejor R² promedio en CV: {grid.best_score_:.4f}")


# ── Evaluación final ──────────────────────────────────────────────────────────
# El test se usa recién aquí, una sola vez.
mejor_modelo = grid.best_estimator_

y_pred_train = mejor_modelo.predict(X_train).ravel()
y_pred_test  = mejor_modelo.predict(X_test).ravel()

# Para mantener comparabilidad entre train/test, se usa el rango global del dataset.
rango = y.max() - y.min()

print("\n=== Evaluación final del mejor PLS ===")
r2_train, mae_train, nrmse_train = mostrar_metricas("Train", y_train, y_pred_train, rango)
r2_test, mae_test, nrmse_test = mostrar_metricas("Test", y_test, y_pred_test, rango)

delta_r2 = abs(r2_train - r2_test)

print(f"\nΔR² train-test = {delta_r2:.4f}")

if delta_r2 < 0.05:
    print("✓ No se observa evidencia clara de sobreajuste en esta evaluación preliminar.")
elif delta_r2 < 0.10:
    print("⚠ Se observa una diferencia moderada entre train y test.")
else:
    print("✗ Posible sobreajuste significativo.")


# ── Guardar modelo y metadatos ────────────────────────────────────────────────
salida = {
    'modelo': mejor_modelo,
    'features': FEATURES,
    'target': TARGET,

    'best_params': grid.best_params_,
    'r2_cv_mean': float(grid.best_score_),

    'r2_train': float(r2_train),
    'mae_train': float(mae_train),
    'nrmse_train': float(nrmse_train),

    'r2_test': float(r2_test),
    'mae_test': float(mae_test),
    'nrmse_test': float(nrmse_test),
    'delta_r2': float(delta_r2),

    'test_size': TEST_SIZE,
    'random_state': RANDOM_STATE,
    'n_train': len(X_train),
    'n_test': len(X_test),
}

joblib.dump(salida, MODELO_OUT)

print(f"\n✓ Modelo PLS guardado en: {MODELO_OUT}")
print(f"  Mejor n_components: {grid.best_params_['regressor__pls__n_components']}")
print(f"  R² test:   {r2_test:.4f}")
print(f"  MAE test:  {mae_test:.4f} V")
print(f"  NRMSE test:{nrmse_test:.4f}")
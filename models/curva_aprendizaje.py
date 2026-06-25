"""
curva_aprendizaje.py
--------------------
Análisis de sensibilidad al tamaño del dataset de entrenamiento.

Para cada tamaño n_train (puntos por temperatura), entrena los 4 modelos
(PLS, KPLS, GPR, GPR Residual) y evalúa sobre un test set fijo balanceado
(TEST_PTS_POR_TEMP puntos por temperatura, siempre los mismos).

Nernst se incluye como referencia horizontal (no depende del tamaño).

Genera:
  - curva_aprendizaje_resultados.csv  → tabla completa con R², MAE, NRMSE
  - curva_aprendizaje_r2.pdf          → gráfico R² vs n_train
  - curva_aprendizaje_mae.pdf         → gráfico MAE vs n_train
  - curva_aprendizaje_nrmse.pdf       → gráfico NRMSE vs n_train

Uso:
    python3 curva_aprendizaje.py

Nota: KPLS usa hiperparámetros fijos (sin GridSearchCV) para que el
      experimento sea manejable en tiempo. Los hiperparámetros se toman
      del mejor resultado conocido del entrenamiento completo.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.cross_decomposition import PLSRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Intentar importar módulos del proyecto ────────────────────────────────────
try:
    from cargar_datos import cargar_dataset
    from modelo_nernst import voltaje_modelo as nernst_voltaje
    MODO = 'postgres'
    print("Modo: PostgreSQL (cargar_datos.py encontrado)")
except ImportError:
    MODO = 'sin_conexion'
    print("AVISO: cargar_datos.py no encontrado.")
    print("Ejecuta este script desde la carpeta del proyecto MCFC.")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

FEATURES     = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²', 'r_1']
TARGET       = 'Experiment'
RANDOM_STATE = 42
TEMPERATURAS = [550, 575, 600, 625, 650]

# Tamaños de entrenamiento a evaluar (puntos por temperatura)
TRAIN_SIZES = [25, 50, 100, 150, 200, 300, 400]

# Test fijo: siempre estos puntos por temperatura (separados antes del loop)
TEST_PTS_POR_TEMP = 80

# Hiperparámetros fijos para KPLS (del mejor resultado conocido)
KPLS_GAMMA        = 0.005
KPLS_N_COMPONENTS_RBF = 200
KPLS_N_COMPONENTS_PLS = 8

# Colores por modelo
COLORES = {
    'PLS':          '#4a7eb5',
    'KPLS':         '#b8922a',
    'GPR':          '#6aa3c8',
    'GPR Residual': '#5a9e6f',
    'Nernst':       '#6b7585',
}
LINESTYLES = {
    'PLS':          '-',
    'KPLS':         '-',
    'GPR':          '-',
    'GPR Residual': '-',
    'Nernst':       '--',
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def calcular_nrmse(y_true, y_pred):
    rango = y_true.max() - y_true.min()
    rmse  = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / rango if rango > 0 else np.nan


def metricas(y_true, y_pred):
    r2    = r2_score(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    nrmse = calcular_nrmse(y_true, y_pred)
    return r2, mae, nrmse


def v_nernst_df(df_sub):
    return nernst_voltaje(
        j    = df_sub['i, A/cm²'].values,
        T_C  = df_sub['T'].values,
        h2a  = df_sub['H2a'].values,
        h2oa = df_sub['H2Oa'].values,
        co2a = df_sub['CO2a'].values,
        o2c  = df_sub['O2c'].values,
        co2c = df_sub['CO2c'].values,
        r1   = df_sub['r_1'].values,
        n2c  = df_sub['N2c'].values,
    )


def muestrear_balanceado(df, n_por_temp, semilla=42):
    """Toma n_por_temp puntos de cada temperatura."""
    grupos = []
    for temp in TEMPERATURAS:
        sub = df[df['T'] == temp]
        n   = min(n_por_temp, len(sub))
        grupos.append(sub.sample(n=n, random_state=semilla))
    return pd.concat(grupos).sample(frac=1, random_state=semilla).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

print("\nCargando dataset balanceado desde PostgreSQL...")
df_full = cargar_dataset(fuentes=['warsaw_ut', 'sintetico'], verbose=True)
df_full = df_full[FEATURES + [TARGET]].dropna().reset_index(drop=True)

print(f"Dataset completo: {len(df_full)} filas")
for t in TEMPERATURAS:
    n = (df_full['T'] == t).sum()
    print(f"  T={t}°C: {n} puntos")

# ══════════════════════════════════════════════════════════════════════════════
# SEPARAR TEST FIJO (antes del loop, siempre los mismos puntos)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSeparando test fijo: {TEST_PTS_POR_TEMP} puntos/temperatura...")

df_test_list  = []
df_pool_list  = []

for temp in TEMPERATURAS:
    sub = df_full[df_full['T'] == temp].copy()
    # Separar test fijo con semilla fija
    if len(sub) >= TEST_PTS_POR_TEMP + TRAIN_SIZES[-1]:
        test_sub  = sub.sample(n=TEST_PTS_POR_TEMP, random_state=RANDOM_STATE)
        pool_sub  = sub.drop(test_sub.index)
    else:
        # Si hay pocos datos, usar 20% como test
        idx_test  = sub.sample(frac=0.2, random_state=RANDOM_STATE).index
        test_sub  = sub.loc[idx_test]
        pool_sub  = sub.drop(idx_test)
    df_test_list.append(test_sub)
    df_pool_list.append(pool_sub)

df_test = pd.concat(df_test_list).reset_index(drop=True)
df_pool = pd.concat(df_pool_list).reset_index(drop=True)

X_test  = df_test[FEATURES].values
y_test  = df_test[TARGET].values
vn_test = v_nernst_df(df_test)

print(f"Test fijo: {len(df_test)} puntos totales")
print(f"Pool disponible para entrenamiento: {len(df_pool)} puntos")

# ── Nernst sobre test (referencia fija) ──────────────────────────────────────
r2_nernst, mae_nernst, nrmse_nernst = metricas(y_test, vn_test)
print(f"\nNernst (referencia): R²={r2_nernst:.4f} | MAE={mae_nernst:.4f} V | NRMSE={nrmse_nernst:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL: entrenar cada modelo para cada tamaño
# ══════════════════════════════════════════════════════════════════════════════

resultados = []

for n_pts in TRAIN_SIZES:
    print(f"\n{'='*60}")
    print(f"  n_train = {n_pts} puntos/temperatura ({n_pts*5} total)")
    print(f"{'='*60}")

    # Muestrear entrenamiento desde el pool
    df_train = muestrear_balanceado(df_pool, n_pts, semilla=RANDOM_STATE)
    X_train  = df_train[FEATURES].values
    y_train  = df_train[TARGET].values
    rango    = y_test.max() - y_test.min()

    fila = {'n_pts_por_temp': n_pts, 'n_train_total': len(df_train)}

    # ── PLS ───────────────────────────────────────────────────────────────────
    print(f"  Entrenando PLS...")
    try:
        pipe_pls = Pipeline([
            ('scaler', StandardScaler()),
            ('pls',    PLSRegression(n_components=min(9, len(FEATURES)), scale=False))
        ])
        modelo_pls = TransformedTargetRegressor(
            regressor=pipe_pls, transformer=StandardScaler()
        )
        modelo_pls.fit(X_train, y_train)
        y_pred = modelo_pls.predict(X_test).ravel()
        r2, mae, nrmse = metricas(y_test, y_pred)
        print(f"    R²={r2:.4f} | MAE={mae:.4f} V | NRMSE={nrmse:.4f}")
    except Exception as ex:
        print(f"    ERROR PLS: {ex}")
        r2, mae, nrmse = np.nan, np.nan, np.nan

    fila.update({'pls_r2': r2, 'pls_mae': mae, 'pls_nrmse': nrmse})

    # ── KPLS (hiperparámetros fijos del mejor conocido) ───────────────────────
    print(f"  Entrenando KPLS (hiperparámetros fijos)...")
    try:
        pipe_kpls = Pipeline([
            ('scaler', StandardScaler()),
            ('rbf',    RBFSampler(gamma=KPLS_GAMMA,
                                  n_components=KPLS_N_COMPONENTS_RBF,
                                  random_state=RANDOM_STATE)),
            ('pls',    PLSRegression(n_components=min(KPLS_N_COMPONENTS_PLS,
                                                       len(FEATURES)),
                                     scale=False))
        ])
        modelo_kpls = TransformedTargetRegressor(
            regressor=pipe_kpls, transformer=StandardScaler()
        )
        modelo_kpls.fit(X_train, y_train)
        y_pred = modelo_kpls.predict(X_test).ravel()
        r2, mae, nrmse = metricas(y_test, y_pred)
        print(f"    R²={r2:.4f} | MAE={mae:.4f} V | NRMSE={nrmse:.4f}")
    except Exception as ex:
        print(f"    ERROR KPLS: {ex}")
        r2, mae, nrmse = np.nan, np.nan, np.nan

    fila.update({'kpls_r2': r2, 'kpls_mae': mae, 'kpls_nrmse': nrmse})

    # ── GPR ───────────────────────────────────────────────────────────────────
    print(f"  Entrenando GPR...")
    try:
        scaler_X_gpr = StandardScaler()
        scaler_y_gpr = StandardScaler()
        X_tr_sc = scaler_X_gpr.fit_transform(X_train)
        X_te_sc = scaler_X_gpr.transform(X_test)
        y_tr_sc = scaler_y_gpr.fit_transform(y_train.reshape(-1,1)).ravel()

        kernel_gpr = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            RBF(np.ones(len(FEATURES)), (1e-2, 1e2)) +
            WhiteKernel(0.01, (1e-5, 1.0))
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel_gpr, n_restarts_optimizer=3,
            random_state=RANDOM_STATE, normalize_y=False
        )
        gpr.fit(X_tr_sc, y_tr_sc)
        y_pred_sc = gpr.predict(X_te_sc)
        y_pred    = scaler_y_gpr.inverse_transform(
                        y_pred_sc.reshape(-1,1)).ravel()
        r2, mae, nrmse = metricas(y_test, y_pred)
        print(f"    R²={r2:.4f} | MAE={mae:.4f} V | NRMSE={nrmse:.4f}")
    except Exception as ex:
        print(f"    ERROR GPR: {ex}")
        r2, mae, nrmse = np.nan, np.nan, np.nan
        scaler_X_gpr = scaler_y_gpr = gpr = None

    fila.update({'gpr_r2': r2, 'gpr_mae': mae, 'gpr_nrmse': nrmse})

    # ── GPR Residual ──────────────────────────────────────────────────────────
    print(f"  Entrenando GPR Residual...")
    try:
        vn_train = v_nernst_df(df_train)
        epsilon  = y_train - vn_train

        scaler_X_res = StandardScaler()
        scaler_e_res = StandardScaler()
        X_tr_sc  = scaler_X_res.fit_transform(X_train)
        X_te_sc  = scaler_X_res.transform(X_test)
        e_tr_sc  = scaler_e_res.fit_transform(epsilon.reshape(-1,1)).ravel()

        kernel_res = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            RBF(np.ones(len(FEATURES)), (1e-2, 1e2)) +
            WhiteKernel(0.01, (1e-5, 1.0))
        )
        gpr_res = GaussianProcessRegressor(
            kernel=kernel_res, n_restarts_optimizer=3,
            random_state=RANDOM_STATE, normalize_y=False
        )
        gpr_res.fit(X_tr_sc, e_tr_sc)

        e_pred_sc = gpr_res.predict(X_te_sc)
        e_pred    = scaler_e_res.inverse_transform(
                        e_pred_sc.reshape(-1,1)).ravel()
        y_pred    = vn_test + e_pred
        r2, mae, nrmse = metricas(y_test, y_pred)
        print(f"    R²={r2:.4f} | MAE={mae:.4f} V | NRMSE={nrmse:.4f}")
    except Exception as ex:
        print(f"    ERROR GPR Residual: {ex}")
        r2, mae, nrmse = np.nan, np.nan, np.nan

    fila.update({'gpr_res_r2': r2, 'gpr_res_mae': mae, 'gpr_res_nrmse': nrmse})

    # Agregar referencia Nernst (constante)
    fila.update({
        'nernst_r2':    r2_nernst,
        'nernst_mae':   mae_nernst,
        'nernst_nrmse': nrmse_nernst,
    })

    resultados.append(fila)
    print(f"\n  Resumen n={n_pts}:")
    print(f"  {'Modelo':<15} {'R²':>8} {'MAE':>9} {'NRMSE':>8}")
    print(f"  {'-'*43}")
    for modelo, key in [('PLS','pls'), ('KPLS','kpls'),
                        ('GPR','gpr'), ('GPR Residual','gpr_res'),
                        ('Nernst','nernst')]:
        print(f"  {modelo:<15} "
              f"{fila[f'{key}_r2']:>8.4f} "
              f"{fila[f'{key}_mae']:>8.4f}V "
              f"{fila[f'{key}_nrmse']:>8.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR CSV
# ══════════════════════════════════════════════════════════════════════════════

df_res = pd.DataFrame(resultados)
csv_path = os.path.join(OUT_DIR, 'curva_aprendizaje_resultados.csv')
df_res.to_csv(csv_path, index=False, float_format='%.4f')
print(f"\nResultados guardados en: {csv_path}")
print(df_res.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

MODELOS_LINEAS = [
    ('PLS',          'pls'),
    ('KPLS',         'kpls'),
    ('GPR',          'gpr'),
    ('GPR Residual', 'gpr_res'),
]
x = df_res['n_pts_por_temp'].values

def estilo_base():
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'font.size':        10,
        'axes.titlesize':   11,
        'axes.labelsize':   10,
        'legend.fontsize':  9,
        'axes.grid':        True,
        'grid.alpha':       0.3,
        'grid.linestyle':   '--',
        'axes.spines.top':  False,
        'axes.spines.right':False,
    })

def guardar_grafico(metrica_label, metrica_key, ylabel, titulo,
                    ylim=None, invert_y=False, filename=None):
    estilo_base()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for nombre, key in MODELOS_LINEAS:
        y_vals = df_res[f'{key}_{metrica_key}'].values
        ax.plot(x, y_vals,
                color=COLORES[nombre],
                linestyle=LINESTYLES[nombre],
                linewidth=2,
                marker='o', markersize=5,
                label=nombre)

    # Nernst como línea horizontal de referencia
    nernst_val = df_res[f'nernst_{metrica_key}'].iloc[0]
    ax.axhline(nernst_val,
               color=COLORES['Nernst'],
               linestyle='--', linewidth=1.5,
               label=f'Nernst ({nernst_val:.3f})')

    ax.set_xlabel('Puntos de entrenamiento por temperatura', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(loc='best', framealpha=0.9)

    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {path}")

# ── R² ────────────────────────────────────────────────────────────────────────
guardar_grafico(
    metrica_label='R²',
    metrica_key='r2',
    ylabel='$R^2$ (conjunto de prueba)',
    titulo='Curva de aprendizaje — $R^2$ vs tamaño del dataset de entrenamiento',
    ylim=(0.5, 1.01),
    filename='curva_aprendizaje_r2.pdf'
)

# ── MAE ───────────────────────────────────────────────────────────────────────
guardar_grafico(
    metrica_label='MAE',
    metrica_key='mae',
    ylabel='MAE (V)',
    titulo='Curva de aprendizaje — MAE vs tamaño del dataset de entrenamiento',
    filename='curva_aprendizaje_mae.pdf'
)

# ── NRMSE ─────────────────────────────────────────────────────────────────────
guardar_grafico(
    metrica_label='NRMSE',
    metrica_key='nrmse',
    ylabel='NRMSE',
    titulo='Curva de aprendizaje — NRMSE vs tamaño del dataset de entrenamiento',
    filename='curva_aprendizaje_nrmse.pdf'
)

print("\n¡Análisis completado!")
print(f"Archivos generados en: {OUT_DIR}")
print("  - curva_aprendizaje_resultados.csv")
print("  - curva_aprendizaje_r2.pdf")
print("  - curva_aprendizaje_mae.pdf")
print("  - curva_aprendizaje_nrmse.pdf")
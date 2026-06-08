"""
evaluar_holdout.py
------------------
Evaluación de generalización a temperaturas bajas (550-625°C) usando
las curvas reales de Milewski que NUNCA fueron vistas durante el
entrenamiento de los modelos holdout.

Compara:
  - Modelos _warsaw_holdout: entrenados solo con 650°C real
  - Modelos _balanceado_holdout: entrenados con 650°C real + sintéticos

Ambos se evalúan sobre las 61 mediciones reales de 550-625°C
(nunca vistas en entrenamiento), midiendo la capacidad de
generalización a temperaturas fuera del dominio de entrenamiento.

Uso:
    python3 evaluar_holdout.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import psycopg2
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import DB_CONFIG
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modelos'))
from modelo_nernst import voltaje_modelo

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.join(BASE_DIR, '..', 'modelos')
HOLDOUT_TEMPS = [550, 575, 600, 625]

OUT_CSV = os.path.join(BASE_DIR, 'evaluacion_holdout.csv')

# ── Cargar modelos ─────────────────────────────────────────────────────────────
print("Cargando modelos holdout...")
modelos = {}
for sufijo in ['warsaw_holdout', 'balanceado_holdout']:
    modelos[sufijo] = {
        'pls':  joblib.load(os.path.join(MODELOS_DIR, f'pls_voltaje_cv_{sufijo}.pkl')),
        'kpls': joblib.load(os.path.join(MODELOS_DIR, f'kpls_voltaje_cv_{sufijo}.pkl')),
        'gpr':  joblib.load(os.path.join(MODELOS_DIR, f'gpr_voltaje_{sufijo}.pkl')),
        'gprr': joblib.load(os.path.join(MODELOS_DIR, f'gpr_residual_{sufijo}.pkl')),
    }
    print(f"  ✓ {sufijo}: PLS, KPLS, GPR, GPR Residual cargados")

# ── Cargar datos holdout desde PostgreSQL ──────────────────────────────────────
print("\nCargando curvas reales holdout (550-625°C)...")
placeholders = ', '.join(['%s'] * len(HOLDOUT_TEMPS))
conn = psycopg2.connect(**DB_CONFIG)
df = pd.read_sql(f"""
    SELECT
        e.id_experimento, e.t AS "T",
        e.h2a AS "H2a", e.h2oa AS "H2Oa", e.co2a AS "CO2a",
        e.n2a AS "N2a", e.co AS "CO",   e.ch4 AS "CH4",
        e.o2c AS "O2c", e.co2c AS "CO2c",
        e.n2c AS "N2c", e.h2oc AS "H2Oc",
        p.r_1,
        m.i_densidad AS "i", m.voltaje AS "E_real"
    FROM experimentos e
    JOIN parametros_modelo p USING(id_experimento)
    JOIN mediciones        m USING(id_experimento)
    WHERE e.fuente = 'warsaw_ut'
      AND e.t IN ({placeholders})
    ORDER BY e.t, m.i_densidad
""", conn, params=tuple(HOLDOUT_TEMPS))
conn.close()

print(f"  {len(df)} mediciones en {df['id_experimento'].nunique()} curvas reales")
print(f"  Temperaturas: {sorted(df['T'].unique().tolist())}")
print(f"  Distribución: {df.groupby('T').size().to_dict()}")


# ── Funciones auxiliares ───────────────────────────────────────────────────────
def nrmse(y_true, y_pred):
    rango = y_true.max() - y_true.min()
    if rango < 1e-10:
        return np.nan
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / rango

def metricas(E_real, y_pred):
    mae = mean_absolute_error(E_real, y_pred)
    nr  = nrmse(E_real, y_pred)
    if len(E_real) < 2:
        return np.nan, mae, np.nan
    ss_res = np.sum((E_real - y_pred) ** 2)
    ss_tot = np.sum((E_real - E_real.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan
    return r2, mae, nr

def predecir(sufijo, grp, i_arr, E_nernst, X):
    m = modelos[sufijo]

    # PLS
    E_pls = m['pls']['modelo'].predict(X).ravel()

    # KPLS
    E_kpls = m['kpls']['modelo'].predict(X).ravel()

    # GPR
    gpr_data = m['gpr']
    X_sc = gpr_data['scaler_X'].transform(X)
    y_sc = gpr_data['modelo'].predict(X_sc)
    E_gpr = gpr_data['scaler_y'].inverse_transform(
                y_sc.reshape(-1,1)).ravel()

    # GPR Residual
    gprr_data = m['gprr']
    X_sc_r = gprr_data['scaler_X'].transform(X)
    e_sc = gprr_data['modelo'].predict(X_sc_r)
    e_pred = gprr_data['scaler_e'].inverse_transform(
                e_sc.reshape(-1,1)).ravel()
    E_gprr = E_nernst + e_pred

    return E_pls, E_kpls, E_gpr, E_gprr


# ── Evaluación por temperatura ─────────────────────────────────────────────────
print("\n" + "="*75)
print("EVALUACIÓN DE GENERALIZACIÓN — CURVAS REALES NUNCA VISTAS (550-625°C)")
print("="*75)

resultados = []
t0 = time.time()

for T, grp in df.groupby('T'):
    row    = grp.iloc[0]
    i_arr  = grp['i'].values
    E_real = grp['E_real'].values

    T_val  = float(row['T'])
    H2a    = float(row['H2a']);  H2Oa = float(row['H2Oa'])
    CO2a   = float(row['CO2a']); O2c  = float(row['O2c'])
    CO2c   = float(row['CO2c']); N2c  = float(row['N2c'])
    N2a    = float(row['N2a']);  CO   = float(row['CO'])
    CH4    = float(row['CH4']);  H2Oc = float(row['H2Oc'])
    r1     = float(row['r_1'])

    # Nernst (igual para ambos sufijos)
    E_nernst = voltaje_modelo(
        j=i_arr, T_C=T_val, h2a=H2a, h2oa=H2Oa, co2a=CO2a,
        o2c=O2c, co2c=CO2c, r1=r1,
        n2a=N2a, co=CO, ch4=CH4, n2c=N2c, h2oc=H2Oc
    )

    # Feature matrix
    X = np.column_stack([
        np.full_like(i_arr, T_val),
        np.full_like(i_arr, H2a),
        np.full_like(i_arr, H2Oa),
        np.full_like(i_arr, CO2a),
        np.full_like(i_arr, O2c),
        np.full_like(i_arr, CO2c),
        np.full_like(i_arr, N2c),
        i_arr,
        np.full_like(i_arr, r1)
    ])

    # Nernst métricas
    r2_n, mae_n, nrmse_n = metricas(E_real, E_nernst)

    fila = {
        'T': int(T_val),
        'n_puntos': len(E_real),
        'r2_nernst':    round(r2_n,    4),
        'mae_nernst':   round(mae_n,   4),
        'nrmse_nernst': round(nrmse_n, 4),
    }

    # Métricas para cada sufijo
    for sufijo in ['warsaw_holdout', 'balanceado_holdout']:
        E_pls, E_kpls, E_gpr, E_gprr = predecir(sufijo, grp, i_arr, E_nernst, X)
        tag = 'w' if sufijo == 'warsaw_holdout' else 'b'

        r2_p,  mae_p,  nrmse_p  = metricas(E_real, E_pls)
        r2_k,  mae_k,  nrmse_k  = metricas(E_real, E_kpls)
        r2_g,  mae_g,  nrmse_g  = metricas(E_real, E_gpr)
        r2_gr, mae_gr, nrmse_gr = metricas(E_real, E_gprr)

        fila.update({
            f'r2_pls_{tag}':      round(r2_p,   4),
            f'mae_pls_{tag}':     round(mae_p,   4),
            f'nrmse_pls_{tag}':   round(nrmse_p, 4),
            f'r2_kpls_{tag}':     round(r2_k,   4),
            f'mae_kpls_{tag}':    round(mae_k,   4),
            f'nrmse_kpls_{tag}':  round(nrmse_k, 4),
            f'r2_gpr_{tag}':      round(r2_g,   4),
            f'mae_gpr_{tag}':     round(mae_g,   4),
            f'nrmse_gpr_{tag}':   round(nrmse_g, 4),
            f'r2_gprr_{tag}':     round(r2_gr,  4),
            f'mae_gprr_{tag}':    round(mae_gr,  4),
            f'nrmse_gprr_{tag}':  round(nrmse_gr,4),
        })

    resultados.append(fila)

df_res = pd.DataFrame(resultados)
df_res.to_csv(OUT_CSV, index=False)

# ── Imprimir resultados ────────────────────────────────────────────────────────
print(f"\n{'T':>5} | {'n':>3} | {'Nernst':>8} | "
      f"{'PLS_w':>7} | {'PLS_b':>7} | "
      f"{'KPLS_w':>7} | {'KPLS_b':>7} | "
      f"{'GPR_w':>7} | {'GPR_b':>7} | "
      f"{'GPRR_w':>7} | {'GPRR_b':>7}")
print("-"*105)
for _, r in df_res.iterrows():
    print(f"{int(r['T']):>5} | {int(r['n_puntos']):>3} | {r['r2_nernst']:>8.4f} | "
          f"{r['r2_pls_w']:>7.4f} | {r['r2_pls_b']:>7.4f} | "
          f"{r['r2_kpls_w']:>7.4f} | {r['r2_kpls_b']:>7.4f} | "
          f"{r['r2_gpr_w']:>7.4f} | {r['r2_gpr_b']:>7.4f} | "
          f"{r['r2_gprr_w']:>7.4f} | {r['r2_gprr_b']:>7.4f}")

# ── Resumen global ─────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("RESUMEN GLOBAL (promedio sobre 550-625°C)")
print("="*75)
print(f"\n{'Modelo':<20} | {'R² medio':>9} | {'MAE medio':>10} | {'NRMSE medio':>12}")
print("-"*58)

cols = [
    ('Nernst',            'r2_nernst',    'mae_nernst',   'nrmse_nernst'),
    ('PLS (warsaw)',       'r2_pls_w',     'mae_pls_w',    'nrmse_pls_w'),
    ('PLS (balanceado)',   'r2_pls_b',     'mae_pls_b',    'nrmse_pls_b'),
    ('KPLS (warsaw)',      'r2_kpls_w',    'mae_kpls_w',   'nrmse_kpls_w'),
    ('KPLS (balanceado)',  'r2_kpls_b',    'mae_kpls_b',   'nrmse_kpls_b'),
    ('GPR (warsaw)',       'r2_gpr_w',     'mae_gpr_w',    'nrmse_gpr_w'),
    ('GPR (balanceado)',   'r2_gpr_b',     'mae_gpr_b',    'nrmse_gpr_b'),
    ('GPR Res (warsaw)',   'r2_gprr_w',    'mae_gprr_w',   'nrmse_gprr_w'),
    ('GPR Res (balanc.)', 'r2_gprr_b',    'mae_gprr_b',   'nrmse_gprr_b'),
]

for nombre, c_r2, c_mae, c_nrmse in cols:
    r2m   = df_res[c_r2].mean()
    maem  = df_res[c_mae].mean()
    nrmsem= df_res[c_nrmse].mean()
    print(f"{nombre:<20} | {r2m:>9.4f} | {maem:>10.4f} | {nrmsem:>12.4f}")

print(f"\nResultados guardados en: {OUT_CSV}")
print(f"Tiempo total: {time.time()-t0:.1f}s")
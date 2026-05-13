"""
evaluar_por_experimento.py
--------------------------
Evalua los tres modelos (Nernst, PLS, KPLS) sobre cada uno de los
111 experimentos reales de Milewski, usando sus datos experimentales
directamente desde PostgreSQL.

Genera un CSV con R2, MAE y NRMSE por experimento y modelo,
y un resumen por temperatura.

Uso:
    python3 evaluar_por_experimento.py
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
OUT_CSV     = os.path.join(BASE_DIR, 'evaluacion_por_experimento.csv')
OUT_RESUMEN = os.path.join(BASE_DIR, 'resumen_por_temperatura.csv')

# ── Cargar modelos ─────────────────────────────────────────────────────────────
print("Cargando modelos...")
pls_data  = joblib.load(os.path.join(MODELOS_DIR, 'pls_voltaje_cv.pkl'))
kpls_data = joblib.load(os.path.join(MODELOS_DIR, 'kpls_voltaje_cv.pkl'))
pls_pipe  = pls_data['modelo']
kpls_pipe = kpls_data['modelo']
PLS_FEATS = pls_data['features']
print(f"  PLS features:  {PLS_FEATS}")
print(f"  KPLS features: {kpls_data['features']}")

# ── Cargar datos desde PostgreSQL ──────────────────────────────────────────────
print("\nCargando datos desde PostgreSQL...")
conn = psycopg2.connect(**DB_CONFIG)
df = pd.read_sql("""
    SELECT
        e.id_experimento, e.fuente, e.t AS "T",
        e.h2a AS "H2a", e.h2oa AS "H2Oa", e.co2a AS "CO2a",
        e.n2a AS "N2a", e.co AS "CO", e.ch4 AS "CH4",
        e.o2c AS "O2c", e.co2c AS "CO2c",
        e.n2c AS "N2c", e.h2oc AS "H2Oc",
        p.r_1, p.r_2,
        m.i_densidad AS "i", m.voltaje AS "E_real"
    FROM experimentos e
    JOIN parametros_modelo p USING(id_experimento)
    JOIN mediciones m USING(id_experimento)
    WHERE e.fuente = 'warsaw_ut'
    ORDER BY e.id_experimento, m.i_densidad
""", conn)
conn.close()

print(f"  {len(df)} mediciones de {df['id_experimento'].nunique()} experimentos")

# ── Funciones auxiliares ───────────────────────────────────────────────────────
def nrmse(y_true, y_pred):
    rango = y_true.max() - y_true.min()
    if rango < 1e-10:
        return np.nan
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / rango

def evaluar_experimento(grp):
    row      = grp.iloc[0]
    i_arr    = grp['i'].values
    E_real   = grp['E_real'].values

    if len(E_real) < 1:
        return None

    T    = float(row['T'])
    H2a  = float(row['H2a'])
    H2Oa = float(row['H2Oa'])
    CO2a = float(row['CO2a'])
    O2c  = float(row['O2c'])
    CO2c = float(row['CO2c'])
    N2c  = float(row['N2c'])
    N2a  = float(row['N2a'])
    CO   = float(row['CO'])
    CH4  = float(row['CH4'])
    H2Oc = float(row['H2Oc'])
    r1   = float(row['r_1'])

    # Nernst
    E_nernst = voltaje_modelo(
        j=i_arr, T_C=T, h2a=H2a, h2oa=H2Oa, co2a=CO2a,
        o2c=O2c, co2c=CO2c, r1=r1,
        n2a=N2a, co=CO, ch4=CH4, n2c=N2c, h2oc=H2Oc
    )

    # PLS y KPLS
    X = np.column_stack([
        np.full_like(i_arr, T),
        np.full_like(i_arr, H2a),
        np.full_like(i_arr, H2Oa),
        np.full_like(i_arr, CO2a),
        np.full_like(i_arr, O2c),
        np.full_like(i_arr, CO2c),
        np.full_like(i_arr, N2c),
        i_arr,
        np.full_like(i_arr, r1)
    ])

    E_pls  = pls_pipe.predict(X).ravel()
    E_kpls = kpls_pipe.predict(X).ravel()

    def metricas(y_pred):
        mae = mean_absolute_error(E_real, y_pred)
        nr  = nrmse(E_real, y_pred)
        if len(E_real) < 2:
            return np.nan, mae, np.nan
        ss_res = np.sum((E_real - y_pred) ** 2)
        ss_tot = np.sum((E_real - E_real.mean()) ** 2)
        r2  = 1 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan
        return r2, mae, nr

    r2_n,  mae_n,  nrmse_n  = metricas(E_nernst)
    r2_p,  mae_p,  nrmse_p  = metricas(E_pls)
    r2_k,  mae_k,  nrmse_k  = metricas(E_kpls)

    return {
        'id_experimento': int(row['id_experimento']),
        'T':              T,
        'n_puntos':       len(E_real),
        'r2_nernst':      round(r2_n,  4),
        'mae_nernst':     round(mae_n, 4),
        'nrmse_nernst':   round(nrmse_n, 4),
        'r2_pls':         round(r2_p,  4),
        'mae_pls':        round(mae_p, 4),
        'nrmse_pls':      round(nrmse_p, 4),
        'r2_kpls':        round(r2_k,  4),
        'mae_kpls':       round(mae_k, 4),
        'nrmse_kpls':     round(nrmse_k, 4),
        'mejor_modelo':   max(
            [('Nernst', r2_n), ('PLS', r2_p), ('KPLS', r2_k)],
            key=lambda x: x[1] if not np.isnan(x[1]) else -999
        )[0]
    }

# ── Evaluar todos los experimentos ─────────────────────────────────────────────
print("\nEvaluando experimentos...")
t0       = time.time()
resultados = []
grupos   = list(df.groupby('id_experimento'))
n_total  = len(grupos)

for idx, (exp_id, grp) in enumerate(grupos):
    res = evaluar_experimento(grp)
    if res:
        resultados.append(res)
    if (idx + 1) % 10 == 0 or (idx + 1) == n_total:
        elapsed = time.time() - t0
        print(f"  [{idx+1:3d}/{n_total}] exp_id={exp_id} — {elapsed:.1f}s")

# ── Guardar resultados ─────────────────────────────────────────────────────────
df_res = pd.DataFrame(resultados)
df_res.to_csv(OUT_CSV, index=False)
print(f"\nResultados guardados en: {OUT_CSV}")

# ── Resumen global ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN GLOBAL")
print("="*60)
print(f"Experimentos evaluados: {len(df_res)}")
print(f"\n{'Modelo':<10} | {'R2 medio':>10} | {'R2 min':>8} | {'R2 max':>8} | {'MAE medio':>10} | {'n_R2':>6}")
print("-"*65)
for m, col_r2, col_mae in [
    ('Nernst', 'r2_nernst', 'mae_nernst'),
    ('PLS',    'r2_pls',    'mae_pls'),
    ('KPLS',   'r2_kpls',   'mae_kpls'),
]:
    r2vals = df_res[col_r2].dropna()
    r2m  = r2vals.mean()
    r2mn = r2vals.min()
    r2mx = r2vals.max()
    maem = df_res[col_mae].mean()
    n_r2 = len(r2vals)
    print(f"{m:<10} | {r2m:>10.4f} | {r2mn:>8.4f} | {r2mx:>8.4f} | {maem:>10.4f} | {n_r2:>6}")

# ── Resumen por temperatura ────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN POR TEMPERATURA")
print("="*60)
resumen_T = []
for T, grp_T in df_res.groupby('T'):
    resumen_T.append({
        'T': T, 'n_exp': len(grp_T),
        'r2_nernst': round(grp_T['r2_nernst'].mean(), 4),
        'r2_pls':    round(grp_T['r2_pls'].mean(),    4),
        'r2_kpls':   round(grp_T['r2_kpls'].mean(),   4),
        'mae_nernst':round(grp_T['mae_nernst'].mean(), 4),
        'mae_pls':   round(grp_T['mae_pls'].mean(),   4),
        'mae_kpls':  round(grp_T['mae_kpls'].mean(),  4),
    })

df_T = pd.DataFrame(resumen_T)
df_T.to_csv(OUT_RESUMEN, index=False)
print(f"\n{'T':>5} | {'n':>4} | {'R2_N':>7} | {'R2_PLS':>7} | {'R2_KPLS':>8} | {'MAE_N':>7} | {'MAE_P':>7} | {'MAE_K':>7}")
print("-"*70)
for _, r in df_T.iterrows():
    print(f"{int(r['T']):>5} | {int(r['n_exp']):>4} | {r['r2_nernst']:>7.4f} | {r['r2_pls']:>7.4f} | {r['r2_kpls']:>8.4f} | {r['mae_nernst']:>7.4f} | {r['mae_pls']:>7.4f} | {r['mae_kpls']:>7.4f}")

# ── Ganadores ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MEJOR MODELO POR EXPERIMENTO")
print("="*60)
print(df_res['mejor_modelo'].value_counts().to_string())

n_1punto = len(df_res[df_res['n_puntos'] == 1])
print(f"\nNota: {n_1punto} experimentos con 1 punto — R² no calculable (NaN), MAE si disponible.")
print(f"R² calculado sobre {len(df_res['r2_nernst'].dropna())} experimentos con n >= 2 puntos.")

elapsed_total = time.time() - t0
print(f"\nTiempo total: {elapsed_total:.1f}s")
"""
evaluar_por_experimento.py
--------------------------
Evalúa los cinco modelos (Nernst, PLS, KPLS, GPR, GPR Residual) sobre
cada una de las 110 curvas de polarización reales de Milewski, usando
sus datos experimentales directamente desde PostgreSQL.

Por defecto usa los modelos _warsaw.pkl (entrenados solo con datos reales).
Con --sufijo balanceado usa los modelos _balanceado.pkl.

Uso:
    python3 evaluar_por_experimento.py
    python3 evaluar_por_experimento.py --sufijo balanceado
"""

import os
import sys
import time
import argparse
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

# ── Argumento sufijo ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluación curva a curva de los 5 modelos MCFC")
parser.add_argument(
    '--sufijo', type=str, default='warsaw',
    choices=['warsaw', 'balanceado'],
    help="Sufijo de los modelos .pkl a usar (default: warsaw)"
)
args = parser.parse_args()

OUT_CSV     = os.path.join(BASE_DIR, f'evaluacion_por_experimento_{args.sufijo}.csv')
OUT_RESUMEN = os.path.join(BASE_DIR, f'resumen_por_temperatura_{args.sufijo}.csv')

print(f"Usando modelos con sufijo: _{args.sufijo}")

# ── Cargar modelos ─────────────────────────────────────────────────────────────
print("\nCargando modelos...")
pls_data   = joblib.load(os.path.join(MODELOS_DIR, f'pls_voltaje_cv_{args.sufijo}.pkl'))
kpls_data  = joblib.load(os.path.join(MODELOS_DIR, f'kpls_voltaje_cv_{args.sufijo}.pkl'))
gpr_data   = joblib.load(os.path.join(MODELOS_DIR, f'gpr_voltaje_{args.sufijo}.pkl'))
gprr_data  = joblib.load(os.path.join(MODELOS_DIR, f'gpr_residual_{args.sufijo}.pkl'))

pls_pipe   = pls_data['modelo']
kpls_pipe  = kpls_data['modelo']
gpr_model  = gpr_data['modelo']
gprr_model = gprr_data['modelo']
scaler_X_gpr  = gpr_data['scaler_X']
scaler_y_gpr  = gpr_data['scaler_y']
scaler_X_gprr = gprr_data['scaler_X']
scaler_e_gprr = gprr_data['scaler_e']

FEATURES_ESPERADAS = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²', 'r_1']
print(f"  PLS features:  {pls_data['features']}")
print(f"  KPLS features: {kpls_data['features']}")
print(f"  GPR features:  {gpr_data['features']}")
print(f"  GPR Res. features: {gprr_data['features']}")

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
print(f"  {len(df)} mediciones de {df['id_experimento'].nunique()} curvas de polarización")

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

def evaluar_experimento(grp):
    row    = grp.iloc[0]
    i_arr  = grp['i'].values
    E_real = grp['E_real'].values
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

    # ── Nernst ─────────────────────────────────────────────────────────────────
    E_nernst = voltaje_modelo(
        j=i_arr, T_C=T, h2a=H2a, h2oa=H2Oa, co2a=CO2a,
        o2c=O2c, co2c=CO2c, r1=r1,
        n2a=N2a, co=CO, ch4=CH4, n2c=N2c, h2oc=H2Oc
    )

    # ── Feature matrix ─────────────────────────────────────────────────────────
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

    # ── PLS y KPLS ─────────────────────────────────────────────────────────────
    E_pls  = pls_pipe.predict(X).ravel()
    E_kpls = kpls_pipe.predict(X).ravel()

    # ── GPR ────────────────────────────────────────────────────────────────────
    X_sc_gpr  = scaler_X_gpr.transform(X)
    y_sc_pred = gpr_model.predict(X_sc_gpr)
    E_gpr     = scaler_y_gpr.inverse_transform(
                    y_sc_pred.reshape(-1,1)).ravel()

    # ── GPR Residual ───────────────────────────────────────────────────────────
    X_sc_gprr  = scaler_X_gprr.transform(X)
    e_sc_pred  = gprr_model.predict(X_sc_gprr)
    e_pred     = scaler_e_gprr.inverse_transform(
                    e_sc_pred.reshape(-1,1)).ravel()
    E_gprr     = E_nernst + e_pred

    # ── Métricas ───────────────────────────────────────────────────────────────
    r2_n,  mae_n,  nrmse_n  = metricas(E_real, E_nernst)
    r2_p,  mae_p,  nrmse_p  = metricas(E_real, E_pls)
    r2_k,  mae_k,  nrmse_k  = metricas(E_real, E_kpls)
    r2_g,  mae_g,  nrmse_g  = metricas(E_real, E_gpr)
    r2_gr, mae_gr, nrmse_gr = metricas(E_real, E_gprr)

    mejor = max(
        [('Nernst', r2_n), ('PLS', r2_p), ('KPLS', r2_k),
         ('GPR', r2_g), ('GPR_Res', r2_gr)],
        key=lambda x: x[1] if not np.isnan(x[1]) else -999
    )[0]

    return {
        'id_experimento': int(row['id_experimento']),
        'T':              T,
        'n_puntos':       len(E_real),
        'r2_nernst':      round(r2_n,  4),  'mae_nernst':  round(mae_n,  4),
        'r2_pls':         round(r2_p,  4),  'mae_pls':     round(mae_p,  4),
        'r2_kpls':        round(r2_k,  4),  'mae_kpls':    round(mae_k,  4),
        'r2_gpr':         round(r2_g,  4),  'mae_gpr':     round(mae_g,  4),
        'r2_gpr_res':     round(r2_gr, 4),  'mae_gpr_res': round(mae_gr, 4),
        'mejor_modelo':   mejor,
    }

# ── Evaluar todos los experimentos ─────────────────────────────────────────────
print("\nEvaluando experimentos...")
t0 = time.time()
resultados = []
grupos     = list(df.groupby('id_experimento'))
n_total    = len(grupos)

for idx, (exp_id, grp) in enumerate(grupos):
    res = evaluar_experimento(grp)
    if res:
        resultados.append(res)
    if (idx + 1) % 10 == 0 or (idx + 1) == n_total:
        print(f"  [{idx+1:3d}/{n_total}] exp_id={exp_id} — {time.time()-t0:.1f}s")

# ── Guardar CSV ────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(resultados)
df_res.to_csv(OUT_CSV, index=False)
print(f"\nResultados guardados en: {OUT_CSV}")

# ── Resumen global ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("RESUMEN GLOBAL")
print("="*70)
print(f"Curvas evaluadas: {len(df_res)}\n")
modelos = [
    ('Nernst',   'r2_nernst',  'mae_nernst'),
    ('PLS',      'r2_pls',     'mae_pls'),
    ('KPLS',     'r2_kpls',    'mae_kpls'),
    ('GPR',      'r2_gpr',     'mae_gpr'),
    ('GPR_Res',  'r2_gpr_res', 'mae_gpr_res'),
]
print(f"{'Modelo':<10} | {'R2 medio':>10} | {'R2 min':>8} | {'R2 max':>8} | {'MAE medio':>10} | {'n_R2':>6}")
print("-"*65)
for m, col_r2, col_mae in modelos:
    r2vals = df_res[col_r2].dropna()
    print(f"{m:<10} | {r2vals.mean():>10.4f} | {r2vals.min():>8.4f} | "
          f"{r2vals.max():>8.4f} | {df_res[col_mae].mean():>10.4f} | {len(r2vals):>6}")

# ── Resumen por temperatura ────────────────────────────────────────────────────
print("\n" + "="*70)
print("RESUMEN POR TEMPERATURA")
print("="*70)
resumen_T = []
for T, grp_T in df_res.groupby('T'):
    resumen_T.append({
        'T': T, 'n_exp': len(grp_T),
        'R2_N':    round(grp_T['r2_nernst'].mean(),  4),
        'R2_PLS':  round(grp_T['r2_pls'].mean(),     4),
        'R2_KPLS': round(grp_T['r2_kpls'].mean(),    4),
        'R2_GPR':  round(grp_T['r2_gpr'].mean(),     4),
        'R2_GPRR': round(grp_T['r2_gpr_res'].mean(), 4),
        'MAE_N':   round(grp_T['mae_nernst'].mean(),  4),
        'MAE_PLS': round(grp_T['mae_pls'].mean(),     4),
        'MAE_K':   round(grp_T['mae_kpls'].mean(),    4),
        'MAE_GPR': round(grp_T['mae_gpr'].mean(),     4),
        'MAE_GPRR':round(grp_T['mae_gpr_res'].mean(), 4),
    })

df_T = pd.DataFrame(resumen_T)
df_T.to_csv(OUT_RESUMEN, index=False)
print(f"\n{'T':>5} | {'n':>4} | {'R2_N':>7} | {'R2_PLS':>7} | {'R2_KPLS':>8} | {'R2_GPR':>7} | {'R2_GPRR':>8}")
print("-"*75)
for _, r in df_T.iterrows():
    print(f"{int(r['T']):>5} | {int(r['n_exp']):>4} | {r['R2_N']:>7.4f} | "
          f"{r['R2_PLS']:>7.4f} | {r['R2_KPLS']:>8.4f} | "
          f"{r['R2_GPR']:>7.4f} | {r['R2_GPRR']:>8.4f}")

# ── Ganadores ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("MEJOR MODELO POR EXPERIMENTO")
print("="*70)
print(df_res['mejor_modelo'].value_counts().to_string())

n_1punto = len(df_res[df_res['n_puntos'] == 1])
print(f"\nNota: {n_1punto} curvas con 1 punto — R² no calculable (NaN), MAE sí disponible.")
print(f"R² calculado sobre {len(df_res['r2_nernst'].dropna())} curvas con n >= 2 puntos.")
print(f"\nTiempo total: {time.time()-t0:.1f}s")
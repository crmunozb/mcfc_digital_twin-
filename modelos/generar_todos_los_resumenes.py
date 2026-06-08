"""
generar_todos_los_resumenes.py
------------------------------
Script unificado que genera TODOS los CSVs de evaluación necesarios
para la tesis v28, usando los modelos .pkl actuales.

Genera:
  1. resumen_por_temperatura_warsaw.csv    → tab:r2_por_temperatura_original
  2. resumen_por_temperatura_balanceado.csv → tab:r2_por_temperatura
  3. evaluacion_holdout.csv               → tab:holdout
  4. resumen_augmentacion.csv             → tab:comparacion_augmentacion
  5. evaluacion_global.csv                → tab:comparacion_modelos_original
                                            tab:comparacion_modelos

Uso:
    cd mcfc_digital_twin/modelos/
    python3 generar_todos_los_resumenes.py

Requiere:
  - PostgreSQL activo con DB_CONFIG configurado
  - Modelos .pkl entrenados en la carpeta modelos/
  - modelo_nernst.py en la misma carpeta
  - cargar_datos.py en la misma carpeta
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import psycopg2
from sklearn.metrics import r2_score, mean_absolute_error

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = BASE_DIR  # los .pkl están en la misma carpeta

sys.path.insert(0, os.path.join(BASE_DIR, '..'))
from config import DB_CONFIG
from modelo_nernst import voltaje_modelo
from cargar_datos import cargar_dataset

FEATURES = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²', 'r_1']
TARGET   = 'Experiment'
HOLDOUT_TEMPS = [550, 575, 600, 625]

t_inicio = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def nrmse_fn(y_true, y_pred):
    rango = y_true.max() - y_true.min()
    if rango < 1e-10:
        return np.nan
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / rango

def metricas(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    nr   = nrmse_fn(y_true, y_pred)
    if len(y_true) < 2:
        return np.nan, mae, np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan
    return r2, mae, nr

def cargar_modelos(sufijo):
    """Carga los 4 modelos para un sufijo dado."""
    m = {}
    m['pls']  = joblib.load(os.path.join(MODELOS_DIR, f'pls_voltaje_cv_{sufijo}.pkl'))
    m['kpls'] = joblib.load(os.path.join(MODELOS_DIR, f'kpls_voltaje_cv_{sufijo}.pkl'))
    m['gpr']  = joblib.load(os.path.join(MODELOS_DIR, f'gpr_voltaje_{sufijo}.pkl'))
    m['gprr'] = joblib.load(os.path.join(MODELOS_DIR, f'gpr_residual_{sufijo}.pkl'))
    print(f"  ✓ Modelos _{sufijo} cargados")
    return m

def predecir_todos(modelos, X, E_nernst):
    """Devuelve predicciones de los 4 modelos data-driven."""
    # PLS
    E_pls = modelos['pls']['modelo'].predict(X).ravel()

    # KPLS
    E_kpls = modelos['kpls']['modelo'].predict(X).ravel()

    # GPR
    gpr = modelos['gpr']
    X_sc = gpr['scaler_X'].transform(X)
    E_gpr = gpr['scaler_y'].inverse_transform(
                gpr['modelo'].predict(X_sc).reshape(-1,1)).ravel()

    # GPR Residual
    gprr = modelos['gprr']
    X_sc_r = gprr['scaler_X'].transform(X)
    e_pred = gprr['scaler_e'].inverse_transform(
                 gprr['modelo'].predict(X_sc_r).reshape(-1,1)).ravel()
    E_gprr = E_nernst + e_pred

    return E_pls, E_kpls, E_gpr, E_gprr

def construir_X(grp):
    """Construye feature matrix y vector Nernst para un grupo de mediciones."""
    row   = grp.iloc[0]
    i_arr = grp['i'].values

    T_val = float(row['T'])
    H2a   = float(row['H2a']);  H2Oa = float(row['H2Oa'])
    CO2a  = float(row['CO2a']); O2c  = float(row['O2c'])
    CO2c  = float(row['CO2c']); N2c  = float(row['N2c'])
    N2a   = float(row.get('N2a', 0)); CO = float(row.get('CO', 0))
    CH4   = float(row.get('CH4', 0)); H2Oc = float(row.get('H2Oc', 0))
    r1    = float(row['r_1'])

    E_nernst = voltaje_modelo(
        j=i_arr, T_C=T_val, h2a=H2a, h2oa=H2Oa, co2a=CO2a,
        o2c=O2c, co2c=CO2c, r1=r1,
        n2a=N2a, co=CO, ch4=CH4, n2c=N2c, h2oc=H2Oc
    )

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
    return X, E_nernst, i_arr

# ══════════════════════════════════════════════════════════════════════════════
# CARGAR DATOS DESDE POSTGRESQL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Cargando datos desde PostgreSQL...")
print("="*70)

conn = psycopg2.connect(**DB_CONFIG)
df_all = pd.read_sql("""
    SELECT
        e.id_experimento, e.fuente, e.t AS "T",
        e.h2a AS "H2a", e.h2oa AS "H2Oa", e.co2a AS "CO2a",
        e.n2a AS "N2a", e.co AS "CO", e.ch4 AS "CH4",
        e.o2c AS "O2c", e.co2c AS "CO2c",
        e.n2c AS "N2c", e.h2oc AS "H2Oc",
        p.r_1,
        m.i_densidad AS "i", m.voltaje AS "E_real"
    FROM experimentos e
    JOIN parametros_modelo p USING(id_experimento)
    JOIN mediciones        m USING(id_experimento)
    WHERE e.fuente = 'warsaw_ut'
    ORDER BY e.t, e.id_experimento, m.i_densidad
""", conn)
conn.close()

df_warsaw = df_all  # alias más claro
print(f"  {len(df_warsaw)} mediciones reales warsaw_ut")
print(f"  Temperaturas: {sorted(df_warsaw['T'].unique().tolist())}")

# ══════════════════════════════════════════════════════════════════════════════
# CARGAR MODELOS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Cargando modelos .pkl...")
print("="*70)

m_warsaw    = cargar_modelos('warsaw')
m_balanceado = cargar_modelos('balanceado')
m_w_holdout = cargar_modelos('warsaw_holdout')
m_b_holdout = cargar_modelos('balanceado_holdout')

# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1: resumen_por_temperatura_warsaw.csv y resumen_por_temperatura_balanceado.csv
# Evaluación curva a curva → promedio por temperatura
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("BLOQUE 1: Evaluación por experimento (warsaw y balanceado)")
print("="*70)

for sufijo, modelos in [('warsaw', m_warsaw), ('balanceado', m_balanceado)]:
    print(f"\n  Procesando _{sufijo}...")
    resultados_exp = []

    for exp_id, grp in df_warsaw.groupby('id_experimento'):
        E_real = grp['E_real'].values
        if len(E_real) < 1:
            continue

        X, E_nernst, i_arr = construir_X(grp)
        T_val = float(grp.iloc[0]['T'])

        E_pls, E_kpls, E_gpr, E_gprr = predecir_todos(modelos, X, E_nernst)

        r2_n,  mae_n,  _ = metricas(E_real, E_nernst)
        r2_p,  mae_p,  _ = metricas(E_real, E_pls)
        r2_k,  mae_k,  _ = metricas(E_real, E_kpls)
        r2_g,  mae_g,  _ = metricas(E_real, E_gpr)
        r2_gr, mae_gr, _ = metricas(E_real, E_gprr)

        resultados_exp.append({
            'id_experimento': int(exp_id),
            'T': T_val, 'n_puntos': len(E_real),
            'r2_nernst': r2_n,  'mae_nernst': mae_n,
            'r2_pls':    r2_p,  'mae_pls':    mae_p,
            'r2_kpls':   r2_k,  'mae_kpls':   mae_k,
            'r2_gpr':    r2_g,  'mae_gpr':     mae_g,
            'r2_gpr_res':r2_gr, 'mae_gpr_res': mae_gr,
        })

    df_exp = pd.DataFrame(resultados_exp)

    # Resumen por temperatura (promedio de R² por experimento)
    resumen_T = []
    for T, grp_T in df_exp.groupby('T'):
        resumen_T.append({
            'T': T, 'n_exp': len(grp_T),
            'R2_N':    round(grp_T['r2_nernst'].mean(), 4),
            'R2_PLS':  round(grp_T['r2_pls'].mean(),    4),
            'R2_KPLS': round(grp_T['r2_kpls'].mean(),   4),
            'R2_GPR':  round(grp_T['r2_gpr'].mean(),    4),
            'R2_GPRR': round(grp_T['r2_gpr_res'].mean(),4),
            'MAE_N':   round(grp_T['mae_nernst'].mean(), 4),
            'MAE_PLS': round(grp_T['mae_pls'].mean(),    4),
            'MAE_K':   round(grp_T['mae_kpls'].mean(),   4),
            'MAE_GPR': round(grp_T['mae_gpr'].mean(),    4),
            'MAE_GPRR':round(grp_T['mae_gpr_res'].mean(),4),
        })

    df_T = pd.DataFrame(resumen_T)
    out_path = os.path.join(BASE_DIR, f'resumen_por_temperatura_{sufijo}.csv')
    df_T.to_csv(out_path, index=False)
    print(f"  ✓ Guardado: {out_path}")
    print(df_T[['T','R2_N','R2_PLS','R2_KPLS','R2_GPR','R2_GPRR']].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2: resumen_augmentacion.csv
# Compara R² por temperatura entre warsaw y balanceado (para tab:comparacion_augmentacion)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("BLOQUE 2: Resumen augmentación (warsaw vs balanceado por temperatura)")
print("="*70)

# Reutilizamos los resúmenes generados en bloque 1
df_w = pd.read_csv(os.path.join(BASE_DIR, 'resumen_por_temperatura_warsaw.csv'))
df_b = pd.read_csv(os.path.join(BASE_DIR, 'resumen_por_temperatura_balanceado.csv'))

df_aug = pd.merge(df_w[['T','R2_PLS','R2_KPLS']], df_b[['T','R2_PLS','R2_KPLS']],
                  on='T', suffixes=('_w', '_b'))
df_aug.columns = ['T', 'PLS_solo_real', 'KPLS_solo_real', 'PLS_balanceado', 'KPLS_balanceado']
df_aug = df_aug.round(4)

out_aug = os.path.join(BASE_DIR, 'resumen_augmentacion.csv')
df_aug.to_csv(out_aug, index=False)
print(f"  ✓ Guardado: {out_aug}")
print(df_aug.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3: evaluacion_holdout.csv
# Evaluación sobre curvas reales nunca vistas (550-625°C)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("BLOQUE 3: Evaluación holdout (550-625°C)")
print("="*70)

df_holdout = df_warsaw[df_warsaw['T'].isin(HOLDOUT_TEMPS)].copy()
print(f"  {len(df_holdout)} mediciones holdout")

resultados_h = []
for T, grp in df_holdout.groupby('T'):
    E_real = grp['E_real'].values
    X, E_nernst, _ = construir_X(grp)
    T_val = float(T)

    r2_n, mae_n, nrmse_n = metricas(E_real, E_nernst)
    fila = {
        'T': int(T_val), 'n_puntos': len(E_real),
        'r2_nernst': round(r2_n, 4), 'mae_nernst': round(mae_n, 4),
        'nrmse_nernst': round(nrmse_n, 4) if not np.isnan(nrmse_n) else np.nan,
    }

    for sufijo, modelos, tag in [
        ('warsaw_holdout',    m_w_holdout, 'w'),
        ('balanceado_holdout', m_b_holdout, 'b'),
    ]:
        E_pls, E_kpls, E_gpr, E_gprr = predecir_todos(modelos, X, E_nernst)
        r2_p,  mae_p,  nr_p  = metricas(E_real, E_pls)
        r2_k,  mae_k,  nr_k  = metricas(E_real, E_kpls)
        r2_g,  mae_g,  nr_g  = metricas(E_real, E_gpr)
        r2_gr, mae_gr, nr_gr = metricas(E_real, E_gprr)

        fila.update({
            f'r2_pls_{tag}':   round(r2_p,  4), f'mae_pls_{tag}':  round(mae_p,  4),
            f'r2_kpls_{tag}':  round(r2_k,  4), f'mae_kpls_{tag}': round(mae_k,  4),
            f'r2_gpr_{tag}':   round(r2_g,  4), f'mae_gpr_{tag}':  round(mae_g,  4),
            f'r2_gprr_{tag}':  round(r2_gr, 4), f'mae_gprr_{tag}': round(mae_gr, 4),
        })

    resultados_h.append(fila)

df_h = pd.DataFrame(resultados_h)
out_h = os.path.join(BASE_DIR, 'evaluacion_holdout.csv')
df_h.to_csv(out_h, index=False)
print(f"  ✓ Guardado: {out_h}")

# Imprimir resumen holdout
cols_show = ['T','r2_nernst','r2_pls_w','r2_pls_b','r2_kpls_w','r2_kpls_b',
             'r2_gpr_w','r2_gpr_b','r2_gprr_w','r2_gprr_b']
print(df_h[cols_show].to_string(index=False))

medias = df_h[cols_show[1:]].mean()
print("\n  Medias:")
for col, val in medias.items():
    print(f"    {col}: {val:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4: evaluacion_global.csv
# Métricas globales test de los 4 modelos (para tab:comparacion_modelos_original
# y tab:comparacion_modelos)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("BLOQUE 4: Métricas globales test (desde .pkl)")
print("="*70)

filas_global = []
for sufijo, modelos in [('warsaw', m_warsaw), ('balanceado', m_balanceado)]:
    for nombre, key in [('PLS', 'pls'), ('KPLS', 'kpls'), ('GPR', 'gpr'), ('GPR_Residual', 'gprr')]:
        d = modelos[key]
        filas_global.append({
            'dataset': sufijo,
            'modelo': nombre,
            'r2_test':    round(d.get('r2_test', np.nan), 4),
            'mae_test':   round(d.get('mae_test', np.nan), 4),
            'nrmse_test': round(d.get('nrmse_test', np.nan), 4),
        })

df_global = pd.DataFrame(filas_global)
out_global = os.path.join(BASE_DIR, 'evaluacion_global.csv')
df_global.to_csv(out_global, index=False)
print(f"  ✓ Guardado: {out_global}")
print(df_global.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print(f"COMPLETADO en {time.time()-t_inicio:.1f}s")
print("="*70)
print("Archivos generados:")
archivos = [
    'resumen_por_temperatura_warsaw.csv',
    'resumen_por_temperatura_balanceado.csv',
    'resumen_augmentacion.csv',
    'evaluacion_holdout.csv',
    'evaluacion_global.csv',
]
for a in archivos:
    path = os.path.join(BASE_DIR, a)
    existe = "✓" if os.path.exists(path) else "✗ NO ENCONTRADO"
    print(f"  {existe}  {a}")

print("\nPróximo paso: subir los CSVs a Claude para actualizar el LaTeX.")
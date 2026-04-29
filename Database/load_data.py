import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'mcfc_digital_twin',
    'user': 'mcfc_user',
    'password': 'Lasvioletas1756'
}

# ── Cargar Excel ───────────────────────────────────────────────────────────────
df = pd.read_excel('../Data/Data_original_PGNN.xlsx')
df['T'] = df['T'].astype(int)

print("Columnas del Excel:")
print(df.columns.tolist())
print(f"\nTotal filas: {len(df)}")
print(f"Temperaturas únicas: {sorted(df['T'].unique())}")

# ── Columnas de condiciones experimentales ────────────────────────────────────
# Usamos T como única columna de agrupación primaria,
# y el resto por nombre exacto del Excel
cond_cols = ['T', 'H2a', 'H2Oa', 'N2a', 'CO', 'CH4', 'CO2a',
             'δNia', 'ρa', 'δLiKe', 'δNiOc', 'ρc',
             'O2c', 'N2c', 'CO2c', 'H2Oc']

# Verificar que todas las columnas existen
missing = [c for c in cond_cols if c not in df.columns]
if missing:
    print(f"\n⚠️  Columnas NO encontradas en el Excel: {missing}")
    print("Columnas disponibles:", df.columns.tolist())
    exit(1)
else:
    print(f"\n✅ Todas las columnas de condiciones encontradas")

# ── Conexión ──────────────────────────────────────────────────────────────────
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

n_exp = 0
n_med = 0

for conds, grupo in df.groupby(cond_cols):
    T, H2a, H2Oa, N2a, CO, CH4, CO2a, dNia, rho_a, dLiKe, dNiOc, rho_c, O2c, N2c, CO2c, H2Oc = conds

    cur.execute('''
        INSERT INTO experimentos
        (fuente, T, H2a, H2Oa, N2a, CO, CH4, CO2a,
         O2c, N2c, CO2c, H2Oc,
         delta_Nia, rho_a, delta_LiKe, delta_NiOc, rho_c)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s)
        RETURNING id_experimento
    ''', ('warsaw_ut', int(T),
          float(H2a), float(H2Oa), float(N2a),
          float(CO),  float(CH4),  float(CO2a),
          float(O2c), float(N2c),  float(CO2c), float(H2Oc),
          float(dNia), float(rho_a), float(dLiKe),
          float(dNiOc), float(rho_c)))

    id_exp = cur.fetchone()[0]
    n_exp += 1

    fila = grupo.iloc[0]
    cur.execute('''
        INSERT INTO parametros_modelo
        (id_experimento, E_max, i_max, r_1, r_2, n_H2_a_in)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (id_exp,
          float(fila['E_max']),    float(fila['i_max']),
          float(fila['r_1']),      float(fila['r_2']),
          float(fila['n_H2_a_in'])))

    mediciones_data = [
        (id_exp,
         float(row['i, A/cm²']),
         float(row['Experiment']),
         float(row['eta']))
        for _, row in grupo.iterrows()
    ]

    execute_values(cur, '''
        INSERT INTO mediciones (id_experimento, i_densidad, voltaje, eta)
        VALUES %s
    ''', mediciones_data)

    n_med += len(mediciones_data)

    # Progreso cada 10 experimentos
    if n_exp % 10 == 0:
        print(f"  Procesados: {n_exp} experimentos, {n_med} mediciones...")

conn.commit()
cur.close()
conn.close()

print(f"\n✅ Base de datos cargada exitosamente")
print(f"   Experimentos: {n_exp}")
print(f"   Mediciones:   {n_med}")
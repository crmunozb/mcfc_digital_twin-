import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from config import DB_CONFIG


# ── Cargar Excel ───────────────────────────────────────────────────────────────
df = pd.read_excel('Data/Data_original_PGNN.xlsx')
df['T'] = df['T'].astype(int)

print("Columnas del Excel:")
print(df.columns.tolist())
print(f"\nTotal filas: {len(df)}")
print(f"Temperaturas únicas: {sorted(df['T'].unique())}")


# ── Columnas para identificar curvas de polarización ──────────────────────────
# Cada curva se define por temperatura y composición gaseosa.
cond_cols = [
    'T', 'H2a', 'H2Oa', 'N2a', 'CO', 'CH4', 'CO2a',
    'O2c', 'N2c', 'CO2c', 'H2Oc'
]

# Columnas adicionales que se guardan como metadatos del experimento,
# pero no se usan para separar una nueva curva.
meta_cols = ['δNia', 'ρa', 'δLiKe', 'δNiOc', 'ρc']

required_cols = cond_cols + meta_cols + [
    'E_max', 'i_max', 'r_1', 'r_2', 'n_H2_a_in',
    'i, A/cm²', 'Experiment', 'eta'
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"\nColumnas NO encontradas en el Excel: {missing}")
    print("Columnas disponibles:", df.columns.tolist())
    exit(1)

print("\nTodas las columnas necesarias fueron encontradas.")


# ── Conexión ──────────────────────────────────────────────────────────────────
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

n_exp = 0
n_med = 0

grupos = df.groupby(cond_cols, dropna=False, sort=False)

print(f"\nCurvas de polarización detectadas: {len(grupos)}")

for conds, grupo in grupos:
    cond = dict(zip(cond_cols, conds))
    fila = grupo.iloc[0]

    cur.execute('''
        INSERT INTO experimentos
        (fuente, T, H2a, H2Oa, N2a, CO, CH4, CO2a,
         O2c, N2c, CO2c, H2Oc,
         delta_Nia, rho_a, delta_LiKe, delta_NiOc, rho_c)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s)
        RETURNING id_experimento
    ''', (
        'warsaw_ut',
        int(cond['T']),
        float(cond['H2a']),
        float(cond['H2Oa']),
        float(cond['N2a']),
        float(cond['CO']),
        float(cond['CH4']),
        float(cond['CO2a']),
        float(cond['O2c']),
        float(cond['N2c']),
        float(cond['CO2c']),
        float(cond['H2Oc']),
        float(fila['δNia']),
        float(fila['ρa']),
        float(fila['δLiKe']),
        float(fila['δNiOc']),
        float(fila['ρc'])
    ))

    id_exp = cur.fetchone()[0]
    n_exp += 1

    cur.execute('''
        INSERT INTO parametros_modelo
        (id_experimento, E_max, i_max, r_1, r_2, n_H2_a_in)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (
        id_exp,
        float(fila['E_max']),
        float(fila['i_max']),
        float(fila['r_1']),
        float(fila['r_2']),
        float(fila['n_H2_a_in'])
    ))

    mediciones_data = [
        (
            id_exp,
            float(row['i, A/cm²']),
            float(row['Experiment']),
            float(row['eta'])
        )
        for _, row in grupo.iterrows()
    ]

    execute_values(cur, '''
        INSERT INTO mediciones (id_experimento, i_densidad, voltaje, eta)
        VALUES %s
    ''', mediciones_data)

    n_med += len(mediciones_data)

    if n_exp % 10 == 0:
        print(f"  Procesados: {n_exp} curvas, {n_med} mediciones...")


conn.commit()
cur.close()
conn.close()

print("\nBase de datos cargada exitosamente")
print(f"Curvas/experimentos: {n_exp}")
print(f"Mediciones:          {n_med}")
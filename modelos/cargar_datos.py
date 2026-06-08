"""
cargar_datos.py
---------------
Módulo centralizado para cargar el dataset MCFC desde PostgreSQL.

Reemplaza la carga directa desde Excel en los scripts de entrenamiento,
permitiendo incorporar tanto los datos reales de Milewski (warsaw_ut)
como los datos sintéticos generados (sintetico).

Las columnas del DataFrame devuelto son compatibles con los scripts de
entrenamiento existentes — mismos nombres que el Excel original:
    T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r_1, i, A/cm², Experiment, eta

Uso:
    from cargar_datos import cargar_dataset

    # Solo datos reales de Milewski (comportamiento original)
    df = cargar_dataset(fuentes=['warsaw_ut'])

    # Datos reales + sintéticos (dataset balanceado)
    df = cargar_dataset(fuentes=['warsaw_ut', 'sintetico'])

    # Con holdout: excluye curvas reales de 550-625°C del train
    df = cargar_dataset(fuentes=['warsaw_ut'], holdout_temps=[550, 575, 600, 625])

    # Con holdout + sintéticos: excluye reales bajas, mantiene sintéticos
    df = cargar_dataset(fuentes=['warsaw_ut', 'sintetico'],
                        holdout_temps=[550, 575, 600, 625])
"""

import os
import sys
import pandas as pd
import psycopg2

# Importar DB_CONFIG desde config.py en la raíz del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..'))
from config import DB_CONFIG


def cargar_dataset(fuentes: list = None,
                   holdout_temps: list = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Carga el dataset MCFC desde PostgreSQL.

    Parámetros
    ----------
    fuentes : list, opcional
        Lista de fuentes a incluir. Valores posibles: 'warsaw_ut', 'sintetico'.
        Si es None, carga todas las fuentes disponibles.
    holdout_temps : list, opcional
        Lista de temperaturas (ej. [550, 575, 600, 625]) cuyas curvas REALES
        (warsaw_ut) se excluyen del dataset de entrenamiento.
        Los datos sintéticos de esas temperaturas NO se excluyen.
        Útil para evaluación de generalización a temperaturas no vistas.
    verbose : bool
        Si True, imprime resumen del dataset cargado.

    Retorna
    -------
    pd.DataFrame con columnas compatibles con los scripts de entrenamiento:
        T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r_1, 'i, A/cm²', Experiment, eta
    """
    # Construir filtro de fuentes
    if fuentes is None:
        filtro_fuentes = ""
        params = []
    else:
        placeholders = ', '.join(['%s'] * len(fuentes))
        filtro_fuentes = f"WHERE e.fuente IN ({placeholders})"
        params = list(fuentes)

    query = f"""
        SELECT
            e.fuente,
            e.id_experimento,
            e.t                AS "T",
            e.h2a              AS "H2a",
            e.h2oa             AS "H2Oa",
            e.co2a             AS "CO2a",
            e.o2c              AS "O2c",
            e.co2c             AS "CO2c",
            e.n2c              AS "N2c",
            p.r_1              AS "r_1",
            m.i_densidad       AS "i, A/cm²",
            m.voltaje          AS "Experiment",
            m.eta              AS "eta",
            m.e_max            AS "E_max"
        FROM experimentos e
        JOIN parametros_modelo p ON p.id_experimento = e.id_experimento
        JOIN mediciones        m ON m.id_experimento = e.id_experimento
        {filtro_fuentes}
        ORDER BY e.t, e.id_experimento, m.i_densidad
    """

    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()

    # ── Aplicar holdout: excluir curvas REALES de temperaturas especificadas ──
    if holdout_temps:
        n_antes = len(df)
        mask_excluir = (
            (df['fuente'] == 'warsaw_ut') &
            (df['T'].isin(holdout_temps))
        )
        df = df[~mask_excluir].copy()
        n_excluidos = n_antes - len(df)
        if verbose:
            print(f"  Holdout aplicado: excluidas {n_excluidos} mediciones reales "
                  f"de T ∈ {holdout_temps}")

    if verbose:
        print(f"Dataset cargado desde PostgreSQL:")
        print(f"  Total filas:  {len(df)}")
        print(f"  Fuentes:      {sorted(df['fuente'].unique().tolist())}")
        print(f"  Temperaturas: {sorted(df['T'].unique().tolist())}")
        resumen = df.groupby(['fuente', 'T']).size().reset_index(name='n_mediciones')
        print(f"\n  Distribución por fuente y temperatura:")
        print(resumen.to_string(index=False))
        print()

    return df


def cargar_holdout(holdout_temps: list = None, verbose: bool = True) -> pd.DataFrame:
    """
    Carga SOLO las curvas reales de las temperaturas holdout.
    Usado para evaluar los modelos sobre datos nunca vistos en entrenamiento.

    Parámetros
    ----------
    holdout_temps : list
        Temperaturas a cargar. Default: [550, 575, 600, 625]

    Retorna
    -------
    pd.DataFrame con las curvas reales de las temperaturas holdout.
    """
    if holdout_temps is None:
        holdout_temps = [550, 575, 600, 625]

    placeholders = ', '.join(['%s'] * len(holdout_temps))

    query = f"""
        SELECT
            e.fuente,
            e.id_experimento,
            e.t                AS "T",
            e.h2a              AS "H2a",
            e.h2oa             AS "H2Oa",
            e.co2a             AS "CO2a",
            e.o2c              AS "O2c",
            e.co2c             AS "CO2c",
            e.n2c              AS "N2c",
            p.r_1              AS "r_1",
            m.i_densidad       AS "i, A/cm²",
            m.voltaje          AS "Experiment",
            m.eta              AS "eta",
            m.e_max            AS "E_max"
        FROM experimentos e
        JOIN parametros_modelo p ON p.id_experimento = e.id_experimento
        JOIN mediciones        m ON m.id_experimento = e.id_experimento
        WHERE e.fuente = 'warsaw_ut'
          AND e.t IN ({placeholders})
        ORDER BY e.t, e.id_experimento, m.i_densidad
    """

    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql_query(query, conn, params=tuple(holdout_temps))
    conn.close()

    if verbose:
        print(f"Dataset holdout cargado:")
        print(f"  Total filas:  {len(df)}")
        print(f"  Temperaturas: {sorted(df['T'].unique().tolist())}")
        resumen = df.groupby('T').size().reset_index(name='n_mediciones')
        print(resumen.to_string(index=False))
        print()

    return df


def cargar_features_target(fuentes: list = None,
                            holdout_temps: list = None,
                            verbose: bool = True):
    """
    Carga el dataset y devuelve X, y, T_vals listos para entrenar.

    Retorna
    -------
    X      : np.ndarray — matriz de features
    y      : np.ndarray — voltaje real (Experiment)
    T_vals : np.ndarray — temperatura por fila
    df     : pd.DataFrame — dataset completo
    """
    FEATURES = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²', 'r_1']
    TARGET   = 'Experiment'

    df = cargar_dataset(fuentes=fuentes, holdout_temps=holdout_temps,
                        verbose=verbose)
    df_clean = df[FEATURES + [TARGET]].dropna()

    X      = df_clean[FEATURES].values
    y      = df_clean[TARGET].values
    T_vals = df_clean['T'].values

    return X, y, T_vals, df_clean


if __name__ == "__main__":
    HOLDOUT_TEMPS = [550, 575, 600, 625]

    print("=== Test 1: datos reales sin holdout ===")
    df1 = cargar_dataset(fuentes=['warsaw_ut'])

    print("=== Test 2: datos reales CON holdout (warsaw_holdout) ===")
    df2 = cargar_dataset(fuentes=['warsaw_ut'], holdout_temps=HOLDOUT_TEMPS)

    print("=== Test 3: balanceado CON holdout (balanceado_holdout) ===")
    df3 = cargar_dataset(fuentes=['warsaw_ut', 'sintetico'],
                         holdout_temps=HOLDOUT_TEMPS)

    print("=== Test 4: curvas holdout para evaluación ===")
    df_test = cargar_holdout(HOLDOUT_TEMPS)

    print(f"\nResumen:")
    print(f"  Warsaw completo:   {len(df1)} filas")
    print(f"  Warsaw holdout:    {len(df2)} filas (excluye {len(df1)-len(df2)} reales de 550-625°C)")
    print(f"  Balanceado holdout:{len(df3)} filas")
    print(f"  Test holdout:      {len(df_test)} filas (curvas reales nunca vistas)")
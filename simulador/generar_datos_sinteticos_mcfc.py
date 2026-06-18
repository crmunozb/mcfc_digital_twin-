"""
generar_datos_sinteticos_mcfc.py
================================
Generador de curvas de polarización sintéticas para la celda MCFC,
calibrado sobre los rangos del dataset experimental de Milewski (650°C).

Estrategia:
- Para cada temperatura {550, 575, 600, 625, 650}°C, se generan N curvas
  variando composiciones gaseosas dentro de los rangos observados a 650°C.
- r_1 se fija al valor experimental de cada temperatura (único valor disponible).
- El voltaje se calcula con el modelo de Nernst con pérdidas + ruido gaussiano
  calibrado sobre el residuo real del modelo a 650°C.
- Las curvas sintéticas se insertan en la base de datos con fuente='sintetico'.

Uso:
    python generar_datos_sinteticos_mcfc.py --n_curvas 20 --insertar
    python generar_datos_sinteticos_mcfc.py --n_curvas 20  # solo preview, sin insertar

Requisitos:
    pip install numpy pandas psycopg2-binary sqlalchemy --break-system-packages
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE BASE DE DATOS
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_CONFIG

# Importar modelo de Nernst centralizado
# Garantiza consistencia con el evaluador y el dashboard
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modelos"))
from modelo_nernst import voltaje_modelo as _voltaje_modelo
from modelo_nernst import e_nernst as _e_nernst

# ─────────────────────────────────────────────
# CONSTANTES FÍSICAS
# ─────────────────────────────────────────────
R  = 8.314    # J/(mol·K)
F  = 96485.0  # C/mol

# Parámetros del modelo Nernst calibrado (mismo que modelo_nernst.py)
R2    = 91.878   # Ω⁻¹·cm²  — pérdidas de activación
DELTA = 0.012    # V        — offset de calibración

# ─────────────────────────────────────────────
# RANGOS OBSERVADOS A 650°C (desde query SQL)
# Usados como espacio de variación para composiciones
# ─────────────────────────────────────────────
RANGOS_650 = {
    "h2a":  (0.2204, 4.4077),
    "h2oa": (0.0514, 1.2896),
    "co2a": (0.0551, 1.1019),
    "o2c":  (0.1295, 5.2504),
    "co2c": (0.2667, 14.2447),
    "n2c":  (0.4872, 29.1086),
}

# ─────────────────────────────────────────────
# PARÁMETROS POR TEMPERATURA
# r_1 fijo al valor experimental real de cada temperatura
# ─────────────────────────────────────────────
TEMP_PARAMS = {
    550: {"r_1": 2.935458, "n2a": 0.0, "co": 0.0, "ch4": 0.0, "h2oc": 0.0},
    575: {"r_1": 2.634815, "n2a": 0.0, "co": 0.0, "ch4": 0.0, "h2oc": 0.0},
    600: {"r_1": 2.379644, "n2a": 0.0, "co": 0.0, "ch4": 0.0, "h2oc": 0.0},
    625: {"r_1": 2.161409, "n2a": 0.0, "co": 0.0, "ch4": 0.0, "h2oc": 0.0},
    650: {"r_1": 1.973445, "n2a": 0.0, "co": 0.0, "ch4": 0.0, "h2oc": 0.0},
}

# ─────────────────────────────────────────────
# Densidad de corriente 
# ─────────────────────────────────────────────
J_MIN  = 0.005   # A/cm²
J_MAX  = 0.200   # A/cm²
N_PUNTOS_CURVA = 11  # puntos por curva (igual al dataset real)

# Ruido gaussiano calibrado sobre residuo del modelo Nernst a 650°C
# mean=-0.1044 V, std=0.0769 V → usamos std como escala de ruido base
RUIDO_STD = 0.005  # V — ruido de medición (conservador)


# ─────────────────────────────────────────────
# Funciones de cálculo — usan modelo_nernst.py centralizado
# ─────────────────────────────────────────────
def calcular_nernst(T_C: float, h2a: float, h2oa: float, co2a: float,
                    o2c: float, co2c: float, n2c: float) -> float:
    """Calcula E_Nernst usando modelo_nernst.py (función centralizada)."""
    return float(_e_nernst(
        T_C=T_C, h2a=h2a, h2oa=h2oa, co2a=co2a,
        o2c=o2c, co2c=co2c, n2c=n2c
    ))


def calcular_voltaje(T_C: float, j: float, r_1: float,
                     h2a: float, h2oa: float, co2a: float,
                     o2c: float, co2c: float, n2c: float,
                     ruido: float = 0.0) -> float:
    """Calcula el voltaje de celda usando modelo_nernst.py (función centralizada)."""
    voltaje = float(_voltaje_modelo(
        j=j, T_C=T_C, h2a=h2a, h2oa=h2oa, co2a=co2a,
        o2c=o2c, co2c=co2c, r1=r_1, n2c=n2c,
        r2=R2, delta=DELTA
    ))
    return voltaje + ruido


def calcular_e_max(T_C: float, h2a: float, h2oa: float, co2a: float,
                   o2c: float, co2c: float, n2c: float) -> float:
    """E_max = Nernst sin pérdidas (j=0), usando modelo_nernst.py."""
    return calcular_nernst(T_C, h2a, h2oa, co2a, o2c, co2c, n2c)


# ─────────────────────────────────────────────
# Generador curvas sinteticas
# ─────────────────────────────────────────────
def generar_composicion_aleatoria(rng: np.random.Generator) -> dict:
    """Muestrea una composición gaseosa dentro de los rangos de 650°C."""
    h2a  = rng.uniform(*RANGOS_650["h2a"])
    h2oa = rng.uniform(*RANGOS_650["h2oa"])
    co2a = rng.uniform(*RANGOS_650["co2a"])
    o2c  = rng.uniform(*RANGOS_650["o2c"])
    co2c = rng.uniform(*RANGOS_650["co2c"])
    n2c  = rng.uniform(*RANGOS_650["n2c"])
    return {"h2a": h2a, "h2oa": h2oa, "co2a": co2a,
            "o2c": o2c, "co2c": co2c, "n2c": n2c}


def generar_curvas(n_curvas: int, seed: int = 42) -> pd.DataFrame:
    """
    Genera n_curvas por cada temperatura {550, 575, 600, 625, 650}°C.
    Retorna un DataFrame con todas las mediciones sintéticas.
    """
    rng    = np.random.default_rng(seed)
    j_vals = np.linspace(J_MIN, J_MAX, N_PUNTOS_CURVA)
    filas  = []
    curva_id = 0

    for T_C, params in TEMP_PARAMS.items():
        r_1 = params["r_1"]
        print(f"\n→ Generando {n_curvas} curvas para T={T_C}°C (r_1={r_1:.4f})")

        for i in range(n_curvas):
            comp = generar_composicion_aleatoria(rng)
            e_max = calcular_e_max(T_C, **comp)

            for j in j_vals:
                ruido   = rng.normal(0, RUIDO_STD)
                voltaje = calcular_voltaje(
                    T_C, j, r_1, ruido=ruido, **comp
                )
                # Eficiencia eléctrica (referencia HHV H2 = 1.48 V)
                eta = voltaje / 1.48 if j > 0 else 0.0
                # Potencia
                potencia = voltaje * j

                filas.append({
                    "curva_id":  curva_id,
                    "T":         T_C,
                    "r_1":       r_1,
                    "h2a":       comp["h2a"],
                    "h2oa":      comp["h2oa"],
                    "co2a":      comp["co2a"],
                    "o2c":       comp["o2c"],
                    "co2c":      comp["co2c"],
                    "n2c":       comp["n2c"],
                    "n2a":       params["n2a"],
                    "co":        params["co"],
                    "ch4":       params["ch4"],
                    "h2oc":      params["h2oc"],
                    "i_densidad": j,
                    "voltaje":   round(voltaje, 6),
                    "eta":       round(eta, 6),
                    "e_max":     round(e_max, 6),
                    "potencia":  round(potencia, 6),
                    "fuente":    "sintetico",
                })

            curva_id += 1

    df = pd.DataFrame(filas)
    print(f"\n✓ Total generado: {len(df)} mediciones en "
          f"{curva_id} curvas ({n_curvas} por temperatura × 5 temperaturas)")
    return df


# ─────────────────────────────────────────────
# Insercion Base de datos
# ─────────────────────────────────────────────
def insertar_en_bd(df: pd.DataFrame):
    """Inserta las curvas sintéticas en experimentos + mediciones."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Verificar que existe el valor 'sintetico' en el ENUM fuente
    try:
        cur.execute("ALTER TYPE fuente_datos ADD VALUE IF NOT EXISTS 'sintetico';")
        conn.commit()
    except Exception:
        conn.rollback()

    n_insertados = 0
    curvas = df["curva_id"].unique()

    for cid in curvas:
        sub = df[df["curva_id"] == cid].iloc[0]  # primera fila = parámetros

        # Insertar experimento
        cur.execute("""
            INSERT INTO experimentos
                (fuente, t, h2a, h2oa, n2a, co, ch4, co2a,
                 o2c, n2c, co2c, h2oc,
                 delta_nia, rho_a, delta_like, delta_nioc, rho_c)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s)
            RETURNING id_experimento
        """, (
            'sintetico',
            int(sub["T"]),
            sub["h2a"], sub["h2oa"], sub["n2a"],
            sub["co"],  sub["ch4"],  sub["co2a"],
            sub["o2c"], sub["n2c"],  sub["co2c"], sub["h2oc"],
            0.0, 0.0, 0.0, 0.0, 0.0   # delta/rho no aplica para sintéticos
        ))
        id_exp = cur.fetchone()[0]

        # Insertar parametros_modelo
        cur.execute("""
            INSERT INTO parametros_modelo
                (id_experimento, e_max, i_max, r_1, r_2, n_h2_a_in)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            id_exp,
            float(sub["e_max"]),
            J_MAX,
            float(sub["r_1"]),
            R2,
            float(sub["h2a"]),
        ))

        # Insertar mediciones de la curva
        mediciones = df[df["curva_id"] == cid]
        rows = [
            (id_exp, float(row["i_densidad"]),
             float(row["voltaje"]), float(row["eta"]),
             datetime.now(timezone.utc), float(row["e_max"]))
            for _, row in mediciones.iterrows()
        ]
        execute_values(cur, """
            INSERT INTO mediciones
                (id_experimento, i_densidad, voltaje, eta,
                 timestamp_medicion, e_max)
            VALUES %s
        """, rows)

        n_insertados += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"✓ Insertados {n_insertados} experimentos sintéticos en la base de datos.")


# ─────────────────────────────────────────────
# PREVIEW (sin insertar)
# ─────────────────────────────────────────────
def mostrar_preview(df: pd.DataFrame):
    """Muestra un resumen estadístico de los datos generados."""
    print("\n" + "="*60)
    print("PREVIEW — Datos sintéticos generados")
    print("="*60)
    resumen = df.groupby("T").agg(
        n_curvas=("curva_id", "nunique"),
        n_mediciones=("i_densidad", "count"),
        voltaje_min=("voltaje", "min"),
        voltaje_max=("voltaje", "max"),
        voltaje_mean=("voltaje", "mean"),
        j_min=("i_densidad", "min"),
        j_max=("i_densidad", "max"),
    ).round(4)
    print(resumen.to_string())
    print()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generador de curvas sintéticas MCFC calibrado en Milewski"
    )
    parser.add_argument(
        "--n_curvas", type=int, default=10,
        help="Número de curvas a generar por temperatura (default: 10)"
    )
    parser.add_argument(
        "--insertar", action="store_true",
        help="Si se especifica, inserta los datos en la base de datos"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria para reproducibilidad (default: 42)"
    )
    parser.add_argument(
        "--exportar_csv", type=str, default=None,
        help="Ruta opcional para exportar los datos a CSV"
    )
    args = parser.parse_args()

    print(f"Generando {args.n_curvas} curvas × 5 temperaturas "
          f"(seed={args.seed})...")

    df = generar_curvas(args.n_curvas, seed=args.seed)
    mostrar_preview(df)

    if args.exportar_csv:
        df.to_csv(args.exportar_csv, index=False)
        print(f"✓ Datos exportados a {args.exportar_csv}")

    if args.insertar:
        print("\nInsertando en base de datos...")
        insertar_en_bd(df)
    else:
        print("ℹ  Modo preview — usa --insertar para cargar en la BD.")
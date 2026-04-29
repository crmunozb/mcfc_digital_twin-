"""
simulador_mcfc.py
-----------------
Simula la operación de una celda MCFC generando datos realistas
e insertándolos en PostgreSQL minuto a minuto.

Modo de operación:
  - Crea un experimento sintético con fuente='simulado'
  - Genera mediciones (i_densidad, voltaje, eta) punto a punto
    recorriendo una curva de polarización con ruido realista
  - Inserta una medición por intervalo (por defecto 60 segundos)
  - Al completar la curva, genera nuevas condiciones operacionales
    y comienza un nuevo experimento simulado

Uso:
  python simulador_mcfc.py [--intervalo SEGUNDOS] [--temperatura CELSIUS]
"""

import argparse
import time
import random
import math
import signal
import sys
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor

# ─────────────────────────────────────────
# CONFIGURACIÓN DE CONEXIÓN
# ─────────────────────────────────────────
DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     5432,
    "dbname":   "mcfc_digital_twin",
    "user":     "mcfc_user",
    "password": "Lasvioletas1756",
}

# ─────────────────────────────────────────
# RANGOS REALISTAS BASADOS EN EL DATASET
# ─────────────────────────────────────────
RANGOS_OPERACION = {
    "t":       [550, 575, 600, 625, 650],   # temperaturas válidas
    "h2a":     (0.22, 2.20),
    "h2oa":    (0.05, 1.30),
    "n2a":     (0.0,  0.5),
    "co":      (0.0,  0.1),
    "ch4":     (0.0,  0.1),
    "co2a":    (0.05, 1.10),
    "o2c":     (0.10, 5.30),
    "n2c":     (0.50, 29.0),
    "co2c":    (0.30, 2.67),
    "h2oc":    (0.0,  0.5),
    "delta_nia":  (0.0, 0.1),
    "rho_a":      (0.0, 0.1),
    "delta_like": (0.0, 0.1),
    "delta_nioc": (0.0, 0.1),
    "rho_c":      (0.0, 0.1),
    "r1":      (1.80, 2.90),   # resistencia óhmica [Ohm·cm2]
}

# Densidades de corriente a simular por experimento [A/cm2]
I_DENSIDADES = [
    0.005, 0.010, 0.020, 0.030, 0.040, 0.050,
    0.060, 0.070, 0.080, 0.090, 0.100, 0.110,
    0.120, 0.130, 0.140, 0.150, 0.160, 0.170,
    0.180, 0.190, 0.200,
]

# Constantes físicas
R_GAS  = 8.314    # J/(mol·K)
F_FAR  = 96485    # C/mol
R2     = 91.878   # parámetro activación (del dataset)
DELTA  = 0.012    # offset calibración [V]

# ─────────────────────────────────────────
# MODELO ELECTROQUÍMICO (igual que el DT)
# ─────────────────────────────────────────

def e0_temperatura(T_K: float) -> float:
    """Potencial estándar en función de T [K]."""
    return 1.2723 - 2.7645e-4 * (T_K - 298.15)


def nernst(T_K: float, h2a: float, h2oa: float, co2a: float,
           o2c: float, co2c: float) -> float:
    """Voltaje de Nernst para MCFC."""
    # Normalizar a fracción molar
    suma_anodo  = h2a + h2oa + co2a + 1e-9
    suma_catodo = o2c + co2c + 1e-9

    xH2  = h2a  / suma_anodo
    xH2O = h2oa / suma_anodo
    xCO2a= co2a / suma_anodo
    xO2  = o2c  / suma_catodo
    xCO2c= co2c / suma_catodo

    # Evitar log(0)
    xH2   = max(xH2,   1e-6)
    xH2O  = max(xH2O,  1e-6)
    xCO2a = max(xCO2a, 1e-6)
    xO2   = max(xO2,   1e-6)
    xCO2c = max(xCO2c, 1e-6)

    arg = (xH2 * math.sqrt(xO2) * xCO2c) / (xH2O * xCO2a)
    return e0_temperatura(T_K) + (R_GAS * T_K / (2 * F_FAR)) * math.log(arg)


def voltaje_modelo(i: float, T_C: float, r1: float,
                   h2a: float, h2oa: float, co2a: float,
                   o2c: float, co2c: float) -> float:
    """Voltaje de celda según modelo semi-empírico."""
    T_K = T_C + 273.15
    E_N = nernst(T_K, h2a, h2oa, co2a, o2c, co2c)
    eta_ohm = r1 * i
    eta_act = (R_GAS * T_K / (2 * F_FAR)) * math.log(1 + R2 * i)
    V = E_N - eta_ohm - eta_act - DELTA
    return max(V, 0.0)


def eta_eficiencia(V: float, i: float) -> float:
    """Eficiencia eléctrica aproximada."""
    lhv_h2 = 241800  # J/mol (LHV hidrógeno)
    n_dot   = i / (2 * F_FAR)
    P = V * i
    if n_dot * lhv_h2 > 0:
        return min(P / (n_dot * lhv_h2), 1.0)
    return 0.0

# ─────────────────────────────────────────
# GENERADOR DE CONDICIONES OPERACIONALES
# ─────────────────────────────────────────

def generar_condiciones() -> dict:
    """Genera condiciones operacionales aleatorias dentro de rangos reales."""
    def rnd(rango):
        return round(random.uniform(*rango), 4)

    T = random.choice(RANGOS_OPERACION["t"])
    r1 = round(random.uniform(*RANGOS_OPERACION["r1"]), 4)

    return {
        "t":          T,
        "h2a":        rnd(RANGOS_OPERACION["h2a"]),
        "h2oa":       rnd(RANGOS_OPERACION["h2oa"]),
        "n2a":        rnd(RANGOS_OPERACION["n2a"]),
        "co":         rnd(RANGOS_OPERACION["co"]),
        "ch4":        rnd(RANGOS_OPERACION["ch4"]),
        "co2a":       rnd(RANGOS_OPERACION["co2a"]),
        "o2c":        rnd(RANGOS_OPERACION["o2c"]),
        "n2c":        rnd(RANGOS_OPERACION["n2c"]),
        "co2c":       rnd(RANGOS_OPERACION["co2c"]),
        "h2oc":       rnd(RANGOS_OPERACION["h2oc"]),
        "delta_nia":  rnd(RANGOS_OPERACION["delta_nia"]),
        "rho_a":      rnd(RANGOS_OPERACION["rho_a"]),
        "delta_like": rnd(RANGOS_OPERACION["delta_like"]),
        "delta_nioc": rnd(RANGOS_OPERACION["delta_nioc"]),
        "rho_c":      rnd(RANGOS_OPERACION["rho_c"]),
        "r1":         r1,
    }

# ─────────────────────────────────────────
# OPERACIONES DE BASE DE DATOS
# ─────────────────────────────────────────

def crear_experimento(conn, cond: dict) -> int:
    """Inserta un nuevo experimento simulado y retorna su id."""
    sql = """
        INSERT INTO experimentos (
            fuente, t, h2a, h2oa, n2a, co, ch4, co2a,
            o2c, n2c, co2c, h2oc,
            delta_nia, rho_a, delta_like, delta_nioc, rho_c
        ) VALUES (
            'udec_lab',
            %(t)s, %(h2a)s, %(h2oa)s, %(n2a)s, %(co)s, %(ch4)s, %(co2a)s,
            %(o2c)s, %(n2c)s, %(co2c)s, %(h2oc)s,
            %(delta_nia)s, %(rho_a)s, %(delta_like)s, %(delta_nioc)s, %(rho_c)s
        )
        RETURNING id_experimento;
    """
    with conn.cursor() as cur:
        cur.execute(sql, cond)
        id_exp = cur.fetchone()[0]
    conn.commit()
    return id_exp


def insertar_medicion(conn, id_exp: int, i: float, V: float, eta: float):
    """Inserta una medición con timestamp actual."""
    sql = """
        INSERT INTO mediciones (id_experimento, i_densidad, voltaje, eta, timestamp_medicion)
        VALUES (%s, %s, %s, %s, %s);
    """
    ts = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(sql, (id_exp, round(i, 6), round(V, 6), round(eta, 6), ts))
    conn.commit()


def parametros_modelo_existe(conn, id_exp: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM parametros_modelo WHERE id_experimento = %s LIMIT 1",
            (id_exp,)
        )
        return cur.fetchone() is not None


def insertar_parametros_modelo(conn, id_exp: int, r1: float,
                               e_max: float = None, i_max: float = None):
    """Inserta parámetros del modelo para el experimento simulado.
    
    Columnas reales de parametros_modelo:
      id_experimento, e_max, i_max, r_1, r_2, n_h2_a_in
    """
    try:
        sql = """
            INSERT INTO parametros_modelo (id_experimento, e_max, i_max, r_1, r_2)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id_experimento) DO UPDATE
              SET r_1 = EXCLUDED.r_1,
                  e_max = EXCLUDED.e_max,
                  i_max = EXCLUDED.i_max;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (
                id_exp,
                round(e_max, 6) if e_max else None,
                round(i_max, 6) if i_max else None,
                round(r1, 6),
                91.878   # r2 fijo del dataset
            ))
        conn.commit()
        print(f"  → parametros_modelo insertado (r1={r1:.4f})")
    except Exception as e:
        print(f"  ⚠ parametros_modelo: {e}")
        conn.rollback()

# ─────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────

def run(intervalo: int, temperatura: int | None):
    print("=" * 55)
    print("  Simulador MCFC — Digital Twin UdeC")
    print("=" * 55)
    print(f"  Intervalo : {intervalo} segundos por medición")
    print(f"  Temperatura fija: {temperatura if temperatura else 'aleatoria'}")
    print(f"  Conectando a PostgreSQL...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("  Conexión exitosa.\n")
    except Exception as e:
        print(f"  ERROR de conexión: {e}")
        sys.exit(1)

    # Manejo de Ctrl+C limpio
    def salir(sig, frame):
        print("\n\n  Simulador detenido por el usuario.")
        conn.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, salir)
    signal.signal(signal.SIGTERM, salir)

    experimento_num = 0

    while True:
        experimento_num += 1

        # Generar condiciones
        cond = generar_condiciones()
        if temperatura:
            cond["t"] = temperatura

        print(f"{'─'*55}")
        print(f"  Experimento simulado #{experimento_num}")
        print(f"  T={cond['t']}°C | H2a={cond['h2a']} | CO2a={cond['co2a']} | "
              f"O2c={cond['o2c']} | r1={cond['r1']}")

        # Crear experimento en BD
        try:
            id_exp = crear_experimento(conn, cond)
            # Calcular e_max e i_max estimados con el modelo
            V_ocv = voltaje_modelo(0.001, cond["t"], cond["r1"],
                                   cond["h2a"], cond["h2oa"], cond["co2a"],
                                   cond["o2c"], cond["co2c"])
            i_max_est = max(I_DENSIDADES)
            insertar_parametros_modelo(conn, id_exp, cond["r1"],
                                       e_max=V_ocv, i_max=i_max_est)
            print(f"  → Experimento creado con ID={id_exp}")
        except Exception as e:
            print(f"  ERROR al crear experimento: {e}")
            conn.rollback()
            time.sleep(intervalo)
            continue

        # Recorrer densidades de corriente
        for idx, i in enumerate(I_DENSIDADES):
            # Calcular voltaje con ruido pequeño realista
            V_modelo = voltaje_modelo(
                i, cond["t"], cond["r1"],
                cond["h2a"], cond["h2oa"], cond["co2a"],
                cond["o2c"], cond["co2c"]
            )
            ruido = random.gauss(0, 0.008)   # σ = 8 mV
            V = max(V_modelo + ruido, 0.0)
            eta = eta_eficiencia(V, i)

            try:
                insertar_medicion(conn, id_exp, i, V, eta)
                ts_str = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts_str}] Medición {idx+1:2d}/{len(I_DENSIDADES)} | "
                      f"i={i:.3f} A/cm² | V={V:.4f} V | η={eta:.4f}")
            except Exception as e:
                print(f"  ERROR al insertar medición: {e}")
                conn.rollback()

            time.sleep(intervalo)

        print(f"  ✓ Experimento #{experimento_num} completado "
              f"({len(I_DENSIDADES)} mediciones)\n")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador MCFC Digital Twin")
    parser.add_argument(
        "--intervalo", type=int, default=5,
        help="Segundos entre mediciones (default: 5 para prueba, usar 60 en producción)"
    )
    parser.add_argument(
        "--temperatura", type=int, default=None,
        choices=[550, 575, 600, 625, 650],
        help="Fijar temperatura en °C (default: aleatoria)"
    )
    args = parser.parse_args()
    run(args.intervalo, args.temperatura)
"""
simulador_mcfc.py
-----------------
Simula la operación continua de una celda MCFC a un punto de
operación fijo, generando mediciones de voltaje con variación
realista e insertándolas en PostgreSQL cada ciertos segundos.

Modos:
  - Modo CONTINUO (default): opera a una densidad de corriente
    fija definida por el usuario. Genera mediciones indefinidamente
    con pequeñas fluctuaciones realistas de voltaje.
  - Modo CURVA: recorre 21 puntos de densidad de corriente (0.005
    a 0.200 A/cm²) como el simulador original.

Uso:
  python simulador_mcfc.py [opciones]

Opciones clave:
  --corriente   FLOAT   Densidad de corriente fija [A/cm²] (modo continuo)
  --modo        curva|continuo  (default: continuo)
  --intervalo   INT     Segundos entre mediciones (default: 3)
  --temperatura INT     Temperatura en °C
  --h2a ... --r1        Parámetros de composición
"""

import argparse
import time
import random
import math
import signal
import sys
import os
from datetime import datetime, timezone

import psycopg2

# ── Credenciales desde config.py ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import DB_CONFIG

# ── Rangos operacionales (basados en dataset experimental) ────────────────────
RANGOS_OPERACION = {
    "t":          [550, 575, 600, 625, 650],
    "h2a":        (0.22, 4.41),
    "h2oa":       (0.05, 1.29),
    "n2a":        (0.0,  0.5),
    "co":         (0.0,  0.1),
    "ch4":        (0.0,  0.1),
    "co2a":       (0.06, 1.10),
    "o2c":        (0.13, 5.25),
    "n2c":        (0.49, 29.11),
    "co2c":       (0.27, 14.24),
    "h2oc":       (0.0,  0.5),
    "delta_nia":  (0.0,  0.1),
    "rho_a":      (0.0,  0.1),
    "delta_like": (0.0,  0.1),
    "delta_nioc": (0.0,  0.1),
    "rho_c":      (0.0,  0.1),
}

# ── r1 calibrado por temperatura (dataset experimental PGNN) ──────────────────
R1_POR_TEMP = {
    550: 2.9355,
    575: 2.6348,
    600: 2.3796,
    625: 2.1614,
    650: 1.9734,
}

# ── Bias residual por temperatura (V) ─────────────────────────────────────────
BIAS_POR_TEMP = {
    550: -0.038,
    575: -0.002,
    600: -0.004,
    625: +0.001,
    650: -0.023,
}

# ── Ruido calibrado por temperatura (std del residual) ────────────────────────
RUIDO_POR_TEMP = {
    550: 0.007,
    575: 0.019,
    600: 0.021,
    625: 0.024,
    650: 0.024,
}

# ── Puntos de densidad de corriente para modo curva ───────────────────────────
I_DENSIDADES = [
    0.005, 0.010, 0.020, 0.030, 0.040, 0.050,
    0.060, 0.070, 0.080, 0.090, 0.100, 0.110,
    0.120, 0.130, 0.140, 0.150, 0.160, 0.170,
    0.180, 0.190, 0.200,
]

R_GAS = 8.314
F_FAR = 96485
R2    = 91.878
DELTA = 0.012

# ── Modelo electroquímico (alineado con dashboard.py) ─────────────────────────

def e0_temperatura(T_K):
    # Fórmula alineada con dashboard.py y dataset PGNN
    return 1.2723 - 2.4516e-4 * T_K

def nernst(T_K, h2a, h2oa, co2a, o2c, co2c):
    eps   = 1e-10
    sa    = max(h2a + h2oa + co2a, eps)
    sc    = max(o2c + co2c, eps)
    xH2   = max(h2a  / sa, eps)
    xH2O  = max(h2oa / sa, eps)
    xCO2a = max(co2a / sa, eps)
    xO2   = max(o2c  / sc, eps)
    xCO2c = max(co2c / sc, eps)
    arg   = (xH2 * math.sqrt(xO2) * xCO2c) / (xH2O * xCO2a)
    return e0_temperatura(T_K) + (R_GAS * T_K / (2 * F_FAR)) * math.log(arg)

def voltaje_modelo(i, T_C, r1, h2a, h2oa, co2a, o2c, co2c):
    T_K     = T_C + 273.15
    E_N     = nernst(T_K, h2a, h2oa, co2a, o2c, co2c)
    eta_ohm = r1 * i
    eta_act = (R_GAS * T_K / (2 * F_FAR)) * math.log(1 + R2 * i)
    return max(E_N - eta_ohm - eta_act - DELTA, 0.0)

def eta_eficiencia(V, i):
    lhv_h2 = 241800
    n_dot  = i / (2 * F_FAR)
    P      = V * i
    if n_dot * lhv_h2 > 0:
        return min(P / (n_dot * lhv_h2), 1.0)
    return 0.0

# ── Generador de condiciones ───────────────────────────────────────────────────

def generar_condiciones(args):
    def rnd(rango): return round(random.uniform(*rango), 4)
    T  = args.temperatura if args.temperatura else random.choice(RANGOS_OPERACION["t"])
    r1 = args.r1 if args.r1 is not None else R1_POR_TEMP.get(T, 1.9734)
    return {
        "t":          T,
        "h2a":        args.h2a   if args.h2a   is not None else rnd(RANGOS_OPERACION["h2a"]),
        "h2oa":       args.h2oa  if args.h2oa  is not None else rnd(RANGOS_OPERACION["h2oa"]),
        "co2a":       args.co2a  if args.co2a  is not None else rnd(RANGOS_OPERACION["co2a"]),
        "o2c":        args.o2c   if args.o2c   is not None else rnd(RANGOS_OPERACION["o2c"]),
        "co2c":       args.co2c  if args.co2c  is not None else rnd(RANGOS_OPERACION["co2c"]),
        "n2c":        args.n2c   if args.n2c   is not None else rnd(RANGOS_OPERACION["n2c"]),
        "r1":         r1,
        "n2a":        rnd(RANGOS_OPERACION["n2a"]),
        "co":         rnd(RANGOS_OPERACION["co"]),
        "ch4":        rnd(RANGOS_OPERACION["ch4"]),
        "h2oc":       rnd(RANGOS_OPERACION["h2oc"]),
        "delta_nia":  rnd(RANGOS_OPERACION["delta_nia"]),
        "rho_a":      rnd(RANGOS_OPERACION["rho_a"]),
        "delta_like": rnd(RANGOS_OPERACION["delta_like"]),
        "delta_nioc": rnd(RANGOS_OPERACION["delta_nioc"]),
        "rho_c":      rnd(RANGOS_OPERACION["rho_c"]),
    }

# ── Base de datos ──────────────────────────────────────────────────────────────

def crear_experimento(conn, cond):
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
        ) RETURNING id_experimento;
    """
    with conn.cursor() as cur:
        cur.execute(sql, cond)
        id_exp = cur.fetchone()[0]
    conn.commit()
    return id_exp

def insertar_medicion(conn, id_exp, i, V, eta):
    sql = """
        INSERT INTO mediciones (id_experimento, i_densidad, voltaje, eta, timestamp_medicion)
        VALUES (%s, %s, %s, %s, %s);
    """
    ts = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(sql, (id_exp, round(i, 6), round(V, 6), round(eta, 6), ts))
    conn.commit()

def insertar_parametros_modelo(conn, id_exp, r1, e_max=None, i_max=None):
    try:
        sql = """
            INSERT INTO parametros_modelo (id_experimento, e_max, i_max, r_1, r_2)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id_experimento) DO UPDATE
              SET r_1=EXCLUDED.r_1, e_max=EXCLUDED.e_max, i_max=EXCLUDED.i_max;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (id_exp,
                              round(e_max, 6) if e_max else None,
                              round(i_max, 6) if i_max else None,
                              round(r1, 6), 91.878))
        conn.commit()
        print(f"  → parametros_modelo insertado (r1={r1:.4f})")
    except Exception as e:
        print(f"  ⚠ parametros_modelo: {e}")
        conn.rollback()

# ── Modo continuo ──────────────────────────────────────────────────────────────

def run_continuo(conn, args, cond, id_exp):
    i_fija = args.corriente
    T      = cond["t"]
    bias   = BIAS_POR_TEMP.get(T, -0.010)
    sigma  = RUIDO_POR_TEMP.get(T, 0.020)

    V_base = voltaje_modelo(
        i_fija, T, cond["r1"],
        cond["h2a"], cond["h2oa"], cond["co2a"],
        cond["o2c"], cond["co2c"]
    )
    V_cal = V_base + bias

    print(f"  Modo CONTINUO — i={i_fija:.3f} A/cm² | T={T}°C")
    print(f"  V_modelo={V_base:.4f} V | bias={bias*1000:+.0f} mV | σ={sigma*1000:.0f} mV")
    print(f"  V_calibrado={V_cal:.4f} V")
    print(f"  Generando mediciones cada {args.intervalo} segundos...\n")

    drift = 0.0
    medicion_num = 0

    while True:
        medicion_num += 1
        ruido  = random.gauss(0, sigma)
        drift += random.gauss(0, sigma * 0.15)
        drift  = max(min(drift, 0.030), -0.030)

        V   = max(V_cal + ruido + drift, 0.0)
        eta = eta_eficiencia(V, i_fija)

        try:
            insertar_medicion(conn, id_exp, i_fija, V, eta)
            ts_str = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts_str}] Med {medicion_num:4d} | "
                  f"i={i_fija:.3f} A/cm² | V={V:.4f} V | η={eta:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            conn.rollback()

        time.sleep(args.intervalo)

# ── Modo curva ─────────────────────────────────────────────────────────────────

def run_curva(conn, args, cond, id_exp):
    T     = cond["t"]
    bias  = BIAS_POR_TEMP.get(T, -0.010)
    sigma = RUIDO_POR_TEMP.get(T, 0.020)

    print(f"  Modo CURVA — {len(I_DENSIDADES)} puntos | T={T}°C | "
          f"bias={bias*1000:+.0f} mV | σ={sigma*1000:.0f} mV\n")

    for idx, i in enumerate(I_DENSIDADES):
        V_mod = voltaje_modelo(
            i, T, cond["r1"],
            cond["h2a"], cond["h2oa"], cond["co2a"],
            cond["o2c"], cond["co2c"]
        )
        ruido = random.gauss(0, sigma)
        V     = max(V_mod + bias + ruido, 0.0)
        eta   = eta_eficiencia(V, i)

        try:
            insertar_medicion(conn, id_exp, i, V, eta)
            ts_str = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts_str}] Medición {idx+1:2d}/{len(I_DENSIDADES)} | "
                  f"i={i:.3f} A/cm² | V={V:.4f} V | η={eta:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            conn.rollback()

        time.sleep(args.intervalo)

# ── Loop principal ─────────────────────────────────────────────────────────────

def run(args):
    print("=" * 55)
    print("  Simulador MCFC — Digital Twin UdeC")
    print("=" * 55)
    print(f"  Modo       : {args.modo.upper()}")
    print(f"  Intervalo  : {args.intervalo} s/medición")
    if args.modo == 'continuo':
        print(f"  Corriente  : {args.corriente:.3f} A/cm²")
    print(f"  Temperatura: {args.temperatura if args.temperatura else 'aleatoria'}")
    print(f"  Conectando a PostgreSQL...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("  Conexión exitosa.\n")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    def salir(sig, frame):
        print("\n  Simulador detenido.")
        conn.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, salir)
    signal.signal(signal.SIGTERM, salir)

    ciclos_max = args.ciclos if args.ciclos else float('inf')
    ciclo = 0

    while ciclo < ciclos_max:
        ciclo += 1
        cond = generar_condiciones(args)

        print(f"{'─'*55}")
        print(f"  Experimento #{ciclo} | T={cond['t']}°C | "
              f"H2a={cond['h2a']} | r1={cond['r1']}")

        try:
            id_exp = crear_experimento(conn, cond)
            V_ocv  = voltaje_modelo(0.001, cond["t"], cond["r1"],
                                    cond["h2a"], cond["h2oa"], cond["co2a"],
                                    cond["o2c"], cond["co2c"])
            i_ref  = args.corriente if args.modo == 'continuo' else max(I_DENSIDADES)
            insertar_parametros_modelo(conn, id_exp, cond["r1"],
                                       e_max=V_ocv, i_max=i_ref)
            print(f"  → Experimento creado ID={id_exp}")
        except Exception as e:
            print(f"  ERROR: {e}")
            conn.rollback()
            time.sleep(args.intervalo)
            continue

        if args.modo == 'continuo':
            run_continuo(conn, args, cond, id_exp)
        else:
            run_curva(conn, args, cond, id_exp)
            print(f"  ✓ Ciclo #{ciclo} completado\n")

    conn.close()
    print("  Simulacion finalizada.")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador MCFC Digital Twin")

    parser.add_argument("--modo",        default="continuo",
                        choices=["continuo", "curva"])
    parser.add_argument("--intervalo",   type=int,   default=3)
    parser.add_argument("--corriente",   type=float, default=0.10)
    parser.add_argument("--temperatura", type=int,   default=None,
                        choices=[550, 575, 600, 625, 650])
    parser.add_argument("--h2a",  type=float, default=None)
    parser.add_argument("--h2oa", type=float, default=None)
    parser.add_argument("--co2a", type=float, default=None)
    parser.add_argument("--o2c",  type=float, default=None)
    parser.add_argument("--co2c", type=float, default=None)
    parser.add_argument("--n2c",  type=float, default=None)
    parser.add_argument("--r1",   type=float, default=None)
    parser.add_argument("--ciclos", type=int, default=None)

    args = parser.parse_args()
    run(args)
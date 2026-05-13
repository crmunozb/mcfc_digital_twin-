"""
reproductor_mcfc.py
-------------------
Reproduce secuencialmente las mediciones reales de un experimento
de la plataforma experimental de Warsaw University of Technology
(Prof. Milewski / Prof. Escalona), insertándolas en PostgreSQL con
timestamps actuales para simular una ingesta en tiempo real.

Esto permite validar la arquitectura de adquisición del Digital Twin
con datos experimentales reales de una celda MCFC equivalente a la
celda piloto de la Universidad de Concepción.

Uso:
    python3 reproductor_mcfc.py [opciones]

Opciones:
    --exp_id    INT   ID del experimento de Milewski a reproducir
                      (si no se indica, selecciona el más cercano a los parámetros)
    --temperatura INT Temperatura para selección automática de experimento
    --intervalo INT   Segundos entre mediciones (default: 3)
    --modo      curva|continuo
    --corriente FLOAT Densidad de corriente para modo continuo
"""

import argparse
import time
import signal
import sys
import os
from datetime import datetime, timezone

import psycopg2
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import DB_CONFIG


# ── Selección de experimento real ──────────────────────────────────────────────

def seleccionar_experimento(conn, temperatura=None, exp_id=None):
    """Selecciona un experimento real de warsaw_ut con al menos 7 mediciones."""
    if exp_id:
        df = pd.read_sql("""
            SELECT e.id_experimento, e.t, e.h2a, e.h2oa, e.co2a,
                   e.o2c, e.co2c, e.n2c, e.n2a, e.co, e.ch4, e.h2oc,
                   p.r_1, p.e_max, p.i_max
            FROM experimentos e
            JOIN parametros_modelo p USING(id_experimento)
            WHERE e.fuente = 'warsaw_ut' AND e.id_experimento = %s
        """, conn, params=(exp_id,))
    elif temperatura:
        df = pd.read_sql("""
            SELECT e.id_experimento, e.t, e.h2a, e.h2oa, e.co2a,
                   e.o2c, e.co2c, e.n2c, e.n2a, e.co, e.ch4, e.h2oc,
                   p.r_1, p.e_max, p.i_max
            FROM experimentos e
            JOIN parametros_modelo p USING(id_experimento)
            WHERE e.fuente = 'warsaw_ut' AND e.t = %s
              AND (SELECT COUNT(*) FROM mediciones m
                   WHERE m.id_experimento = e.id_experimento) >= 7
            ORDER BY RANDOM() LIMIT 1
        """, conn, params=(temperatura,))
    else:
        df = pd.read_sql("""
            SELECT e.id_experimento, e.t, e.h2a, e.h2oa, e.co2a,
                   e.o2c, e.co2c, e.n2c, e.n2a, e.co, e.ch4, e.h2oc,
                   p.r_1, p.e_max, p.i_max
            FROM experimentos e
            JOIN parametros_modelo p USING(id_experimento)
            WHERE e.fuente = 'warsaw_ut'
              AND (SELECT COUNT(*) FROM mediciones m
                   WHERE m.id_experimento = e.id_experimento) >= 7
            ORDER BY RANDOM() LIMIT 1
        """, conn)

    if df.empty:
        return None
    return df.iloc[0]


def cargar_mediciones(conn, id_experimento):
    """Carga las mediciones reales del experimento seleccionado."""
    return pd.read_sql("""
        SELECT i_densidad, voltaje, eta
        FROM mediciones
        WHERE id_experimento = %s
        ORDER BY i_densidad ASC
    """, conn, params=(id_experimento,))


# ── Crear experimento udec_lab como copia del real ─────────────────────────────

def crear_experimento_reproduccion(conn, exp_real):
    """Crea un nuevo experimento warsaw_ut para la reproducción."""
    sql = """
        INSERT INTO experimentos (
            fuente, t, h2a, h2oa, n2a, co, ch4, co2a,
            o2c, n2c, co2c, h2oc
        ) VALUES (
            'warsaw_ut',
            %(t)s, %(h2a)s, %(h2oa)s, %(n2a)s, %(co)s, %(ch4)s, %(co2a)s,
            %(o2c)s, %(n2c)s, %(co2c)s, %(h2oc)s
        ) RETURNING id_experimento;
    """
    params = {
        't':    float(exp_real['t']),
        'h2a':  float(exp_real['h2a']),
        'h2oa': float(exp_real['h2oa']),
        'n2a':  float(exp_real.get('n2a', 0.0)),
        'co':   float(exp_real.get('co',  0.0)),
        'ch4':  float(exp_real.get('ch4', 0.0)),
        'co2a': float(exp_real['co2a']),
        'o2c':  float(exp_real['o2c']),
        'n2c':  float(exp_real['n2c']),
        'co2c': float(exp_real['co2c']),
        'h2oc': float(exp_real.get('h2oc', 0.0)),
    }
    with conn.cursor() as cur:
        cur.execute(sql, params)
        id_exp = cur.fetchone()[0]

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO parametros_modelo (id_experimento, e_max, i_max, r_1, r_2)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id_experimento) DO UPDATE
              SET r_1=EXCLUDED.r_1, e_max=EXCLUDED.e_max, i_max=EXCLUDED.i_max;
        """, (
            id_exp,
            float(exp_real['e_max']) if exp_real['e_max'] else None,
            float(exp_real['i_max']) if exp_real['i_max'] else None,
            float(exp_real['r_1']),
            91.878
        ))
    conn.commit()
    return id_exp


def insertar_medicion(conn, id_exp, i, V, eta):
    ts = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO mediciones (id_experimento, i_densidad, voltaje, eta, timestamp_medicion)
            VALUES (%s, %s, %s, %s, %s)
        """, (id_exp, round(float(i), 6), round(float(V), 6), round(float(eta), 6), ts))
    conn.commit()


# ── Modos de reproducción ──────────────────────────────────────────────────────

def run_curva(conn, args, exp_real, mediciones, id_exp):
    """Reproduce todas las mediciones del experimento en orden."""
    print(f"  Modo CURVA — {len(mediciones)} puntos reales | T={exp_real['t']}°C")
    print(f"  Reproduciendo experimento original ID={exp_real['id_experimento']}\n")

    for idx, row in mediciones.iterrows():
        try:
            insertar_medicion(conn, id_exp, row['i_densidad'], row['voltaje'], row['eta'])
            ts_str = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts_str}] Punto {idx+1:2d}/{len(mediciones)} | "
                  f"i={row['i_densidad']:.3f} A/cm² | "
                  f"V={row['voltaje']:.4f} V | η={row['eta']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            conn.rollback()

        time.sleep(args.intervalo)

    print(f"\n  Reproduccion completada.")


def run_continuo(conn, args, exp_real, mediciones, id_exp):
    """Reproduce el punto de corriente más cercano al solicitado, en bucle."""
    i_target = args.corriente
    # Seleccionar el punto más cercano a la corriente solicitada
    idx_closest = (mediciones['i_densidad'] - i_target).abs().idxmin()
    row = mediciones.loc[idx_closest]

    print(f"  Modo CONTINUO — punto i={row['i_densidad']:.3f} A/cm² | T={exp_real['t']}°C")
    print(f"  Reproduciendo datos reales del exp ID={exp_real['id_experimento']}\n")

    medicion_num = 0
    while True:
        medicion_num += 1
        try:
            insertar_medicion(conn, id_exp, row['i_densidad'], row['voltaje'], row['eta'])
            ts_str = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts_str}] Med {medicion_num:4d} | "
                  f"i={row['i_densidad']:.3f} A/cm² | "
                  f"V={row['voltaje']:.4f} V | η={row['eta']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            conn.rollback()

        time.sleep(args.intervalo)


# ── Loop principal ─────────────────────────────────────────────────────────────

def run(args):
    print("=" * 55)
    print("  Reproductor MCFC — Digital Twin UdeC")
    print("  Datos: Warsaw University of Technology")
    print("=" * 55)
    print(f"  Modo      : {args.modo.upper()}")
    print(f"  Intervalo : {args.intervalo} s/medición")
    print(f"  Conectando a PostgreSQL...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("  Conexión exitosa.\n")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    def salir(sig, frame):
        print("\n  Reproductor detenido.")
        conn.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, salir)
    signal.signal(signal.SIGTERM, salir)

    # Seleccionar experimento real
    exp_real = seleccionar_experimento(
        conn,
        temperatura=args.temperatura,
        exp_id=args.exp_id
    )

    if exp_real is None:
        print("  ERROR: No se encontró experimento real con esos parámetros.")
        conn.close()
        sys.exit(1)

    print(f"  Experimento seleccionado: ID={exp_real['id_experimento']} | "
          f"T={exp_real['t']}°C | r1={exp_real['r_1']:.4f}")

    # Cargar mediciones reales
    mediciones = cargar_mediciones(conn, exp_real['id_experimento'])
    print(f"  Mediciones disponibles: {len(mediciones)}")

    if mediciones.empty:
        print("  ERROR: El experimento no tiene mediciones.")
        conn.close()
        sys.exit(1)

    # Crear nuevo experimento para esta reproducción
    id_exp = crear_experimento_reproduccion(conn, exp_real)
    id_original = int(exp_real['id_experimento'])

    # Escribir ID original en archivo para que el dashboard lo lea
    try:
        with open('/tmp/mcfc_exp_original.txt', 'w') as f:
            f.write(str(id_original))
    except Exception:
        pass

    print(f"  EXP_ORIGINAL={id_original}")
    print(f"  Nuevo experimento creado: ID={id_exp} (basado en Exp {id_original})\n")

    if args.modo == 'curva':
        run_curva(conn, args, exp_real, mediciones, id_exp)
    else:
        run_continuo(conn, args, exp_real, mediciones, id_exp)

    conn.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproductor MCFC — datos reales")

    parser.add_argument("--exp_id",     type=int,   default=None,
                        help="ID del experimento de Milewski a reproducir")
    parser.add_argument("--temperatura", type=int,  default=None,
                        choices=[550, 575, 600, 625, 650])
    parser.add_argument("--modo",       default="curva",
                        choices=["continuo", "curva"])
    parser.add_argument("--intervalo",  type=int,   default=3)
    parser.add_argument("--corriente",  type=float, default=0.10)
    parser.add_argument("--ciclos",     type=int,   default=None)

    args = parser.parse_args()
    run(args)
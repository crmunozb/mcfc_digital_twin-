"""
optimizador_robusto.py
----------------------
Optimizador operacional robusto para la celda MCFC del Digital Twin.

Dado un experimento real de la base de datos (por ID), calcula para
cada modelo disponible (Nernst, PLS, KPLS, GPR, GPR Residual):

    1. Curva de densidad de potencia p(j) = V(j) × j
    2. Punto óptimo puntual:
           j* = argmax p(j)     dentro de [J_MIN, J_MAX]
    3. Región óptima de operación:
           [j_low, j_high] = rango donde p(j) >= umbral × p(j*)
           (por defecto umbral = 0.95, es decir 95% del óptimo)
    4. Para modelos con incertidumbre (GPR y GPR Residual):
           p_lower(j) = (μ(j) - 2σ(j)) × j   ← peor caso estadístico
           p_garantizada = max p_lower(j) para j en región óptima
           Interpreta: potencia mínima garantizada con 95% de confianza

El script reporta una tabla comparativa de resultados y genera un
gráfico con las curvas p(j), las bandas de incertidumbre ±2σ y la
región óptima sombreada.

Uso:
    python3 optimizador_robusto.py --exp_id 9
    python3 optimizador_robusto.py --exp_id 9 --umbral 0.90
    python3 optimizador_robusto.py --exp_id 9 --guardar_grafico
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
import psycopg2

from scipy.optimize import minimize_scalar

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from config import DB_CONFIG

RUTAS_MODELOS = {
    'PLS':          os.path.join(BASE_DIR, 'modelos', 'pls_voltaje_cv.pkl'),
    'KPLS':         os.path.join(BASE_DIR, 'modelos', 'kpls_voltaje_cv.pkl'),
    'GPR':          os.path.join(BASE_DIR, 'modelos', 'gpr_voltaje.pkl'),
    'GPR Residual': os.path.join(BASE_DIR, 'modelos', 'gpr_residual.pkl'),
}

# ── Constantes ─────────────────────────────────────────────────────────────────
J_MIN      = 0.005    # A/cm² — límite inferior del rango experimental
J_MAX      = 0.200    # A/cm² — límite superior del rango experimental
N_PUNTOS   = 500      # resolución de la grilla j para optimización
R_GAS      = 8.314
F_FAR      = 96485.0
FEATURES   = ['T', 'H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c', 'i, A/cm²', 'r_1']


# ── Modelo de Nernst ───────────────────────────────────────────────────────────
def nernst_voltaje(j, T_C, h2a, h2oa, co2a, o2c, co2c, n2c, r1,
                   r2=91.878, delta=0.012):
    """Calcula V(j) con el modelo semi-empírico de Nernst con pérdidas."""
    j   = np.asarray(j, dtype=float)
    T_K = T_C + 273.15
    eps = 1e-10
    sa  = max(h2a + h2oa + co2a, eps)
    sc  = max(o2c + co2c + n2c,  eps)
    xH2   = max(h2a  / sa, eps)
    xH2O  = max(h2oa / sa, eps)
    xCO2a = max(co2a / sa, eps)
    xO2   = max(o2c  / sc, eps)
    xCO2c = max(co2c / sc, eps)
    E0    = 1.2723 - 2.4516e-4 * T_K
    EN    = E0 + (R_GAS * T_K) / (2 * F_FAR) * np.log(
        xH2 * np.sqrt(xO2) * xCO2c / (xH2O * xCO2a + eps))
    V = EN - r1 * j - (R_GAS * T_K) / (2 * F_FAR) * np.log(1 + r2 * j) - delta
    return np.maximum(V, 0.0)


# ── Cargar experimento desde PostgreSQL ───────────────────────────────────────
def listar_experimentos():
    """Muestra una tabla de experimentos disponibles en la BD."""
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT e.id_experimento, e.t, e.h2a, e.co2a, e.o2c, e.co2c,
               COUNT(m.id_medicion) AS n_mediciones
        FROM experimentos e
        LEFT JOIN mediciones m USING(id_experimento)
        WHERE e.fuente = 'warsaw_ut'
        GROUP BY e.id_experimento, e.t, e.h2a, e.co2a, e.o2c, e.co2c
        ORDER BY e.id_experimento ASC
    """, conn)
    conn.close()
    return df


def seleccionar_experimento_interactivo():
    """Muestra la lista de experimentos y pide al usuario que elija uno."""
    df = listar_experimentos()

    print("=" * 65)
    print("  Experimentos disponibles en la base de datos")
    print("=" * 65)
    print(f"{'ID':>4} | {'T (°C)':>6} | {'H2a':>6} | {'CO2a':>6} | "
          f"{'O2c':>6} | {'CO2c':>6} | {'n med':>6}")
    print("-" * 55)
    for _, row in df.iterrows():
        print(f"{int(row['id_experimento']):>4} | "
              f"{int(row['t']):>6} | "
              f"{row['h2a']:>6.3f} | "
              f"{row['co2a']:>6.3f} | "
              f"{row['o2c']:>6.3f} | "
              f"{row['co2c']:>6.3f} | "
              f"{int(row['n_mediciones']):>6}")
    print("-" * 55)
    print(f"  Total: {len(df)} experimentos\n")

    ids_validos = df['id_experimento'].tolist()
    while True:
        try:
            entrada = input("  Ingresa el ID del experimento a optimizar: ")
            exp_id = int(entrada.strip())
            if exp_id in ids_validos:
                return exp_id
            else:
                print(f"  ID {exp_id} no existe. Elige un ID de la lista.")
        except ValueError:
            print("  Ingresa un número entero válido.")
        except KeyboardInterrupt:
            print("\n  Cancelado.")
            sys.exit(0)


def cargar_experimento(exp_id):
    """Lee las condiciones operacionales del experimento desde la BD."""
    conn = psycopg2.connect(**DB_CONFIG)
    df_exp = pd.read_sql("""
        SELECT e.id_experimento, e.t, e.h2a, e.h2oa, e.co2a,
               e.o2c, e.co2c, e.n2c, p.r_1
        FROM experimentos e
        JOIN parametros_modelo p USING(id_experimento)
        WHERE e.id_experimento = %s
    """, conn, params=(exp_id,))
    df_med = pd.read_sql("""
        SELECT i_densidad, voltaje
        FROM mediciones
        WHERE id_experimento = %s
        ORDER BY i_densidad ASC
    """, conn, params=(exp_id,))
    conn.close()

    if df_exp.empty:
        raise ValueError(f"Experimento {exp_id} no encontrado en la BD.")

    row = df_exp.iloc[0]
    condiciones = {
        'T':    float(row['t']),
        'H2a':  float(row['h2a']),
        'H2Oa': float(row['h2oa']),
        'CO2a': float(row['co2a']),
        'O2c':  float(row['o2c']),
        'CO2c': float(row['co2c']),
        'N2c':  float(row['n2c']),
        'r_1':  float(row['r_1']),
    }
    return condiciones, df_med
    """Muestra la lista de experimentos y pide al usuario que elija uno."""
    df = listar_experimentos()

    print("=" * 65)
    print("  Experimentos disponibles en la base de datos")
    print("=" * 65)
    print(f"{'ID':>4} | {'T (°C)':>6} | {'H2a':>6} | {'CO2a':>6} | "
          f"{'O2c':>6} | {'CO2c':>6} | {'n med':>6}")
    print("-" * 55)
    for _, row in df.iterrows():
        print(f"{int(row['id_experimento']):>4} | "
              f"{int(row['t']):>6} | "
              f"{row['h2a']:>6.3f} | "
              f"{row['co2a']:>6.3f} | "
              f"{row['o2c']:>6.3f} | "
              f"{row['co2c']:>6.3f} | "
              f"{int(row['n_mediciones']):>6}")
    print("-" * 55)
    print(f"  Total: {len(df)} experimentos\n")

    ids_validos = df['id_experimento'].tolist()
    while True:
        try:
            entrada = input("  Ingresa el ID del experimento a optimizar: ")
            exp_id = int(entrada.strip())
            if exp_id in ids_validos:
                return exp_id
            else:
                print(f"  ID {exp_id} no existe. Elige un ID de la lista.")
        except ValueError:
            print("  Ingresa un número entero válido.")
        except KeyboardInterrupt:
            print("\n  Cancelado.")
            sys.exit(0)



# ── Predicciones con incertidumbre ────────────────────────────────────────────
def predecir_con_incertidumbre(nombre, modelo_pkl, condiciones, j_arr):
    """
    Retorna (mu, sigma) para el arreglo j_arr dado las condiciones.
    sigma=None para modelos sin incertidumbre (PLS, KPLS, Nernst).
    """
    T    = condiciones['T']
    H2a  = condiciones['H2a']
    H2Oa = condiciones['H2Oa']
    CO2a = condiciones['CO2a']
    O2c  = condiciones['O2c']
    CO2c = condiciones['CO2c']
    N2c  = condiciones['N2c']
    r1   = condiciones['r_1']

    if nombre == 'Nernst':
        mu = nernst_voltaje(j_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1)
        return mu, None

    d = joblib.load(modelo_pkl)

    # Construir matriz de features para todos los puntos j
    X = np.column_stack([
        np.full(len(j_arr), T),
        np.full(len(j_arr), H2a),
        np.full(len(j_arr), H2Oa),
        np.full(len(j_arr), CO2a),
        np.full(len(j_arr), O2c),
        np.full(len(j_arr), CO2c),
        np.full(len(j_arr), N2c),
        j_arr,
        np.full(len(j_arr), r1),
    ])

    if nombre in ('PLS', 'KPLS'):
        mu = d['modelo'].predict(X).ravel()
        return mu, None

    if nombre == 'GPR':
        X_sc = d['scaler_X'].transform(X)
        mu_sc, sigma_sc = d['modelo'].predict(X_sc, return_std=True)
        sigma_scale = d['scaler_y'].scale_[0]
        mu    = d['scaler_y'].inverse_transform(
            mu_sc.reshape(-1, 1)).ravel()
        sigma = sigma_sc * sigma_scale
        return mu, sigma

    if nombre == 'GPR Residual':
        # V_Nernst + GPR(ε)
        V_nernst = nernst_voltaje(
            j_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1)
        X_sc = d['scaler_X'].transform(X)
        e_sc, sigma_sc = d['modelo'].predict(X_sc, return_std=True)
        sigma_scale = d['scaler_e'].scale_[0]
        epsilon = d['scaler_e'].inverse_transform(
            e_sc.reshape(-1, 1)).ravel()
        sigma   = sigma_sc * sigma_scale
        mu      = V_nernst + epsilon
        return mu, sigma

    raise ValueError(f"Modelo desconocido: {nombre}")


# ── Optimización robusta ───────────────────────────────────────────────────────
def optimizar_robusto(mu, sigma, j_arr, umbral=0.95):
    """
    Calcula el óptimo puntual, la región óptima y la potencia garantizada.

    Parámetros
    ----------
    mu     : ndarray — voltaje medio predicho
    sigma  : ndarray o None — desviación estándar (None si no aplica)
    j_arr  : ndarray — grilla de densidades de corriente
    umbral : float — fracción del óptimo para definir región (default 0.95)

    Retorna
    -------
    dict con j_star, p_star, j_low, j_high, p_garantizada, p_lower
    """
    p_media = mu * j_arr

    # Óptimo puntual
    idx_star  = int(np.argmax(p_media))
    j_star    = float(j_arr[idx_star])
    p_star    = float(p_media[idx_star])
    V_star    = float(mu[idx_star])

    # Región óptima: rango donde p_media >= umbral × p_star
    threshold = umbral * p_star
    en_region = p_media >= threshold
    idx_region = np.where(en_region)[0]

    if len(idx_region) > 0:
        j_low  = float(j_arr[idx_region[0]])
        j_high = float(j_arr[idx_region[-1]])
    else:
        j_low  = j_star
        j_high = j_star

    # Potencia garantizada (solo si hay incertidumbre)
    p_lower = None
    p_garantizada = None
    if sigma is not None:
        p_lower = np.maximum(mu - 2 * sigma, 0.0) * j_arr
        # Potencia garantizada en la región óptima
        mask_region = (j_arr >= j_low) & (j_arr <= j_high)
        if mask_region.any():
            p_garantizada = float(p_lower[mask_region].max())

    return {
        'j_star':         j_star,
        'p_star':         p_star,
        'V_star':         V_star,
        'j_low':          j_low,
        'j_high':         j_high,
        'amplitud_region': j_high - j_low,
        'p_garantizada':  p_garantizada,
        'p_lower':        p_lower,
        'p_media':        p_media,
        'en_limite':      abs(j_star - J_MAX) < 1e-4,
    }


# ── Programa principal ─────────────────────────────────────────────────────────
def run(args):
    print("=" * 65)
    print("  Optimizador Operacional MCFC — Digital Twin UdeC")
    print("=" * 65)
    print(f"  Experimento : {args.exp_id}")
    print(f"  Umbral      : {args.umbral * 100:.0f}% del óptimo")
    print(f"  Rango j     : [{J_MIN:.3f}, {J_MAX:.3f}] A/cm²")
    print()

    # Cargar condiciones desde la BD
    condiciones, df_med = cargar_experimento(args.exp_id)
    print("Condiciones operacionales del experimento:")
    for k, v in condiciones.items():
        print(f"  {k:<6} = {v:.4f}")
    print(f"  Mediciones reales disponibles: {len(df_med)}")
    print()

    # Grilla de j para optimización
    j_arr = np.linspace(J_MIN, J_MAX, N_PUNTOS)

    # Modelos a evaluar
    modelos_eval = [('Nernst', None)] + [
        (n, r) for n, r in RUTAS_MODELOS.items()
        if os.path.exists(r)
    ]

    resultados = []
    datos_curvas = {}  # para el gráfico

    for nombre, ruta in modelos_eval:
        try:
            mu, sigma = predecir_con_incertidumbre(
                nombre, ruta, condiciones, j_arr)
            res = optimizar_robusto(mu, sigma, j_arr, umbral=args.umbral)
            res['nombre'] = nombre
            res['mu']     = mu
            res['sigma']  = sigma
            resultados.append(res)
            datos_curvas[nombre] = (mu, sigma, res)
        except Exception as e:
            print(f"  [{nombre}] Error: {e}")

    # ── Tabla de resultados ────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  RESULTADOS — Umbral {args.umbral*100:.0f}% del óptimo")
    print("=" * 65)
    print(f"{'Modelo':<14} | {'j* (A/cm²)':>10} | {'V* (V)':>8} | "
          f"{'Pmax (W/cm²)':>12} | {'Región óptima':>16} | "
          f"{'P_garantizada':>14}")
    print("-" * 85)

    for r in resultados:
        region = f"[{r['j_low']:.3f}, {r['j_high']:.3f}]"
        p_gar  = f"{r['p_garantizada']:.4f}" \
                 if r['p_garantizada'] is not None else "   N/A   "
        limite = " ⚠" if r['en_limite'] else ""
        print(f"{r['nombre']:<14} | {r['j_star']:>10.4f} | "
              f"{r['V_star']:>8.4f} | {r['p_star']:>12.5f} | "
              f"{region:>16} | {p_gar:>14}{limite}")

    print("-" * 85)
    print("  ⚠ j* en límite: Pmax real podría estar fuera del rango validado")
    print()

    # ── Interpretación de la región óptima ────────────────────────────────────
    print("=" * 65)
    print("  INTERPRETACIÓN — Región óptima de operación")
    print("=" * 65)
    for r in resultados:
        if r['sigma'] is not None:
            print(f"\n  {r['nombre']}:")
            print(f"    Operar en j ∈ [{r['j_low']:.3f}, {r['j_high']:.3f}] A/cm²")
            print(f"    garantiza ≥ {args.umbral*100:.0f}% de la potencia óptima "
                  f"({r['p_star']:.5f} W/cm²)")
            print(f"    Potencia mínima garantizada (peor caso ±2σ): "
                  f"{r['p_garantizada']:.5f} W/cm²")
            print(f"    Amplitud de la región: {r['amplitud_region']*1000:.1f} mA/cm²")

    # ── Gráfico ────────────────────────────────────────────────────────────────
    if args.guardar_grafico or args.mostrar_grafico:
        try:
            import matplotlib
            if not args.mostrar_grafico:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            COLORES = {
                'Nernst':       '#e74c3c',
                'PLS':          '#2980b9',
                'KPLS':         '#e67e22',
                'GPR':          '#8e44ad',
                'GPR Residual': '#27ae60',
            }

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(
                f"Optimizador Operacional MCFC — Exp {args.exp_id} "
                f"(T={condiciones['T']:.0f}°C)",
                fontsize=13, fontweight='bold'
            )

            ax_V, ax_P = axes

            # Panel izquierdo: curvas E(j)
            ax_V.set_title("Curvas de polarización E(j)")
            ax_V.set_xlabel("Densidad de corriente j [A/cm²]")
            ax_V.set_ylabel("Voltaje E [V]")

            for nombre, (mu, sigma, res) in datos_curvas.items():
                color = COLORES.get(nombre, '#333')
                ax_V.plot(j_arr, mu, color=color, lw=2, label=nombre)
                if sigma is not None:
                    ax_V.fill_between(
                        j_arr, mu - 2*sigma, mu + 2*sigma,
                        color=color, alpha=0.15,
                        label=f'{nombre} ±2σ')
                ax_V.axvline(res['j_star'], color=color,
                             lw=1, ls='--', alpha=0.6)

            # Datos reales
            if not df_med.empty:
                ax_V.scatter(df_med['i_densidad'], df_med['voltaje'],
                             color='#27ae60', s=40, zorder=5,
                             label='Datos reales (Milewski)')

            ax_V.legend(fontsize=7, loc='upper right')
            ax_V.grid(True, alpha=0.3)

            # Panel derecho: curvas p(j) con región óptima
            ax_P.set_title(
                f"Densidad de potencia p(j) — "
                f"región óptima ≥{args.umbral*100:.0f}% Pmax")
            ax_P.set_xlabel("Densidad de corriente j [A/cm²]")
            ax_P.set_ylabel("Densidad de potencia p [W/cm²]")

            for nombre, (mu, sigma, res) in datos_curvas.items():
                color  = COLORES.get(nombre, '#333')
                p_med  = res['p_media']
                ax_P.plot(j_arr, p_med, color=color, lw=2, label=nombre)

                # Banda de incertidumbre en p(j)
                if sigma is not None and res['p_lower'] is not None:
                    ax_P.fill_between(
                        j_arr, res['p_lower'], p_med,
                        color=color, alpha=0.12)

                # Región óptima sombreada
                mask = (j_arr >= res['j_low']) & (j_arr <= res['j_high'])
                ax_P.fill_between(
                    j_arr[mask], 0, p_med[mask],
                    color=color, alpha=0.18)

                # Marcador j*
                ax_P.plot(res['j_star'], res['p_star'],
                          marker='*', color=color, ms=12, zorder=5)

                # Potencia garantizada
                if res['p_garantizada'] is not None:
                    ax_P.axhline(
                        res['p_garantizada'],
                        color=color, lw=1, ls=':', alpha=0.7)

            ax_P.legend(fontsize=7, loc='upper left')
            ax_P.grid(True, alpha=0.3)

            plt.tight_layout()

            if args.guardar_grafico:
                nombre_archivo = (
                    f"optimizador_exp{args.exp_id}_"
                    f"umbral{int(args.umbral*100)}.png"
                )
                plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
                print(f"\nGráfico guardado en: {nombre_archivo}")

            if args.mostrar_grafico:
                plt.show()

        except ImportError:
            print("\nmatplotlib no disponible — omitiendo gráfico")

    print("\nOptimización completada.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimizador Operacional MCFC — Digital Twin UdeC"
    )
    parser.add_argument(
        "--exp_id", type=int, default=None,
        help="ID del experimento en PostgreSQL (si no se indica, "
             "se muestra lista interactiva)"
    )
    parser.add_argument(
        "--umbral", type=float, default=0.95,
        help="Fracción del Pmax para definir región óptima (default: 0.95)"
    )
    parser.add_argument(
        "--guardar_grafico", action="store_true",
        help="Guardar gráfico como PNG en el directorio actual"
    )
    parser.add_argument(
        "--mostrar_grafico", action="store_true",
        help="Mostrar gráfico interactivo"
    )
    parser.add_argument(
        "--listar", action="store_true",
        help="Listar experimentos disponibles y salir"
    )
    args = parser.parse_args()

    # Modo --listar: solo muestra la tabla y sale
    if args.listar:
        df = listar_experimentos()
        print("=" * 65)
        print("  Experimentos disponibles en la base de datos")
        print("=" * 65)
        print(f"{'ID':>4} | {'T (°C)':>6} | {'H2a':>6} | {'CO2a':>6} | "
              f"{'O2c':>6} | {'CO2c':>6} | {'n med':>6}")
        print("-" * 55)
        for _, row in df.iterrows():
            print(f"{int(row['id_experimento']):>4} | "
                  f"{int(row['t']):>6} | "
                  f"{row['h2a']:>6.3f} | "
                  f"{row['co2a']:>6.3f} | "
                  f"{row['o2c']:>6.3f} | "
                  f"{row['co2c']:>6.3f} | "
                  f"{int(row['n_mediciones']):>6}")
        print("-" * 55)
        print(f"  Total: {len(df)} experimentos")
        sys.exit(0)

    # Si no se pasó --exp_id, modo interactivo
    if args.exp_id is None:
        args.exp_id = seleccionar_experimento_interactivo()

    run(args)
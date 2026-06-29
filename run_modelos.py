"""
run_modelos.py
==============
Runner de verificación y optimización del Digital Twin MCFC.

NO reentrena ni toca la base de datos: carga los modelos serializados (.pkl)
ya existentes, ejecuta los cinco modelos sobre una condición operacional dada
y corre el optimizador operacional sobre cada uno.

Sirve como "smoke test" reproducible: si este script corre sin errores y las
curvas son monótonas decrecientes, los modelos del repo están sanos y son
consistentes con los reportados en la memoria.

Convención de predicción idéntica a optimizador_mcfc.py (misma fórmula de
Nernst inline, mismos scalers por familia de modelo).

Uso:
    python3 run_modelos.py                  # condición por defecto, T=650°C
    python3 run_modelos.py --temp 550       # otra temperatura (usa su r_1 real)
    python3 run_modelos.py --variante balanceado   # warsaw (def.) | balanceado
    python3 run_modelos.py --csv salida.csv # exporta la tabla de predicciones
"""

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "models")

# ── Constantes físicas (idénticas a optimizador_mcfc.py) ─────────────────────
R_GAS = 8.314      # J/(mol·K)
F_FAR = 96485.0    # C/mol
R2_ACT = 91.878    # cm²/A — parámetro de pérdidas de activación
DELTA = 0.012      # V     — offset de calibración

# ── r_1 experimental real por temperatura (desde el dataset de Milewski) ─────
R1_POR_TEMP = {
    550: 2.935458,
    575: 2.634815,
    600: 2.379644,
    625: 2.161409,
    650: 1.973445,
}

# ── Condición operacional por defecto (composición media a 650°C) ────────────
COND_DEFAULT = dict(H2a=2.0, H2Oa=0.40, CO2a=0.55, O2c=2.15, CO2c=4.27, N2c=4.87)

# ── Grilla de densidad de corriente ──────────────────────────────────────────
J_MIN, J_MAX, N_J = 0.005, 0.200, 400

FAMILIAS = ["Nernst", "PLS", "KPLS", "GPR", "GPR Residual"]

# Sufijo de archivo por familia y variante
PKL_NOMBRE = {
    "PLS": "pls_voltaje_cv_{var}.pkl",
    "KPLS": "kpls_voltaje_cv_{var}.pkl",
    "GPR": "gpr_voltaje_{var}.pkl",
    "GPR Residual": "gpr_residual_{var}.pkl",
}


# ── Modelo de Nernst (inline, idéntico a optimizador_mcfc.py) ────────────────
def nernst_voltaje(j, T_C, h2a, h2oa, co2a, o2c, co2c, n2c, r1,
                   r2=R2_ACT, delta=DELTA):
    j = np.asarray(j, dtype=float)
    T_K = T_C + 273.15
    eps = 1e-10
    sa = max(h2a + h2oa + co2a, eps)
    sc = max(o2c + co2c + n2c, eps)
    xH2 = max(h2a / sa, eps)
    xH2O = max(h2oa / sa, eps)
    xCO2a = max(co2a / sa, eps)
    xO2 = max(o2c / sc, eps)
    xCO2c = max(co2c / sc, eps)
    E0 = 1.2723 - 2.4516e-4 * T_K
    EN = E0 + (R_GAS * T_K) / (2 * F_FAR) * np.log(
        xH2 * np.sqrt(xO2) * xCO2c / (xH2O * xCO2a + eps))
    V = EN - r1 * j - (R_GAS * T_K) / (2 * F_FAR) * np.log(1 + r2 * j) - delta
    return np.maximum(V, 0.0)


def _matriz_features(cond, j_arr):
    return np.column_stack([
        np.full(len(j_arr), cond["T"]),
        np.full(len(j_arr), cond["H2a"]),
        np.full(len(j_arr), cond["H2Oa"]),
        np.full(len(j_arr), cond["CO2a"]),
        np.full(len(j_arr), cond["O2c"]),
        np.full(len(j_arr), cond["CO2c"]),
        np.full(len(j_arr), cond["N2c"]),
        j_arr,
        np.full(len(j_arr), cond["r_1"]),
    ])


def predecir(nombre, cond, j_arr, variante):
    """Devuelve (mu, sigma). sigma=None para Nernst, PLS, KPLS."""
    import joblib
    if nombre == "Nernst":
        mu = nernst_voltaje(j_arr, cond["T"], cond["H2a"], cond["H2Oa"],
                            cond["CO2a"], cond["O2c"], cond["CO2c"],
                            cond["N2c"], cond["r_1"])
        return mu, None

    pkl = os.path.join(MODELS_DIR, PKL_NOMBRE[nombre].format(var=variante))
    d = joblib.load(pkl)
    X = _matriz_features(cond, j_arr)

    if nombre in ("PLS", "KPLS"):
        return d["modelo"].predict(X).ravel(), None

    if nombre == "GPR":
        X_sc = d["scaler_X"].transform(X)
        mu_sc, sig_sc = d["modelo"].predict(X_sc, return_std=True)
        escala = d["scaler_y"].scale_[0]
        mu = d["scaler_y"].inverse_transform(mu_sc.reshape(-1, 1)).ravel()
        return mu, sig_sc * escala

    if nombre == "GPR Residual":
        X_sc = d["scaler_X"].transform(X)
        e_sc, sig_sc = d["modelo"].predict(X_sc, return_std=True)
        escala = d["scaler_e"].scale_[0]
        eps_corr = d["scaler_e"].inverse_transform(e_sc.reshape(-1, 1)).ravel()
        v_nernst = nernst_voltaje(j_arr, cond["T"], cond["H2a"], cond["H2Oa"],
                                  cond["CO2a"], cond["O2c"], cond["CO2c"],
                                  cond["N2c"], cond["r_1"])
        return v_nernst + eps_corr, sig_sc * escala

    raise ValueError(f"Modelo desconocido: {nombre}")


def optimizar(mu, sigma, j_arr, umbral=0.95):
    """Punto óptimo j*, Pmax, región óptima y potencia garantizada (±2σ)."""
    p = mu * j_arr
    idx = int(np.argmax(p))
    j_star, p_max = float(j_arr[idx]), float(p[idx])
    en_region = p >= umbral * p_max
    reg = np.where(en_region)[0]
    j_low, j_high = float(j_arr[reg[0]]), float(j_arr[reg[-1]])
    p_gar = None
    if sigma is not None:
        p_lower = (mu - 2 * sigma) * j_arr
        mask = (j_arr >= j_low) & (j_arr <= j_high)
        if mask.any():
            p_gar = float(p_lower[mask].max())
    return dict(j_star=j_star, v_star=float(mu[idx]), p_max=p_max,
                j_low=j_low, j_high=j_high, p_gar=p_gar)


def main():
    ap = argparse.ArgumentParser(description="Runner de verificación + optimización MCFC")
    ap.add_argument("--temp", type=int, default=650, choices=[550, 575, 600, 625, 650],
                    help="Temperatura de operación (usa su r_1 experimental real)")
    ap.add_argument("--variante", default="warsaw", choices=["warsaw", "balanceado"],
                    help="Variante de modelos a usar (default: warsaw)")
    ap.add_argument("--csv", default=None, help="Exportar tabla de predicciones a CSV")
    args = ap.parse_args()

    cond = dict(COND_DEFAULT)
    cond["T"] = float(args.temp)
    cond["r_1"] = R1_POR_TEMP[args.temp]

    print("=" * 70)
    print(f"  Digital Twin MCFC — verificación de modelos (variante: {args.variante})")
    print("=" * 70)
    print(f"  Condición: T={args.temp}°C  r_1={cond['r_1']:.4f}  "
          f"H2a={cond['H2a']} O2c={cond['O2c']} CO2c={cond['CO2c']}")
    print()

    j_arr = np.linspace(J_MIN, J_MAX, N_J)
    j_show = np.linspace(J_MIN, J_MAX, 11)

    resultados = {}
    sanos = True
    for nombre in FAMILIAS:
        mu, sigma = predecir(nombre, cond, j_arr, args.variante)
        resultados[nombre] = (mu, sigma)

    # Tabla de curvas (11 puntos legibles)
    print("  Curva de polarización V(j) [V]:")
    header = "    j      " + "".join(f"{n[:8]:>10}" for n in FAMILIAS)
    print(header)
    idx_show = [int(np.argmin(np.abs(j_arr - js))) for js in j_show]
    for k in idx_show:
        fila = f"  {j_arr[k]:.4f}  "
        for nombre in FAMILIAS:
            fila += f"{resultados[nombre][0][k]:>10.4f}"
        print(fila)

    # Chequeos de sanidad física
    print("\n  Chequeos de sanidad física:")
    for nombre in FAMILIAS:
        mu, sigma = resultados[nombre]
        mono = bool(np.all(np.diff(mu) <= 1e-6))
        rango = bool(mu.min() > 0.0 and mu.max() < 1.4)
        sig_ok = sigma is None or bool(np.all(sigma > 0))
        ok = mono and rango and sig_ok
        sanos = sanos and ok
        marca = "OK " if ok else "XX "
        extra = "" if sigma is None else f"  σ∈[{sigma.min():.4f},{sigma.max():.4f}]"
        print(f"    {marca}{nombre:13s} monótona={mono}  rango={rango}{extra}")

    # Optimización por modelo
    print("\n  Optimización operacional (P = V·j):")
    print(f"    {'Modelo':13s} {'j* (A/cm²)':>11} {'V* (V)':>8} "
          f"{'Pmax (W/cm²)':>13} {'región óptima':>18} {'P_gar':>8}")
    for nombre in FAMILIAS:
        mu, sigma = resultados[nombre]
        o = optimizar(mu, sigma, j_arr)
        reg = f"[{o['j_low']:.3f}, {o['j_high']:.3f}]"
        pg = f"{o['p_gar']:.4f}" if o["p_gar"] is not None else "  —"
        print(f"    {nombre:13s} {o['j_star']:>11.4f} {o['v_star']:>8.4f} "
              f"{o['p_max']:>13.4f} {reg:>18} {pg:>8}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["j"] + FAMILIAS)
            for k in range(len(j_arr)):
                w.writerow([f"{j_arr[k]:.5f}"] +
                           [f"{resultados[n][0][k]:.5f}" for n in FAMILIAS])
        print(f"\n  ✓ Predicciones exportadas a {args.csv}")

    print("\n" + "=" * 70)
    if sanos:
        print("  ✓ TODOS LOS MODELOS PASAN LOS CHEQUEOS DE SANIDAD FÍSICA")
        print("=" * 70)
        return 0
    else:
        print("  ✗ ALGÚN MODELO FALLÓ LOS CHEQUEOS — revisar arriba")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
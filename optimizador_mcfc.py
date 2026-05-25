"""
optimizador_mcfc.py
-------------------
Optimizador Nivel 2 para el Digital Twin de la celda MCFC.

Dado un conjunto de condiciones operacionales fijas (temperatura,
composición gaseosa, resistencia óhmica r_1), encuentra la densidad
de corriente j* que maximiza la densidad de potencia p = V·j para
cada uno de los tres modelos disponibles: Nernst, PLS y KPLS.

Uso independiente:
    python3 optimizador_mcfc.py

Uso como módulo (desde dashboard u otro script):
    from optimizador_mcfc import optimizar_pmax, OptimizadorMCFC
"""

import os
import numpy as np
import joblib
from scipy.optimize import minimize_scalar

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PKL_PLS   = os.path.join(BASE_DIR, 'modelos', 'pls_voltaje_cv.pkl')
PKL_KPLS  = os.path.join(BASE_DIR, 'modelos', 'kpls_voltaje_cv.pkl')

# ── Rango experimental de j ────────────────────────────────────────────────────
J_MIN = 0.005   # A/cm²
J_MAX = 0.200   # A/cm²


# ──────────────────────────────────────────────────────────────────────────────
# Importar modelo Nernst desde módulo centralizado
# ──────────────────────────────────────────────────────────────────────────────
try:
    from modelos.modelo_nernst import voltaje_modelo as nernst_voltaje
    NERNST_OK = True
except ImportError:
    try:
        from modelo_nernst import voltaje_modelo as nernst_voltaje
        NERNST_OK = True
    except ImportError:
        NERNST_OK = False
        print("AVISO: modelo_nernst.py no encontrado. Nernst no disponible.")


# ──────────────────────────────────────────────────────────────────────────────
# Carga de modelos PLS / KPLS
# ──────────────────────────────────────────────────────────────────────────────
def _cargar_pkl(ruta, nombre):
    """Carga un modelo .pkl y devuelve (modelo_sklearn, features) o (None, None)."""
    if not os.path.exists(ruta):
        print(f"AVISO: {nombre} no encontrado en {ruta}")
        return None, None
    datos = joblib.load(ruta)
    return datos['modelo'], datos['features']


_modelo_pls,  _features_pls  = _cargar_pkl(PKL_PLS,  'PLS')
_modelo_kpls, _features_kpls = _cargar_pkl(PKL_KPLS, 'KPLS')


# ──────────────────────────────────────────────────────────────────────────────
# Funciones de potencia por modelo
# ──────────────────────────────────────────────────────────────────────────────

def _potencia_nernst(j, T_C, h2a, h2oa, co2a, o2c, co2c, n2c, r1):
    """Densidad de potencia según modelo Nernst con pérdidas."""
    V = nernst_voltaje(
        j=j, T_C=T_C,
        h2a=h2a, h2oa=h2oa, co2a=co2a,
        o2c=o2c, co2c=co2c, n2c=n2c,
        r1=r1
    )
    return float(V * j)


def _potencia_datamodel(j, modelo, features, T_C, h2a, h2oa, co2a,
                        o2c, co2c, n2c, r1):
    """Densidad de potencia según modelo basado en datos (PLS o KPLS)."""
    fila = {
        'T':        T_C,
        'H2a':      h2a,
        'H2Oa':     h2oa,
        'CO2a':     co2a,
        'O2c':      o2c,
        'CO2c':     co2c,
        'N2c':      n2c,
        'i, A/cm²': j,
        'r_1':      r1,
    }
    X = np.array([[fila[f] for f in features]])
    V = float(modelo.predict(X).ravel()[0])
    V = max(V, 0.0)
    return float(V * j)


# ──────────────────────────────────────────────────────────────────────────────
# Función principal de optimización
# ──────────────────────────────────────────────────────────────────────────────

def optimizar_pmax(T_C, h2a, h2oa, co2a, o2c, co2c, n2c, r1,
                   j_min=J_MIN, j_max=J_MAX):
    """
    Encuentra la densidad de corriente j* que maximiza p = V·j
    para cada modelo disponible.

    Retorna dict con claves 'Nernst', 'PLS', 'KPLS', cada una con:
        j_opt  : densidad de corriente óptima [A/cm²]
        V_opt  : voltaje en ese punto [V]
        Pmax   : densidad de potencia máxima [W/cm²]
        exito  : bool
    """
    resultados = {}

    # ── Nernst ────────────────────────────────────────────────────────────────
    if NERNST_OK:
        try:
            res = minimize_scalar(
                fun=lambda j: -_potencia_nernst(
                    j, T_C, h2a, h2oa, co2a, o2c, co2c, n2c, r1
                ),
                bounds=(j_min, j_max),
                method='bounded',
                options={'xatol': 1e-5}
            )
            j_opt = float(res.x)
            V_opt = float(nernst_voltaje(
                j=j_opt, T_C=T_C,
                h2a=h2a, h2oa=h2oa, co2a=co2a,
                o2c=o2c, co2c=co2c, n2c=n2c, r1=r1
            ))
            resultados['Nernst'] = {
                'j_opt':    round(j_opt, 5),
                'V_opt':    round(V_opt, 4),
                'Pmax':     round(float(V_opt * j_opt), 5),
                'en_limite': abs(j_opt - j_max) < 1e-4,
                'exito':    True
            }
        except Exception as e:
            resultados['Nernst'] = {'exito': False, 'error': str(e)}
    else:
        resultados['Nernst'] = {'exito': False, 'error': 'modelo_nernst.py no disponible'}

    # ── PLS ───────────────────────────────────────────────────────────────────
    if _modelo_pls is not None:
        try:
            res = minimize_scalar(
                fun=lambda j: -_potencia_datamodel(
                    j, _modelo_pls, _features_pls,
                    T_C, h2a, h2oa, co2a, o2c, co2c, n2c, r1
                ),
                bounds=(j_min, j_max),
                method='bounded',
                options={'xatol': 1e-5}
            )
            j_opt = float(res.x)
            X_opt = np.array([[T_C, h2a, h2oa, co2a, o2c, co2c, n2c, j_opt, r1]])
            V_opt = max(float(_modelo_pls.predict(X_opt).ravel()[0]), 0.0)
            resultados['PLS'] = {
                'j_opt':    round(j_opt, 5),
                'V_opt':    round(V_opt, 4),
                'Pmax':     round(float(V_opt * j_opt), 5),
                'en_limite': abs(j_opt - j_max) < 1e-4,
                'exito':    True
            }
        except Exception as e:
            resultados['PLS'] = {'exito': False, 'error': str(e)}
    else:
        resultados['PLS'] = {'exito': False, 'error': 'pls_voltaje_cv.pkl no disponible'}

    # ── KPLS ──────────────────────────────────────────────────────────────────
    if _modelo_kpls is not None:
        try:
            res = minimize_scalar(
                fun=lambda j: -_potencia_datamodel(
                    j, _modelo_kpls, _features_kpls,
                    T_C, h2a, h2oa, co2a, o2c, co2c, n2c, r1
                ),
                bounds=(j_min, j_max),
                method='bounded',
                options={'xatol': 1e-5}
            )
            j_opt = float(res.x)
            X_opt = np.array([[T_C, h2a, h2oa, co2a, o2c, co2c, n2c, j_opt, r1]])
            V_opt = max(float(_modelo_kpls.predict(X_opt).ravel()[0]), 0.0)
            resultados['KPLS'] = {
                'j_opt':    round(j_opt, 5),
                'V_opt':    round(V_opt, 4),
                'Pmax':     round(float(V_opt * j_opt), 5),
                'en_limite': abs(j_opt - j_max) < 1e-4,
                'exito':    True
            }
        except Exception as e:
            resultados['KPLS'] = {'exito': False, 'error': str(e)}
    else:
        resultados['KPLS'] = {'exito': False, 'error': 'kpls_voltaje_cv.pkl no disponible'}

    return resultados


# ──────────────────────────────────────────────────────────────────────────────
# Clase conveniente para uso repetido
# ──────────────────────────────────────────────────────────────────────────────

class OptimizadorMCFC:
    """
    Wrapper para llamadas repetidas al optimizador.
    Los modelos se cargan una sola vez al instanciar la clase.

    Ejemplo:
        opt = OptimizadorMCFC()
        r = opt.optimizar(T_C=650, h2a=2.2, h2oa=0.41, co2a=0.55,
                          o2c=1.3, co2c=2.15, n2c=4.87, r1=1.9734)
    """
    def __init__(self, j_min=J_MIN, j_max=J_MAX):
        self.j_min = j_min
        self.j_max = j_max

    def optimizar(self, T_C, h2a, h2oa, co2a, o2c, co2c, n2c, r1):
        return optimizar_pmax(
            T_C=T_C, h2a=h2a, h2oa=h2oa, co2a=co2a,
            o2c=o2c, co2c=co2c, n2c=n2c, r1=r1,
            j_min=self.j_min, j_max=self.j_max
        )

    def tabla_resultados(self, T_C, h2a, h2oa, co2a, o2c, co2c, n2c, r1):
        r = self.optimizar(T_C, h2a, h2oa, co2a, o2c, co2c, n2c, r1)
        print(f"\n{'='*62}")
        print(f"  Optimización Nivel 2 — Pmax restringido al rango experimental")
        print(f"  Rango válido: j ∈ [{self.j_min:.3f}, {self.j_max:.3f}] A/cm²")
        print(f"{'='*62}")
        print(f"  T={T_C}°C  r1={r1} Ohm·cm²")
        print(f"  Ánodo  → H2={h2a}, H2O={h2oa}, CO2={co2a}")
        print(f"  Cátodo → O2={o2c}, CO2={co2c}, N2={n2c}")
        print(f"{'-'*62}")
        print(f"  {'Modelo':<8} | {'j* (A/cm²)':>10} | {'V* (V)':>8} | "
              f"{'Pmax (W/cm²)':>12} | {'En límite':>9}")
        print(f"{'-'*62}")
        for nombre, res in r.items():
            if res.get('exito', False):
                en_limite = "Sí ⚠" if abs(res['j_opt'] - self.j_max) < 1e-4 else "No"
                print(f"  {nombre:<8} | {res['j_opt']:>10.5f} | "
                      f"{res['V_opt']:>8.4f} | {res['Pmax']:>12.5f} | {en_limite:>9}")
            else:
                print(f"  {nombre:<8} | ERROR — {res.get('error','')}")
        print(f"{'-'*62}")
        print(f"  ⚠ j* en límite: Pmax real podría estar fuera del rango validado.")
        print(f"{'='*62}\n")
        return r


# ──────────────────────────────────────────────────────────────────────────────
# Prueba rápida
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    opt = OptimizadorMCFC()

    # Condiciones típicas a 650°C
    opt.tabla_resultados(
        T_C=650, h2a=2.2, h2oa=0.41, co2a=0.55,
        o2c=1.3, co2c=2.15, n2c=4.87, r1=1.9734
    )

    # Segunda prueba a 600°C
    opt.tabla_resultados(
        T_C=600, h2a=2.5, h2oa=0.50, co2a=0.60,
        o2c=1.5, co2c=2.00, n2c=5.00, r1=2.380
    )
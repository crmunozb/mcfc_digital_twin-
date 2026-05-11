"""
modelo_nernst.py
----------------
Modelo semi-empírico de voltaje para una celda de combustible de
carbonatos fundidos (MCFC).

Incluye:
- Potencial de equilibrio de referencia E0(T)
- Voltaje de Nernst para MCFC
- Modelo de voltaje con pérdidas óhmicas y de activación
- Cálculo de densidad de potencia

Este módulo puede ser importado desde dashboard.py y simulador_mcfc.py,
o ejecutado directamente para realizar una prueba rápida.

Uso directo:
    python3 modelos/modelo_nernst.py
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constantes físicas
# ─────────────────────────────────────────────────────────────────────────────

R_GAS = 8.314          # Constante universal de gases [J/(mol·K)]
F_FAR = 96485.0        # Constante de Faraday [C/mol]
N_ELEC = 2             # Número de electrones transferidos


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros semi-empíricos por defecto
# ─────────────────────────────────────────────────────────────────────────────

R2_DEFAULT = 91.878    # Parámetro asociado a pérdidas de activación
DELTA_DEFAULT = 0.012  # Offset de calibración [V]


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad interna
# ─────────────────────────────────────────────────────────────────────────────

def _salida_escalar_si_corresponde(valor, entrada_original):
    """
    Devuelve float si la entrada original era escalar.
    Devuelve ndarray si la entrada original era arreglo.
    """
    if np.isscalar(entrada_original):
        return float(np.asarray(valor))
    return valor


# ─────────────────────────────────────────────────────────────────────────────
# Potencial de equilibrio de referencia
# ─────────────────────────────────────────────────────────────────────────────

def e0_temperatura(T_K):
    """
    Calcula el potencial de equilibrio de referencia E0(T).

    Modelo:
        E0(T) = 1.2723 - 2.4516e-4 * T_K

    Parámetros
    ----------
    T_K : float or array-like
        Temperatura absoluta en Kelvin.

    Retorna
    -------
    float or ndarray
        Potencial de equilibrio de referencia [V].
    """
    T_K_arr = np.asarray(T_K, dtype=float)
    E0 = 1.2723 - 2.4516e-4 * T_K_arr
    return _salida_escalar_si_corresponde(E0, T_K)


# ─────────────────────────────────────────────────────────────────────────────
# Voltaje de Nernst para MCFC
# ─────────────────────────────────────────────────────────────────────────────

def e_nernst(
    T_C,
    h2a,
    h2oa,
    co2a,
    o2c,
    co2c,
    n2a=0.0,
    co=0.0,
    ch4=0.0,
    n2c=0.0,
    h2oc=0.0
):
    """
    Calcula el voltaje reversible de Nernst para una celda MCFC.

    La composición gaseosa se normaliza como fracción molar relativa
    en ánodo y cátodo.

    Ánodo considerado:
        H2, H2O, CO2, N2, CO, CH4

    Cátodo considerado:
        O2, CO2, N2, H2O

    Fórmula:
        E_Nernst = E0(T) + (RT/2F) ln[
            (x_H2 * sqrt(x_O2) * x_CO2,c) / (x_H2O * x_CO2,a)
        ]

    Parámetros
    ----------
    T_C : float
        Temperatura en grados Celsius.

    h2a, h2oa, co2a : float
        Flujos molares relativos o composiciones del ánodo.

    o2c, co2c : float
        Flujos molares relativos o composiciones del cátodo.

    n2a, co, ch4, n2c, h2oc : float
        Especies adicionales consideradas en la normalización.

    Retorna
    -------
    float or ndarray
        Voltaje reversible de Nernst [V].
    """
    eps = 1e-10

    T_C_original = T_C
    T_C = np.asarray(T_C, dtype=float)
    T_K = T_C + 273.15

    suma_anodo = h2a + h2oa + co2a + n2a + co + ch4
    suma_catodo = o2c + co2c + n2c + h2oc

    suma_anodo = np.maximum(suma_anodo, eps)
    suma_catodo = np.maximum(suma_catodo, eps)

    x_h2 = np.maximum(h2a / suma_anodo, eps)
    x_h2o = np.maximum(h2oa / suma_anodo, eps)
    x_co2a = np.maximum(co2a / suma_anodo, eps)

    x_o2 = np.maximum(o2c / suma_catodo, eps)
    x_co2c = np.maximum(co2c / suma_catodo, eps)

    argumento = (x_h2 * np.sqrt(x_o2) * x_co2c) / (x_h2o * x_co2a)
    argumento = np.maximum(argumento, eps)

    E = e0_temperatura(T_K) + (R_GAS * T_K) / (N_ELEC * F_FAR) * np.log(argumento)

    return _salida_escalar_si_corresponde(E, T_C_original)


# ─────────────────────────────────────────────────────────────────────────────
# Modelo de voltaje con pérdidas
# ─────────────────────────────────────────────────────────────────────────────

def voltaje_modelo(
    j,
    T_C,
    h2a,
    h2oa,
    co2a,
    o2c,
    co2c,
    r1,
    r2=R2_DEFAULT,
    delta=DELTA_DEFAULT,
    n2a=0.0,
    co=0.0,
    ch4=0.0,
    n2c=0.0,
    h2oc=0.0
):
    """
    Calcula el voltaje de celda usando Nernst con pérdidas.

    Modelo:
        E = E_Nernst - r1*j - (RT/2F)*ln(1 + r2*j) - delta

    Parámetros
    ----------
    j : float or array-like
        Densidad de corriente [A/cm²].

    T_C : float
        Temperatura [°C].

    h2a, h2oa, co2a, o2c, co2c : float
        Variables principales de composición gaseosa.

    r1 : float
        Resistencia óhmica específica [Ohm·cm²].

    r2 : float
        Parámetro semi-empírico de activación.

    delta : float
        Offset de calibración [V].

    n2a, co, ch4, n2c, h2oc : float
        Especies adicionales para normalización de fracciones molares.

    Retorna
    -------
    float or ndarray
        Voltaje predicho [V].
    """
    j_original = j
    j = np.asarray(j, dtype=float)

    T_K = np.asarray(T_C, dtype=float) + 273.15

    e_n = e_nernst(
        T_C=T_C,
        h2a=h2a,
        h2oa=h2oa,
        co2a=co2a,
        o2c=o2c,
        co2c=co2c,
        n2a=n2a,
        co=co,
        ch4=ch4,
        n2c=n2c,
        h2oc=h2oc
    )

    eta_ohm = r1 * j
    eta_act = (R_GAS * T_K) / (N_ELEC * F_FAR) * np.log(1 + r2 * j)

    voltaje = e_n - eta_ohm - eta_act - delta

    voltaje = np.maximum(voltaje, 0.0)

    return _salida_escalar_si_corresponde(voltaje, j_original)


# ─────────────────────────────────────────────────────────────────────────────
# Densidad de potencia
# ─────────────────────────────────────────────────────────────────────────────

def densidad_potencia(V, j):
    """
    Calcula la densidad de potencia.

    Modelo:
        p = V * j

    Parámetros
    ----------
    V : float or array-like
        Voltaje [V].

    j : float or array-like
        Densidad de corriente [A/cm²].

    Retorna
    -------
    float or ndarray
        Densidad de potencia [W/cm²].
    """
    V_original = V
    p = np.asarray(V, dtype=float) * np.asarray(j, dtype=float)
    return _salida_escalar_si_corresponde(p, V_original)


# ─────────────────────────────────────────────────────────────────────────────
# Prueba rápida del módulo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    T = 650
    j = np.linspace(0.005, 0.200, 21)

    h2a = 2.2
    h2oa = 0.41
    co2a = 0.55
    o2c = 1.3
    co2c = 2.15
    n2c = 4.87
    r1 = 1.9734

    E_N = e_nernst(
        T_C=T,
        h2a=h2a,
        h2oa=h2oa,
        co2a=co2a,
        o2c=o2c,
        co2c=co2c,
        n2c=n2c
    )

    V = voltaje_modelo(
        j=j,
        T_C=T,
        h2a=h2a,
        h2oa=h2oa,
        co2a=co2a,
        o2c=o2c,
        co2c=co2c,
        r1=r1,
        n2c=n2c
    )

    p = densidad_potencia(V, j)

    print("=" * 70)
    print("Prueba modelo_nernst.py")
    print("=" * 70)
    print(f"Temperatura: {T} °C")
    print(f"E_Nernst:    {E_N:.4f} V")
    print(f"r1:          {r1:.4f} Ohm·cm²")
    print("-" * 70)
    print("Primeros 5 puntos calculados:")
    print("-" * 70)

    for jj, vv, pp in zip(j[:5], V[:5], p[:5]):
        print(f"j = {jj:.3f} A/cm² | V = {vv:.4f} V | p = {pp:.4f} W/cm²")

    print("-" * 70)
    print("Último punto:")
    print(f"j = {j[-1]:.3f} A/cm² | V = {V[-1]:.4f} V | p = {p[-1]:.4f} W/cm²")
    print("=" * 70)
    print("OK: modelo_nernst.py funcionando correctamente.")
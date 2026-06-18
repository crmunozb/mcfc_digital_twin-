"""
generar_graficos_curva.py
--------------------------
Regenera los gráficos de curva de aprendizaje desde el CSV ya existente.
No requiere reentrenar los modelos.

Uso:
    python3 generar_graficos_curva.py

Genera:
    curva_aprendizaje_r2.pdf
    curva_aprendizaje_mae.pdf
    curva_aprendizaje_nrmse.pdf
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'curva_aprendizaje_resultados.csv')

df_res = pd.read_csv(CSV_PATH)
x      = df_res['n_pts_por_temp'].values
print(f"CSV cargado: {len(df_res)} filas")

# ── Paleta y estilos ──────────────────────────────────────────────────────────
COLORES = {
    'PLS':          '#4a7eb5',
    'KPLS':         '#b8922a',
    'GPR':          '#6aa3c8',
    'GPR Residual': '#2e8b57',
    'Nernst':       '#888888',
}

# Marcadores distintos por modelo para distinguir en impresión B/N
MARCADORES = {
    'PLS':          'o',
    'KPLS':         's',
    'GPR':          '^',
    'GPR Residual': 'D',
}

# Estilos de línea distintos
LINESTYLES_MODELOS = {
    'PLS':          (0, (5, 2)),      # guión largo
    'KPLS':         (0, (3, 1, 1, 1)), # punto-guión
    'GPR':          (0, (1, 1)),       # punteada
    'GPR Residual': '-',               # sólida
}

MODELOS_LINEAS = [
    ('PLS',          'pls'),
    ('KPLS',         'kpls'),
    ('GPR',          'gpr'),
    ('GPR Residual', 'gpr_res'),
]

# ── Estilo global ─────────────────────────────────────────────────────────────
def estilo_base():
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.titlesize':    12,
        'axes.labelsize':    11,
        'legend.fontsize':   10,
        'axes.grid':         True,
        'grid.alpha':        0.25,
        'grid.linestyle':    '--',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'figure.dpi':        150,
    })

# ── Función de gráfico ────────────────────────────────────────────────────────
def guardar_grafico(metrica_key, ylabel, titulo,
                    ylim=None, legend_loc='upper right',
                    filename=None, ytick_format=None):
    estilo_base()
    fig, ax = plt.subplots(figsize=(8, 5))

    for nombre, key in MODELOS_LINEAS:
        y_vals = df_res[f'{key}_{metrica_key}'].values
        ax.plot(x, y_vals,
                color=COLORES[nombre],
                linestyle=LINESTYLES_MODELOS[nombre],
                linewidth=2.5,
                marker=MARCADORES[nombre],
                markersize=7,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=COLORES[nombre],
                label=nombre,
                zorder=3)

    # Nernst como línea horizontal de referencia
    nernst_val = df_res[f'nernst_{metrica_key}'].iloc[0]
    ax.axhline(nernst_val,
               color=COLORES['Nernst'],
               linestyle='--',
               linewidth=1.8,
               alpha=0.8,
               label=f'Nernst (ref. = {nernst_val:.3f})',
               zorder=2)

    ax.set_xlabel('Puntos de entrenamiento por temperatura', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], fontsize=10)

    if ylim:
        ax.set_ylim(ylim)

    if ytick_format:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(ytick_format))

    # Leyenda arriba a la derecha con borde suave
    legend = ax.legend(
        loc=legend_loc,
        framealpha=0.92,
        edgecolor='#cccccc',
        fancybox=True,
        fontsize=10,
    )

    # Anotación del punto de convergencia para GPR Residual
    if metrica_key == 'r2':
        y_gpr_res = df_res['gpr_res_r2'].values
        # Marcar n=100 como punto de convergencia aproximado
        idx_conv = list(x).index(100) if 100 in x else 2
        ax.annotate(
            'Convergencia\naproximada',
            xy=(x[idx_conv], y_gpr_res[idx_conv]),
            xytext=(x[idx_conv] + 40, y_gpr_res[idx_conv] - 0.003),
            fontsize=8.5,
            color=COLORES['GPR Residual'],
            arrowprops=dict(arrowstyle='->', color=COLORES['GPR Residual'],
                           lw=1.2),
            ha='left',
        )

    plt.tight_layout()
    path = os.path.join(BASE_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {path}")


# ── R² ────────────────────────────────────────────────────────────────────────
guardar_grafico(
    metrica_key='r2',
    ylabel='$R^2$ (conjunto de prueba)',
    titulo='Curva de aprendizaje — $R^2$ vs puntos de entrenamiento por temperatura',
    ylim=(0.930, 1.004),
    legend_loc='lower right',
    ytick_format='%.3f',
    filename='curva_aprendizaje_r2.pdf'
)

# ── MAE ───────────────────────────────────────────────────────────────────────
guardar_grafico(
    metrica_key='mae',
    ylabel='MAE (V)',
    titulo='Curva de aprendizaje — MAE vs puntos de entrenamiento por temperatura',
    legend_loc='upper right',
    ytick_format='%.3f',
    filename='curva_aprendizaje_mae.pdf'
)

# ── NRMSE ─────────────────────────────────────────────────────────────────────
guardar_grafico(
    metrica_key='nrmse',
    ylabel='NRMSE',
    titulo='Curva de aprendizaje — NRMSE vs puntos de entrenamiento por temperatura',
    legend_loc='upper right',
    ytick_format='%.3f',
    filename='curva_aprendizaje_nrmse.pdf'
)

print("\n¡Gráficos regenerados!")
print(f"Archivos en: {BASE_DIR}")
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
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'curva_aprendizaje_resultados.csv')

df_res = pd.read_csv(CSV_PATH)
x      = df_res['n_pts_por_temp'].values
print(f"CSV cargado: {len(df_res)} filas")

# ── Paleta ────────────────────────────────────────────────────────────────────
COLORES = {
    'PLS':          '#4a7eb5',
    'KPLS':         '#c9952d',
    'GPR':          '#6aa3c8',
    'GPR Residual': '#3a8f5c',
    'Nernst':       '#999999',
}

MARCADORES = {
    'PLS':          'o',
    'KPLS':         's',
    'GPR':          '^',
    'GPR Residual': 'D',
}

LINESTYLES_MODELOS = {
    'PLS':          (0, (6, 2)),
    'KPLS':         (0, (3, 1, 1, 1)),
    'GPR':          (0, (2, 1)),
    'GPR Residual': '-',
}

MODELOS_LINEAS = [
    ('GPR Residual', 'gpr_res'),
    ('GPR',          'gpr'),
    ('KPLS',         'kpls'),
    ('PLS',          'pls'),
]

# ── Estilo global ─────────────────────────────────────────────────────────────
def estilo_base():
    plt.rcParams.update({
        'font.family':        'DejaVu Sans',
        'font.size':          11,
        'axes.titlesize':     12,
        'axes.labelsize':     11,
        'legend.fontsize':    10,
        'axes.grid':          True,
        'grid.alpha':         0.20,
        'grid.linestyle':     ':',
        'grid.color':         '#aaaaaa',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.spines.left':   True,
        'axes.spines.bottom': True,
        'axes.linewidth':     0.8,
        'figure.dpi':         200,
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'xtick.major.size':   4,
        'ytick.major.size':   4,
    })

# ── Función principal ─────────────────────────────────────────────────────────
def guardar_grafico(metrica_key, ylabel, ylim=None,
                    legend_loc='best', filename=None,
                    ytick_format=None, anotaciones=None,
                    broken_axis=None):
    """
    broken_axis: dict con claves 'bottom_lim' y 'top_lim' para hacer eje roto.
    Ej: {'bottom_lim': (0.86, 0.90), 'top_lim': (0.955, 1.025)}
    """
    estilo_base()

    if broken_axis:
        # Eje roto: dos subplots apilados
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(8, 5.5), sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06}
        )
        fig.patch.set_facecolor('white')
        axes = [ax_top, ax_bot]
        lims = [broken_axis['top_lim'], broken_axis['bottom_lim']]
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('white')
        axes = [ax]
        lims  = [ylim]

    for ax_i, lim_i in zip(axes, lims):
        nernst_val = df_res[f'nernst_{metrica_key}'].iloc[0]

        # Zona gris bajo/sobre Nernst
        if metrica_key == 'r2':
            ax_i.axhspan(lim_i[0], nernst_val,
                         color='#f5f5f5', zorder=0)
        else:
            ax_i.axhspan(nernst_val, lim_i[1],
                         color='#f5f5f5', zorder=0)

        # Línea Nernst
        ax_i.axhline(nernst_val, color=COLORES['Nernst'],
                     linestyle='--', linewidth=1.6, alpha=0.85, zorder=2)

        # Curvas
        for nombre, key in MODELOS_LINEAS:
            y_vals = df_res[f'{key}_{metrica_key}'].values
            ax_i.plot(x, y_vals,
                      color=COLORES[nombre],
                      linestyle=LINESTYLES_MODELOS[nombre],
                      linewidth=2.2,
                      marker=MARCADORES[nombre],
                      markersize=6.5,
                      markerfacecolor='white',
                      markeredgewidth=1.8,
                      markeredgecolor=COLORES[nombre],
                      label=nombre,
                      zorder=4,
                      solid_capstyle='round')

        ax_i.set_ylim(lim_i)
        ax_i.set_xlim(x[0] - 15, x[-1] + 55)
        if ytick_format:
            ax_i.yaxis.set_major_formatter(ticker.FormatStrFormatter(ytick_format))
        ax_i.grid(True, linestyle=':', alpha=0.20, color='#aaaaaa')
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)

    if broken_axis:
        # Ocultar spines del corte
        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(bottom=False)

        # Marcas de corte diagonales
        d = 0.012
        kwargs = dict(transform=ax_top.transAxes, color='#888', clip_on=False, lw=1.2)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1-d, 1+d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax_bot.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

        # Etiqueta Nernst solo en ax_top
        nernst_val = df_res[f'nernst_{metrica_key}'].iloc[0]
        ax_top.text(x[-1] + 8, nernst_val,
                    f'Nernst\n{nernst_val:.4f}',
                    va='center', ha='left',
                    fontsize=8.5, color=COLORES['Nernst'], style='italic')

        # Anotaciones solo en ax_top
        if anotaciones:
            for an in anotaciones:
                ax_top.annotate(
                    an['texto'], xy=an['xy'], xytext=an['xytext'],
                    fontsize=8.5, color=an['color'],
                    arrowprops=dict(arrowstyle='->', color=an['color'],
                                    lw=1.1, connectionstyle='arc3,rad=0.1'),
                    ha=an.get('ha', 'left'), va=an.get('va', 'center'),
                    bbox=dict(boxstyle='round,pad=0.25', fc='white',
                              ec=an['color'], alpha=0.85, lw=0.8),
                )

        # Eje Y compartido centrado
        fig.text(0.02, 0.55, ylabel, va='center', rotation='vertical', fontsize=11)
        ax_bot.set_xlabel('Puntos de entrenamiento por temperatura (n)', fontsize=11, labelpad=8)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([str(v) for v in x], fontsize=10)

        # Leyenda en ax_top
        handles = []
        for nombre, _ in MODELOS_LINEAS:
            handles.append(Line2D([0],[0], color=COLORES[nombre],
                linestyle=LINESTYLES_MODELOS[nombre], linewidth=2,
                marker=MARCADORES[nombre], markersize=6,
                markerfacecolor='white', markeredgewidth=1.8,
                markeredgecolor=COLORES[nombre], label=nombre))
        handles.append(Line2D([0],[0], color=COLORES['Nernst'],
            linestyle='--', linewidth=1.6, label='Nernst (ref.)'))
        ax_top.legend(handles=handles, loc=legend_loc,
                      framealpha=0.95, edgecolor='#dddddd',
                      fancybox=False, fontsize=10,
                      borderpad=0.7, labelspacing=0.4)

        plt.tight_layout(rect=[0.04, 0, 0.93, 1])

    else:
        nernst_val = df_res[f'nernst_{metrica_key}'].iloc[0]
        ax.text(x[-1] + 8, nernst_val,
                f'Nernst\n{nernst_val:.4f}',
                va='center', ha='left',
                fontsize=8.5, color=COLORES['Nernst'], style='italic')

        if anotaciones:
            for an in anotaciones:
                ax.annotate(
                    an['texto'], xy=an['xy'], xytext=an['xytext'],
                    fontsize=8.5, color=an['color'],
                    arrowprops=dict(arrowstyle='->', color=an['color'],
                                    lw=1.1, connectionstyle='arc3,rad=0.1'),
                    ha=an.get('ha', 'left'), va=an.get('va', 'center'),
                    bbox=dict(boxstyle='round,pad=0.25', fc='white',
                              ec=an['color'], alpha=0.85, lw=0.8),
                )

        ax.set_xlabel('Puntos de entrenamiento por temperatura (n)', fontsize=11, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x], fontsize=10)

        handles = []
        for nombre, _ in MODELOS_LINEAS:
            handles.append(Line2D([0],[0], color=COLORES[nombre],
                linestyle=LINESTYLES_MODELOS[nombre], linewidth=2,
                marker=MARCADORES[nombre], markersize=6,
                markerfacecolor='white', markeredgewidth=1.8,
                markeredgecolor=COLORES[nombre], label=nombre))
        handles.append(Line2D([0],[0], color=COLORES['Nernst'],
            linestyle='--', linewidth=1.6, label='Nernst (ref.)'))
        ax.legend(handles=handles, loc=legend_loc,
                  framealpha=0.95, edgecolor='#dddddd',
                  fancybox=False, fontsize=10,
                  borderpad=0.7, labelspacing=0.4)

        plt.tight_layout(rect=[0, 0, 0.93, 1])

    path = os.path.join(BASE_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Guardado: {path}")


# ── R² ────────────────────────────────────────────────────────────────────────
gpr_r2_200    = df_res.loc[df_res['n_pts_por_temp'] == 200, 'gpr_r2'].values[0]
gpr_res_r2_25 = df_res.loc[df_res['n_pts_por_temp'] == 25,  'gpr_res_r2'].values[0]

guardar_grafico(
    metrica_key='r2',
    ylabel='$R^2$ (conjunto de prueba)',
    legend_loc='lower right',
    ytick_format='%.3f',
    filename='curva_aprendizaje_r2.pdf',
    broken_axis={
        'top_lim':    (0.955, 1.025),
        'bottom_lim': (0.860, 0.900),
    },
    anotaciones=[
        dict(
            texto='GPR Residual:\nconvergencia\ndesde n=25',
            xy=(25, 0.9975),
            xytext=(85, 0.980),
            color=COLORES['GPR Residual'],
            ha='left', va='top',
        ),
        dict(
            texto='GPR supera\na Nernst\nen n=200',
            xy=(200, 0.9971),
            xytext=(230, 0.973),
            color=COLORES['GPR'],
            ha='left', va='top',
        ),
    ]
)

# ── MAE ───────────────────────────────────────────────────────────────────────
gpr_mae_25  = df_res.loc[df_res['n_pts_por_temp'] == 25,  'gpr_mae'].values[0]
gpr_mae_200 = df_res.loc[df_res['n_pts_por_temp'] == 200, 'gpr_mae'].values[0]

guardar_grafico(
    metrica_key='mae',
    ylabel='MAE (V)',
    ylim=(0.000, 0.045),
    legend_loc='upper right',
    ytick_format='%.3f',
    filename='curva_aprendizaje_mae.pdf',
    anotaciones=[
        dict(
            texto=f'GPR: {gpr_mae_25:.3f}→{gpr_mae_200:.3f} V\nen 25→200 pts',
            xy=(200, gpr_mae_200),
            xytext=(225, gpr_mae_200 + 0.012),
            color=COLORES['GPR'],
            ha='left', va='bottom',
        ),
    ]
)

# ── NRMSE ─────────────────────────────────────────────────────────────────────
guardar_grafico(
    metrica_key='nrmse',
    ylabel='NRMSE',
    ylim=(0.000, 0.065),
    legend_loc='upper right',
    ytick_format='%.3f',
    filename='curva_aprendizaje_nrmse.pdf',
)

print("\n¡Gráficos regenerados!")
print(f"Archivos en: {BASE_DIR}")
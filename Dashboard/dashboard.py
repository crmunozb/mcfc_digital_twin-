import warnings
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy')

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import psycopg2
import subprocess
import os
import signal
from datetime import datetime, timezone
from scipy.optimize import minimize_scalar as _minimize_scalar

# ── Constantes fisicas ─────────────────────────────────────────────────────────
R_GAS  = 8.314
F_FAR  = 96485.0
N_ELEC = 2

def E0_T(T_K):
    return 1.2723 - 2.4516e-4 * T_K

def E_nernst(T_C, H2a, H2Oa, CO2a, O2c, CO2c,
             N2a=0.0, CO=0.0, CH4=0.0, N2c=0.0, H2Oc=0.0):
    T_K = T_C + 273.15
    eps = 1e-10
    s_a = max(H2a + H2Oa + CO2a + N2a + CO + CH4, eps)
    s_c = max(O2c + CO2c + N2c + H2Oc, eps)
    xH2   = max(H2a  / s_a, eps)
    xH2O  = max(H2Oa / s_a, eps)
    xCO2a = max(CO2a / s_a, eps)
    xO2   = max(O2c  / s_c, eps)
    xCO2c = max(CO2c / s_c, eps)
    log_t = np.log(xH2 * (xO2 ** 0.5) * xCO2c / (xH2O * xCO2a))
    return E0_T(T_K) + (R_GAS * T_K) / (N_ELEC * F_FAR) * log_t

def voltaje_modelo(i, T_C, H2a, H2Oa, CO2a, O2c, CO2c, r_1, r_2=91.878,
                   N2a=0.0, CO=0.0, CH4=0.0, N2c=0.0, H2Oc=0.0):
    En      = E_nernst(T_C, H2a, H2Oa, CO2a, O2c, CO2c, N2a, CO, CH4, N2c, H2Oc)
    eta_ohm = r_1 * i
    eta_act = (R_GAS * (T_C + 273.15)) / (N_ELEC * F_FAR) * np.log(1 + i * r_2)
    return En - eta_ohm - eta_act - 0.012

def metricas(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    r2    = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae   = np.mean(np.abs(y_real - y_pred))
    rmse  = np.sqrt(np.mean((y_real - y_pred) ** 2))
    rango = y_real.max() - y_real.min()
    nrmse = rmse / rango if rango > 0 else 0.0
    return r2, mae, nrmse

# ── Configuracion PostgreSQL ───────────────────────────────────────────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from config import DB_CONFIG

# ── Proceso simulador global ───────────────────────────────────────────────────
_sim_process = None

# ── Cargar modelos PLS y KPLS ────────────────────────────────────────────────
import joblib as _joblib

_MODELOS_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'modelos')
)

# PLS
try:
    _pls_data  = _joblib.load(_os.path.join(_MODELOS_DIR, 'pls_voltaje_cv.pkl'))
    _pls_pipe  = _pls_data['modelo']
    _PLS_FEATS = _pls_data['features']
    _PLS_OK    = True
    print(f"✓ Modelo PLS cargado — R²_test={_pls_data['r2_test']:.4f}")
except Exception as _e:
    _PLS_OK = False
    print(f"⚠ Modelo PLS no disponible: {_e}")

# KPLS
try:
    _kpls_data  = _joblib.load(_os.path.join(_MODELOS_DIR, 'kpls_voltaje_cv.pkl'))
    _kpls_pipe  = _kpls_data['modelo']
    _KPLS_FEATS = _kpls_data['features']
    _KPLS_OK    = True
    print(f"✓ Modelo KPLS cargado — R²_test={_kpls_data['r2_test']:.4f}")
except Exception as _e:
    _KPLS_OK = False
    print(f"⚠ Modelo KPLS no disponible: {_e}")


def _construir_X(i_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r_1):
    """Construye matriz de features para los modelos PLS/KPLS (incluye r_1)."""
    i_arr = np.atleast_1d(i_arr)
    return np.column_stack([
        np.full_like(i_arr, T),
        np.full_like(i_arr, H2a),
        np.full_like(i_arr, H2Oa),
        np.full_like(i_arr, CO2a),
        np.full_like(i_arr, O2c),
        np.full_like(i_arr, CO2c),
        np.full_like(i_arr, N2c),
        i_arr,
        np.full_like(i_arr, r_1)
    ])


def voltaje_pls(i_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r_1):
    """Predice voltaje usando el modelo PLS."""
    if not _PLS_OK:
        return None
    X = _construir_X(i_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r_1)
    return _pls_pipe.predict(X).ravel()


def voltaje_kpls(i_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r_1):
    """Predice voltaje usando el modelo KPLS."""
    if not _KPLS_OK:
        return None
    X = _construir_X(i_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r_1)
    return _kpls_pipe.predict(X).ravel()

def get_data():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql('''
        SELECT e.id_experimento,
               e.fuente,
               e.t    AS "T",
               e.h2a  AS "H2a",  e.h2oa AS "H2Oa", e.co2a AS "CO2a",
               e.n2a  AS "N2a",  e.co   AS "CO",   e.ch4  AS "CH4",
               e.o2c  AS "O2c",  e.co2c AS "CO2c",
               e.n2c  AS "N2c",  e.h2oc AS "H2Oc",
               p.e_max AS "E_max", p.i_max AS "i_max",
               p.r_1, p.r_2,
               m.i_densidad, m.voltaje, m.eta
        FROM experimentos e
        JOIN parametros_modelo p USING(id_experimento)
        JOIN mediciones m USING(id_experimento)
    ''', conn)
    conn.close()
    return df

df = get_data()
temps = sorted(df['T'].unique())
exp_summary = df.drop_duplicates('id_experimento')[
    ['id_experimento', 'fuente', 'T',
     'H2a', 'H2Oa', 'CO2a', 'N2a', 'CO', 'CH4',
     'O2c', 'CO2c', 'N2c', 'H2Oc',
     'E_max', 'i_max', 'r_1', 'r_2']
].reset_index(drop=True)

# ── Opciones del dropdown del optimizador (calculadas al arrancar) ─────────────
_OPT_OPTIONS = [
    {
        'label': (f"Exp {row.id_experimento} | T={int(row.T)}°C | "
                  f"H2a={row.H2a:.2f} | CO2a={row.CO2a:.2f} | "
                  f"O2c={row.O2c:.2f} | r1={row.r_1:.3f}"),
        'value': int(row.id_experimento)
    }
    for row in exp_summary.sort_values('id_experimento').itertuples()
]
_OPT_DEFAULT = _OPT_OPTIONS[0]['value'] if _OPT_OPTIONS else None

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Digital Twin MCFC"

SL = {'fontFamily': 'Arial', 'fontWeight': 'bold', 'marginTop': '12px'}
SP = {'padding': '20px', 'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top'}
SG = {'width': '70%', 'display': 'inline-block'}

# ── Estilos comunes ────────────────────────────────────────────────────────────
CARD_STYLE = {
    'backgroundColor': 'white',
    'borderRadius': '10px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
    'padding': '16px',
    'marginBottom': '12px'
}

app.layout = html.Div([

    html.H1("Digital Twin — Celda MCFC",
            style={'textAlign': 'center', 'fontFamily': 'Arial',
                   'color': '#2c3e50', 'marginBottom': '6px'}),
    html.P("Plataforma de analisis operacional — Universidad de Concepcion",
           style={'textAlign': 'center', 'fontFamily': 'Arial',
                  'color': '#7f8c8d', 'marginBottom': '24px'}),

    dcc.Tabs(style={'fontFamily': 'Arial'}, children=[

        # ── TAB 1: Curvas de polarizacion ──────────────────────────
        dcc.Tab(label='Curvas de Polarizacion', children=[
            html.Div([
                html.Div([
                    html.Label('Temperatura (°C)', style=SL),
                    dcc.Checklist(
                        id='temp-filter',
                        options=[{'label': f' {t} °C', 'value': t} for t in temps],
                        value=[650], inline=True,
                        style={'fontFamily': 'Arial', 'marginBottom': '12px'}
                    ),
                    html.Label('Numero de curvas', style=SL),
                    dcc.Slider(id='n-curvas', min=1, max=20, step=1, value=5,
                               marks={i: str(i) for i in [1, 5, 10, 15, 20]}),
                    html.Label('Mostrar curva de potencia', style=SL),
                    dcc.Checklist(
                        id='show-power',
                        options=[{'label': ' P = V·i [W/cm²]', 'value': 'power'}],
                        value=[],
                        style={'fontFamily': 'Arial', 'marginTop': '6px'}
                    ),
                ], style=SP),
                html.Div([
                    dcc.Graph(id='polar-graph', style={'height': '520px'})
                ], style=SG)
            ])
        ]),

        # ── TAB 2: Efecto de variables ─────────────────────────────
        dcc.Tab(label='Efecto de Variables', children=[
            html.Div([
                html.Div([
                    html.Label('Variable a analizar', style=SL),
                    dcc.Dropdown(
                        id='var-selector',
                        options=[
                            {'label': 'H2 anodo (H2a)',    'value': 'H2a'},
                            {'label': 'CO2 anodo (CO2a)',  'value': 'CO2a'},
                            {'label': 'H2O anodo (H2Oa)',  'value': 'H2Oa'},
                            {'label': 'O2 catodo (O2c)',   'value': 'O2c'},
                            {'label': 'CO2 catodo (CO2c)', 'value': 'CO2c'},
                            {'label': 'N2 catodo (N2c)',   'value': 'N2c'},
                        ],
                        value='H2a', clearable=False,
                        style={'fontFamily': 'Arial'}
                    ),
                ], style=SP),
                html.Div([
                    dcc.Graph(id='var-graph', style={'height': '520px'})
                ], style=SG)
            ])
        ]),

        # ── TAB 3: Tabla de experimentos ───────────────────────────
        dcc.Tab(label='Experimentos', children=[
            html.Div([
                # Filtros superiores
                html.Div([
                    html.Div([
                        html.Label('Fuente de datos', style=SL),
                        dcc.Dropdown(
                            id='exp-fuente-filter',
                            options=[
                                {'label': 'Todas las fuentes',   'value': 'todas'},
                                {'label': 'Warsaw (Polonia)',     'value': 'warsaw_ut'},
                                {'label': 'Simulador UdeC',      'value': 'udec_lab'},
                                {'label': 'Laboratorio UdeC',    'value': 'udec_real'},
                            ],
                            value='todas', clearable=False,
                            style={'fontFamily': 'Arial', 'width': '220px'}
                        ),
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label('Temperatura (°C)', style=SL),
                        dcc.Dropdown(
                            id='exp-temp-filter',
                            options=[{'label': 'Todas', 'value': 'todas'}] +
                                    [{'label': f'{t}°C', 'value': t}
                                     for t in sorted(df['T'].unique())],
                            value='todas', clearable=False,
                            style={'fontFamily': 'Arial', 'width': '160px'}
                        ),
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label('Experimentos por página', style=SL),
                        dcc.Dropdown(
                            id='exp-pagesize',
                            options=[
                                {'label': '20',   'value': 20},
                                {'label': '50',   'value': 50},
                                {'label': '100',  'value': 100},
                                {'label': 'Todos','value': 9999},
                            ],
                            value=20, clearable=False,
                            style={'fontFamily': 'Arial', 'width': '160px'}
                        ),
                    ], style={'display': 'inline-block'}),
                ], style={'padding': '16px 20px 8px 20px',
                          'backgroundColor': '#f8f9fa', 'borderRadius': '8px',
                          'margin': '16px 20px 8px 20px'}),

                # Contador
                html.Div(id='exp-contador', style={
                    'fontFamily': 'Arial', 'fontSize': '12px',
                    'color': '#7f8c8d', 'margin': '4px 20px 8px 20px'
                }),

                # Tabla
                dash_table.DataTable(
                    id='exp-table',
                    columns=[
                        {'name': 'ID',           'id': 'id_experimento'},
                        {'name': 'Fuente',       'id': 'fuente'},
                        {'name': 'T (°C)',        'id': 'T'},
                        {'name': 'H2a',           'id': 'H2a'},
                        {'name': 'CO2a',          'id': 'CO2a'},
                        {'name': 'O2c',           'id': 'O2c'},
                        {'name': 'CO2c',          'id': 'CO2c'},
                        {'name': 'E_max (V)',     'id': 'E_max'},
                        {'name': 'i_max (A/cm2)', 'id': 'i_max'},
                        {'name': 'r1 (Ohm·cm2)', 'id': 'r_1'},
                    ],
                    data=exp_summary[[
                        'id_experimento', 'fuente', 'T', 'H2a', 'CO2a',
                        'O2c', 'CO2c', 'E_max', 'i_max', 'r_1'
                    ]].round(4).to_dict('records'),
                    filter_action='native', sort_action='native',
                    page_size=20,
                    page_action='native',
                    style_table={'overflowX': 'auto', 'margin': '0 20px 20px 20px'},
                    style_header={'backgroundColor': '#2c3e50', 'color': 'white',
                                  'fontWeight': 'bold', 'fontFamily': 'Arial'},
                    style_cell={'fontFamily': 'Arial', 'fontSize': '13px', 'padding': '8px'},
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'},
                        {'if': {'filter_query': '{fuente} = "udec_lab"'},
                         'backgroundColor': '#eafaf1', 'color': '#1e8449'},
                        {'if': {'filter_query': '{fuente} = "udec_real"'},
                         'backgroundColor': '#eaf4fb', 'color': '#1a5276'},
                    ]
                )
            ])
        ]),

        # ── TAB 4: Digital Twin (unificado) ───────────────────────
        dcc.Tab(label='Digital Twin', children=[
            html.Div([

                dcc.Interval(id='dt-sim-interval', interval=5_000,
                             n_intervals=0, disabled=True),

                # Store para guardar el id del experimento simulado activo
                dcc.Store(id='dt-sim-exp-id', data=None),

                # Store para guardar el último exp_id reproducido (persiste al detener)
                dcc.Store(id='dt-last-exp-id', data=None),

                # ══ BANNER DE ESTADO ══════════════════════════════════════════
                html.Div([
                    html.Div([
                        html.Div(id='dt-banner-identidad',
                                 style={'fontWeight': 'bold', 'fontSize': '15px',
                                        'color': 'white'}),
                        html.Div(id='dt-banner-detalle',
                                 style={'fontSize': '12px', 'color': '#d5f5e3',
                                        'marginTop': '2px'}),
                    ], style={'flex': '1'}),
                    html.Div(id='dt-banner-semaforo',
                             style={'fontSize': '28px', 'marginLeft': '16px'}),
                ], style={
                    'display': 'flex', 'alignItems': 'center',
                    'backgroundColor': '#1abc9c', 'borderRadius': '10px',
                    'padding': '12px 20px', 'marginBottom': '16px',
                }),

                # ══ FILA CENTRAL: Panel control + Gemelo digital ══════════════
                html.Div([

                    # ── Panel de control (izquierda) ──────────────────────────
                    html.Div([
                        html.Div([

                            # Selector de experimento
                            html.H4("Panel de control",
                                    style={'fontFamily': 'Arial', 'color': '#2c3e50',
                                           'marginBottom': '10px', 'marginTop': '0px'}),
                            html.Label('Experimento', style=SL),
                            dcc.Dropdown(
                                id='dt-exp-selector',
                                options=_OPT_OPTIONS,
                                value=_OPT_DEFAULT,
                                clearable=False,
                                style={'fontFamily': 'Arial', 'fontSize': '12px',
                                       'marginTop': '4px', 'marginBottom': '8px'}
                            ),
                            html.Button(
                                '↓ Cargar condiciones',
                                id='btn-cargar-exp',
                                n_clicks=0,
                                style={
                                    'width': '100%', 'padding': '8px',
                                    'backgroundColor': '#2980b9',
                                    'color': 'white', 'border': 'none',
                                    'borderRadius': '6px', 'fontFamily': 'Arial',
                                    'fontSize': '12px', 'cursor': 'pointer',
                                    'marginBottom': '4px'
                                }
                            ),
                            html.Div(id='btn-feedback', style={
                                'fontFamily': 'Arial', 'fontSize': '11px',
                                'color': '#27ae60', 'marginBottom': '8px',
                                'textAlign': 'center'
                            }),

                            # Sliders de condiciones
                            html.Div([
                                html.Label('Condiciones (ajustables)',
                                           style={**SL, 'marginTop': '4px',
                                                  'fontSize': '12px', 'color': '#555'}),
                                html.Label('T (°C)', style={**SL, 'fontSize': '11px'}),
                                dcc.Slider(id='dt-T', min=550, max=650, step=25,
                                           value=650,
                                           marks={t: f'{t}' for t in [550,575,600,625,650]},
                                           tooltip={'placement': 'bottom',
                                                    'always_visible': False}),
                                html.Label('H2a', style={**SL, 'fontSize': '11px'}),
                                dcc.Slider(id='dt-H2a', min=0.5, max=4.5, step=0.1,
                                           value=2.2, marks=None,
                                           tooltip={'placement': 'bottom',
                                                    'always_visible': False}),
                                html.Label('H2Oa', style={**SL, 'fontSize': '11px'}),
                                dcc.Slider(id='dt-H2Oa', min=0.05, max=1.3, step=0.05,
                                           value=0.41, marks=None,
                                           tooltip={'placement': 'bottom',
                                                    'always_visible': False}),
                                html.Label('CO2a', style={**SL, 'fontSize': '11px'}),
                                dcc.Slider(id='dt-CO2a', min=0.05, max=1.1, step=0.05,
                                           value=0.55, marks=None,
                                           tooltip={'placement': 'bottom',
                                                    'always_visible': False}),
                                html.Label('O2c', style={**SL, 'fontSize': '11px'}),
                                dcc.Slider(id='dt-O2c', min=0.1, max=5.3, step=0.1,
                                           value=1.3, marks=None,
                                           tooltip={'placement': 'bottom',
                                                    'always_visible': False}),
                                html.Label('CO2c', style={**SL, 'fontSize': '11px'}),
                                dcc.Slider(id='dt-CO2c', min=0.3, max=14.0, step=0.2,
                                           value=2.15, marks=None,
                                           tooltip={'placement': 'bottom',
                                                    'always_visible': False}),
                                html.Label('N2c', style={**SL, 'fontSize': '11px'}),
                                dcc.Slider(id='dt-N2c', min=0.5, max=29.0, step=0.5,
                                           value=4.87, marks=None,
                                           tooltip={'placement': 'bottom',
                                                    'always_visible': False}),
                                html.Label('r₁ (Ω·cm²)', style={**SL, 'fontSize': '11px'}),
                                dcc.Slider(id='dt-r1', min=1.8, max=3.0, step=0.05,
                                           value=1.97, marks=None,
                                           tooltip={'placement': 'bottom',
                                                    'always_visible': False}),
                                html.Label('i (A/cm²)', style={**SL, 'fontSize': '11px'}),
                                dcc.Slider(id='dt-corriente', min=0.005, max=0.200,
                                           step=0.005, value=0.100, marks=None,
                                           tooltip={'placement': 'bottom',
                                                    'always_visible': False}),
                            ], style={'backgroundColor': '#f8f9fa', 'borderRadius': '8px',
                                      'padding': '10px', 'marginBottom': '10px'}),

                            # Selector de modelo
                            html.Label('Modelo', style={**SL, 'marginTop': '4px'}),
                            dcc.RadioItems(
                                id='dt-modelo',
                                options=[
                                    {'label': ' Nernst', 'value': 'nernst'},
                                    {'label': ' PLS',    'value': 'pls'},
                                    {'label': ' KPLS',   'value': 'kpls'},
                                    {'label': ' Todos',  'value': 'ambos'},
                                ],
                                value='ambos',
                                style={'fontFamily': 'Arial', 'fontSize': '12px',
                                       'color': '#2c3e50'},
                                labelStyle={'display': 'inline-block',
                                            'marginRight': '8px'}
                            ),

                            # Selector de modo
                            html.Label('Modo', style={**SL, 'marginTop': '8px'}),
                            dcc.RadioItems(
                                id='dt-modo',
                                options=[
                                    {'label': ' Continuo', 'value': 'continuo'},
                                    {'label': ' Curva',    'value': 'curva'},
                                ],
                                value='continuo',
                                style={'fontFamily': 'Arial', 'fontSize': '12px',
                                       'color': '#2c3e50'},
                                labelStyle={'display': 'inline-block',
                                            'marginRight': '12px'}
                            ),

                            # Botones iniciar / detener
                            html.Div([
                                html.Button('▶ Iniciar', id='btn-iniciar-sim',
                                            n_clicks=0,
                                            style={
                                                'marginTop': '12px', 'width': '48%',
                                                'padding': '9px',
                                                'backgroundColor': '#27ae60',
                                                'color': 'white', 'border': 'none',
                                                'borderRadius': '6px',
                                                'fontFamily': 'Arial', 'fontSize': '12px',
                                                'cursor': 'pointer', 'fontWeight': 'bold'
                                            }),
                                html.Button('■ Detener', id='btn-detener-sim',
                                            n_clicks=0,
                                            style={
                                                'marginTop': '12px', 'width': '48%',
                                                'marginLeft': '4%', 'padding': '9px',
                                                'backgroundColor': '#e74c3c',
                                                'color': 'white', 'border': 'none',
                                                'borderRadius': '6px',
                                                'fontFamily': 'Arial', 'fontSize': '12px',
                                                'cursor': 'pointer', 'fontWeight': 'bold'
                                            }),
                            ], style={'display': 'flex'}),

                            # Estado del simulador
                            html.Div(id='sim-status', style={
                                'fontFamily': 'Arial', 'fontSize': '11px',
                                'marginTop': '8px', 'textAlign': 'center',
                                'padding': '6px', 'borderRadius': '6px',
                                'backgroundColor': '#f8f9fa', 'color': '#7f8c8d'
                            }),

                        ], style={**CARD_STYLE, 'padding': '14px'})
                    ], style={'width': '26%', 'display': 'inline-block',
                              'verticalAlign': 'top'}),

                    # ── Gemelo digital (derecha) ──────────────────────────────
                    html.Div([

                        # Curva DT + métricas
                        html.Div([
                            html.Div([
                                dcc.Graph(id='dt-graph', style={'height': '340px'})
                            ], style={'width': '65%', 'display': 'inline-block',
                                      'verticalAlign': 'top'}),
                            html.Div([
                                html.Div(id='dt-metricas', style={
                                    'fontFamily': 'Arial', 'fontSize': '13px',
                                    'height': '340px', 'overflowY': 'auto'
                                })
                            ], style={'width': '33%', 'display': 'inline-block',
                                      'verticalAlign': 'top', 'marginLeft': '2%'}),
                        ], style=CARD_STYLE),

                        # Variables en tiempo real
                        html.Div([
                            html.H5("Variables en tiempo real",
                                    style={'fontFamily': 'Arial', 'color': '#2c3e50',
                                           'marginBottom': '10px', 'marginTop': '0px'}),
                            html.Div(id='sim-kpis', style={
                                'display': 'flex', 'gap': '8px',
                                'marginBottom': '10px'
                            }),
                            html.Div([
                                html.Div([
                                    dcc.Graph(id='sim-voltaje-graph',
                                              style={'height': '180px'})
                                ], style={'width': '49%', 'display': 'inline-block'}),
                                html.Div([
                                    dcc.Graph(id='sim-potencia-graph',
                                              style={'height': '180px'})
                                ], style={'width': '49%', 'display': 'inline-block',
                                          'marginLeft': '2%'}),
                            ]),
                        ], style=CARD_STYLE),

                    ], style={'width': '72%', 'display': 'inline-block',
                              'verticalAlign': 'top', 'paddingLeft': '14px'}),

                ], style={'display': 'flex', 'alignItems': 'flex-start',
                          'marginBottom': '16px'}),

                # ══ ZONA OPERACIÓN ÓPTIMA ═════════════════════════════════════
                html.Div([
                    html.Div([
                        html.H4("Operación óptima",
                                style={'fontFamily': 'Arial', 'color': '#2c3e50',
                                       'marginBottom': '4px', 'marginTop': '0px'}),
                        html.P("Para las condiciones activas, calcula j* que maximiza "
                               "p = V · j dentro del rango experimental [0.005, 0.200] A/cm².",
                               style={'fontFamily': 'Arial', 'color': '#7f8c8d',
                                      'fontSize': '12px', 'marginBottom': '12px'}),
                        html.Div([
                            # Resultados tabla
                            html.Div([
                                html.Div(id='opt-resultados',
                                         style={'fontFamily': 'Arial'})
                            ], style={'width': '42%', 'display': 'inline-block',
                                      'verticalAlign': 'top'}),
                            # Gráfico p(j)
                            html.Div([
                                dcc.Graph(id='opt-graph', style={'height': '280px'})
                            ], style={'width': '56%', 'display': 'inline-block',
                                      'verticalAlign': 'top', 'marginLeft': '2%'}),
                        ]),
                    ], style=CARD_STYLE),
                ]),

            ], style={'padding': '16px'})
        ]),

    ])
], style={'maxWidth': '1280px', 'margin': 'auto', 'padding': '20px'})


# ── Callback Tab 3: Filtros de experimentos ───────────────────────────────────
@app.callback(
    Output('exp-table',    'data'),
    Output('exp-table',    'page_size'),
    Output('exp-contador', 'children'),
    Input('exp-fuente-filter', 'value'),
    Input('exp-temp-filter',   'value'),
    Input('exp-pagesize',      'value'),
)
def filtrar_experimentos(fuente, temp, page_size):
    sub = exp_summary.copy()

    # Filtro por fuente
    if fuente == 'warsaw_ut':
        sub = sub[sub['fuente'] == 'warsaw_ut']
    elif fuente == 'udec_lab':
        sub = sub[sub['fuente'] == 'udec_lab']
    elif fuente == 'udec_real':
        sub = sub[sub['fuente'] == 'udec_real']

    # Filtro por temperatura
    if temp != 'todas':
        sub = sub[sub['T'] == temp]

    total = len(sub)
    data = sub[['id_experimento', 'fuente', 'T', 'H2a', 'CO2a',
                'O2c', 'CO2c', 'E_max', 'i_max', 'r_1']].round(4).to_dict('records')

    # Fuente legible
    fuente_label = {
        'todas':      'todas las fuentes',
        'warsaw_ut':  'Warsaw (Polonia)',
        'udec_lab':   'Simulador UdeC',
        'udec_real':  'Laboratorio UdeC',
    }.get(fuente, fuente)

    contador = f"Mostrando {total} experimento{'s' if total != 1 else ''} — {fuente_label}"
    if temp != 'todas':
        contador += f" — T={temp}°C"

    ps = min(page_size, total) if page_size < 9999 else total
    return data, ps, contador


# ── Callbacks Tab 1 ────────────────────────────────────────────────────────────
@app.callback(
    Output('polar-graph', 'figure'),
    Input('temp-filter', 'value'),
    Input('n-curvas', 'value'),
    Input('show-power', 'value')
)
def update_polar(temps_sel, n_curvas, show_power):
    fig = go.Figure()
    if not temps_sel:
        return fig
    mostrar_p = 'power' in (show_power or [])
    subset = df[df['T'].isin(temps_sel)]
    ids = subset['id_experimento'].unique()[:n_curvas]
    for exp_id in ids:
        curva = subset[subset['id_experimento'] == exp_id].sort_values('i_densidad')
        t_val = int(curva['T'].iloc[0])
        nombre = f'Exp {exp_id} ({t_val}°C)'
        fig.add_trace(go.Scatter(
            x=curva['i_densidad'], y=curva['voltaje'],
            mode='lines+markers', name=nombre,
            marker=dict(size=5), yaxis='y1'
        ))
        if mostrar_p:
            potencia = curva['voltaje'] * curva['i_densidad']
            fig.add_trace(go.Scatter(
                x=curva['i_densidad'], y=potencia,
                mode='lines', name=f'P — {nombre}',
                line=dict(dash='dash'), yaxis='y2'
            ))
    layout = dict(
        title='Curvas de Polarizacion' + (' y Potencia' if mostrar_p else ' (V vs i)'),
        xaxis_title='Densidad de corriente i [A/cm2]',
        yaxis=dict(title='Voltaje E [V]', side='left'),
        hovermode='x unified', template='plotly_white',
        font=dict(family='Arial'), legend=dict(orientation='v', x=1.08)
    )
    if mostrar_p:
        layout['yaxis2'] = dict(title='Potencia P [W/cm2]',
                                overlaying='y', side='right', showgrid=False)
    fig.update_layout(**layout)
    return fig


# ── Callback Tab 2 ─────────────────────────────────────────────────────────────
@app.callback(
    Output('var-graph', 'figure'),
    Input('var-selector', 'value')
)
def update_var(var):
    fig = go.Figure()
    if not var:
        return fig
    summary = df.groupby(['id_experimento', var, 'T'])['voltaje'].max().reset_index()
    summary.columns = ['id_experimento', 'var_val', 'T', 'E_max']
    fig = px.scatter(
        summary, x='var_val', y='E_max', color='T',
        labels={'var_val': var, 'E_max': 'Voltaje maximo E_max [V]', 'T': 'T (°C)'},
        title=f'Efecto de {var} sobre E_max',
        template='plotly_white', color_continuous_scale='RdYlGn'
    )
    fig.update_traces(marker=dict(size=9))
    fig.update_layout(font=dict(family='Arial'))
    return fig


# ── Callbacks Tab 4: Digital Twin con simulacion ───────────────────────────────

# Iniciar / Detener simulador
@app.callback(
    Output('dt-sim-interval', 'disabled'),
    Output('dt-sim-exp-id',   'data'),
    Output('dt-last-exp-id',  'data'),
    Output('sim-status',      'children'),
    Output('sim-status',      'style'),
    Input('btn-iniciar-sim',  'n_clicks'),
    Input('btn-detener-sim',  'n_clicks'),
    State('dt-exp-selector',  'value'),
    State('dt-modo',          'value'),
    State('dt-sim-exp-id',    'data'),
    State('dt-last-exp-id',   'data'),
    prevent_initial_call=True
)
def controlar_simulador(n_ini, n_det, exp_id_sel, modo,
                         exp_id_actual, last_exp_id):
    from dash import ctx
    triggered = ctx.triggered_id

    status_base = {
        'fontFamily': 'Arial', 'fontSize': '12px',
        'marginTop': '10px', 'textAlign': 'center',
        'padding': '8px', 'borderRadius': '6px',
    }

    if triggered == 'btn-iniciar-sim':
        if exp_id_sel is None:
            style = {**status_base,
                     'backgroundColor': '#fde8e8', 'color': '#e74c3c'}
            return True, None, last_exp_id, \
                   'Selecciona un experimento primero', style

        # Leer info del experimento para el label
        fila = exp_summary[exp_summary['id_experimento'] == exp_id_sel]
        T_label   = int(fila.iloc[0]['T']) if not fila.empty else '?'
        modo_label = modo if modo else 'curva'

        style = {**status_base,
                 'backgroundColor': '#d5f5e3', 'color': '#1e8449'}
        label = (f'● Reproduciendo Exp {exp_id_sel} (Milewski) — '
                 f'T={T_label}°C | modo {modo_label}')

        # Usar directamente el experimento original de Milewski
        return False, exp_id_sel, exp_id_sel, label, style

    elif triggered == 'btn-detener-sim':
        style = {**status_base,
                 'backgroundColor': '#f8f9fa', 'color': '#7f8c8d'}
        ultimo = exp_id_actual or last_exp_id
        label  = '■ Simulación detenida'
        if ultimo:
            label += f' — último exp: Exp {ultimo} (Milewski)'
        return True, None, ultimo, label, style

    style = {**status_base, 'backgroundColor': '#f8f9fa', 'color': '#7f8c8d'}
    return True, None, last_exp_id, 'Sin simulación activa', style



# ── Callback: cargar experimento desde dropdown ────────────────────────────────
# ── Callback: cargar experimento desde dropdown ────────────────────────────────
@app.callback(
    Output('dt-T',       'value'),
    Output('dt-H2a',     'value'),
    Output('dt-H2Oa',    'value'),
    Output('dt-CO2a',    'value'),
    Output('dt-O2c',     'value'),
    Output('dt-CO2c',    'value'),
    Output('dt-N2c',     'value'),
    Output('dt-r1',      'value'),
    Output('btn-feedback','children'),
    Input('btn-cargar-exp',   'n_clicks'),
    State('dt-exp-selector',  'value'),
    prevent_initial_call=True
)
def cargar_experimento_desde_dropdown(n_clicks, exp_id):
    if exp_id is None:
        return (650, 2.2, 0.41, 0.55, 1.3, 2.15, 4.87, 1.97,
                'Selecciona un experimento')
    fila = exp_summary[exp_summary['id_experimento'] == exp_id]
    if fila.empty:
        return (650, 2.2, 0.41, 0.55, 1.3, 2.15, 4.87, 1.97,
                'Experimento no encontrado')
    row = fila.iloc[0]
    return (
        int(row['T']),
        round(float(row['H2a']),  3),
        round(float(row['H2Oa']), 3),
        round(float(row['CO2a']), 3),
        round(float(row['O2c']),  3),
        round(float(row['CO2c']), 3),
        round(float(row['N2c']),  3),
        round(float(row['r_1']),  3),
        f"✓ Cargado: Exp {exp_id} (T={int(row['T'])}°C)"
    )




# ── Callback: banner de estado ─────────────────────────────────────────────────
@app.callback(
    Output('dt-banner-identidad', 'children'),
    Output('dt-banner-detalle',   'children'),
    Output('dt-banner-semaforo',  'children'),
    Input('dt-sim-interval', 'n_intervals'),
    Input('dt-sim-exp-id',   'data'),
    State('dt-T',     'value'),
    State('dt-modelo','value'),
)
def actualizar_banner(n, exp_id_sim, T, modelo_sel):
    modelo_label = {
        'nernst': 'Nernst (físico)',
        'pls':    'PLS (datos)',
        'kpls':   'KPLS (kernel)',
        'ambos':  'Todos los modelos',
    }.get(modelo_sel or 'ambos', 'Todos los modelos')

    identidad = "Digital Twin — Celda MCFC · Universidad de Concepción"

    if exp_id_sim is not None:
        ts = datetime.now().strftime('%H:%M:%S')
        detalle = (f"Exp {exp_id_sim} activo · T={T}°C · "
                   f"Modelo: {modelo_label} · Última actualización: {ts}")
        semaforo = "🟢"
    else:
        detalle = f"Sin simulación activa · T={T}°C · Modelo: {modelo_label}"
        semaforo = "🔴"

    return identidad, detalle, semaforo


# ── Callback: Digital Twin principal (curvas + métricas + KPIs + tiempo real) ──
@app.callback(
    Output('dt-graph',           'figure'),
    Output('dt-metricas',        'children'),
    Output('sim-kpis',           'children'),
    Output('sim-voltaje-graph',  'figure'),
    Output('sim-potencia-graph', 'figure'),
    Output('opt-resultados',     'children'),
    Output('opt-graph',          'figure'),
    Input('dt-sim-interval', 'n_intervals'),
    Input('dt-T',    'value'),
    Input('dt-H2a',  'value'),
    Input('dt-H2Oa', 'value'),
    Input('dt-CO2a', 'value'),
    Input('dt-O2c',  'value'),
    Input('dt-CO2c', 'value'),
    Input('dt-N2c',  'value'),
    Input('dt-r1',   'value'),
    Input('dt-sim-exp-id', 'data'),
    Input('dt-modelo',     'value'),
    State('dt-exp-selector',  'value'),
    State('dt-last-exp-id',   'data'),
)
def actualizar_dt_completo(n, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1,
                            exp_id_sim, modelo_sel, exp_id_dropdown,
                            last_exp_id):

    fig_empty = go.Figure()
    fig_empty.update_layout(
        template='plotly_white', font=dict(family='Arial'),
        annotations=[dict(text='Inicia la simulación para ver datos',
                         showarrow=False, font=dict(size=13, color='#aaa'),
                         xref='paper', yref='paper', x=0.5, y=0.5)]
    )

    card_kpi = {
        'display': 'inline-block', 'textAlign': 'center',
        'width': '22%', 'padding': '10px',
        'backgroundColor': 'white', 'borderRadius': '10px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.08)',
    }

    def kpi(titulo, valor, unidad, color='#2c3e50'):
        return html.Div([
            html.Div(titulo, style={'fontWeight': 'bold', 'fontSize': '11px',
                                    'color': '#7f8c8d', 'marginBottom': '4px'}),
            html.Div(valor,  style={'fontSize': '20px', 'fontWeight': 'bold',
                                    'color': color}),
            html.Div(unidad, style={'fontSize': '10px', 'color': '#aaa'}),
        ], style=card_kpi)

    # ── Curva DT (responde a sliders siempre) ─────────────────────────────────
    i_range   = np.linspace(0.005, 0.35, 150)
    modelo_sel = modelo_sel or 'ambos'
    fig_dt    = go.Figure()

    if modelo_sel in ('nernst', 'ambos'):
        V_nernst = voltaje_modelo(i_range, T, H2a, H2Oa, CO2a, O2c, CO2c, r1)
        P_nernst = V_nernst * i_range
        fig_dt.add_trace(go.Scatter(x=i_range, y=V_nernst, mode='lines',
            line=dict(color='#e74c3c', width=2.5, dash='dot'),
            name='Nernst (físico)', yaxis='y1'))
        fig_dt.add_trace(go.Scatter(x=i_range, y=P_nernst, mode='lines',
            line=dict(color='#8e44ad', width=2, dash='dash'),
            name='P Nernst', yaxis='y2'))
        idx_n = int(np.argmax(P_nernst))
        fig_dt.add_trace(go.Scatter(
            x=[i_range[idx_n]], y=[P_nernst[idx_n]], mode='markers+text',
            marker=dict(color='#8e44ad', size=12, symbol='star'),
            text=[f"Pmax={P_nernst[idx_n]:.4f}"], textposition='top center',
            textfont=dict(size=9, color='#8e44ad'),
            name='Pmax Nernst', yaxis='y2', showlegend=False))

    if modelo_sel in ('pls', 'ambos') and _PLS_OK:
        V_pls = voltaje_pls(i_range, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1)
        if V_pls is not None:
            P_pls = V_pls * i_range
            fig_dt.add_trace(go.Scatter(x=i_range, y=V_pls, mode='lines',
                line=dict(color='#2980b9', width=2.5), name='PLS (datos)', yaxis='y1'))
            fig_dt.add_trace(go.Scatter(x=i_range, y=P_pls, mode='lines',
                line=dict(color='#16a085', width=2, dash='dash'),
                name='P PLS', yaxis='y2'))
            idx_p = int(np.argmax(P_pls))
            fig_dt.add_trace(go.Scatter(
                x=[i_range[idx_p]], y=[P_pls[idx_p]], mode='markers+text',
                marker=dict(color='#16a085', size=12, symbol='star'),
                text=[f"Pmax={P_pls[idx_p]:.4f}"], textposition='top center',
                textfont=dict(size=9, color='#16a085'),
                name='Pmax PLS', yaxis='y2', showlegend=False))

    if modelo_sel in ('kpls', 'ambos') and _KPLS_OK:
        V_kpls = voltaje_kpls(i_range, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1)
        if V_kpls is not None:
            P_kpls = V_kpls * i_range
            fig_dt.add_trace(go.Scatter(x=i_range, y=V_kpls, mode='lines',
                line=dict(color='#e67e22', width=2.5),
                name='KPLS (kernel)', yaxis='y1'))
            fig_dt.add_trace(go.Scatter(x=i_range, y=P_kpls, mode='lines',
                line=dict(color='#d35400', width=2, dash='dash'),
                name='P KPLS', yaxis='y2'))
            idx_k = int(np.argmax(P_kpls))
            fig_dt.add_trace(go.Scatter(
                x=[i_range[idx_k]], y=[P_kpls[idx_k]], mode='markers+text',
                marker=dict(color='#d35400', size=12, symbol='star'),
                text=[f"Pmax={P_kpls[idx_k]:.4f}"], textposition='top center',
                textfont=dict(size=9, color='#d35400'),
                name='Pmax KPLS', yaxis='y2', showlegend=False))

    metricas_out = html.P("Selecciona un experimento y carga las condiciones.",
                          style={'color': '#888', 'fontFamily': 'Arial',
                                 'fontSize': '12px'})
    kpis_out = []
    fig_v    = fig_empty
    fig_p    = fig_empty

    # ── Helpers de métricas ────────────────────────────────────────────────────
    def color_m(val, bueno, malo, inv=False):
        if val is None: return '#aaa'
        if not inv:
            return '#27ae60' if val >= bueno else (
                '#e67e22' if val >= malo else '#e74c3c')
        return '#27ae60' if val <= bueno else (
            '#e67e22' if val <= malo else '#e74c3c')

    card_met = {'display': 'inline-block', 'textAlign': 'center',
                'width': '30%', 'padding': '8px',
                'backgroundColor': 'white', 'borderRadius': '8px',
                'boxShadow': '0 1px 4px #ccc', 'marginRight': '4px'}

    def met_card(titulo, val_n, val_p, val_k, bueno, malo, inv=False):
        return html.Div([
            html.Div(titulo, style={'fontWeight': 'bold', 'fontSize': '11px',
                                    'marginBottom': '4px', 'color': '#555'}),
            html.Div("Nernst", style={'fontSize': '10px', 'color': '#e74c3c'}),
            html.Div(f"{val_n:.4f}" if val_n is not None else "—",
                     style={'fontSize': '14px', 'fontWeight': 'bold',
                            'color': color_m(val_n, bueno, malo, inv)}),
            html.Div("PLS", style={'fontSize': '10px', 'color': '#2980b9',
                                   'marginTop': '3px'}),
            html.Div(f"{val_p:.4f}" if val_p is not None else "—",
                     style={'fontSize': '14px', 'fontWeight': 'bold',
                            'color': color_m(val_p, bueno, malo, inv)}),
            html.Div("KPLS", style={'fontSize': '10px', 'color': '#e67e22',
                                    'marginTop': '3px'}),
            html.Div(f"{val_k:.4f}" if val_k is not None else "—",
                     style={'fontSize': '14px', 'fontWeight': 'bold',
                            'color': color_m(val_k, bueno, malo, inv)}),
        ], style=card_met)

    def calcular_metricas_bloque(df_puntos, T_m, H2a_m, H2Oa_m, CO2a_m,
                                  O2c_m, CO2c_m, N2c_m, r1_m,
                                  exp_id_label, sufijo=""):
        """Calcula y retorna el panel de métricas para un DataFrame de puntos."""
        if len(df_puntos) < 2:
            return html.P(
                f"Exp {exp_id_label}: acumulando puntos... ({len(df_puntos)}/2 mín.)",
                style={'color': '#e67e22', 'fontFamily': 'Arial',
                       'fontSize': '12px'})

        V_pred_n = voltaje_modelo(
            df_puntos['i_densidad'].values,
            T_m, H2a_m, H2Oa_m, CO2a_m, O2c_m, CO2c_m, r1_m)
        r2_n, mae_n, nrmse_n = metricas(df_puntos['voltaje'].values, V_pred_n)

        r2_p = mae_p = nrmse_p = None
        r2_k = mae_k = nrmse_k = None
        if _PLS_OK:
            V_pred_p = voltaje_pls(df_puntos['i_densidad'].values,
                                   T_m, H2a_m, H2Oa_m, CO2a_m,
                                   O2c_m, CO2c_m, N2c_m, r1_m)
            if V_pred_p is not None:
                r2_p, mae_p, nrmse_p = metricas(
                    df_puntos['voltaje'].values, V_pred_p)
        if _KPLS_OK:
            V_pred_k = voltaje_kpls(df_puntos['i_densidad'].values,
                                    T_m, H2a_m, H2Oa_m, CO2a_m,
                                    O2c_m, CO2c_m, N2c_m, r1_m)
            if V_pred_k is not None:
                r2_k, mae_k, nrmse_k = metricas(
                    df_puntos['voltaje'].values, V_pred_k)

        n = len(df_puntos)
        return html.Div([
            html.Div(
                f"Exp {exp_id_label}{sufijo} — {n} punto{'s' if n != 1 else ''}",
                style={'fontWeight': 'bold', 'fontSize': '11px',
                       'color': '#2c3e50', 'marginBottom': '8px',
                       'fontFamily': 'Arial'}),
            html.Div([
                met_card("R²",    r2_n, r2_p, r2_k, 0.95, 0.85),
                met_card("MAE",   mae_n, mae_p, mae_k, 0.02, 0.05, inv=True),
                met_card("NRMSE", nrmse_n, nrmse_p, nrmse_k,
                         0.05, 0.10, inv=True),
            ], style={'display': 'flex', 'flexWrap': 'nowrap'})
        ])

    # ── Leer condiciones del experimento de referencia (dropdown) ─────────────
    T_m = T; H2a_m = H2a; H2Oa_m = H2Oa; CO2a_m = CO2a
    O2c_m = O2c; CO2c_m = CO2c; N2c_m = N2c; r1_m = r1

    exp_id_ref = exp_id_sim if exp_id_sim is not None else exp_id_dropdown
    if exp_id_ref is not None:
        fila_ref = exp_summary[exp_summary['id_experimento'] == exp_id_ref]
        if not fila_ref.empty:
            row_ref = fila_ref.iloc[0]
            T_m     = float(row_ref['T'])
            H2a_m   = float(row_ref['H2a'])
            H2Oa_m  = float(row_ref['H2Oa'])
            CO2a_m  = float(row_ref['CO2a'])
            O2c_m   = float(row_ref['O2c'])
            CO2c_m  = float(row_ref['CO2c'])
            N2c_m   = float(row_ref['N2c'])
            r1_m    = float(row_ref['r_1'])

    # ── CASO 1: Simulación activa → reproducción directa del exp original ────
    if exp_id_sim is not None:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            df_todos = pd.read_sql("""
                SELECT i_densidad, voltaje, eta
                FROM mediciones WHERE id_experimento = %s
                ORDER BY i_densidad ASC
            """, conn, params=(exp_id_sim,))
            conn.close()

            total = len(df_todos)

            # n_intervals como índice: liberar de a 1 punto por tick
            # Empezar desde 1, llegar hasta total
            n_mostrar = min(max(n or 1, 1), total)
            df_m = df_todos.iloc[:n_mostrar].copy()

            if not df_m.empty:
                fig_dt.add_trace(go.Scatter(
                    x=df_m['i_densidad'], y=df_m['voltaje'],
                    mode='markers', name='Datos reales (Milewski)',
                    marker=dict(color='#27ae60', size=8, symbol='circle'),
                    yaxis='y1'))

                en_curso = n_mostrar < total
                sufijo   = (f" · en curso 🔄 ({n_mostrar}/{total})"
                            if en_curso
                            else f" · completo ✓ ({total} pts)")

                metricas_out = calcular_metricas_bloque(
                    df_m[['i_densidad', 'voltaje']],
                    T_m, H2a_m, H2Oa_m, CO2a_m,
                    O2c_m, CO2c_m, N2c_m, r1_m,
                    exp_id_sim, sufijo=sufijo)

                ultima  = df_m.iloc[-1]
                V_u     = float(ultima['voltaje'])
                i_u     = float(ultima['i_densidad'])
                P_u     = V_u * i_u
                eta_u   = float(ultima['eta']) if ultima['eta'] else 0.0
                kpis_out = [
                    kpi("Voltaje",      f"{V_u:.4f}", "V",     '#27ae60'),
                    kpi("Corriente",    f"{i_u:.4f}", "A/cm²", '#2980b9'),
                    kpi("Potencia",     f"{P_u:.4f}", "W/cm²", '#8e44ad'),
                    kpi("Eficiencia η", f"{eta_u:.3f}", "—",   '#16a085'),
                ]

                df_idx = df_m.reset_index(drop=True)
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(
                    x=df_idx.index + 1, y=df_idx['voltaje'],
                    mode='lines+markers',
                    line=dict(color='#27ae60', width=2),
                    marker=dict(size=5), name='Voltaje'))
                fig_v.update_layout(
                    title='Voltaje E [V]', xaxis_title='Punto N°',
                    yaxis_title='E [V]', template='plotly_white',
                    font=dict(family='Arial'),
                    margin=dict(t=30, b=40, l=50, r=20))

                df_idx['potencia'] = df_idx['voltaje'] * df_idx['i_densidad']
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(
                    x=df_idx.index + 1, y=df_idx['potencia'],
                    mode='lines+markers',
                    line=dict(color='#8e44ad', width=2),
                    marker=dict(size=5), name='Potencia'))
                fig_p.update_layout(
                    title='Potencia P [W/cm²]', xaxis_title='Punto N°',
                    yaxis_title='P [W/cm²]', template='plotly_white',
                    font=dict(family='Arial'),
                    margin=dict(t=30, b=40, l=50, r=20))

        except Exception:
            pass

    # ── CASO 2: Sin simulación → métricas del experimento completo ────────────
    else:
        # Prioridad: último exp reproducido > experimento del dropdown
        exp_id_metricas = last_exp_id if last_exp_id else exp_id_dropdown
        sufijo_label = (" · resultado final ✓"
                        if last_exp_id else " · referencia completa")

        if exp_id_metricas is not None:
            try:
                # Actualizar condiciones si el exp de métricas difiere del dropdown
                if exp_id_metricas != exp_id_dropdown:
                    fila_last = exp_summary[
                        exp_summary['id_experimento'] == exp_id_metricas]
                    if not fila_last.empty:
                        row_last = fila_last.iloc[0]
                        T_m    = float(row_last['T'])
                        H2a_m  = float(row_last['H2a'])
                        H2Oa_m = float(row_last['H2Oa'])
                        CO2a_m = float(row_last['CO2a'])
                        O2c_m  = float(row_last['O2c'])
                        CO2c_m = float(row_last['CO2c'])
                        N2c_m  = float(row_last['N2c'])
                        r1_m   = float(row_last['r_1'])

                conn_m = psycopg2.connect(**DB_CONFIG)
                df_real = pd.read_sql("""
                    SELECT i_densidad, voltaje
                    FROM mediciones
                    WHERE id_experimento = %s
                    ORDER BY i_densidad ASC
                """, conn_m, params=(exp_id_metricas,))
                conn_m.close()

                if not df_real.empty:
                    fig_dt.add_trace(go.Scatter(
                        x=df_real['i_densidad'], y=df_real['voltaje'],
                        mode='markers', name='Datos reales (Milewski)',
                        marker=dict(color='#27ae60', size=8, symbol='circle'),
                        yaxis='y1'))

                    metricas_out = calcular_metricas_bloque(
                        df_real,
                        T_m, H2a_m, H2Oa_m, CO2a_m,
                        O2c_m, CO2c_m, N2c_m, r1_m,
                        exp_id_metricas,
                        sufijo=sufijo_label
                    )
            except Exception as e:
                metricas_out = html.P(
                    f"Error: {e}",
                    style={'color': '#e74c3c', 'fontFamily': 'Arial',
                           'fontSize': '12px'})

    fig_dt.update_layout(
        title=f'Digital Twin — T={T}°C',
        xaxis_title='Densidad de corriente i [A/cm²]',
        yaxis=dict(title='Voltaje E [V]', side='left'),
        yaxis2=dict(title='Potencia P [W/cm²]', overlaying='y',
                    side='right', showgrid=False),
        hovermode='x unified', template='plotly_white',
        font=dict(family='Arial'), legend=dict(orientation='h', y=-0.25))

    # ── Optimizador integrado (lee los sliders actuales) ──────────────────────
    J_MIN, J_MAX  = 0.005, 0.200
    j_arr         = np.linspace(J_MIN, J_MAX, 200)
    fig_opt       = go.Figure()
    filas_opt     = []
    COLORES       = {'Nernst': '#e74c3c', 'PLS': '#2980b9', 'KPLS': '#27ae60'}

    def p_nernst(j):
        V = voltaje_modelo(j, T, H2a, H2Oa, CO2a, O2c, CO2c, r1, N2c=N2c)
        return max(float(V), 0.0) * float(j)

    def p_pls(j):
        V = float(voltaje_pls(np.array([float(j)]),
                              T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1)[0])
        return max(V, 0.0) * float(j)

    def p_kpls(j):
        V = float(voltaje_kpls(np.array([float(j)]),
                               T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1)[0])
        return max(V, 0.0) * float(j)

    FN = {'Nernst': p_nernst, 'PLS': p_pls, 'KPLS': p_kpls}

    for nombre in ['Nernst', 'PLS', 'KPLS']:
        if nombre == 'PLS'  and not _PLS_OK:  continue
        if nombre == 'KPLS' and not _KPLS_OK: continue
        try:
            fn    = FN[nombre]
            color = COLORES[nombre]
            p_arr = np.array([fn(j) for j in j_arr])
            res   = _minimize_scalar(lambda j: -fn(j), bounds=(J_MIN, J_MAX),
                                     method='bounded', options={'xatol': 1e-5})
            j_opt     = float(res.x)
            pmax      = round(fn(j_opt), 5)
            V_opt     = round(pmax / j_opt if j_opt > 0 else 0.0, 4)
            en_limite = abs(j_opt - J_MAX) < 1e-4

            fig_opt.add_trace(go.Scatter(
                x=j_arr, y=p_arr, mode='lines',
                name=f'p(j) {nombre}', line=dict(color=color, width=2)))
            fig_opt.add_trace(go.Scatter(
                x=[j_opt], y=[pmax], mode='markers',
                name=f'j* {nombre}',
                marker=dict(color=color, size=11, symbol='star',
                            line=dict(color='white', width=1))))
            filas_opt.append({
                'Modelo': nombre,
                'j* (A/cm²)': round(j_opt, 5),
                'V* (V)': V_opt,
                'Pmax (W/cm²)': pmax,
                'En límite': '⚠' if en_limite else '—',
            })
        except Exception:
            pass

    fig_opt.update_layout(
        title=f'p(j) — T={T}°C',
        xaxis_title='j [A/cm²]', yaxis_title='p [W/cm²]',
        template='plotly_white', font=dict(family='Arial'),
        hovermode='x unified', legend=dict(orientation='h', y=-0.25),
        margin=dict(t=40, b=60, l=50, r=20))

    cols_opt = ['Modelo', 'j* (A/cm²)', 'V* (V)', 'Pmax (W/cm²)', 'En límite']
    tabla_opt = dash_table.DataTable(
        columns=[{'name': c, 'id': c} for c in cols_opt],
        data=filas_opt,
        style_table={'fontFamily': 'Arial'},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white',
                      'fontWeight': 'bold', 'textAlign': 'center',
                      'fontSize': '12px'},
        style_cell={'textAlign': 'center', 'padding': '8px',
                    'fontFamily': 'Arial', 'fontSize': '12px'},
        style_data_conditional=[{
            'if': {'filter_query': '{En límite} = "⚠"'},
            'backgroundColor': '#fef9e7', 'color': '#d35400'
        }]
    )
    nota_opt = html.P(
        "⚠ j*=0.200: Pmax real podría estar fuera del rango validado.",
        style={'fontFamily': 'Arial', 'fontSize': '11px',
               'color': '#e67e22', 'marginTop': '6px'})

    return (fig_dt, metricas_out, kpis_out, fig_v, fig_p,
            html.Div([tabla_opt, nota_opt]), fig_opt)


if __name__ == '__main__':
    app.run(debug=True)
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


def _construir_X(i_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c):
    """Construye matriz de features para los modelos PLS/KPLS."""
    i_arr = np.atleast_1d(i_arr)
    return np.column_stack([
        np.full_like(i_arr, T),
        np.full_like(i_arr, H2a),
        np.full_like(i_arr, H2Oa),
        np.full_like(i_arr, CO2a),
        np.full_like(i_arr, O2c),
        np.full_like(i_arr, CO2c),
        np.full_like(i_arr, N2c),
        i_arr
    ])


def voltaje_pls(i_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c):
    """Predice voltaje usando el modelo PLS."""
    if not _PLS_OK:
        return None
    X = _construir_X(i_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c)
    return _pls_pipe.predict(X).ravel()


def voltaje_kpls(i_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c):
    """Predice voltaje usando el modelo KPLS."""
    if not _KPLS_OK:
        return None
    X = _construir_X(i_arr, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c)
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

        # ── TAB 4: Digital Twin ────────────────────────────────────
        dcc.Tab(label='Digital Twin', children=[
            html.Div([

                # Intervalo de auto-refresh para la simulacion
                dcc.Interval(id='dt-sim-interval', interval=5_000,
                             n_intervals=0, disabled=True),

                # Store para guardar el id del experimento simulado activo
                dcc.Store(id='dt-sim-exp-id', data=None),

                html.Div([

                    # ── Panel izquierdo: sliders + botones ────────
                    html.Div([
                        html.H4("Condiciones de operacion",
                                style={'fontFamily': 'Arial', 'color': '#2c3e50',
                                       'marginBottom': '8px'}),

                        html.Label('Temperatura T (°C)', style=SL),
                        dcc.Slider(id='dt-T', min=550, max=650, step=25, value=650,
                                   marks={t: f'{t}' for t in [550, 575, 600, 625, 650]}),

                        html.Label('H2 anodo (H2a)', style=SL),
                        dcc.Slider(id='dt-H2a', min=0.5, max=4.5, step=0.1, value=2.2,
                                   marks={v: str(v) for v in [0.5, 1.5, 2.5, 3.5, 4.5]}),

                        html.Label('H2O anodo (H2Oa)', style=SL),
                        dcc.Slider(id='dt-H2Oa', min=0.05, max=1.3, step=0.05, value=0.41,
                                   marks={v: str(round(v, 2)) for v in [0.05, 0.5, 1.0, 1.3]}),

                        html.Label('CO2 anodo (CO2a)', style=SL),
                        dcc.Slider(id='dt-CO2a', min=0.05, max=1.1, step=0.05, value=0.55,
                                   marks={v: str(round(v, 2)) for v in [0.05, 0.3, 0.6, 1.1]}),

                        html.Label('O2 catodo (O2c)', style=SL),
                        dcc.Slider(id='dt-O2c', min=0.1, max=5.3, step=0.1, value=1.3,
                                   marks={v: str(v) for v in [0.1, 1.5, 3.0, 5.3]}),

                        html.Label('CO2 catodo (CO2c)', style=SL),
                        dcc.Slider(id='dt-CO2c', min=0.3, max=14.0, step=0.2, value=2.15,
                                   marks={v: str(v) for v in [0.3, 3.0, 7.0, 14.0]}),

                        html.Label('N2 catodo (N2c)', style=SL),
                        dcc.Slider(id='dt-N2c', min=0.5, max=29.0, step=0.5, value=4.87,
                                   marks={v: str(v) for v in [0.5, 5, 15, 29]}),

                        html.Label('Resistencia ohmica r1 (Ohm·cm2)', style=SL),
                        dcc.Slider(id='dt-r1', min=1.8, max=3.0, step=0.05, value=1.97,
                                   marks={v: str(round(v, 2)) for v in [1.8, 2.0, 2.5, 3.0]}),

                        html.Label('Densidad de corriente de operacion i (A/cm²)', style=SL),
                        dcc.Slider(id='dt-corriente', min=0.005, max=0.200, step=0.005, value=0.100,
                                   marks={v: str(round(v, 3)) for v in [0.005, 0.05, 0.10, 0.15, 0.200]}),

                        html.Label('Modelo de prediccion', style={**SL, 'marginTop': '12px'}),
                        dcc.RadioItems(
                            id='dt-modelo',
                            options=[
                                {'label': '  Nernst (físico)', 'value': 'nernst'},
                                {'label': '  PLS (datos)',     'value': 'pls'},
                                {'label': '  KPLS (kernel)',   'value': 'kpls'},
                                {'label': '  Todos',           'value': 'ambos'},
                            ],
                            value='ambos',
                            style={'fontFamily': 'Arial', 'fontSize': '12px',
                                   'color': '#2c3e50', 'marginBottom': '8px'},
                            labelStyle={'display': 'block', 'marginBottom': '4px'}
                        ),

                        html.Label('Modo de simulacion', style={**SL, 'marginTop': '4px'}),
                        dcc.RadioItems(
                            id='dt-modo',
                            options=[
                                {'label': '  Continuo (corriente fija)', 'value': 'continuo'},
                                {'label': '  Curva (21 puntos i)',        'value': 'curva'},
                            ],
                            value='continuo',
                            style={'fontFamily': 'Arial', 'fontSize': '12px',
                                   'color': '#2c3e50', 'marginBottom': '8px'},
                            labelStyle={'display': 'block', 'marginBottom': '4px'}
                        ),

                        # Botones simulacion
                        html.Div([
                            html.Button(
                                '▶  Iniciar Simulacion',
                                id='btn-iniciar-sim',
                                n_clicks=0,
                                style={
                                    'marginTop': '20px', 'width': '100%',
                                    'padding': '10px',
                                    'backgroundColor': '#27ae60',
                                    'color': 'white', 'border': 'none',
                                    'borderRadius': '6px', 'fontFamily': 'Arial',
                                    'fontSize': '13px', 'cursor': 'pointer',
                                    'fontWeight': 'bold'
                                }
                            ),
                            html.Button(
                                '■  Detener',
                                id='btn-detener-sim',
                                n_clicks=0,
                                style={
                                    'marginTop': '8px', 'width': '100%',
                                    'padding': '10px',
                                    'backgroundColor': '#e74c3c',
                                    'color': 'white', 'border': 'none',
                                    'borderRadius': '6px', 'fontFamily': 'Arial',
                                    'fontSize': '13px', 'cursor': 'pointer',
                                    'fontWeight': 'bold'
                                }
                            ),
                        ]),

                        # Estado simulador
                        html.Div(id='sim-status', style={
                            'fontFamily': 'Arial', 'fontSize': '12px',
                            'marginTop': '10px', 'textAlign': 'center',
                            'padding': '8px', 'borderRadius': '6px',
                            'backgroundColor': '#f8f9fa', 'color': '#7f8c8d'
                        }),

                        # Boton cargar experimento cercano
                        html.Button(
                            'Cargar experimento mas cercano',
                            id='btn-cargar-exp',
                            n_clicks=0,
                            style={
                                'marginTop': '16px', 'width': '100%',
                                'padding': '10px',
                                'backgroundColor': '#2980b9',
                                'color': 'white', 'border': 'none',
                                'borderRadius': '6px', 'fontFamily': 'Arial',
                                'fontSize': '13px', 'cursor': 'pointer',
                                'fontWeight': 'bold'
                            }
                        ),
                        html.Div(id='btn-feedback', style={
                            'fontFamily': 'Arial', 'fontSize': '12px',
                            'color': '#27ae60', 'marginTop': '6px',
                            'textAlign': 'center'
                        }),

                    ], style={**SP, 'width': '28%'}),

                    # ── Panel derecho: curva DT + visor variables ──
                    html.Div([

                        # Curva DT
                        html.Div([
                            dcc.Graph(id='dt-graph', style={'height': '360px'}),
                            html.Div(id='dt-metricas', style={
                                'fontFamily': 'Arial', 'fontSize': '14px',
                                'backgroundColor': '#f0f4f8', 'borderRadius': '8px',
                                'padding': '16px', 'marginTop': '8px'
                            })
                        ], style=CARD_STYLE),

                        # ── Visor de variables en tiempo real ─────
                        html.Div([
                            html.H4("Variables en tiempo real",
                                    style={'fontFamily': 'Arial', 'color': '#2c3e50',
                                           'marginBottom': '12px'}),

                            # KPIs instantaneos
                            html.Div(id='sim-kpis', style={
                                'display': 'flex', 'gap': '10px',
                                'flexWrap': 'nowrap', 'marginBottom': '12px'
                            }),

                            # Graficos voltaje y potencia en tiempo real
                            html.Div([
                                html.Div([
                                    dcc.Graph(id='sim-voltaje-graph',
                                              style={'height': '220px'})
                                ], style={'width': '49%', 'display': 'inline-block'}),
                                html.Div([
                                    dcc.Graph(id='sim-potencia-graph',
                                              style={'height': '220px'})
                                ], style={'width': '49%', 'display': 'inline-block',
                                          'marginLeft': '2%'}),
                            ]),

                            # Tabla de mediciones recientes
                            html.Div([
                                html.H5("Ultimas mediciones",
                                        style={'fontFamily': 'Arial',
                                               'color': '#555', 'marginBottom': '8px'}),
                                html.Div(id='sim-tabla-mediciones')
                            ], style={'marginTop': '12px'})

                        ], style=CARD_STYLE),

                    ], style={'width': '70%', 'display': 'inline-block',
                              'verticalAlign': 'top', 'paddingLeft': '16px'})

                ], style={'display': 'flex', 'alignItems': 'flex-start'})

            ], style={'padding': '20px'})
        ]),

        # ── TAB 5: Monitoreo Live ──────────────────────────────────
        dcc.Tab(label='Monitoreo Live', children=[
            html.Div([
                dcc.Interval(id='live-interval', interval=10_000, n_intervals=0),
                html.Div([
                    html.Div([
                        html.H4("Sesión activa",
                                style={'fontFamily': 'Arial', 'color': '#2c3e50',
                                       'marginBottom': '8px'}),
                        html.Label('Experimento (udec_lab)', style=SL),
                        dcc.Dropdown(
                            id='live-exp-selector',
                            placeholder='Seleccionar experimento...',
                            clearable=False,
                            style={'fontFamily': 'Arial', 'marginTop': '6px'}
                        ),
                        html.Div(id='live-exp-info', style={
                            'fontFamily': 'Arial', 'fontSize': '12px',
                            'color': '#555', 'marginTop': '10px',
                            'backgroundColor': '#f8f9fa', 'borderRadius': '6px',
                            'padding': '10px'
                        }),
                        html.Div([
                            html.Span("● ", style={'color': '#27ae60'}),
                            html.Span(id='live-status',
                                      style={'fontFamily': 'Arial',
                                             'fontSize': '12px', 'color': '#27ae60'})
                        ], style={'marginTop': '12px'}),
                    ], style={**SP, 'width': '26%'}),
                    html.Div([
                        html.H4("Última medición",
                                style={'fontFamily': 'Arial', 'color': '#2c3e50',
                                       'marginBottom': '12px'}),
                        html.Div(id='live-kpis')
                    ], style={'width': '70%', 'display': 'inline-block',
                              'verticalAlign': 'top', 'padding': '20px'})
                ], style={'display': 'flex', 'alignItems': 'flex-start',
                          'marginBottom': '16px'}),
                html.Div([
                    html.Div([
                        dcc.Graph(id='live-voltaje', style={'height': '280px'})
                    ], style={'width': '49%', 'display': 'inline-block'}),
                    html.Div([
                        dcc.Graph(id='live-potencia', style={'height': '280px'})
                    ], style={'width': '49%', 'display': 'inline-block',
                              'marginLeft': '2%'}),
                ]),
                html.Div([
                    dcc.Graph(id='live-polar', style={'height': '320px'})
                ], style={'marginTop': '10px'})
            ], style={'padding': '20px'})
        ]),

    ])
], style={'maxWidth': '1200px', 'margin': 'auto', 'padding': '20px'})


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
    Output('sim-status',      'children'),
    Output('sim-status',      'style'),
    Input('btn-iniciar-sim',  'n_clicks'),
    Input('btn-detener-sim',  'n_clicks'),
    State('dt-T',    'value'),
    State('dt-H2a',  'value'),
    State('dt-H2Oa', 'value'),
    State('dt-CO2a', 'value'),
    State('dt-O2c',  'value'),
    State('dt-CO2c', 'value'),
    State('dt-N2c',  'value'),
    State('dt-r1',   'value'),
    State('dt-corriente', 'value'),
    State('dt-modo',       'value'),
    State('dt-sim-exp-id', 'data'),
    prevent_initial_call=True
)
def controlar_simulador(n_ini, n_det, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1, corriente, modo, exp_id_actual):
    global _sim_process

    from dash import ctx
    triggered = ctx.triggered_id

    status_base = {
        'fontFamily': 'Arial', 'fontSize': '12px',
        'marginTop': '10px', 'textAlign': 'center',
        'padding': '8px', 'borderRadius': '6px',
    }

    if triggered == 'btn-iniciar-sim':
        # Detener proceso anterior si existe
        if _sim_process and _sim_process.poll() is None:
            try:
                os.killpg(os.getpgid(_sim_process.pid), signal.SIGTERM)
            except Exception:
                pass

        # Lanzar simulador con los parametros definidos
        # Dashboard en .../Dashboard/ — simulador en .../simulador/ (un nivel arriba)
        _dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        sim_path = os.path.normpath(
            os.path.join(_dashboard_dir, '..', 'simulador', 'simulador_mcfc.py')
        )
        modo_sim = modo if modo else 'continuo'
        cmd = [
            'python3', sim_path,
            '--modo',        modo_sim,
            '--temperatura', str(int(T)),
            '--h2a',   str(H2a),
            '--h2oa',  str(H2Oa),
            '--co2a',  str(CO2a),
            '--o2c',   str(O2c),
            '--co2c',  str(CO2c),
            '--n2c',   str(N2c),
            '--r1',    str(r1),
            '--intervalo', '3'
        ]
        if modo_sim == 'continuo':
            cmd += ['--corriente', str(corriente)]
        if modo_sim == 'curva':
            cmd += ['--ciclos', '1']
        try:
            import time as _time
            t_inicio = datetime.now(timezone.utc)
            # Redirigir stderr a log para diagnóstico
            import tempfile
            _log = open('/tmp/simulador_mcfc.log', 'w')
            _sim_process = subprocess.Popen(
                cmd, stdout=_log,
                stderr=_log,
                preexec_fn=os.setsid
            )
            # Esperar activamente hasta que aparezca un experimento creado DESPUES del inicio
            new_exp_id = None
            for _ in range(15):          # hasta ~15 s
                _time.sleep(1)
                try:
                    conn = psycopg2.connect(**DB_CONFIG)
                    row = pd.read_sql(
                        "SELECT id_experimento FROM experimentos "
                        "WHERE fuente='udec_lab' AND created_at >= %s "
                        "ORDER BY created_at DESC LIMIT 1",
                        conn, params=(t_inicio,)
                    )
                    conn.close()
                    if not row.empty:
                        new_exp_id = int(row['id_experimento'].iloc[0])
                        break
                except Exception:
                    pass
            style = {**status_base, 'backgroundColor': '#d5f5e3', 'color': '#1e8449'}
            if new_exp_id:
                if modo_sim == 'continuo':
                    label = f'● Simulando (continuo) — T={T}°C | i={corriente:.3f} A/cm² | Exp {new_exp_id}'
                else:
                    label = f'● Simulando (curva) — T={T}°C | Exp {new_exp_id}'
            else:
                label = f'● Simulando ({modo_sim}) — T={T}°C | exp pendiente'
            return False, new_exp_id, label, style
        except Exception as ex:
            style = {**status_base, 'backgroundColor': '#fde8e8', 'color': '#e74c3c'}
            return True, None, f'Error al iniciar: {str(ex)[:60]}', style

    elif triggered == 'btn-detener-sim':
        if _sim_process and _sim_process.poll() is None:
            try:
                os.killpg(os.getpgid(_sim_process.pid), signal.SIGTERM)
            except Exception:
                pass
        style = {**status_base, 'backgroundColor': '#f8f9fa', 'color': '#7f8c8d'}
        return True, None, '■ Simulacion detenida', style

    style = {**status_base, 'backgroundColor': '#f8f9fa', 'color': '#7f8c8d'}
    return True, None, 'Sin simulacion activa', style


# Actualizar curva DT + visor de variables en tiempo real
@app.callback(
    Output('dt-graph',            'figure'),
    Output('dt-metricas',         'children'),
    Output('sim-kpis',            'children'),
    Output('sim-voltaje-graph',   'figure'),
    Output('sim-potencia-graph',  'figure'),
    Output('sim-tabla-mediciones','children'),
    Input('dt-sim-interval', 'n_intervals'),
    Input('dt-T',    'value'),
    Input('dt-H2a',  'value'),
    Input('dt-H2Oa', 'value'),
    Input('dt-CO2a', 'value'),
    Input('dt-O2c',  'value'),
    Input('dt-CO2c', 'value'),
    Input('dt-N2c',  'value'),
    Input('dt-r1',   'value'),
    Input('dt-sim-exp-id', 'data'),   # Input (no State) → dispara al cambiar exp_id
    Input('dt-modelo',     'value'),
)
def actualizar_dt(n, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1, exp_id_sim, modelo_sel):

    fig_empty = go.Figure()
    fig_empty.update_layout(
        template='plotly_white', font=dict(family='Arial'),
        annotations=[dict(text='Inicia la simulacion para ver datos',
                         showarrow=False, font=dict(size=13, color='#aaa'),
                         xref='paper', yref='paper', x=0.5, y=0.5)]
    )

    card_kpi = {
        'display': 'inline-block', 'textAlign': 'center',
        'width': '22%', 'padding': '12px',
        'backgroundColor': 'white', 'borderRadius': '10px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.08)',
    }

    def kpi(titulo, valor, unidad, color='#2c3e50'):
        return html.Div([
            html.Div(titulo, style={'fontWeight': 'bold', 'fontSize': '11px',
                                    'color': '#7f8c8d', 'marginBottom': '4px'}),
            html.Div(valor,  style={'fontSize': '22px', 'fontWeight': 'bold',
                                    'color': color}),
            html.Div(unidad, style={'fontSize': '10px', 'color': '#aaa'}),
        ], style=card_kpi)

    # ── Curva DT estática (responde a sliders siempre) ─────────────────────────
    i_range = np.linspace(0.005, 0.25, 100)
    modelo_sel = modelo_sel or 'ambos'

    fig_dt = go.Figure()

    # Curva Nernst
    if modelo_sel in ('nernst', 'ambos'):
        V_nernst = voltaje_modelo(i_range, T, H2a, H2Oa, CO2a, O2c, CO2c, r1)
        P_nernst = V_nernst * i_range
        fig_dt.add_trace(go.Scatter(
            x=i_range, y=V_nernst, mode='lines',
            line=dict(color='#e74c3c', width=2.5, dash='dot'),
            name='Nernst (físico)', yaxis='y1'
        ))
        fig_dt.add_trace(go.Scatter(
            x=i_range, y=P_nernst, mode='lines',
            line=dict(color='#8e44ad', width=2, dash='dash'),
            name='P Nernst', yaxis='y2'
        ))

    # Curva PLS
    if modelo_sel in ('pls', 'ambos') and _PLS_OK:
        V_pls = voltaje_pls(i_range, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c)
        if V_pls is not None:
            P_pls = V_pls * i_range
            fig_dt.add_trace(go.Scatter(
                x=i_range, y=V_pls, mode='lines',
                line=dict(color='#2980b9', width=2.5),
                name='PLS (datos)', yaxis='y1'
            ))
            fig_dt.add_trace(go.Scatter(
                x=i_range, y=P_pls, mode='lines',
                line=dict(color='#16a085', width=2, dash='dash'),
                name='P PLS', yaxis='y2'
            ))

    # Curva KPLS
    if modelo_sel in ('kpls', 'ambos') and _KPLS_OK:
        V_kpls = voltaje_kpls(i_range, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c)
        if V_kpls is not None:
            P_kpls = V_kpls * i_range
            fig_dt.add_trace(go.Scatter(
                x=i_range, y=V_kpls, mode='lines',
                line=dict(color='#e67e22', width=2.5),
                name='KPLS (kernel)', yaxis='y1'
            ))
            fig_dt.add_trace(go.Scatter(
                x=i_range, y=P_kpls, mode='lines',
                line=dict(color='#d35400', width=2, dash='dash'),
                name='P KPLS', yaxis='y2'
            ))

    metricas_out = html.P("Ajusta los sliders para ver la curva del modelo DT.",
                          style={'color': '#888', 'fontFamily': 'Arial'})
    kpis_out = []
    fig_v = fig_empty
    fig_p = fig_empty
    tabla_out = html.P("Sin mediciones aún.", style={'color': '#aaa', 'fontFamily': 'Arial'})

    # ── Si hay simulacion activa, cargar mediciones reales ─────────────────────
    if exp_id_sim is not None:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            df_m = pd.read_sql("""
                SELECT id_medicion, i_densidad, voltaje, eta, timestamp_medicion
                FROM mediciones
                WHERE id_experimento = %s
                ORDER BY timestamp_medicion ASC
            """, conn, params=(exp_id_sim,))
            conn.close()

            if not df_m.empty:
                # Superponer datos reales en curva DT
                fig_dt.add_trace(go.Scatter(
                    x=df_m['i_densidad'], y=df_m['voltaje'],
                    mode='markers', name='Datos simulados',
                    marker=dict(color='#27ae60', size=8), yaxis='y1'
                ))

                # Metricas si hay suficientes puntos y hay variedad de corrientes
                es_modo_continuo = df_m['i_densidad'].nunique() <= 2
                if es_modo_continuo:
                    metricas_out = html.Div([
                        html.P(
                            "⚠ Métricas no disponibles en modo continuo.",
                            style={'color': '#e67e22', 'fontFamily': 'Arial',
                                   'fontWeight': 'bold', 'marginBottom': '4px'}
                        ),
                        html.P(
                            "Use modo Curva (21 puntos i) para evaluar el modelo con R², MAE y NRMSE.",
                            style={'color': '#7f8c8d', 'fontFamily': 'Arial', 'fontSize': '12px'}
                        ),
                    ], style={'padding': '12px', 'backgroundColor': '#fef9e7',
                              'borderRadius': '8px', 'border': '1px solid #f39c12'})

                if len(df_m) >= 3 and not es_modo_continuo:
                    V_pred_n = voltaje_modelo(
                        df_m['i_densidad'].values, T, H2a, H2Oa, CO2a, O2c, CO2c, r1
                    )
                    r2_n, mae_n, nrmse_n = metricas(df_m['voltaje'].values, V_pred_n)

                    # Métricas PLS
                    r2_p = mae_p = nrmse_p = None
                    if _PLS_OK:
                        V_pred_p = voltaje_pls(df_m['i_densidad'].values, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c)
                        if V_pred_p is not None:
                            r2_p, mae_p, nrmse_p = metricas(df_m['voltaje'].values, V_pred_p)

                    # Métricas KPLS
                    r2_k = mae_k = nrmse_k = None
                    if _KPLS_OK:
                        V_pred_k = voltaje_kpls(df_m['i_densidad'].values, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c)
                        if V_pred_k is not None:
                            r2_k, mae_k, nrmse_k = metricas(df_m['voltaje'].values, V_pred_k)

                    def color_m(val, bueno, malo, inv=False):
                        if val is None: return '#aaa'
                        if not inv:
                            return '#27ae60' if val >= bueno else ('#e67e22' if val >= malo else '#e74c3c')
                        return '#27ae60' if val <= bueno else ('#e67e22' if val <= malo else '#e74c3c')

                    card = {'display': 'inline-block', 'textAlign': 'center', 'width': '14%',
                            'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '8px',
                            'boxShadow': '0 1px 4px #ccc', 'marginRight': '8px'}

                    def met_card(titulo, val_n, val_p, val_k, bueno, malo, inv=False):
                        return html.Div([
                            html.Div(titulo, style={'fontWeight': 'bold', 'fontSize': '11px',
                                                    'marginBottom': '6px', 'color': '#555'}),
                            html.Div("Nernst", style={'fontSize': '10px', 'color': '#e74c3c'}),
                            html.Div(f"{val_n:.4f}" if val_n is not None else "—",
                                     style={'fontSize': '16px', 'fontWeight': 'bold',
                                            'color': color_m(val_n, bueno, malo, inv)}),
                            html.Div("PLS", style={'fontSize': '10px', 'color': '#2980b9',
                                                   'marginTop': '4px'}),
                            html.Div(f"{val_p:.4f}" if val_p is not None else "—",
                                     style={'fontSize': '16px', 'fontWeight': 'bold',
                                            'color': color_m(val_p, bueno, malo, inv)}),
                            html.Div("KPLS", style={'fontSize': '10px', 'color': '#e67e22',
                                                    'marginTop': '4px'}),
                            html.Div(f"{val_k:.4f}" if val_k is not None else "—",
                                     style={'fontSize': '16px', 'fontWeight': 'bold',
                                            'color': color_m(val_k, bueno, malo, inv)}),
                        ], style=card)

                    metricas_out = html.Div([
                        html.H4(f"Evaluacion del modelo — Exp {exp_id_sim}",
                                style={'color': '#2c3e50', 'marginBottom': '8px',
                                       'fontFamily': 'Arial'}),
                        html.Div([
                            met_card("R²",    r2_n,    r2_p,    r2_k,    0.95, 0.85),
                            met_card("MAE[V]",mae_n,   mae_p,   mae_k,   0.02, 0.05, inv=True),
                            met_card("NRMSE", nrmse_n, nrmse_p, nrmse_k, 0.05, 0.10, inv=True),
                        ], style={'display': 'flex', 'flexWrap': 'nowrap'})
                    ])

                # KPIs ultima medicion
                ultima = df_m.iloc[-1]
                V_u = float(ultima['voltaje'])
                i_u = float(ultima['i_densidad'])
                P_u = V_u * i_u
                eta_u = float(ultima['eta']) if ultima['eta'] else 0.0
                kpis_out = [
                    kpi("Voltaje",     f"{V_u:.4f}", "V",      '#27ae60'),
                    kpi("Corriente",   f"{i_u:.4f}", "A/cm²",  '#2980b9'),
                    kpi("Potencia",    f"{P_u:.4f}", "W/cm²",  '#8e44ad'),
                    kpi("Eficiencia η",f"{eta_u:.3f}", "—",    '#16a085'),
                ]

                # Grafico voltaje vs tiempo
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(
                    x=df_m['timestamp_medicion'], y=df_m['voltaje'],
                    mode='lines+markers',
                    line=dict(color='#27ae60', width=2),
                    marker=dict(size=5), name='Voltaje'
                ))
                fig_v.update_layout(
                    title='Voltaje E [V]',
                    xaxis_title='Tiempo', yaxis_title='E [V]',
                    template='plotly_white', font=dict(family='Arial'),
                    margin=dict(t=35, b=40, l=50, r=20)
                )

                # Grafico potencia vs tiempo
                df_m['potencia'] = df_m['voltaje'] * df_m['i_densidad']
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(
                    x=df_m['timestamp_medicion'], y=df_m['potencia'],
                    mode='lines+markers',
                    line=dict(color='#8e44ad', width=2),
                    marker=dict(size=5), name='Potencia'
                ))
                fig_p.update_layout(
                    title='Potencia P [W/cm²]',
                    xaxis_title='Tiempo', yaxis_title='P [W/cm²]',
                    template='plotly_white', font=dict(family='Arial'),
                    margin=dict(t=35, b=40, l=50, r=20)
                )

                # Tabla ultimas 10 mediciones
                df_tabla = df_m.tail(10).copy()
                df_tabla['potencia'] = (df_tabla['voltaje'] * df_tabla['i_densidad']).round(5)
                df_tabla['voltaje']    = df_tabla['voltaje'].round(4)
                df_tabla['i_densidad'] = df_tabla['i_densidad'].round(4)
                df_tabla['eta']        = df_tabla['eta'].round(4)
                df_tabla['timestamp_medicion'] = df_tabla['timestamp_medicion'].astype(str).str[:19]
                df_tabla = df_tabla[['timestamp_medicion','i_densidad','voltaje','potencia','eta']]
                df_tabla.columns = ['Timestamp','i [A/cm²]','V [V]','P [W/cm²]','η']

                tabla_out = dash_table.DataTable(
                    data=df_tabla.to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in df_tabla.columns],
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#2c3e50', 'color': 'white',
                                  'fontWeight': 'bold', 'fontFamily': 'Arial',
                                  'fontSize': '12px'},
                    style_cell={'fontFamily': 'Arial', 'fontSize': '12px',
                                'padding': '6px', 'textAlign': 'center'},
                    style_data_conditional=[{
                        'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'
                    }]
                )

        except Exception:
            pass

    fig_dt.update_layout(
        title=f'Digital Twin — T={T}°C',
        xaxis_title='Densidad de corriente i [A/cm²]',
        yaxis=dict(title='Voltaje E [V]', side='left'),
        yaxis2=dict(title='Potencia P [W/cm²]', overlaying='y',
                    side='right', showgrid=False),
        hovermode='x unified', template='plotly_white',
        font=dict(family='Arial'),
        legend=dict(orientation='h', y=-0.22)
    )

    return fig_dt, metricas_out, kpis_out, fig_v, fig_p, tabla_out


# ── Callback: cargar experimento cercano ───────────────────────────────────────
@app.callback(
    Output('dt-T',    'value'),
    Output('dt-H2a',  'value'),
    Output('dt-H2Oa', 'value'),
    Output('dt-CO2a', 'value'),
    Output('dt-O2c',  'value'),
    Output('dt-CO2c', 'value'),
    Output('dt-N2c',  'value'),
    Output('dt-r1',   'value'),
    Output('btn-feedback', 'children'),
    Input('btn-cargar-exp', 'n_clicks'),
    State('dt-T',    'value'),
    State('dt-H2a',  'value'),
    State('dt-H2Oa', 'value'),
    State('dt-CO2a', 'value'),
    State('dt-O2c',  'value'),
    State('dt-CO2c', 'value'),
    State('dt-N2c',  'value'),
    State('dt-r1',   'value'),
    prevent_initial_call=True
)
def cargar_experimento_cercano(n_clicks, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1):
    tol_T = 13
    cands = exp_summary[abs(exp_summary['T'] - T) <= tol_T].copy()
    if len(cands) == 0:
        return T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1, 'Sin experimentos cercanos'
    ref = np.array([H2a, H2Oa, CO2a, O2c, CO2c, N2c])
    cands['dist'] = cands[['H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c']].apply(
        lambda row: np.linalg.norm(row.values - ref), axis=1
    )
    mejor = cands.sort_values('dist').iloc[0]
    eid   = int(mejor['id_experimento'])
    return (
        int(mejor['T']),
        round(float(mejor['H2a']),  3),
        round(float(mejor['H2Oa']), 3),
        round(float(mejor['CO2a']), 3),
        round(float(mejor['O2c']),  3),
        round(float(mejor['CO2c']), 3),
        round(float(mejor['N2c']),  3),
        round(float(mejor['r_1']),  3),
        f'Cargado: Exp {eid} (T={int(mejor["T"])}°C)'
    )


# ── Helpers y Callbacks Tab 5: Monitoreo Live ──────────────────────────────────
def get_experimentos_udec():
    conn = psycopg2.connect(**DB_CONFIG)
    df_exp = pd.read_sql("""
        SELECT e.id_experimento, e.t, e.h2a, e.co2a, e.o2c, e.co2c,
               e.created_at,
               COUNT(m.id_medicion) AS n_mediciones
        FROM experimentos e
        LEFT JOIN mediciones m USING(id_experimento)
        WHERE e.fuente = 'udec_lab'
        GROUP BY e.id_experimento, e.t, e.h2a, e.co2a, e.o2c, e.co2c, e.created_at
        ORDER BY e.created_at DESC
        LIMIT 50
    """, conn)
    conn.close()
    return df_exp


def get_mediciones_live(id_exp: int):
    conn = psycopg2.connect(**DB_CONFIG)
    df_m = pd.read_sql("""
        SELECT id_medicion, i_densidad, voltaje, eta, timestamp_medicion
        FROM mediciones
        WHERE id_experimento = %s
        ORDER BY timestamp_medicion ASC
    """, conn, params=(id_exp,))
    conn.close()
    return df_m


@app.callback(
    Output('live-exp-selector', 'options'),
    Output('live-exp-selector', 'value'),
    Input('live-interval', 'n_intervals'),
    State('live-exp-selector', 'value'),
)
def actualizar_selector(n, current_val):
    df_exp = get_experimentos_udec()
    if df_exp.empty:
        return [], None
    options = [
        {
            'label': (f"Exp {row.id_experimento} | T={int(row.t)}°C | "
                      f"H2a={row.h2a:.2f} | {row.n_mediciones} med. | "
                      f"{str(row.created_at)[:16]}"),
            'value': int(row.id_experimento)
        }
        for row in df_exp.itertuples()
    ]
    ids_disponibles = [o['value'] for o in options]
    if options:
        options[0]['label'] = "▶ " + options[0]['label']
    val = current_val if current_val in ids_disponibles else options[0]['value']
    return options, val


@app.callback(
    Output('live-voltaje',  'figure'),
    Output('live-potencia', 'figure'),
    Output('live-polar',    'figure'),
    Output('live-kpis',     'children'),
    Output('live-exp-info', 'children'),
    Output('live-status',   'children'),
    Input('live-interval',      'n_intervals'),
    Input('live-exp-selector',  'value'),
)
def actualizar_live(n, id_exp):
    card_base = {
        'display': 'inline-block', 'textAlign': 'center',
        'width': '22%', 'padding': '14px',
        'backgroundColor': 'white', 'borderRadius': '10px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.08)',
        'marginRight': '12px', 'verticalAlign': 'top'
    }
    fig_empty = go.Figure()
    fig_empty.update_layout(template='plotly_white', font=dict(family='Arial'),
                            annotations=[dict(text='Sin datos', showarrow=False,
                                             font=dict(size=14, color='#aaa'),
                                             xref='paper', yref='paper', x=0.5, y=0.5)])
    if id_exp is None:
        return fig_empty, fig_empty, fig_empty, html.P("Sin experimento seleccionado."), "", "Esperando datos..."

    df_m = get_mediciones_live(id_exp)
    if df_m.empty:
        return fig_empty, fig_empty, fig_empty, \
               html.P("Sin mediciones aún.", style={'color': '#888'}), \
               "", "Sin mediciones"

    ultima = df_m.iloc[-1]
    n_med  = len(df_m)
    ts_str = str(ultima['timestamp_medicion'])[:19] if ultima['timestamp_medicion'] else '—'

    def kpi_card(titulo, valor, unidad, color='#2c3e50'):
        return html.Div([
            html.Div(titulo, style={'fontWeight': 'bold', 'fontSize': '12px',
                                    'color': '#7f8c8d', 'marginBottom': '6px'}),
            html.Div(valor,  style={'fontSize': '26px', 'fontWeight': 'bold', 'color': color}),
            html.Div(unidad, style={'fontSize': '11px', 'color': '#aaa', 'marginTop': '2px'}),
        ], style=card_base)

    V_last   = float(ultima['voltaje'])
    i_last   = float(ultima['i_densidad'])
    P_last   = V_last * i_last
    eta_last = float(ultima['eta']) if ultima['eta'] else 0.0

    def color_v_relativo(v, i):
        v_esperado = max(1.05 - 3.5 * i, 0.4)
        ratio = v / v_esperado if v_esperado > 0 else 1.0
        if ratio >= 0.97:   return '#27ae60'
        elif ratio >= 0.90: return '#e67e22'
        else:               return '#e74c3c'

    kpis = html.Div([
        kpi_card("Voltaje",            f"{V_last:.4f}", "V",     color_v_relativo(V_last, i_last)),
        kpi_card("Densidad corriente", f"{i_last:.4f}", "A/cm²", '#2980b9'),
        kpi_card("Potencia",           f"{P_last:.4f}", "W/cm²", '#8e44ad'),
        kpi_card("Eficiencia η",  f"{eta_last:.3f}", "—",   '#16a085'),
    ], style={'display': 'flex', 'flexWrap': 'nowrap', 'gap': '12px', 'alignItems': 'stretch'})

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        row_exp = pd.read_sql(
            "SELECT t, h2a, co2a, o2c, co2c, n2c FROM experimentos WHERE id_experimento=%s",
            conn, params=(id_exp,)
        ).iloc[0]
        conn.close()
        info = html.Div([
            html.Div(f"T = {int(row_exp.t)} °C",   style={'marginBottom': '3px'}),
            html.Div(f"H2a = {row_exp.h2a:.3f}",   style={'marginBottom': '3px'}),
            html.Div(f"CO2a = {row_exp.co2a:.3f}",  style={'marginBottom': '3px'}),
            html.Div(f"O2c = {row_exp.o2c:.3f}",   style={'marginBottom': '3px'}),
            html.Div(f"CO2c = {row_exp.co2c:.3f}",  style={'marginBottom': '3px'}),
            html.Div(f"Mediciones: {n_med}",
                     style={'marginTop': '8px', 'fontWeight': 'bold', 'color': '#2980b9'}),
            html.Div(f"Última: {ts_str}",
                     style={'fontSize': '11px', 'color': '#aaa', 'marginTop': '3px'}),
        ])
    except Exception:
        info = html.P("—")

    status = f"Actualizando... {n_med} mediciones recibidas"

    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=df_m['timestamp_medicion'], y=df_m['voltaje'],
                               mode='lines+markers', line=dict(color='#2980b9', width=2),
                               marker=dict(size=6), name='Voltaje'))
    fig_v.update_layout(title='Voltaje E [V] — tiempo real',
                        xaxis_title='Tiempo', yaxis_title='E [V]',
                        template='plotly_white', font=dict(family='Arial'),
                        margin=dict(t=40, b=40, l=50, r=20))

    df_m['potencia'] = df_m['voltaje'] * df_m['i_densidad']
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=df_m['timestamp_medicion'], y=df_m['potencia'],
                               mode='lines+markers', line=dict(color='#8e44ad', width=2),
                               marker=dict(size=6), name='Potencia'))
    fig_p.update_layout(title='Potencia P [W/cm²] — tiempo real',
                        xaxis_title='Tiempo', yaxis_title='P [W/cm²]',
                        template='plotly_white', font=dict(family='Arial'),
                        margin=dict(t=40, b=40, l=50, r=20))

    fig_polar = go.Figure()
    fig_polar.add_trace(go.Scatter(x=df_m['i_densidad'], y=df_m['voltaje'],
                                   mode='lines+markers', line=dict(color='#27ae60', width=2),
                                   marker=dict(size=7, color='#27ae60'),
                                   name='Curva experimental'))
    if n_med >= 3:
        try:
            conn2 = psycopg2.connect(**DB_CONFIG)
            row_e = pd.read_sql(
                "SELECT t, h2a, h2oa, co2a, o2c, co2c, n2c, h2oc, co, ch4 "
                "FROM experimentos WHERE id_experimento=%s", conn2, params=(id_exp,)
            ).iloc[0]
            pm = pd.read_sql(
                "SELECT r_1 FROM parametros_modelo WHERE id_experimento=%s LIMIT 1",
                conn2, params=(id_exp,)
            )
            conn2.close()
            r1_val = float(pm['r_1'].iloc[0]) if not pm.empty and pm['r_1'].iloc[0] is not None else 1.973
            i_range = np.linspace(0, df_m['i_densidad'].max() * 1.1, 80)
            V_mod = voltaje_modelo(i_range, float(row_e.t), float(row_e.h2a), float(row_e.h2oa),
                                   float(row_e.co2a), float(row_e.o2c), float(row_e.co2c), r1_val,
                                   N2a=float(row_e.co), CO=float(row_e.co), CH4=float(row_e.ch4),
                                   N2c=float(row_e.n2c), H2Oc=float(row_e.h2oc))
            fig_polar.add_trace(go.Scatter(x=i_range, y=V_mod, mode='lines',
                                           line=dict(color='#e74c3c', width=2, dash='dot'),
                                           name='Modelo DT'))
        except Exception:
            pass

    fig_polar.update_layout(
        title=f'Curva de Polarización acumulada — Exp {id_exp}',
        xaxis_title='Densidad de corriente i [A/cm²]',
        yaxis_title='Voltaje E [V]',
        template='plotly_white', font=dict(family='Arial'),
        hovermode='x unified', margin=dict(t=40, b=50, l=50, r=20),
        legend=dict(orientation='h', y=-0.2)
    )

    return fig_v, fig_p, fig_polar, kpis, info, status


if __name__ == '__main__':
    app.run(debug=True)
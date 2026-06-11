import warnings
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy')

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import psycopg2
import joblib as _joblib
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import DB_CONFIG

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
R_GAS  = 8.314
F_FAR  = 96485.0
J_MIN, J_MAX = 0.005, 0.200

_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modelos'))

# ── Paleta industrial seria — gris acero + azul técnico ──────────────────────
C = {
    'bg':       '#1a1d23',
    'surface':  '#20242c',
    'panel':    '#252a33',
    'border':   '#3d4455',
    'border2':  '#2c3140',
    'accent':   '#5a8ec8',
    'accent2':  '#3567a0',
    'accent_bg':'#1e2d40',
    'ok':       '#6ab87f',
    'ok_bg':    '#1a2e22',
    'warn':     '#d4a843',
    'warn_bg':  '#2a2010',
    'red':      '#cc6360',
    'red_bg':   '#2a1818',
    'teal':     '#5ab5cc',
    'text':     '#e8ecf2',
    'muted':    '#9ba8bc',
    'dim':      '#5e6b80',
}

COLORES_MODELOS = {
    'Nernst':       '#6b7585',
    'PLS':          '#4a7eb5',
    'KPLS':         '#b8922a',
    'GPR':          '#6aa3c8',
    'GPR Residual': '#5a9e6f',
}

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════
def _cargar(f):
    try:
        d = _joblib.load(os.path.join(_DIR, f))
        print(f"✓ {f} — R²={d.get('r2_test','?'):.4f}")
        return d, True
    except Exception as e:
        print(f"⚠ {f}: {e}")
        return None, False

_pls,  _PLS_OK  = _cargar('pls_voltaje_cv_balanceado.pkl')
_kpls, _KPLS_OK = _cargar('kpls_voltaje_cv_balanceado.pkl')
_gpr,  _GPR_OK  = _cargar('gpr_voltaje_balanceado.pkl')
_gprr, _GPRR_OK = _cargar('gpr_residual_balanceado.pkl')

# ══════════════════════════════════════════════════════════════════════════════
# PREDICCIONES
# ══════════════════════════════════════════════════════════════════════════════
def v_nernst(j, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1):
    j   = np.atleast_1d(np.array(j, dtype=float))
    T_K = T + 273.15
    eps = 1e-10
    sa  = max(H2a+H2Oa+CO2a, eps); sc = max(O2c+CO2c+N2c, eps)
    xH2=max(H2a/sa,eps); xH2O=max(H2Oa/sa,eps); xCO2a=max(CO2a/sa,eps)
    xO2=max(O2c/sc,eps); xCO2c=max(CO2c/sc,eps)
    EN = (1.2723-2.4516e-4*T_K) + (R_GAS*T_K)/(2*F_FAR)*np.log(
        xH2*xO2**0.5*xCO2c/(xH2O*xCO2a+eps))
    return np.maximum(EN - r1*j - (R_GAS*T_K)/(2*F_FAR)*np.log(1+91.878*j) - 0.012, 0.0)

def _X(j, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1):
    j = np.atleast_1d(j)
    return np.column_stack([np.full_like(j,T), np.full_like(j,H2a),
        np.full_like(j,H2Oa), np.full_like(j,CO2a), np.full_like(j,O2c),
        np.full_like(j,CO2c), np.full_like(j,N2c), j, np.full_like(j,r1)])

def v_pls(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1):
    if not _PLS_OK: return None, None
    return _pls['modelo'].predict(_X(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1)).ravel(), None

def v_kpls(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1):
    if not _KPLS_OK: return None, None
    return _kpls['modelo'].predict(_X(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1)).ravel(), None

def v_gpr(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1):
    if not _GPR_OK: return None, None
    X = _gpr['scaler_X'].transform(_X(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1))
    mu_sc,sig_sc = _gpr['modelo'].predict(X, return_std=True)
    sc = _gpr['scaler_y'].scale_[0]
    mu = _gpr['scaler_y'].inverse_transform(mu_sc.reshape(-1,1)).ravel()
    return np.maximum(mu, 0.0), sig_sc*sc

def v_gprr(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1):
    if not _GPRR_OK: return None, None
    Vn = v_nernst(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1)
    X  = _gprr['scaler_X'].transform(_X(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1))
    e_sc,sig_sc = _gprr['modelo'].predict(X, return_std=True)
    sc  = _gprr['scaler_e'].scale_[0]
    eps = _gprr['scaler_e'].inverse_transform(e_sc.reshape(-1,1)).ravel()
    return np.maximum(Vn+eps, 0.0), sig_sc*sc

MODELOS_FN = {
    'Nernst':       v_nernst,
    'PLS':          v_pls,
    'KPLS':         v_kpls,
    'GPR':          v_gpr,
    'GPR Residual': v_gprr,
}

# ══════════════════════════════════════════════════════════════════════════════
# DATOS DESDE BD
# ══════════════════════════════════════════════════════════════════════════════
def get_experimentos():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql('''
        SELECT e.id_experimento,
               e.t AS "T", e.h2a AS "H2a", e.h2oa AS "H2Oa",
               e.co2a AS "CO2a", e.o2c AS "O2c",
               e.co2c AS "CO2c", e.n2c AS "N2c", p.r_1
        FROM experimentos e
        JOIN parametros_modelo p USING(id_experimento)
        ORDER BY e.id_experimento
    ''', conn)
    conn.close()
    return df

exp_df = get_experimentos()
EXP_OPTS = [
    {'label': f"EXP-{r.id_experimento:03d}  T={int(r.T)}°C  "
              f"H2a={r.H2a:.2f}  r₁={r.r_1:.3f}",
     'value': int(r.id_experimento)}
    for r in exp_df.itertuples()
]

# ARD values del modelo GPR Residual balanceado
ARD_VARS = ['CO₂c', 'N₂c', 'H₂Oa', 'T', 'j', 'O₂c', 'CO₂a', 'H₂a', 'r₁']
ARD_VALS = [1.000, 0.538, 0.457, 0.449, 0.326, 0.279, 0.201, 0.034, 0.028]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE ESTILO
# ══════════════════════════════════════════════════════════════════════════════
def mono(size=11, color=None, **kw):
    s = {'fontFamily': "'JetBrains Mono', 'Courier New', monospace",
         'fontSize': f'{size}px', 'color': color or C['text']}
    s.update(kw)
    return s

def label_style():
    return mono(11, C['muted'], letterSpacing='0.06em',
                textTransform='uppercase', marginBottom='6px')

def panel_style():
    return {'backgroundColor': C['panel'], 'border': f"1px solid {C['border']}",
            'borderRadius': '10px', 'padding': '16px', 'marginBottom': '12px'}

def kpi_card(label, value, unit, sub=None, color=None):
    col = color or C['ok']
    return html.Div([
        html.Div(label, style=mono(10, C['muted'], letterSpacing='0.06em',
                                   textTransform='uppercase', marginBottom='8px')),
        html.Div(value, style=mono(26, col, fontWeight='500', lineHeight='1')),
        html.Div(unit,  style=mono(11, C['muted'], marginTop='5px')),
        html.Div(sub,   style=mono(11, C['dim'],   marginTop='3px')) if sub else html.Div(),
    ], style={'backgroundColor': C['surface'],
              'border': f"1px solid {C['border']}",
              'borderRadius': '8px', 'padding': '16px'})

def section_title(icon_char, text):
    return html.Div([
        html.Span(icon_char, style={'marginRight': '7px', 'color': C['accent']}),
        html.Span(text, style=mono(11, C['text'], letterSpacing='0.06em',
                                   textTransform='uppercase', fontWeight='500')),
    ], style={'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center'})

def slider_with_label(label, id, min, max, step, value, unit=''):
    return html.Div([
        html.Div([
            html.Span(label, style=mono(12, C['muted'], fontWeight='500')),
            html.Span(id={'type': 'sl-display', 'index': id},
                      style=mono(13, C['teal'], fontWeight='500')),
            html.Span(f' {unit}', style=mono(11, C['dim'])),
        ], style={'display': 'flex', 'justifyContent': 'space-between',
                  'marginBottom': '4px'}),
        dcc.Slider(id=id, min=min, max=max, step=step, value=value,
                   marks=None, updatemode='drag',
                   tooltip={'always_visible': False},
                   className='industrial-slider'),
    ], style={'marginBottom': '10px'})

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY TEMPLATE INDUSTRIAL
# ══════════════════════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor=C['surface'],
    font=dict(family="'JetBrains Mono', monospace", color=C['text'], size=12),
    xaxis=dict(gridcolor=C['border2'], linecolor=C['border'],
               tickcolor=C['border'], zerolinecolor=C['border2'],
               tickfont=dict(size=11), title_font=dict(size=12)),
    yaxis=dict(gridcolor=C['border2'], linecolor=C['border'],
               tickcolor=C['border'], zerolinecolor=C['border2'],
               tickfont=dict(size=11), title_font=dict(size=12)),
    legend=dict(bgcolor='rgba(30,35,45,0.9)', font=dict(size=11),
                orientation='h', y=-0.28,
                bordercolor=C['border'], borderwidth=1),
    margin=dict(t=35, b=75, l=55, r=25),
    hovermode='x unified',
)

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PANEL MODELOS — fichas simples
# ══════════════════════════════════════════════════════════════════════════════
MODELOS_INFO = [
    {
        'nombre':  'Nernst (Semi-empírico)',
        'tipo':    'Referencia física',
        'color':   '#6b7585',
        'desc':    'Modelo termodinámico basado en la ecuación de Nernst con correcciones de pérdidas óhmicas y cinéticas. No requiere entrenamiento.',
        'r2':      '~0.970',
        'mae':     '~0.028 V',
        'nrmse':   '~0.046',
        'dataset': 'Fórmula analítica',
    },
    {
        'nombre':  'PLS',
        'tipo':    'Regresión lineal de componentes',
        'color':   '#4a7eb5',
        'desc':    'Partial Least Squares. Proyecta las variables en componentes latentes que maximizan la covarianza con el voltaje.',
        'r2':      '0.9694',
        'mae':     '0.0246 V',
        'nrmse':   '0.0348',
        'dataset': 'Balanceado (Warsaw + sintéticos)',
    },
    {
        'nombre':  'KPLS',
        'tipo':    'Regresión kernel no lineal',
        'color':   '#b8922a',
        'desc':    'Kernel PLS con función RBF. Captura relaciones no lineales entre variables y voltaje mediante transformación en espacio de características.',
        'r2':      '0.9829',
        'mae':     '0.0181 V',
        'nrmse':   '0.0260',
        'dataset': 'Balanceado (Warsaw + sintéticos)',
    },
    {
        'nombre':  'GPR',
        'tipo':    'Proceso gaussiano',
        'color':   '#6aa3c8',
        'desc':    'Gaussian Process Regression. Modelo bayesiano no paramétrico que además de predecir el voltaje, entrega una banda de incertidumbre ±2σ.',
        'r2':      '0.9968',
        'mae':     '0.0074 V',
        'nrmse':   '0.0113',
        'dataset': 'Balanceado (Warsaw + sintéticos)',
    },
    {
        'nombre':  'GPR Residual',
        'tipo':    'Híbrido físico + GPR',
        'color':   '#5a9e6f',
        'desc':    'Contribución original: combina la base física de Nernst con un GPR que aprende el error sistemático residual. Mejor desempeño global.',
        'r2':      '0.9981',
        'mae':     '0.0056 V',
        'nrmse':   '0.0086',
        'dataset': 'Balanceado (Warsaw + sintéticos)',
    },
]

def build_panel_modelos():
    cards = []
    for m in MODELOS_INFO:
        cards.append(html.Div([
            html.Div([
                html.Div([
                    html.Span('●', style={'color': m['color'], 'marginRight':'8px',
                                          'fontSize':'12px'}),
                    html.Span(m['nombre'], style=mono(13, C['text'], fontWeight='500')),
                ]),
                html.Div(m['tipo'], style=mono(10, C['muted'], marginTop='2px')),
            ], style={'marginBottom':'8px'}),

            html.Div(m['desc'], style={**mono(10, C['muted']), 'lineHeight':'1.5',
                                        'marginBottom':'10px'}),

            html.Div([
                html.Div([
                    html.Div('R²', style=mono(9, C['dim'], textTransform='uppercase',
                                              letterSpacing='0.06em')),
                    html.Div(m['r2'], style=mono(13, m['color'], fontWeight='500')),
                ], style={'textAlign':'center','flex':'1'}),
                html.Div([
                    html.Div('MAE', style=mono(9, C['dim'], textTransform='uppercase',
                                               letterSpacing='0.06em')),
                    html.Div(m['mae'], style=mono(11, C['text'])),
                ], style={'textAlign':'center','flex':'1'}),
                html.Div([
                    html.Div('NRMSE', style=mono(9, C['dim'], textTransform='uppercase',
                                                 letterSpacing='0.06em')),
                    html.Div(m['nrmse'], style=mono(11, C['text'])),
                ], style={'textAlign':'center','flex':'1'}),
            ], style={'display':'flex','gap':'4px',
                      'backgroundColor':C['surface'],
                      'borderRadius':'4px','padding':'8px',
                      'marginBottom':'6px'}),

            html.Div([
                html.Span('Dataset: ', style=mono(9, C['dim'])),
                html.Span(m['dataset'], style=mono(9, C['muted'])),
            ]),

        ], style={**panel_style(),
                  'borderLeft': f"2px solid {m['color']}",
                  'marginBottom':'8px'}))

    return html.Div([
        section_title('⊟', 'Modelos implementados'),
        html.Div(cards, style={'overflowY':'auto', 'maxHeight':'600px',
                               'paddingRight':'4px'}),
    ])

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "MCFC Digital Twin — Sistema de Análisis Operacional"

app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: #1a1d23;
            color: #d0d6e0;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #20242c; }
        ::-webkit-scrollbar-thumb { background: #363c48; border-radius: 2px; }

        .industrial-slider .rc-slider-rail { background: #2c3140; height: 3px; }
        .industrial-slider .rc-slider-track { background: #3567a0; height: 3px; }
        .industrial-slider .rc-slider-handle {
            background: #1a1d23; border: 2px solid #4a7eb5;
            width: 13px; height: 13px; margin-top: -5px;
        }
        .industrial-slider .rc-slider-handle:hover { border-color: '#6aa3c8'; }
        .industrial-slider .rc-slider-handle-dragging {
            border-color: #6aa3c8 !important;
            box-shadow: 0 0 0 3px rgba(74,126,181,0.15) !important;
        }

        .temp-btn {
            flex: 1; padding: 7px 0; font-size: 12px;
            border: 1px solid #3d4455; border-radius: 5px;
            text-align: center; cursor: pointer;
            color: #9ba8bc; background: #20242c;
            font-family: 'JetBrains Mono', monospace;
            transition: all 0.15s;
        }
        .temp-btn:hover { border-color: #5a8ec8; color: #5ab5cc; }
        .temp-btn.active {
            background: #1e2d40; color: #e8ecf2;
            border-color: #5a8ec8; font-weight: 500;
        }

        .run-btn {
            width: 100%; padding: 12px; border: 1px solid #3567a0;
            border-radius: 7px; background: #1e2d40; color: #5ab5cc;
            font-family: 'JetBrains Mono', monospace; font-size: 13px;
            font-weight: 500; cursor: pointer; letter-spacing: 0.08em;
            text-transform: uppercase; margin-top: 6px;
            transition: all 0.15s;
        }
        .run-btn:hover { background: #3567a0; color: #e8ecf2; border-color: #5a8ec8; }
        .run-btn:active { transform: scale(0.98); }

        .nav-item {
            font-size: 12px; color: #9ba8bc; cursor: pointer;
            padding: 6px 14px; border-radius: 5px;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.04em; transition: all 0.15s;
            border: 0.5px solid transparent;
        }
        .nav-item:hover { color: #e8ecf2; border-color: #3d4455; }
        .nav-item.active {
            color: #5ab5cc; background: #1e2d40;
            border: 1px solid #3567a0; font-weight: 500;
        }

        .status-pulse {
            width: 7px; height: 7px; border-radius: 50%;
            background: #5a9e6f; display: inline-block;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%,100% { opacity: 1; }
            50% { opacity: 0.35; }
        }

        .model-row { transition: background 0.1s; }
        .model-row:hover { background: #252a33; }
        .model-row.best { background: #1e2d40; }

        .ard-bar-fill { transition: width 0.4s ease; }

        .Select-control, .VirtualizedSelectFocusedOption,
        .Select-menu-outer, .Select-menu,
        .VirtualizedSelectOption { background: #252a33 !important; }
        .Select-control {
            background: #252a33 !important;
            border: 0.5px solid #363c48 !important;
            color: #d0d6e0 !important;
            border-radius: 4px !important;
        }
        .Select-value-label, .Select-placeholder,
        .Select-input input { color: #d0d6e0 !important; }
        .Select-arrow { border-top-color: #7a8494 !important; }
        .Select-menu-outer {
            background: #252a33 !important;
            border: 0.5px solid #363c48 !important;
            border-radius: 4px !important;
            z-index: 9999 !important;
        }
        .Select-option {
            background: #252a33 !important;
            color: #d0d6e0 !important;
            font-size: 10px !important;
        }
        .Select-option:hover, .Select-option.is-focused {
            background: #1e2d40 !important;
            color: #6aa3c8 !important;
        }
        .Select-option.is-selected {
            background: #1e2d40 !important;
            color: #6aa3c8 !important;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
app.layout = html.Div([

    # ── TOPBAR ─────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div(style={'width':'8px','height':'8px','borderRadius':'50%',
                            'backgroundColor':C['ok'],
                            'animation':'pulse 2s infinite',
                            'marginRight':'10px','flexShrink':'0'}),
            html.Div([
                html.Div("MCFC DIGITAL TWIN — SISTEMA DE ANÁLISIS OPERACIONAL",
                         style=mono(13, C['text'], fontWeight='500',
                                    letterSpacing='0.05em')),
                html.Div("Universidad de Concepción · Dpto. Ing. Informática · v2.0",
                         style=mono(11, C['muted'])),
            ]),
            html.Span("● SISTEMA ACTIVO",
                      style=mono(9, C['ok'], letterSpacing='0.05em',
                                 backgroundColor=C['ok_bg'],
                                 padding='3px 8px', borderRadius='4px',
                                 marginLeft='12px')),
            html.Span("◈ MODO SIMULACIÓN",
                      style=mono(9, C['warn'], letterSpacing='0.05em',
                                 backgroundColor=C['warn_bg'],
                                 padding='3px 8px', borderRadius='4px',
                                 marginLeft='6px')),
        ], style={'display':'flex','alignItems':'center'}),

        html.Div([
            html.Button("MONITOREO",    id='tab-monitoreo',    n_clicks=0, className='nav-item'),
            html.Button("OPTIMIZACIÓN", id='tab-optimizacion', n_clicks=0, className='nav-item active'),
            html.Button("HISTORIAL",    id='tab-historial',    n_clicks=0, className='nav-item'),
            html.Button("MODELOS",      id='tab-modelos',      n_clicks=0, className='nav-item'),
        ], style={'display':'flex','gap':'4px'}),

    ], style={'backgroundColor': C['panel'],
              'border': f"1px solid {C['border']}",
              'borderRadius': '10px', 'padding': '12px 20px',
              'display': 'flex', 'alignItems': 'center',
              'justifyContent': 'space-between', 'marginBottom': '12px',
              'flexWrap': 'wrap', 'gap': '8px'}),

    # ── MAIN GRID ──────────────────────────────────────────────────────────────
    html.Div([

        # ── PANEL IZQUIERDO ────────────────────────────────────────────────────
        html.Div([

            # Modo entrada
            html.Div([
                section_title('⊕', 'Fuente de condiciones'),
                dcc.RadioItems(
                    id='modo',
                    options=[
                        {'label': '  Cargar desde experimento (BD)', 'value': 'bd'},
                        {'label': '  Ingresar condiciones libres',   'value': 'libre'},
                    ],
                    value='bd',
                    style=mono(12, C['muted']),
                    labelStyle={'display': 'block', 'marginBottom': '8px',
                                'cursor': 'pointer'}),

                html.Div(id='div-bd', children=[
                    html.Div(style={'height':'8px'}),
                    dcc.Dropdown(
                        id='exp-selector',
                        options=EXP_OPTS,
                        value=EXP_OPTS[0]['value'] if EXP_OPTS else None,
                        clearable=False,
                        style={'fontFamily':"'JetBrains Mono', monospace",
                               'fontSize': '10px',
                               'backgroundColor': C['surface'],
                               'color': C['text'],
                               'border': f"0.5px solid {C['border']}"}),
                    html.Button('↓ CARGAR EXPERIMENTO', id='btn-cargar', n_clicks=0,
                        style={**mono(10, C['accent'], letterSpacing='0.05em'),
                               'width':'100%','padding':'7px','marginTop':'6px',
                               'backgroundColor':C['accent_bg'],
                               'border':f"0.5px solid {C['accent']}",
                               'borderRadius':'4px','cursor':'pointer',
                               'textTransform':'uppercase'}),
                    html.Div(id='feedback-cargar',
                             style=mono(10, C['ok'], textAlign='center',
                                        marginTop='4px')),
                ]),
            ], style=panel_style()),

            # Temperatura
            html.Div([
                section_title('◎', 'Temperatura operacional'),
                html.Div([
                    html.Button(str(t), id=f'btn-T-{t}', n_clicks=0,
                                className='temp-btn' + (' active' if t==650 else ''))
                    for t in [550, 575, 600, 625, 650]
                ], style={'display':'flex','gap':'4px','marginBottom':'4px'}),
                dcc.Store(id='store-T', data=650),
            ], style=panel_style()),

            # Ánodo
            html.Div([
                section_title('⬡', 'Composición ánodo'),
                slider_with_label('H₂a',   'sl-H2a',  0.2,  4.5,  0.05, 2.20),
                slider_with_label('H₂Oa',  'sl-H2Oa', 0.05, 1.3,  0.05, 0.41),
                slider_with_label('CO₂a',  'sl-CO2a', 0.05, 1.1,  0.05, 0.55),
            ], style=panel_style()),

            # Cátodo
            html.Div([
                section_title('⬡', 'Composición cátodo'),
                slider_with_label('O₂c',  'sl-O2c',  0.1,  5.3,  0.1,  1.30),
                slider_with_label('CO₂c', 'sl-CO2c', 0.3,  14.0, 0.2,  2.15),
                slider_with_label('N₂c',  'sl-N2c',  0.5,  29.0, 0.5,  4.87),
            ], style=panel_style()),

            # Parámetro óhmico + umbral
            html.Div([
                section_title('⊟', 'Parámetro óhmico'),
                slider_with_label('r₁', 'sl-r1', 1.8, 3.0, 0.05, 1.97, 'Ω·cm²'),
                html.Div(style={'height':'6px'}),
                section_title('◈', 'Umbral región óptima'),
                slider_with_label('umbral', 'sl-umbral', 0.80, 0.99, 0.01, 0.95, '%'),
            ], style=panel_style()),

            # Botón calcular
            html.Button('▶  CALCULAR Y OPTIMIZAR', id='btn-calcular', n_clicks=0,
                        className='run-btn'),
            html.Div(id='status-calcular',
                     style=mono(11, C['muted'], textAlign='center', marginTop='8px')),

        ], style={'width':'285px','flexShrink':'0'}),

        # ── PANEL DERECHO — contenido dinámico según tab ────────────────────
        html.Div([

            # ── TAB: OPTIMIZACIÓN ──────────────────────────────────────────────
            html.Div(id='panel-optimizacion', children=[

                html.Div(id='kpi-row', children=[
                    kpi_card("Esperando cálculo", "—", "Presione calcular"),
                ], style={'display':'grid',
                          'gridTemplateColumns':'repeat(4,1fr)',
                          'gap':'8px','marginBottom':'10px'}),

                html.Div([
                    section_title('◈', 'Curva de polarización E(j) y densidad de potencia'),
                    dcc.Graph(id='fig-curvas', style={'height':'260px'},
                              config={'displayModeBar':False}),
                ], style=panel_style()),

                html.Div([
                    html.Div([
                        section_title('⊞', 'Comparación de modelos'),
                        html.Div(id='tabla-industrial'),
                        html.Div(id='opt-box'),
                    ], style={**panel_style(), 'flex':'1','marginBottom':'0',
                               'marginRight':'10px'}),
                    html.Div([
                        section_title('◉', 'Relevancia ARD — GPR Residual'),
                        html.Div(id='ard-panel', children=[]),
                    ], style={**panel_style(), 'width':'270px','flexShrink':'0',
                               'marginBottom':'0'}),
                ], style={'display':'flex','gap':'0','alignItems':'flex-start'}),

            ]),

            # ── TAB: HISTORIAL ─────────────────────────────────────────────────
            html.Div(id='panel-historial', style={'display':'none'}, children=[
                html.Div([
                    section_title('◷', 'Historial de cálculos — sesión actual'),
                    html.Div(id='tabla-historial',
                             children=[html.Div("Sin cálculos aún. Presione Calcular en Optimización.",
                                                style=mono(11, C['muted'], textAlign='center',
                                                           padding='30px 0'))]),
                ], style=panel_style()),
            ]),

            # ── TAB: MODELOS ───────────────────────────────────────────────────
            html.Div(id='panel-modelos', style={'display':'none'}, children=[
                build_panel_modelos(),
            ]),

            # ── TAB: MONITOREO ─────────────────────────────────────────────────
            html.Div(id='panel-monitoreo', style={'display':'none'}, children=[
                html.Div([
                    section_title('◈', 'Tendencias de variables operacionales'),
                    html.Div(id='monitor-info',
                             style=mono(10, C['muted'], marginBottom='8px')),
                    dcc.Graph(id='fig-monitoreo', style={'height':'340px'},
                              config={'displayModeBar':False}),
                ], style=panel_style()),
            ]),

            # Stores
            dcc.Store(id='store-tab',        data='optimizacion'),
            dcc.Store(id='store-historial',  data=[]),
            dcc.Store(id='store-best-model', data='—'),

        ], style={'flex':'1','minWidth':'0','paddingLeft':'10px',
                  'display':'flex','flexDirection':'column','gap':'0'}),

    ], style={'display':'flex','alignItems':'flex-start','gap':'0'}),

], style={'maxWidth':'1400px','margin':'auto','padding':'16px',
          'backgroundColor':C['bg'],'minHeight':'100vh'})


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE UI
# ══════════════════════════════════════════════════════════════════════════════
def build_ard_static():
    items = []
    for var, val in zip(ARD_VARS, ARD_VALS):
        pct = int(val * 100)
        col = C['ok'] if val > 0.5 else (C['teal'] if val > 0.25 else C['dim'])
        items.append(html.Div([
            html.Div(var, style=mono(11, C['muted'], width='48px',
                                     textAlign='right', flexShrink='0')),
            html.Div([
                html.Div(style={'height':'7px','borderRadius':'4px',
                                'width':f'{pct}%','backgroundColor':col,
                                'transition':'width 0.4s ease'}),
            ], style={'flex':'1','backgroundColor':C['surface'],
                      'borderRadius':'4px','height':'7px'}),
            html.Div(f'{val:.3f}', style=mono(11, C['text'], width='38px',
                                               flexShrink='0', textAlign='right',
                                               fontWeight='500')),
        ], style={'display':'flex','alignItems':'center','gap':'10px',
                  'marginBottom':'9px'}))
    items.append(html.Div([
        html.Div(style={'width':'7px','height':'7px','borderRadius':'50%',
                        'backgroundColor':C['ok'],'flexShrink':'0',
                        'marginTop':'2px'}),
        html.Span("Importancia relativa para explicar el error de Nernst",
                  style={**mono(10, C['muted']), 'lineHeight':'1.5',
                         'whiteSpace':'normal'}),
    ], style={'display':'flex','alignItems':'flex-start','gap':'8px',
              'marginTop':'12px','paddingTop':'10px',
              'borderTop':f"1px solid {C['border2']}"}))
    return items

def build_tabla(filas):
    headers = ['Modelo','j* (A/cm²)','V* (V)','P (W/cm²)','Región','P_gar','⚠']
    COLS = '1.4fr 1fr 0.9fr 0.9fr 1.3fr 1fr 0.4fr'
    rows = []
    for i, f in enumerate(filas):
        is_best = f.get('_best', False)
        bg = C['ok_bg'] if is_best else ('rgba(255,255,255,0.02)' if i%2==0 else 'transparent')
        row_style = {'display':'grid',
                     'gridTemplateColumns': COLS,
                     'gap':'4px','padding':'9px 10px',
                     'borderBottom':f"0.5px solid {C['border2']}",
                     'backgroundColor': bg,
                     'borderLeft': f"2px solid {C['ok']}" if is_best else '2px solid transparent',
                     'transition':'background 0.1s'}
        cols_vals = [
            html.Div([
                html.Span('★ ' if is_best else '', style={'color':C['ok']}),
                html.Span(f['Modelo']),
            ], style=mono(12, C['ok'] if is_best else C['text'], fontWeight='500' if is_best else 'normal')),
            html.Div(f"j*={f['j*']:.3f}", style=mono(12, C['teal'])),
            html.Div(f"{f['V*']:.4f} V",  style=mono(12, C['text'])),
            html.Div(f"{f['P']:.4f}",     style=mono(12, C['text'])),
            html.Div(f['Region'],          style=mono(11, C['muted'])),
            html.Div(f['Pgar'],            style=mono(11, C['muted'])),
            html.Div(f['warn'],            style=mono(12, C['warn'])),
        ]
        rows.append(html.Div(cols_vals, style=row_style))

    header_style = {'display':'grid',
                    'gridTemplateColumns': COLS,
                    'gap':'4px','padding':'8px 10px',
                    'borderBottom':f"1px solid {C['border']}",
                    'backgroundColor': C['surface'],
                    'borderRadius':'6px 6px 0 0',
                    'marginBottom':'2px'}
    header = html.Div(
        [html.Div(h, style=mono(10, C['text'], letterSpacing='0.05em',
                                textTransform='uppercase', fontWeight='500'))
         for h in headers],
        style=header_style)

    return html.Div([header] + rows)

def build_kpis(best_model, j_star, v_star, p_star, sigma, T):
    e0 = 1.2723 - 2.4516e-4 * (T + 273.15)
    return [
        kpi_card("Voltaje óptimo",
                 f"{v_star:.4f}",
                 f"V  @ j*={j_star:.3f} A/cm²",
                 f"Modelo: {best_model}", C['ok']),
        kpi_card("Potencia máxima",
                 f"{p_star:.4f}",
                 "W/cm²",
                 sub=None, color=C['teal']),
        kpi_card("Incertidumbre ±2σ",
                 f"±{2*sigma:.4f}" if sigma and sigma > 0 else "Solo GPR",
                 "V  (95% confianza)" if sigma and sigma > 0 else "GPR/GPR Residual",
                 sub=None, color=C['accent']),
        kpi_card("E₀ Nernst (ref.)",
                 f"{e0:.4f}",
                 "V  (potencial teórico)",
                 f"η_total ≈ {e0-v_star:.3f} V", C['muted']),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# Inicializar panel ARD al cargar
@app.callback(
    Output('ard-panel', 'children'),
    Input('ard-panel',  'id'),
)
def init_ard(_):
    return build_ard_static()



# Mostrar valores de sliders en tiempo real
@app.callback(
    Output({'type': 'sl-display', 'index': 'sl-H2a'},  'children'),
    Output({'type': 'sl-display', 'index': 'sl-H2Oa'}, 'children'),
    Output({'type': 'sl-display', 'index': 'sl-CO2a'}, 'children'),
    Output({'type': 'sl-display', 'index': 'sl-O2c'},  'children'),
    Output({'type': 'sl-display', 'index': 'sl-CO2c'}, 'children'),
    Output({'type': 'sl-display', 'index': 'sl-N2c'},  'children'),
    Output({'type': 'sl-display', 'index': 'sl-r1'},   'children'),
    Output({'type': 'sl-display', 'index': 'sl-umbral'},'children'),
    Input('sl-H2a',   'value'),
    Input('sl-H2Oa',  'value'),
    Input('sl-CO2a',  'value'),
    Input('sl-O2c',   'value'),
    Input('sl-CO2c',  'value'),
    Input('sl-N2c',   'value'),
    Input('sl-r1',    'value'),
    Input('sl-umbral','value'),
)
def update_slider_displays(h2a, h2oa, co2a, o2c, co2c, n2c, r1, umbral):
    return (f"{h2a:.2f}", f"{h2oa:.2f}", f"{co2a:.2f}",
            f"{o2c:.2f}", f"{co2c:.2f}", f"{n2c:.2f}",
            f"{r1:.2f}", f"{int(umbral*100)}%")


# Toggle BD
@app.callback(Output('div-bd','style'), Input('modo','value'))
def toggle_bd(modo):
    return {'display':'block'} if modo == 'bd' else {'display':'none'}


# Callback unificado: botones de temperatura + cargar BD
# Maneja store-T y clases de botones desde un solo punto
@app.callback(
    Output('store-T',          'data'),
    Output('btn-T-550',        'className'),
    Output('btn-T-575',        'className'),
    Output('btn-T-600',        'className'),
    Output('btn-T-625',        'className'),
    Output('btn-T-650',        'className'),
    Output('sl-H2a',           'value'),
    Output('sl-H2Oa',          'value'),
    Output('sl-CO2a',          'value'),
    Output('sl-O2c',           'value'),
    Output('sl-CO2c',          'value'),
    Output('sl-N2c',           'value'),
    Output('sl-r1',            'value'),
    Output('feedback-cargar',  'children'),
    Input('btn-T-550',         'n_clicks'),
    Input('btn-T-575',         'n_clicks'),
    Input('btn-T-600',         'n_clicks'),
    Input('btn-T-625',         'n_clicks'),
    Input('btn-T-650',         'n_clicks'),
    Input('btn-cargar',        'n_clicks'),
    State('store-T',           'data'),
    State('exp-selector',      'value'),
    State('sl-H2a',            'value'),
    State('sl-H2Oa',           'value'),
    State('sl-CO2a',           'value'),
    State('sl-O2c',            'value'),
    State('sl-CO2c',           'value'),
    State('sl-N2c',            'value'),
    State('sl-r1',             'value'),
    prevent_initial_call=True,
)
def actualizar_temperatura_y_bd(
        n550, n575, n600, n625, n650, n_cargar,
        current_T, exp_id,
        h2a, h2oa, co2a, o2c, co2c, n2c, r1):

    triggered = ctx.triggered_id
    temps = [550, 575, 600, 625, 650]
    btn_map = {
        'btn-T-550': 550, 'btn-T-575': 575,
        'btn-T-600': 600, 'btn-T-625': 625, 'btn-T-650': 650,
    }

    # ── Botón de temperatura presionado ───────────────────────────────────────
    if triggered in btn_map:
        new_T = btn_map[triggered]
        clases = ['temp-btn active' if t == new_T else 'temp-btn' for t in temps]
        return new_T, *clases, h2a, h2oa, co2a, o2c, co2c, n2c, r1, ''

    # ── Cargar desde BD ───────────────────────────────────────────────────────
    if triggered == 'btn-cargar':
        if exp_id is None:
            clases = ['temp-btn active' if t == 650 else 'temp-btn' for t in temps]
            return 650, *clases, h2a, h2oa, co2a, o2c, co2c, n2c, r1, '—'
        r    = exp_df[exp_df['id_experimento'] == exp_id].iloc[0]
        new_T = int(r['T'])
        clases = ['temp-btn active' if t == new_T else 'temp-btn' for t in temps]
        return (new_T, *clases,
                round(float(r['H2a']),3),  round(float(r['H2Oa']),3),
                round(float(r['CO2a']),3), round(float(r['O2c']),3),
                round(float(r['CO2c']),3), round(float(r['N2c']),3),
                round(float(r['r_1']),3),
                f"✓ EXP-{exp_id:03d} cargado (T={new_T}°C)")

    # Fallback
    clases = ['temp-btn active' if t == (current_T or 650) else 'temp-btn'
              for t in temps]
    return current_T or 650, *clases, h2a, h2oa, co2a, o2c, co2c, n2c, r1, ''

# Calcular
@app.callback(
    Output('fig-curvas',       'figure'),
    Output('kpi-row',          'children'),
    Output('tabla-industrial', 'children'),
    Output('opt-box',          'children'),
    Output('status-calcular',    'children'),
    Output('store-best-model',   'data'),
    Input('btn-calcular', 'n_clicks'),
    State('store-T',    'data'),
    State('sl-H2a',     'value'),
    State('sl-H2Oa',    'value'),
    State('sl-CO2a',    'value'),
    State('sl-O2c',     'value'),
    State('sl-CO2c',    'value'),
    State('sl-N2c',     'value'),
    State('sl-r1',      'value'),
    State('sl-umbral',  'value'),
    State('modo',       'value'),
    State('exp-selector','value'),
    prevent_initial_call=True,
)
def calcular(n, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1,
             umbral, modo, exp_id):

    args  = (T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1)
    j_arr = np.linspace(J_MIN, J_MAX, 400)

    fig = go.Figure()
    filas = []
    best_p = -np.inf
    best_info = {}

    for nombre, fn in MODELOS_FN.items():
        color = COLORES_MODELOS[nombre]
        dash_style = 'dash' if nombre == 'Nernst' else 'solid'
        width = 2.5 if nombre == 'GPR Residual' else 1.8

        try:
            if nombre == 'Nernst':
                mu, sig = fn(j_arr, *args), None
            else:
                mu, sig = fn(j_arr, *args)
            if mu is None:
                continue

            # Banda ±2σ
            if sig is not None:
                r_int = int(color[1:3],16)
                g_int = int(color[3:5],16)
                b_int = int(color[5:7],16)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([j_arr, j_arr[::-1]]),
                    y=np.concatenate([mu+2*sig, (mu-2*sig)[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba({r_int},{g_int},{b_int},0.08)',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False, hoverinfo='skip'))

            fig.add_trace(go.Scatter(
                x=j_arr, y=mu, mode='lines', name=nombre,
                line=dict(color=color, width=width, dash=dash_style),
                hovertemplate=f'<b>{nombre}</b><br>j=%{{x:.3f}} A/cm²'
                              f'<br>V=%{{y:.4f}} V<extra></extra>'))

            # Optimización
            p_arr  = mu * j_arr
            idx_s  = int(np.argmax(p_arr))
            j_star = float(j_arr[idx_s])
            p_star = float(p_arr[idx_s])
            v_star = float(mu[idx_s])
            en_lim = abs(j_star - J_MAX) < 1e-4

            mask_r = p_arr >= umbral * p_star
            idx_r  = np.where(mask_r)[0]
            j_low  = float(j_arr[idx_r[0]])  if len(idx_r) else j_star
            j_high = float(j_arr[idx_r[-1]]) if len(idx_r) else j_star

            p_gar = None
            sigma_mean = 0.0
            if sig is not None:
                p_low = np.maximum(mu-2*sig,0.0)*j_arr
                mask_g = (j_arr >= j_low) & (j_arr <= j_high)
                if mask_g.any():
                    p_gar = float(p_low[mask_g].max())
                sigma_mean = float(sig.mean())

            # Guardar para marcar solo el mejor al final
            if nombre == 'GPR Residual' or nombre == 'GPR':
                _jstar_marker = dict(x=j_star, y=v_star, color=color)

            filas.append({
                'Modelo': nombre,
                'j*':     j_star,
                'V*':     v_star,
                'P':      p_star,
                'Region': f"[{j_low:.3f}–{j_high:.3f}]",
                'Pgar':   f"{p_gar:.4f}" if p_gar else "N/A",
                'warn':   '⚠' if en_lim else '—',
                'sigma':  sigma_mean,
                '_best':  False,
            })

            if p_star > best_p:
                best_p = p_star
                best_info = {'model': nombre, 'j': j_star,
                             'v': v_star, 'p': p_star,
                             'sigma': sigma_mean,
                             'jlow': j_low, 'jhigh': j_high}

        except Exception as e:
            print(f"Error {nombre}: {e}")
            continue

    # Marcar mejor modelo
    for f in filas:
        if f['Modelo'] == best_info.get('model'):
            f['_best'] = True

    # Marcador único j* del mejor modelo
    if best_info:
        fig.add_trace(go.Scatter(
            x=[best_info['j']], y=[best_info['v']],
            mode='markers',
            marker=dict(color=C['warn'], size=10, symbol='diamond',
                        line=dict(color=C['bg'], width=1.5)),
            showlegend=False,
            hovertemplate=f"j*={best_info['j']:.3f} A/cm²"
                          f"<br>V*={best_info['v']:.4f} V<extra></extra>"))

    # Línea j*
    if best_info:
        fig.add_vline(x=best_info['j'], line_color=C['warn'],
                      line_dash='dot', line_width=1,
                      annotation_text=f"j*={best_info['j']:.3f}",
                      annotation_font=dict(color=C['warn'], size=9))

    # Datos reales
    if modo == 'bd' and exp_id is not None:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            df_r = pd.read_sql(
                "SELECT i_densidad, voltaje FROM mediciones "
                "WHERE id_experimento=%s ORDER BY i_densidad",
                conn, params=(exp_id,))
            conn.close()
            if not df_r.empty:
                fig.add_trace(go.Scatter(
                    x=df_r['i_densidad'], y=df_r['voltaje'],
                    mode='markers', name='Datos reales (Milewski)',
                    marker=dict(color='white', size=6, symbol='circle',
                                line=dict(color=C['ok'],width=1.5),
                                opacity=0.9)))
        except Exception:
            pass

    # Layout figura
    layout = {**PLOT_LAYOUT}
    layout['xaxis'] = {**layout['xaxis'],
                       'title': dict(text='Densidad de corriente j [A/cm²]',
                                     font=dict(size=10, color=C['muted']))}
    layout['yaxis'] = {**layout['yaxis'],
                       'title': dict(text='Voltaje E [V]',
                                     font=dict(size=10, color=C['muted']))}
    fig.update_layout(**layout)

    # Construir outputs
    kpis = build_kpis(
        best_info.get('model','—'),
        best_info.get('j', 0),
        best_info.get('v', 0),
        best_info.get('p', 0),
        best_info.get('sigma', 0),
        T)

    tabla = build_tabla(filas)

    opt_box = html.Div([
        html.Div([
            html.Div([
                html.Div("PUNTO ÓPTIMO", style=mono(9, C['ok'],
                                                     letterSpacing='0.06em',
                                                     textTransform='uppercase')),
                html.Div(f"j* = {best_info.get('j',0):.3f} A/cm²",
                         style=mono(14, C['ok'], fontWeight='500')),
            ]),
            html.Div([
                html.Div("REGIÓN ÓPTIMA", style=mono(9, C['ok'],
                                                      letterSpacing='0.06em',
                                                      textTransform='uppercase')),
                html.Div(f"[{best_info.get('jlow',0):.3f} – {best_info.get('jhigh',0):.3f}] A/cm²",
                         style=mono(11, C['teal'], fontWeight='500')),
            ]),
        ], style={'display':'flex','justifyContent':'space-between',
                  'alignItems':'center'}),
    ], style={'backgroundColor': C['ok_bg'],
              'border': f"0.5px solid {C['accent2']}",
              'borderRadius': '6px', 'padding': '10px 14px',
              'marginTop': '10px'})

    status = f"✓ T={T}°C · {len(filas)} modelos calculados · j*={best_info.get('j',0):.3f} A/cm²"
    modelo_ganador = best_info.get('model', '—')

    return fig, kpis, tabla, opt_box, status, modelo_ganador



# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS DE TABS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('store-tab',         'data'),
    Output('tab-monitoreo',     'className'),
    Output('tab-optimizacion',  'className'),
    Output('tab-historial',     'className'),
    Output('tab-modelos',       'className'),
    Output('panel-optimizacion','style'),
    Output('panel-historial',   'style'),
    Output('panel-modelos',     'style'),
    Output('panel-monitoreo',   'style'),
    Input('tab-monitoreo',      'n_clicks'),
    Input('tab-optimizacion',   'n_clicks'),
    Input('tab-historial',      'n_clicks'),
    Input('tab-modelos',        'n_clicks'),
    State('store-tab',          'data'),
    prevent_initial_call=True,
)
def cambiar_tab(n_mon, n_opt, n_his, n_mod, current):
    triggered = ctx.triggered_id
    tab_map = {
        'tab-monitoreo':    'monitoreo',
        'tab-optimizacion': 'optimizacion',
        'tab-historial':    'historial',
        'tab-modelos':      'modelos',
    }
    tab = tab_map.get(triggered, current or 'optimizacion')

    def cls(t): return 'nav-item active' if tab == t else 'nav-item'
    def vis(t): return {} if tab == t else {'display': 'none'}

    return (tab,
            cls('monitoreo'), cls('optimizacion'),
            cls('historial'), cls('modelos'),
            vis('optimizacion'), vis('historial'),
            vis('modelos'),      vis('monitoreo'))


# Guardar en historial tras cada cálculo
@app.callback(
    Output('store-historial',  'data'),
    Output('tabla-historial',  'children'),
    Input('status-calcular',   'children'),
    State('store-historial',   'data'),
    State('store-best-model',  'data'),
    State('store-T',           'data'),
    State('sl-H2a',            'value'),
    State('sl-H2Oa',           'value'),
    State('sl-CO2a',           'value'),
    State('sl-O2c',            'value'),
    State('sl-CO2c',           'value'),
    State('sl-N2c',            'value'),
    State('sl-r1',             'value'),
    prevent_initial_call=True,
)
def actualizar_historial(status, historial, modelo_ganador, T,
                         H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1):
    import datetime
    if not status or '✓' not in status:
        raise dash.exceptions.PreventUpdate

    historial = historial or []

    # Extraer j* del status string
    j_star = '—'
    for p in status.split('·'):
        if 'j*=' in p:
            j_star = p.strip().replace('j*=', '').replace(' A/cm²', '').strip()

    entrada = {
        'hora':   datetime.datetime.now().strftime('%H:%M:%S'),
        'T':      f"{T}°C",
        'H2a':    f"{H2a:.2f}",
        'H2Oa':   f"{H2Oa:.2f}",
        'CO2a':   f"{CO2a:.2f}",
        'O2c':    f"{O2c:.2f}",
        'CO2c':   f"{CO2c:.2f}",
        'N2c':    f"{N2c:.2f}",
        'r1':     f"{r1:.2f}",
        'modelo': modelo_ganador or '—',
        'j_star': j_star,
    }
    historial = [entrada] + historial[:19]  # máx 20 entradas

    # Construir tabla visual — legible
    COLS = '80px 60px 55px 55px 55px 55px 55px 55px 55px 160px 90px'
    headers = ['Hora', 'T', 'H₂a', 'H₂Oa', 'CO₂a', 'O₂c', 'CO₂c', 'N₂c', 'r₁', 'Mejor modelo', 'j* (A/cm²)']

    hdr = html.Div([
        html.Div(h, style={
            'fontFamily': "'JetBrains Mono', monospace",
            'fontSize': '10px',
            'color': C['text'],
            'fontWeight': '500',
            'letterSpacing': '0.05em',
            'textTransform': 'uppercase',
            'padding': '8px 6px',
        }) for h in headers
    ], style={
        'display': 'grid',
        'gridTemplateColumns': COLS,
        'borderBottom': f"1px solid {C['border']}",
        'backgroundColor': C['surface'],
        'borderRadius': '6px 6px 0 0',
        'marginBottom': '2px',
    })

    rows = []
    for i, e in enumerate(historial):
        is_new = (i == 0)
        bg     = C['accent_bg'] if is_new else ('rgba(255,255,255,0.02)' if i % 2 == 0 else 'transparent')
        border = f"1px solid {C['accent2']}" if is_new else f"0.5px solid {C['border2']}"
        txt    = C['text'] if is_new else '#b0bac8'

        def cell(val, col=None):
            return html.Div(val, style={
                'fontFamily': "'JetBrains Mono', monospace",
                'fontSize': '12px',
                'color': col or txt,
                'padding': '9px 6px',
                'lineHeight': '1.2',
            })

        row = html.Div([
            cell(e['hora'],   C['muted'] if not is_new else C['accent']),
            cell(e['T'],      C['teal']),
            cell(e['H2a']),
            cell(e['H2Oa']),
            cell(e['CO2a']),
            cell(e['O2c']),
            cell(e['CO2c']),
            cell(e['N2c']),
            cell(e['r1']),
            cell(e['modelo'], C['ok']),
            cell(e['j_star'], C['warn']),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': COLS,
            'backgroundColor': bg,
            'border': border,
            'borderRadius': '4px',
            'marginBottom': '3px',
        })
        rows.append(row)

    # Contador de entradas
    contador = html.Div(
        f"{len(historial)} cálculo{'s' if len(historial)!=1 else ''} registrado{'s' if len(historial)!=1 else ''} en esta sesión",
        style={**mono(10, C['muted']), 'marginBottom': '10px', 'textAlign': 'right'}
    )

    tabla = html.Div([contador, hdr] + rows)
    return historial, tabla


# Monitoreo — actualizar gráfico de tendencias cuando cambian sliders
@app.callback(
    Output('fig-monitoreo',  'figure'),
    Output('monitor-info',   'children'),
    Input('sl-H2a',    'value'),
    Input('sl-H2Oa',   'value'),
    Input('sl-CO2a',   'value'),
    Input('sl-O2c',    'value'),
    Input('sl-CO2c',   'value'),
    Input('sl-N2c',    'value'),
    Input('sl-r1',     'value'),
    Input('store-T',   'data'),
    State('store-tab', 'data'),
)
def actualizar_monitoreo(H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1, T, tab):
    args = (T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1)
    j_arr = np.linspace(J_MIN, J_MAX, 200)

    fig = go.Figure()
    layout = {**PLOT_LAYOUT}
    layout['xaxis'] = {**layout['xaxis'],
                       'title': dict(text='Densidad de corriente j [A/cm²]',
                                     font=dict(size=10, color=C['muted']))}
    layout['yaxis'] = {**layout['yaxis'],
                       'title': dict(text='Voltaje E [V]',
                                     font=dict(size=10, color=C['muted']))}

    # Curva Nernst base (referencia)
    vn = v_nernst(j_arr, *args)
    fig.add_trace(go.Scatter(x=j_arr, y=vn, mode='lines', name='Nernst',
                             line=dict(color=C['muted'], width=1.5, dash='dash')))

    # Sweeps de ±20% sobre CO₂c (variable más relevante por ARD)
    for factor, lbl, col in [(0.8, 'CO₂c −20%', C['red']),
                              (1.0, 'CO₂c actual', C['ok']),
                              (1.2, 'CO₂c +20%', C['teal'])]:
        co2c_var = np.clip(CO2c * factor, 0.3, 14.0)
        args_v = (T, H2a, H2Oa, CO2a, O2c, co2c_var, N2c, r1)
        try:
            mu, _ = v_gprr(j_arr, *args_v)
            if mu is not None:
                fig.add_trace(go.Scatter(x=j_arr, y=mu, mode='lines', name=lbl,
                                         line=dict(color=col, width=2 if factor==1.0 else 1.2)))
        except Exception:
            pass

    # Línea vertical en j actual (midpoint)
    j_mid = (J_MIN + J_MAX) / 2
    fig.add_vline(x=j_mid, line_color=C['dim'], line_dash='dot', line_width=1)

    fig.update_layout(**layout)

    info = (f"T = {T}°C  ·  H₂a = {H2a:.2f}  ·  CO₂c = {CO2c:.2f}  ·  "
            f"r₁ = {r1:.2f} Ω·cm²  ·  Sensibilidad: sweep CO₂c ±20%")
    return fig, info


if __name__ == '__main__':
    app.run(debug=True, port=8050)
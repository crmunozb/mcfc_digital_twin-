import warnings
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy')

import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import psycopg2
import joblib as _joblib
import os, sys
from scipy.optimize import minimize_scalar

# ── Config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import DB_CONFIG

R_GAS  = 8.314
F_FAR  = 96485.0
J_MIN, J_MAX = 0.005, 0.200

_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modelos'))

def _cargar(f):
    try:
        d = _joblib.load(os.path.join(_DIR, f))
        print(f"✓ {f} — R²={d.get('r2_test','?'):.4f}")
        return d, True
    except Exception as e:
        print(f"⚠ {f}: {e}")
        return None, False

_pls,  _PLS_OK  = _cargar('pls_voltaje_cv.pkl')
_kpls, _KPLS_OK = _cargar('kpls_voltaje_cv.pkl')
_gpr,  _GPR_OK  = _cargar('gpr_voltaje.pkl')
_gprr, _GPRR_OK = _cargar('gpr_residual.pkl')

# ── Predicciones ───────────────────────────────────────────────────────────────
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
    if not _PLS_OK: return None,None
    return _pls['modelo'].predict(_X(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1)).ravel(), None

def v_kpls(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1):
    if not _KPLS_OK: return None,None
    return _kpls['modelo'].predict(_X(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1)).ravel(), None

def v_gpr(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1):
    if not _GPR_OK: return None,None
    X = _gpr['scaler_X'].transform(_X(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1))
    mu_sc,sig_sc = _gpr['modelo'].predict(X, return_std=True)
    sc = _gpr['scaler_y'].scale_[0]
    mu = _gpr['scaler_y'].inverse_transform(mu_sc.reshape(-1,1)).ravel()
    return np.maximum(mu,0.0), sig_sc*sc

def v_gprr(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1):
    if not _GPRR_OK: return None,None
    Vn = v_nernst(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1)
    X  = _gprr['scaler_X'].transform(_X(j,T,H2a,H2Oa,CO2a,O2c,CO2c,N2c,r1))
    e_sc,sig_sc = _gprr['modelo'].predict(X, return_std=True)
    sc  = _gprr['scaler_e'].scale_[0]
    eps = _gprr['scaler_e'].inverse_transform(e_sc.reshape(-1,1)).ravel()
    return np.maximum(Vn+eps,0.0), sig_sc*sc

MODELOS_FN = {
    'Nernst':       v_nernst,
    'PLS':          v_pls,
    'KPLS':         v_kpls,
    'GPR':          v_gpr,
    'GPR Residual': v_gprr,
}
COLORES = {
    'Nernst':'#e74c3c','PLS':'#2980b9','KPLS':'#e67e22',
    'GPR':'#8e44ad','GPR Residual':'#27ae60',
}

# ── Datos BD ───────────────────────────────────────────────────────────────────
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
    {'label': f"Exp {r.id_experimento} | T={int(r.T)}°C | "
              f"H2a={r.H2a:.2f} | O2c={r.O2c:.2f} | r₁={r.r_1:.3f}",
     'value': int(r.id_experimento)}
    for r in exp_df.itertuples()
]

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Digital Twin MCFC — Optimizador Operacional"

SL = {'fontFamily':'Arial','fontWeight':'bold',
      'fontSize':'12px','color':'#2c3e50','marginTop':'8px'}
CARD = {'backgroundColor':'white','borderRadius':'10px',
        'boxShadow':'0 2px 8px rgba(0,0,0,0.08)',
        'padding':'16px','marginBottom':'12px'}

app.layout = html.Div([

    # Título
    html.H1("Digital Twin — Celda MCFC",
            style={'textAlign':'center','fontFamily':'Arial',
                   'color':'#2c3e50','marginBottom':'4px'}),
    html.P("Optimizador Operacional · Universidad de Concepción",
           style={'textAlign':'center','fontFamily':'Arial',
                  'color':'#7f8c8d','marginBottom':'20px'}),

    html.Div([

        # ── Panel izquierdo ────────────────────────────────────────────────────
        html.Div([html.Div([

            html.H4("Condiciones operacionales",
                    style={'fontFamily':'Arial','color':'#2c3e50',
                           'margin':'0 0 12px 0','fontSize':'14px'}),

            # Modo entrada
            dcc.RadioItems(
                id='modo',
                options=[
                    {'label':' Cargar desde experimento (BD)', 'value':'bd'},
                    {'label':' Ingresar condiciones libres',   'value':'libre'},
                ],
                value='bd',
                style={'fontFamily':'Arial','fontSize':'12px'},
                labelStyle={'display':'block','marginBottom':'4px'}),

            # Selector BD
            html.Div(id='div-bd', children=[
                html.Label('Experimento', style=SL),
                dcc.Dropdown(
                    id='exp-selector',
                    options=EXP_OPTS,
                    value=EXP_OPTS[0]['value'] if EXP_OPTS else None,
                    clearable=False,
                    style={'fontFamily':'Arial','fontSize':'11px',
                           'marginTop':'4px'}),
                html.Button('↓ Cargar condiciones', id='btn-cargar',
                    n_clicks=0,
                    style={'width':'100%','padding':'7px','marginTop':'6px',
                           'backgroundColor':'#2980b9','color':'white',
                           'border':'none','borderRadius':'6px',
                           'fontFamily':'Arial','fontSize':'11px',
                           'cursor':'pointer'}),
                html.Div(id='feedback-cargar',
                         style={'fontSize':'11px','color':'#27ae60',
                                'textAlign':'center','marginTop':'4px'}),
            ]),

            # Sliders
            html.Div([
                html.Label('T (°C)', style={**SL,'marginTop':'10px'}),
                dcc.Slider(id='sl-T', min=550, max=650, step=25, value=650,
                    marks={t:str(t) for t in [550,575,600,625,650]},
                    tooltip={'placement':'bottom','always_visible':False}),
                html.Label('H2a', style=SL),
                dcc.Slider(id='sl-H2a', min=0.2, max=4.5, step=0.1,
                    value=2.2, marks=None,
                    tooltip={'placement':'bottom','always_visible':False}),
                html.Label('H2Oa', style=SL),
                dcc.Slider(id='sl-H2Oa', min=0.05, max=1.3, step=0.05,
                    value=0.41, marks=None,
                    tooltip={'placement':'bottom','always_visible':False}),
                html.Label('CO2a', style=SL),
                dcc.Slider(id='sl-CO2a', min=0.05, max=1.1, step=0.05,
                    value=0.55, marks=None,
                    tooltip={'placement':'bottom','always_visible':False}),
                html.Label('O2c', style=SL),
                dcc.Slider(id='sl-O2c', min=0.1, max=5.3, step=0.1,
                    value=1.3, marks=None,
                    tooltip={'placement':'bottom','always_visible':False}),
                html.Label('CO2c', style=SL),
                dcc.Slider(id='sl-CO2c', min=0.3, max=14.0, step=0.2,
                    value=2.15, marks=None,
                    tooltip={'placement':'bottom','always_visible':False}),
                html.Label('N2c', style=SL),
                dcc.Slider(id='sl-N2c', min=0.5, max=29.0, step=0.5,
                    value=4.87, marks=None,
                    tooltip={'placement':'bottom','always_visible':False}),
                html.Label('r₁ (Ω·cm²)', style=SL),
                dcc.Slider(id='sl-r1', min=1.8, max=3.0, step=0.05,
                    value=1.97, marks=None,
                    tooltip={'placement':'bottom','always_visible':False}),
            ], style={'backgroundColor':'#f8f9fa','borderRadius':'8px',
                      'padding':'10px','marginTop':'10px'}),

            # Umbral
            html.Label('Umbral región óptima', style=SL),
            dcc.Slider(id='sl-umbral', min=0.80, max=0.99, step=0.01,
                value=0.95,
                marks={v:f'{int(v*100)}%' for v in [0.80,0.85,0.90,0.95,0.99]},
                tooltip={'placement':'bottom','always_visible':False}),

            # Botón calcular
            html.Button('Calcular', id='btn-calcular', n_clicks=0,
                style={'width':'100%','padding':'10px','marginTop':'14px',
                       'backgroundColor':'#2c3e50','color':'white',
                       'border':'none','borderRadius':'8px',
                       'fontFamily':'Arial','fontSize':'13px',
                       'cursor':'pointer','fontWeight':'bold'}),

            html.Div(id='status-calcular',
                     style={'textAlign':'center','fontFamily':'Arial',
                            'fontSize':'11px','color':'#7f8c8d',
                            'marginTop':'6px'}),

        ], style={**CARD,'padding':'16px'})],
        style={'width':'24%','display':'inline-block','verticalAlign':'top'}),

        # ── Panel derecho ──────────────────────────────────────────────────────
        html.Div([

            # Curvas E(j)
            html.Div([
                html.H5("Curvas de polarización predichas",
                        style={'fontFamily':'Arial','color':'#2c3e50',
                               'margin':'0 0 8px 0','fontSize':'13px'}),
                dcc.Graph(id='fig-curvas', style={'height':'350px'}),
            ], style=CARD),

            # Optimización
            html.Div([
                html.H5("Optimización operacional — región óptima de operación",
                        style={'fontFamily':'Arial','color':'#2c3e50',
                               'margin':'0 0 4px 0','fontSize':'13px'}),
                html.P(id='txt-umbral',
                       style={'fontFamily':'Arial','fontSize':'11px',
                              'color':'#7f8c8d','margin':'0 0 10px 0'}),
                html.Div([
                    html.Div([
                        html.Div(id='tabla-opt')
                    ], style={'width':'46%','display':'inline-block',
                              'verticalAlign':'top'}),
                    html.Div([
                        dcc.Graph(id='fig-opt', style={'height':'280px'})
                    ], style={'width':'52%','display':'inline-block',
                              'verticalAlign':'top','marginLeft':'2%'}),
                ]),
            ], style=CARD),

        ], style={'width':'74%','display':'inline-block',
                  'verticalAlign':'top','paddingLeft':'14px'}),

    ], style={'display':'flex','alignItems':'flex-start'}),

], style={'maxWidth':'1400px','margin':'auto','padding':'20px'})


# ══ CALLBACKS ══════════════════════════════════════════════════════════════════

# Mostrar/ocultar selector BD
@app.callback(
    Output('div-bd','style'),
    Input('modo','value'),
)
def toggle_bd(modo):
    return {'display':'block'} if modo == 'bd' else {'display':'none'}


# Cargar condiciones desde BD
@app.callback(
    Output('sl-T',   'value'),
    Output('sl-H2a', 'value'),
    Output('sl-H2Oa','value'),
    Output('sl-CO2a','value'),
    Output('sl-O2c', 'value'),
    Output('sl-CO2c','value'),
    Output('sl-N2c', 'value'),
    Output('sl-r1',  'value'),
    Output('feedback-cargar','children'),
    Input('btn-cargar',   'n_clicks'),
    State('exp-selector', 'value'),
    prevent_initial_call=True,
)
def cargar_bd(n, exp_id):
    if exp_id is None:
        return 650,2.2,0.41,0.55,1.3,2.15,4.87,1.97,'—'
    r = exp_df[exp_df['id_experimento']==exp_id].iloc[0]
    return (int(r['T']), round(float(r['H2a']),3),
            round(float(r['H2Oa']),3), round(float(r['CO2a']),3),
            round(float(r['O2c']),3),  round(float(r['CO2c']),3),
            round(float(r['N2c']),3),  round(float(r['r_1']),3),
            f"✓ Cargado: Exp {exp_id} (T={int(r['T'])}°C)")


# Calcular y mostrar resultados
@app.callback(
    Output('fig-curvas',      'figure'),
    Output('fig-opt',         'figure'),
    Output('tabla-opt',       'children'),
    Output('txt-umbral',      'children'),
    Output('status-calcular', 'children'),
    Input('btn-calcular', 'n_clicks'),
    State('sl-T',      'value'),
    State('sl-H2a',    'value'),
    State('sl-H2Oa',   'value'),
    State('sl-CO2a',   'value'),
    State('sl-O2c',    'value'),
    State('sl-CO2c',   'value'),
    State('sl-N2c',    'value'),
    State('sl-r1',     'value'),
    State('sl-umbral', 'value'),
    State('modo',      'value'),
    State('exp-selector','value'),
    prevent_initial_call=True,
)
def calcular(n, T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1,
             umbral, modo, exp_id):

    args = (T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1)
    j_arr = np.linspace(J_MIN, J_MAX, 400)

    fig_c = go.Figure()   # curvas E(j)
    fig_o = go.Figure()   # curvas p(j)
    filas = []

    for nombre, fn in MODELOS_FN.items():
        color = COLORES[nombre]

        try:
            if nombre == 'Nernst':
                mu  = fn(j_arr, *args)
                sig = None
            else:
                mu, sig = fn(j_arr, *args)

            if mu is None:
                continue

            # ── Curva E(j) ────────────────────────────────────────────────────
            if sig is not None:
                # Banda ±2σ sombreada
                r_int = int(color[1:3],16)
                g_int = int(color[3:5],16)
                b_int = int(color[5:7],16)
                fig_c.add_trace(go.Scatter(
                    x=np.concatenate([j_arr, j_arr[::-1]]),
                    y=np.concatenate([mu+2*sig, (mu-2*sig)[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba({r_int},{g_int},{b_int},0.12)',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False, hoverinfo='skip'))

            fig_c.add_trace(go.Scatter(
                x=j_arr, y=mu, mode='lines', name=nombre,
                line=dict(color=color, width=2.5)))

            # ── Optimización ──────────────────────────────────────────────────
            p_med = mu * j_arr

            # Curva potencia en fig_o
            fig_o.add_trace(go.Scatter(
                x=j_arr, y=p_med, mode='lines', name=nombre,
                line=dict(color=color, width=2)))

            if sig is not None:
                p_low = np.maximum(mu - 2*sig, 0.0) * j_arr
                fig_o.add_trace(go.Scatter(
                    x=np.concatenate([j_arr, j_arr[::-1]]),
                    y=np.concatenate([p_med, p_low[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba({r_int},{g_int},{b_int},0.10)',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False, hoverinfo='skip'))

            # Óptimo puntual
            idx_s  = int(np.argmax(p_med))
            j_star = float(j_arr[idx_s])
            p_star = float(p_med[idx_s])
            V_star = float(mu[idx_s])
            en_lim = abs(j_star - J_MAX) < 1e-4

            # Región óptima
            mask_r = p_med >= umbral * p_star
            idx_r  = np.where(mask_r)[0]
            j_low  = float(j_arr[idx_r[0]])  if len(idx_r) else j_star
            j_high = float(j_arr[idx_r[-1]]) if len(idx_r) else j_star

            # Sombrear región óptima
            r_int2 = int(color[1:3],16)
            g_int2 = int(color[3:5],16)
            b_int2 = int(color[5:7],16)
            fig_o.add_trace(go.Scatter(
                x=j_arr[mask_r], y=p_med[mask_r],
                fill='tozeroy',
                fillcolor=f'rgba({r_int2},{g_int2},{b_int2},0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False, hoverinfo='skip'))

            # Marcador j*
            fig_o.add_trace(go.Scatter(
                x=[j_star], y=[p_star], mode='markers',
                name=f'j* {nombre}',
                marker=dict(color=color, size=12, symbol='star',
                            line=dict(color='white',width=1)),
                showlegend=False))

            # P garantizada
            p_gar = None
            if sig is not None:
                p_low2 = np.maximum(mu-2*sig,0.0)*j_arr
                mask_g = (j_arr >= j_low) & (j_arr <= j_high)
                if mask_g.any():
                    p_gar = float(p_low2[mask_g].max())
                    fig_o.add_hline(y=p_gar, line_dash='dot',
                                    line_color=color, opacity=0.5)

            filas.append({
                'Modelo':         nombre,
                'j* (A/cm²)':    round(j_star, 4),
                'V* (V)':         round(V_star, 4),
                'Pmax (W/cm²)':  round(p_star, 5),
                'Región óptima':  f"[{j_low:.3f}, {j_high:.3f}]",
                'P garantizada':  f"{p_gar:.5f}" if p_gar else "N/A",
                '⚠':              '⚠' if en_lim else '—',
            })

        except Exception as e:
            print(f"Error {nombre}: {e}")
            continue

    # Datos reales si modo BD
    if modo == 'bd' and exp_id is not None:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            df_r = pd.read_sql(
                "SELECT i_densidad, voltaje FROM mediciones "
                "WHERE id_experimento=%s ORDER BY i_densidad",
                conn, params=(exp_id,))
            conn.close()
            if not df_r.empty:
                fig_c.add_trace(go.Scatter(
                    x=df_r['i_densidad'], y=df_r['voltaje'],
                    mode='markers', name='Datos reales (Milewski)',
                    marker=dict(color='#27ae60', size=8, symbol='circle',
                                opacity=0.8)))
        except Exception:
            pass

    # Layout curvas
    fig_c.update_layout(
        title=f'Curvas E(j) — T={T}°C',
        xaxis_title='Densidad de corriente j [A/cm²]',
        yaxis_title='Voltaje E [V]',
        template='plotly_white', font=dict(family='Arial'),
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.25, font=dict(size=10)),
        margin=dict(t=40,b=80,l=50,r=20))

    # Layout optimización
    fig_o.update_layout(
        title=f'Densidad de potencia p(j) — T={T}°C',
        xaxis_title='j [A/cm²]', yaxis_title='p [W/cm²]',
        template='plotly_white', font=dict(family='Arial'),
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.35, font=dict(size=10)),
        margin=dict(t=40,b=90,l=50,r=20))

    # Tabla
    cols = ['Modelo','j* (A/cm²)','V* (V)','Pmax (W/cm²)',
            'Región óptima','P garantizada','⚠']
    tabla = dash_table.DataTable(
        columns=[{'name':c,'id':c} for c in cols],
        data=filas,
        style_table={'fontFamily':'Arial','overflowX':'auto'},
        style_header={'backgroundColor':'#2c3e50','color':'white',
                      'fontWeight':'bold','textAlign':'center',
                      'fontSize':'11px','padding':'6px'},
        style_cell={'textAlign':'center','padding':'6px',
                    'fontFamily':'Arial','fontSize':'11px'},
        style_data_conditional=[
            {'if':{'filter_query':'{⚠} = "⚠"'},
             'backgroundColor':'#fef9e7','color':'#d35400'},
            {'if':{'row_index':'odd'},'backgroundColor':'#f8f9fa'},
        ])

    nota = html.Div([
        html.P("⚠ j*=0.200 A/cm²: el máximo real podría estar fuera del "
               "rango experimental validado.",
               style={'fontFamily':'Arial','fontSize':'10px',
                      'color':'#e67e22','margin':'4px 0 0 0'}),
        html.P("P garantizada: potencia mínima con 95% de confianza "
               "estadística (solo GPR y GPR Residual).",
               style={'fontFamily':'Arial','fontSize':'10px',
                      'color':'#7f8c8d','margin':'2px 0 0 0'}),
    ])

    txt_umbral = (f"Región óptima: rango de j donde p(j) ≥ "
                  f"{int(umbral*100)}% de Pmax")
    status = f"✓ Calculado para T={T}°C | {len(filas)} modelos"

    return fig_c, fig_o, html.Div([tabla, nota]), txt_umbral, status


if __name__ == '__main__':
    app.run(debug=True)
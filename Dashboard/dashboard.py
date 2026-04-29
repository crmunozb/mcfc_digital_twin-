import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import psycopg2

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
DB_CONFIG = {
    'host':     'localhost',
    'port':     5432,
    'database': 'mcfc_digital_twin',
    'user':     'mcfc_user',
    'password': 'Lasvioletas1756'
}

# ── Carga de datos ─────────────────────────────────────────────────────────────
def get_data():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql('''
        SELECT e.id_experimento,
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
    ['id_experimento', 'T',
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
                dash_table.DataTable(
                    id='exp-table',
                    columns=[
                        {'name': 'ID',           'id': 'id_experimento'},
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
                        'id_experimento', 'T', 'H2a', 'CO2a',
                        'O2c', 'CO2c', 'E_max', 'i_max', 'r_1'
                    ]].round(4).to_dict('records'),
                    filter_action='native', sort_action='native',
                    page_size=20,
                    style_table={'overflowX': 'auto', 'margin': '20px'},
                    style_header={'backgroundColor': '#2c3e50', 'color': 'white',
                                  'fontWeight': 'bold', 'fontFamily': 'Arial'},
                    style_cell={'fontFamily': 'Arial', 'fontSize': '13px', 'padding': '8px'},
                    style_data_conditional=[{
                        'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'
                    }]
                )
            ])
        ]),

        # ── TAB 4: Digital Twin ────────────────────────────────────
        dcc.Tab(label='Digital Twin', children=[
            html.Div([

                html.Div([
                    html.H4("Condiciones de operacion",
                            style={'fontFamily': 'Arial', 'color': '#2c3e50'}),

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
                    dcc.Slider(id='dt-CO2c', min=0.3, max=14.0, step=0.2, value=2.67,
                               marks={v: str(v) for v in [0.3, 3.0, 7.0, 14.0]}),

                    html.Label('N2 catodo (N2c)', style=SL),
                    dcc.Slider(id='dt-N2c', min=0.5, max=29.0, step=0.5, value=4.87,
                               marks={v: str(v) for v in [0.5, 5, 15, 29]}),

                    html.Label('Resistencia ohmica r1 (Ohm·cm2)', style=SL),
                    dcc.Slider(id='dt-r1', min=1.8, max=3.0, step=0.05, value=1.97,
                               marks={v: str(round(v, 2)) for v in [1.8, 2.0, 2.5, 3.0]}),

                    html.Div([
                        html.Button(
                            'Cargar experimento mas cercano',
                            id='btn-cargar-exp',
                            n_clicks=0,
                            style={
                                'marginTop': '20px',
                                'width': '100%',
                                'padding': '10px',
                                'backgroundColor': '#2980b9',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'fontFamily': 'Arial',
                                'fontSize': '13px',
                                'cursor': 'pointer',
                                'fontWeight': 'bold'
                            }
                        ),
                        html.Div(id='btn-feedback', style={
                            'fontFamily': 'Arial', 'fontSize': '12px',
                            'color': '#27ae60', 'marginTop': '6px',
                            'textAlign': 'center'
                        })
                    ]),

                ], style={**SP, 'width': '30%'}),

                html.Div([
                    dcc.Graph(id='dt-graph', style={'height': '420px'}),
                    html.Div(id='dt-metricas', style={
                        'fontFamily': 'Arial', 'fontSize': '14px',
                        'backgroundColor': '#f0f4f8', 'borderRadius': '8px',
                        'padding': '16px', 'marginTop': '12px'
                    })
                ], style={**SG, 'width': '68%'})

            ])
        ]),

        # ── TAB 5: Monitoreo Live ──────────────────────────────
        dcc.Tab(label='Monitoreo Live', children=[
            html.Div([

                # Intervalo de autorefresh (10 segundos)
                dcc.Interval(id='live-interval', interval=10_000, n_intervals=0),

                # Fila superior: selector + indicadores
                html.Div([

                    # Panel izquierdo: selector de experimento
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

                    # Panel derecho: KPIs en tiempo real
                    html.Div([
                        html.H4("Última medición",
                                style={'fontFamily': 'Arial', 'color': '#2c3e50',
                                       'marginBottom': '12px'}),
                        html.Div(id='live-kpis')
                    ], style={'width': '70%', 'display': 'inline-block',
                              'verticalAlign': 'top', 'padding': '20px'})

                ], style={'display': 'flex', 'alignItems': 'flex-start',
                          'marginBottom': '16px'}),

                # Gráficos en tiempo real
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


# ── Callbacks ──────────────────────────────────────────────────────────────────

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


@app.callback(
    Output('var-graph', 'figure'),
    Input('var-selector', 'value')
)
def update_var(variable):
    resumen = df.drop_duplicates('id_experimento')[
        ['id_experimento', 'T', variable, 'E_max']
    ].dropna()
    fig = px.scatter(
        resumen, x=variable, y='E_max', color='T',
        color_continuous_scale='plasma',
        labels={variable: variable, 'E_max': 'Voltaje maximo E_max [V]',
                'T': 'Temperatura (°C)'},
        title=f'Efecto de {variable} sobre el voltaje maximo',
        template='plotly_white'
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(font=dict(family='Arial'))
    return fig


@app.callback(
    Output('dt-graph', 'figure'),
    Output('dt-metricas', 'children'),
    Input('dt-T',    'value'),
    Input('dt-H2a',  'value'),
    Input('dt-H2Oa', 'value'),
    Input('dt-CO2a', 'value'),
    Input('dt-O2c',  'value'),
    Input('dt-CO2c', 'value'),
    Input('dt-N2c',  'value'),
    Input('dt-r1',   'value'),
)
def update_dt(T, H2a, H2Oa, CO2a, O2c, CO2c, N2c, r1):
    fig = go.Figure()

    i_pred = np.linspace(0, 0.30, 120)

    V_pred = voltaje_modelo(i_pred, T, H2a, H2Oa, CO2a, O2c, CO2c, r1, N2c=N2c)
    P_pred = V_pred * i_pred

    fig.add_trace(go.Scatter(
        x=i_pred, y=V_pred, mode='lines',
        name=f'Modelo libre (r1={r1:.2f})',
        line=dict(color='#e74c3c', width=2, dash='dot'), yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=i_pred, y=P_pred, mode='lines',
        name='P modelo libre',
        line=dict(color='#e74c3c', width=1.5, dash='dot'), yaxis='y2'
    ))

    tol_T = 13
    cands = exp_summary[abs(exp_summary['T'] - T) <= tol_T].copy()
    metricas_out = html.P("No hay experimentos cercanos para comparar.",
                          style={'color': '#888'})

    if len(cands) > 0:
        ref = np.array([H2a, H2Oa, CO2a, O2c, CO2c, N2c])
        cands = cands.copy()
        cands['dist'] = cands[['H2a', 'H2Oa', 'CO2a', 'O2c', 'CO2c', 'N2c']].apply(
            lambda row: np.linalg.norm(row.values - ref), axis=1
        )
        mejor = cands.sort_values('dist').iloc[0]
        eid   = int(mejor['id_experimento'])
        r1_real = float(mejor['r_1'])

        datos_exp = df[df['id_experimento'] == eid].sort_values('i_densidad')
        i_exp = datos_exp['i_densidad'].values
        V_exp = datos_exp['voltaje'].values

        V_ajustado = voltaje_modelo(
            i_pred, mejor['T'], mejor['H2a'], mejor['H2Oa'],
            mejor['CO2a'], mejor['O2c'], mejor['CO2c'], r1_real,
            N2a=mejor.get('N2a', 0), CO=mejor.get('CO', 0),
            CH4=mejor.get('CH4', 0), N2c=mejor['N2c'], H2Oc=mejor.get('H2Oc', 0)
        )
        fig.add_trace(go.Scatter(
            x=i_pred, y=V_ajustado, mode='lines',
            name=f'Modelo ajustado (r1={r1_real:.3f})',
            line=dict(color='#27ae60', width=2.5), yaxis='y1'
        ))
        fig.add_trace(go.Scatter(
            x=i_pred, y=V_ajustado * i_pred, mode='lines',
            name='P modelo ajustado',
            line=dict(color='#27ae60', width=2, dash='dash'), yaxis='y2'
        ))

        V_mod = voltaje_modelo(
            i_exp, mejor['T'], mejor['H2a'], mejor['H2Oa'],
            mejor['CO2a'], mejor['O2c'], mejor['CO2c'], r1_real,
            N2a=mejor.get('N2a', 0), CO=mejor.get('CO', 0),
            CH4=mejor.get('CH4', 0), N2c=mejor['N2c'], H2Oc=mejor.get('H2Oc', 0)
        )

        r2, mae, nrmse = metricas(V_exp, V_mod)

        fig.add_trace(go.Scatter(
            x=i_exp, y=V_exp, mode='markers',
            name=f'Exp {eid} (T={int(mejor["T"])}°C)',
            marker=dict(color='#2980b9', size=7), yaxis='y1'
        ))
        fig.add_trace(go.Scatter(
            x=i_exp, y=V_exp * i_exp, mode='markers',
            name=f'P exp {eid}',
            marker=dict(color='#2980b9', size=6, symbol='diamond'), yaxis='y2'
        ))

        def color_m(val, bueno, malo, inv=False):
            if not inv:
                return '#27ae60' if val >= bueno else ('#e67e22' if val >= malo else '#e74c3c')
            return '#27ae60' if val <= bueno else ('#e67e22' if val <= malo else '#e74c3c')

        card = {'display': 'inline-block', 'textAlign': 'center', 'width': '30%',
                'padding': '12px', 'backgroundColor': 'white', 'borderRadius': '8px',
                'boxShadow': '0 1px 4px #ccc', 'marginRight': '10px'}

        metricas_out = html.Div([
            html.H4("Evaluacion del modelo vs. experimento mas cercano",
                    style={'color': '#2c3e50', 'marginBottom': '8px'}),
            html.P(
                f"Referencia: ID {eid} | T={int(mejor['T'])}°C | "
                f"dist. composicion = {mejor['dist']:.3f}",
                style={'color': '#555', 'marginBottom': '12px'}
            ),
            html.Div([
                html.Div([
                    html.Div("R²", style={'fontWeight': 'bold', 'marginBottom': '4px'}),
                    html.Div(f"{r2:.4f}",
                             style={'fontSize': '24px', 'color': color_m(r2, 0.95, 0.85)})
                ], style=card),
                html.Div([
                    html.Div("MAE [V]", style={'fontWeight': 'bold', 'marginBottom': '4px'}),
                    html.Div(f"{mae:.4f}",
                             style={'fontSize': '24px',
                                    'color': color_m(mae, 0.02, 0.05, inv=True)})
                ], style=card),
                html.Div([
                    html.Div("NRMSE", style={'fontWeight': 'bold', 'marginBottom': '4px'}),
                    html.Div(f"{nrmse:.4f}",
                             style={'fontSize': '24px',
                                    'color': color_m(nrmse, 0.05, 0.10, inv=True)})
                ], style={**card, 'marginRight': '0'})
            ])
        ])

    fig.update_layout(
        title=f'Digital Twin — T={T}°C',
        xaxis_title='Densidad de corriente i [A/cm2]',
        yaxis=dict(title='Voltaje E [V]', side='left'),
        yaxis2=dict(title='Potencia P [W/cm2]', overlaying='y',
                    side='right', showgrid=False),
        hovermode='x unified', template='plotly_white',
        font=dict(family='Arial'), legend=dict(orientation='h', y=-0.22)
    )

    return fig, metricas_out


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




# ── Helpers Live ───────────────────────────────────────────────────────────────

def get_experimentos_udec():
    """Retorna lista de experimentos udec_lab ordenados por más reciente."""
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
    """Retorna todas las mediciones de un experimento ordenadas por timestamp."""
    conn = psycopg2.connect(**DB_CONFIG)
    df_m = pd.read_sql("""
        SELECT id_medicion, i_densidad, voltaje, eta, timestamp_medicion
        FROM mediciones
        WHERE id_experimento = %s
        ORDER BY timestamp_medicion ASC
    """, conn, params=(id_exp,))
    conn.close()
    return df_m


# ── Callback: actualizar selector de experimentos ──────────────────────────────
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

    # Siempre mostrar badge "ACTIVO" en el más reciente (primer elemento)
    if options:
        options[0]['label'] = "▶ " + options[0]['label']

    # Mantener selección actual si sigue existiendo, sino auto-seleccionar el más reciente
    val = current_val if current_val in ids_disponibles else options[0]['value']
    return options, val


# ── Callback: actualizar gráficos y KPIs ──────────────────────────────────────
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

    # Estilos tarjeta KPI
    card_base = {
        'display': 'inline-block', 'textAlign': 'center',
        'width': '22%', 'padding': '14px',
        'backgroundColor': 'white', 'borderRadius': '10px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.08)',
        'marginRight': '12px', 'verticalAlign': 'top'
    }

    fig_empty = go.Figure()
    fig_empty.update_layout(template='plotly_white',
                            font=dict(family='Arial'),
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

    # ── KPIs ──────────────────────────────────────────────────────────────────
    def kpi_card(titulo, valor, unidad, color='#2c3e50'):
        return html.Div([
            html.Div(titulo, style={'fontWeight': 'bold', 'fontSize': '12px',
                                    'color': '#7f8c8d', 'marginBottom': '6px'}),
            html.Div(valor,  style={'fontSize': '26px', 'fontWeight': 'bold',
                                    'color': color}),
            html.Div(unidad, style={'fontSize': '11px', 'color': '#aaa',
                                    'marginTop': '2px'}),
        ], style=card_base)

    V_last   = float(ultima['voltaje'])
    i_last   = float(ultima['i_densidad'])
    P_last   = V_last * i_last
    eta_last = float(ultima['eta']) if ultima['eta'] else 0.0

    def color_v_relativo(v, i):
        # Color dinámico: compara voltaje real vs esperado para esa densidad de corriente
        v_esperado = max(1.05 - 3.5 * i, 0.4)
        ratio = v / v_esperado if v_esperado > 0 else 1.0
        if ratio >= 0.97:
            return '#27ae60'   # verde: dentro del 3% del esperado
        elif ratio >= 0.90:
            return '#e67e22'   # naranjo: 90-97% del esperado
        else:
            return '#e74c3c'   # rojo: más del 10% bajo lo esperado

    kpis = html.Div([
        kpi_card("Voltaje",            f"{V_last:.4f}", "V",
                 color_v_relativo(V_last, i_last)),
        kpi_card("Densidad corriente", f"{i_last:.4f}", "A/cm²",  '#2980b9'),
        kpi_card("Potencia",           f"{P_last:.4f}", "W/cm²",  '#8e44ad'),
        kpi_card("Eficiencia η",  f"{eta_last:.3f}", "—",    '#16a085'),
    ], style={'display': 'flex', 'flexWrap': 'nowrap', 'gap': '12px',
              'alignItems': 'stretch'})

    # ── Info experimento ──────────────────────────────────────────────────────
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        row_exp = pd.read_sql(
            "SELECT t, h2a, co2a, o2c, co2c, n2c FROM experimentos WHERE id_experimento=%s",
            conn, params=(id_exp,)
        ).iloc[0]
        conn.close()
        info = html.Div([
            html.Div(f"T = {int(row_exp.t)} °C",       style={'marginBottom': '3px'}),
            html.Div(f"H2a = {row_exp.h2a:.3f}",       style={'marginBottom': '3px'}),
            html.Div(f"CO2a = {row_exp.co2a:.3f}",     style={'marginBottom': '3px'}),
            html.Div(f"O2c = {row_exp.o2c:.3f}",       style={'marginBottom': '3px'}),
            html.Div(f"CO2c = {row_exp.co2c:.3f}",     style={'marginBottom': '3px'}),
            html.Div(f"Mediciones: {n_med}",
                     style={'marginTop': '8px', 'fontWeight': 'bold', 'color': '#2980b9'}),
            html.Div(f"Última: {ts_str}",
                     style={'fontSize': '11px', 'color': '#aaa', 'marginTop': '3px'}),
        ])
    except Exception:
        info = html.P("—")

    status = f"Actualizando... {n_med} mediciones recibidas"

    # ── Figura: Voltaje vs tiempo ─────────────────────────────────────────────
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(
        x=df_m['timestamp_medicion'], y=df_m['voltaje'],
        mode='lines+markers',
        line=dict(color='#2980b9', width=2),
        marker=dict(size=6),
        name='Voltaje'
    ))
    fig_v.update_layout(
        title='Voltaje E [V] — tiempo real',
        xaxis_title='Tiempo', yaxis_title='E [V]',
        template='plotly_white', font=dict(family='Arial'),
        margin=dict(t=40, b=40, l=50, r=20)
    )

    # ── Figura: Potencia vs tiempo ────────────────────────────────────────────
    df_m['potencia'] = df_m['voltaje'] * df_m['i_densidad']
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=df_m['timestamp_medicion'], y=df_m['potencia'],
        mode='lines+markers',
        line=dict(color='#8e44ad', width=2),
        marker=dict(size=6),
        name='Potencia'
    ))
    fig_p.update_layout(
        title='Potencia P [W/cm²] — tiempo real',
        xaxis_title='Tiempo', yaxis_title='P [W/cm²]',
        template='plotly_white', font=dict(family='Arial'),
        margin=dict(t=40, b=40, l=50, r=20)
    )

    # ── Figura: Curva de polarización acumulada ───────────────────────────────
    fig_polar = go.Figure()
    fig_polar.add_trace(go.Scatter(
        x=df_m['i_densidad'], y=df_m['voltaje'],
        mode='lines+markers',
        line=dict(color='#27ae60', width=2),
        marker=dict(size=7, color='#27ae60'),
        name='Curva experimental'
    ))
    # Curva del modelo sobre los mismos puntos (si hay suficientes datos)
    if n_med >= 3:
        try:
            conn2 = psycopg2.connect(**DB_CONFIG)
            row_e = pd.read_sql(
                "SELECT t, h2a, h2oa, co2a, o2c, co2c, n2c, h2oc, co, ch4 "
                "FROM experimentos WHERE id_experimento=%s",
                conn2, params=(id_exp,)
            ).iloc[0]
            # Buscar r1: primero en parametros_modelo, fallback a valor típico MCFC
            pm = pd.read_sql(
                "SELECT r_1 FROM parametros_modelo WHERE id_experimento=%s LIMIT 1",
                conn2, params=(id_exp,)
            )
            conn2.close()
            if not pm.empty and pm['r_1'].iloc[0] is not None:
                r1_val = float(pm['r_1'].iloc[0])
            else:
                # Fallback: valor típico MCFC a 650°C
                r1_val = 1.973
            i_range = np.linspace(0, df_m['i_densidad'].max() * 1.1, 80)
            V_mod = voltaje_modelo(
                i_range, float(row_e.t), float(row_e.h2a), float(row_e.h2oa),
                float(row_e.co2a), float(row_e.o2c), float(row_e.co2c), r1_val,
                N2a=float(row_e.co), CO=float(row_e.co), CH4=float(row_e.ch4),
                N2c=float(row_e.n2c), H2Oc=float(row_e.h2oc)
            )
            fig_polar.add_trace(go.Scatter(
                x=i_range, y=V_mod, mode='lines',
                line=dict(color='#e74c3c', width=2, dash='dot'),
                name='Modelo DT'
            ))
        except Exception:
            pass

    fig_polar.update_layout(
        title=f'Curva de Polarización acumulada — Exp {id_exp}',
        xaxis_title='Densidad de corriente i [A/cm²]',
        yaxis_title='Voltaje E [V]',
        template='plotly_white', font=dict(family='Arial'),
        hovermode='x unified',
        margin=dict(t=40, b=50, l=50, r=20),
        legend=dict(orientation='h', y=-0.2)
    )

    return fig_v, fig_p, fig_polar, kpis, info, status


if __name__ == '__main__':
    app.run(debug=True)
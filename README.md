# Digital Twin — Celda de Combustible de Carbonatos Fundidos (MCFC)

> **Memoria de Título** — Ingeniería Civil Informática, Universidad de Concepción  
> Autor: Cristóbal Muñoz Barrios · Profesor Guía: Hugo Garcés Hernández  
> Co-Guía: Andrés Escalona (Depto. Ingeniería Mecánica, UdeC)  
> Datos experimentales: Prof. Jarosław Milewski — Warsaw University of Technology

---

## ¿Qué es este proyecto?

Este repositorio implementa un **Digital Twin** (gemelo digital) para una celda de combustible de carbonatos fundidos (MCFC, *Molten Carbonate Fuel Cell*). El sistema integra modelos de machine learning con un modelo semi-empírico basado en la ecuación de Nernst para predecir y optimizar la densidad de potencia de la celda bajo distintas condiciones operacionales.

La MCFC opera a alta temperatura (550–650 °C) con electrolito de carbonato fundido Li₂CO₃/K₂CO₃, siendo relevante para generación de energía estacionaria y sistemas de hidrógeno.

### Componentes principales

| Componente | Descripción |
|---|---|
| Base de datos | PostgreSQL 14 con 111 experimentos y 1.171 mediciones reales |
| Modelos | Nernst, PLS, KPLS, GPR, GPR Residual (4 variantes c/u) |
| Dashboard | Aplicación Plotly Dash — optimizador interactivo |
| Despliegue remoto | Render + Gunicorn (`https://mcfc-digital-twin.onrender.com`) |

---

## Inicio rápido

### Prerrequisitos

- Python 3.10+
- PostgreSQL 14
- Git

### 1. Clonar el repositorio

```bash
git clone https://github.com/crmunozb/mcfc_digital_twin.git
cd mcfc_digital_twin
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configurar la base de datos

```bash
# Crear la base de datos en PostgreSQL
createdb mcfc_digital_twin

# Cargar el esquema
psql -d mcfc_digital_twin -f Database/schema.sql

# Cargar los datos experimentales desde el Excel
python Database/load_data.py
```

> **Nota:** Los datos experimentales originales se encuentran en `Data/Data_original_PGNN.xlsx`
> (111 experimentos, 1.171 mediciones — cortesía del Prof. Milewski, Warsaw University of Technology).

### 4. Configurar la conexión a la base de datos

Copia el archivo de ejemplo y edita tus credenciales:

```bash
cp config.example.py config.py
```

Edita `config.py`:

```python
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mcfc_digital_twin",
    "user": "TU_USUARIO",
    "password": "TU_PASSWORD"
}
```

### 5. Ejecutar el dashboard localmente

```bash
python Dashboard/dashboard.py
```

Abre `http://localhost:8050` en tu navegador.

---

## Estructura del repositorio

```
mcfc_digital_twin/
│
├── Dashboard/
│   └── dashboard.py                        # Aplicación Plotly Dash (punto de entrada)
│
├── Data/
│   └── Data_original_PGNN.xlsx             # Dataset experimental (Milewski, WUT)
│
├── Database/
│   ├── schema.sql                          # Definición del esquema PostgreSQL
│   └── load_data.py                        # Script de carga de datos desde Excel
│
├── modelos/
│   ├── modelo_nernst.py                    # Modelo semi-empírico (ecuación de Nernst)
│   │
│   ├── entrenar_pls_cv.py                  # Entrenamiento PLS con validación cruzada
│   ├── entrenar_kpls_cv.py                 # Entrenamiento KPLS
│   ├── entrenar_gpr.py                     # Entrenamiento GPR (kernel ARD)
│   ├── entrenar_gpr_residual.py            # Entrenamiento GPR Residual (híbrido Nernst)
│   │
│   ├── evaluar_nernst.py                   # Evaluación modelo Nernst
│   ├── evaluar_holdout.py                  # Evaluación en experimento holdout
│   ├── evaluar_por_experimento.py          # Métricas por experimento
│   │
│   ├── curva_aprendizaje.py                # Curvas de aprendizaje
│   ├── curva_aprendizaje_mae.pdf           # Gráfico MAE
│   ├── curva_aprendizaje_nrmse.pdf         # Gráfico NRMSE
│   ├── curva_aprendizaje_r2.pdf            # Gráfico R²
│   │
│   ├── generar_graficos_curva.py           # Generación de gráficos de curvas
│   ├── generar_todos_los_resumenes.py      # Resúmenes globales de métricas
│   ├── cargar_datos.py                     # Utilidades de carga
│   │
│   ├── evaluacion_global.csv               # Métricas globales todos los modelos
│   ├── evaluacion_holdout.csv              # Métricas en holdout
│   ├── evaluacion_por_experimento_*.csv    # Métricas por experimento (warsaw/balanceado)
│   ├── resumen_augmentacion.csv            # Resumen generación de datos sintéticos
│   ├── resumen_por_temperatura*.csv        # Métricas desagregadas por temperatura
│   │
│   ├── pls_voltaje_cv_warsaw.pkl           # Modelo PLS — dataset warsaw
│   ├── pls_voltaje_cv_balanceado.pkl       # Modelo PLS — dataset balanceado
│   ├── pls_voltaje_cv_warsaw_holdout.pkl   # Modelo PLS — warsaw sin holdout
│   ├── pls_voltaje_cv_balanceado_holdout.pkl
│   ├── kpls_voltaje_cv_warsaw.pkl          # Modelos KPLS (ídem variantes)
│   ├── kpls_voltaje_cv_balanceado.pkl
│   ├── kpls_voltaje_cv_warsaw_holdout.pkl
│   ├── kpls_voltaje_cv_balanceado_holdout.pkl
│   ├── gpr_voltaje_warsaw.pkl              # Modelos GPR (ídem variantes)
│   ├── gpr_voltaje_balanceado.pkl
│   ├── gpr_voltaje_warsaw_holdout.pkl
│   ├── gpr_voltaje_balanceado_holdout.pkl
│   ├── gpr_residual_warsaw.pkl             # Modelos GPR Residual (ídem variantes)
│   ├── gpr_residual_balanceado.pkl
│   ├── gpr_residual_warsaw_holdout.pkl
│   └── gpr_residual_balanceado_holdout.pkl
│
├── simulador/
│   ├── generar_datos_sinteticos_mcfc.py    # Generador de curvas sintéticas (Nernst)
│   └── preview_sintetico.csv              # Muestra de datos generados
│
├── optimizador_mcfc.py                     # Optimizador j* (scipy.minimize_scalar)
├── config.py                               # Credenciales BD (NO incluir en git)
├── config.example.py                       # Plantilla de configuración
├── Procfile                                # Gunicorn: --chdir Dashboard dashboard:server
├── requirements.txt                        # Dependencias Python
└── README.md
```

---

## Descripción técnica detallada

### Base de datos

El esquema PostgreSQL contiene tres tablas principales:

```sql
experimentos       -- Condiciones operacionales por experimento (T, presiones parciales)
mediciones         -- Curvas de polarización (j, V, P) por experimento
parametros_modelo  -- Parámetros ajustados del modelo Nernst por experimento
```

La conexión se gestiona con `psycopg2` mediante el diccionario `DB_CONFIG` definido en `config.py`.

En el entorno de Render (producción), la base de datos del laboratorio no es accesible. El dashboard opera automáticamente en **modo BD OFFLINE**, cargando predicciones directamente desde los archivos `.pkl`.

### Modelos de machine learning

Se entrenaron cinco familias de modelos en cuatro variantes cada una (warsaw / balanceado / warsaw_holdout / balanceado_holdout):

| Modelo | Descripción | R² destacado |
|---|---|---|
| **Nernst** | Semi-empírico — baseline físico | — |
| **PLS** | Partial Least Squares (n_components=7) | 0,966 |
| **KPLS** | Kernel PLS (kernel RBF) | 0,983 |
| **GPR** | Gaussian Process Regression (kernel ARD) | 0,997 |
| **GPR Residual** | Híbrido: Nernst + GPR sobre residuos | 0,998 |

**Variables de entrada (features):**

| Variable | Descripción | Unidad |
|---|---|---|
| `T` | Temperatura de operación | °C |
| `p_H2` | Presión parcial de H₂ | bar |
| `p_CO2_c` | Presión parcial de CO₂ (cátodo) | bar |
| `p_O2` | Presión parcial de O₂ | bar |
| `j` | Densidad de corriente | A/m² |
| `r1` | Resistencia óhmica | Ω·m² |

**Variable objetivo:** Voltaje de celda `V` [V]

Las variantes **warsaw** usan el dataset experimental original (94,8% datos a 650 °C). Las variantes **balanceado** incorporan datos sintéticos para equilibrar la representación de las tres temperaturas.

### Generación de datos sintéticos

Para mitigar el severo desbalance de temperatura, se generaron **500 curvas de polarización sintéticas** mediante el modelo de Nernst con perturbación controlada en `simulador/generar_datos_sinteticos_mcfc.py`. Esto permite entrenar modelos con representación balanceada a 550, 600 y 650 °C.

Un experimento holdout (excluido del entrenamiento) valida que los modelos balanceados generalizan coherentemente a temperaturas no vistas en datos reales.

### Optimizador

`optimizador_mcfc.py` encuentra el punto j* que maximiza la densidad de potencia P = j · V(j) usando `scipy.optimize.minimize_scalar` con método *bounded*:

```python
from optimizador_mcfc import optimizar_punto_operacion

resultado = optimizar_punto_operacion(
    modelo=modelo_gpr,
    T=650,          # °C
    p_H2=0.8,       # bar
    p_CO2_c=0.3,    # bar
    p_O2=0.18,      # bar
    r1=0.001        # Ω·m²
)
# → {'j_optimo': ..., 'V_optimo': ..., 'P_max': ...}
```

### Dashboard (Plotly Dash)

El dashboard está diseñado como **herramienta de investigación remota** para explorar el Digital Twin sin acceso al laboratorio. Sus funcionalidades principales:

- **Optimizador:** Ingresa condiciones operacionales y obtiene j* de máxima potencia con curva de polarización predicha.
- **Comparación de modelos:** Visualiza métricas R², RMSE y MAE entre modelos y variantes.
- **Reproductor de experimentos:** Reproduce curvas reales almacenadas en la base de datos.

> **Distinción importante — laboratorio vs. producción:**
>
> | Entorno | Acceso BD | Modelos | Uso previsto |
> |---|---|---|---|
> | **Laboratorio (este repo)** | Si — PostgreSQL en vivo | Si — `.pkl` locales | Desarrollo, reentrenamiento, DAQ |
> | **Render (remoto)** | No — BD OFFLINE | Si — `.pkl` embebidos | Demostracion e investigacion remota |

### Despliegue en Render

```
# Procfile
web: gunicorn --chdir Dashboard dashboard:server
```

Variable de entorno requerida en Render:

```
BD_OFFLINE=true
```

---

## Flujo de replicabilidad completo

```
1.  git clone https://github.com/crmunozb/mcfc_digital_twin.git
2.  pip install -r requirements.txt
3.  createdb mcfc_digital_twin
4.  psql -d mcfc_digital_twin -f Database/schema.sql
5.  cp config.example.py config.py  →  editar credenciales
6.  python Database/load_data.py    →  carga Data_original_PGNN.xlsx
7.  python Dashboard/dashboard.py   →  dashboard en localhost:8050
```

**Para re-entrenar modelos** (opcional — los `.pkl` ya están incluidos):

```bash
# En orden sugerido
python modelos/entrenar_pls_cv.py
python modelos/entrenar_kpls_cv.py
python modelos/entrenar_gpr.py
python modelos/entrenar_gpr_residual.py
```

**Para regenerar datos sintéticos:**

```bash
python simulador/generar_datos_sinteticos_mcfc.py
```

---

## Dependencias principales

```
dash>=2.14
plotly>=5.18
scikit-learn==1.5.0
psycopg2-binary>=2.9
scipy>=1.11
numpy>=1.26
pandas>=2.1
gunicorn>=21.2
joblib>=1.3
openpyxl>=3.1        # lectura de Data_original_PGNN.xlsx
```

Versión completa en `requirements.txt`.

---

## Contexto experimental

Los datos provienen de **111 experimentos** realizados en el laboratorio del Prof. Jarosław Milewski (Warsaw University of Technology), con una MCFC de electrolito Li₂CO₃/K₂CO₃ operando entre 550 y 650 °C. El dataset contiene **1.171 mediciones** de curvas de polarización bajo distintas composiciones de gas y temperaturas.

En paralelo, el equipo de la Universidad de Concepción desarrolló un sistema de adquisición de datos (DAQ) y SCADA para monitoreo en tiempo real de la misma celda, utilizando termopares tipo K, protocolo Modbus TCP y una Raspberry Pi como unidad de control — documentado en la tesis complementaria de Luciano Toledo (UdeC, 2025).

---

## Citación

Si utilizas este trabajo en una publicación, por favor cita:

```
Muñoz Barrios, C. (2026). Desarrollo de un Digital Twin para el Modelamiento
y Análisis Operacional de una Celda de Combustible de Carbonatos Fundidos (MCFC).
Memoria de Título, Ingeniería Civil Informática, Universidad de Concepción.
```

---

## Contacto

**Cristóbal Muñoz Barrios** — Autor  
GitHub: [@crmunozb](https://github.com/crmunozb)

**Prof. Hugo Garcés Hernández** — Profesor Guía  
Departamento de Ingeniería Eléctrica, Universidad de Concepción
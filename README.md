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
| Base de datos | PostgreSQL 14 con 110 experimentos y 1.171 mediciones reales |
| Modelos | Nernst, PLS, KPLS, GPR, GPR Residual (4 variantes c/u) |
| Dashboard | Aplicación Plotly Dash — optimizador interactivo |
| Despliegue remoto | Render + Gunicorn (`https://mcfc-digital-twin.onrender.com`) |

---

## Inicio rápido

### Prerrequisitos

- Python 3.10 (probado en 3.10.12 — ver `.python-version`)
- PostgreSQL 14
- Git

### Verificación rápida con `start.sh` (sin base de datos)

El repositorio incluye un lanzador único que **verifica los modelos y corre el
optimizador** usando los `.pkl` ya entrenados, sin necesidad de configurar la
base de datos. Es la forma más rápida de comprobar que todo funciona tras clonar:

```bash
chmod +x start.sh          # solo la primera vez
./start.sh                 # verifica los 5 modelos + optimizador (T=650 °C)
./start.sh --temp 550      # otra temperatura (550 / 575 / 600 / 625 / 650)
./start.sh --variante balanceado   # usa los modelos del dataset balanceado
./start.sh --dashboard     # lanza el dashboard interactivo (localhost:8050)
./start.sh --retrain       # reentrena todos los modelos (requiere BD — ver aviso)
./start.sh --help          # muestra la ayuda
```

> El modo por defecto es **no destructivo**: carga los modelos serializados,
> comprueba que sus predicciones son físicamente coherentes (curva de
> polarización monótona decreciente, incertidumbre positiva) y ejecuta el
> optimizador operacional. No toca la base de datos ni los archivos `.pkl`.
>
> El modo `--retrain` sí reentrena y **sobrescribe** los modelos que respaldan
> las tablas de la memoria; úsalo solo de forma deliberada y con PostgreSQL
> configurado.

`run_modelos.py` puede invocarse directamente con las mismas opciones
(`--temp`, `--variante`, y además `--csv salida.csv` para exportar las predicciones).

### 1. Clonar el repositorio

```bash
git clone https://github.com/crmunozb/mcfc_digital_twin-.git
cd mcfc_digital_twin-
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python3 -m venv venv
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
python3 Database/load_data.py
```

> **Nota:** Los datos experimentales originales se encuentran en `Data/Data_original_PGNN.xlsx`
> (110 experimentos, 1.171 mediciones — cortesía del Prof. Milewski, Warsaw University of Technology).

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
python3 Dashboard/dashboard.py
```

También se puede lanzar con el botón único:

```bash
./start.sh --dashboard
```

Abre `http://localhost:8050` en tu navegador.

---

## Estructura del repositorio

```
mcfc_digital_twin-/
│
├── start.sh                                # Lanzador: verificación + optimizador / dashboard / retrain
├── run_modelos.py                          # Runner de verificación de modelos + optimizador
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
├── models/
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
│   ├── curva_aprendizaje.py                # Script de curvas de aprendizaje
│   ├── curva_aprendizaje_mae.pdf           # Gráfico MAE
│   ├── curva_aprendizaje_nrmse.pdf         # Gráfico NRMSE
│   ├── curva_aprendizaje_r2.pdf            # Gráfico R²
│   ├── curva_aprendizaje_resultados.csv    # Datos de curvas de aprendizaje
│   │
│   ├── optimizador_mcfc.py                 # Optimizador j* (scipy.minimize_scalar)
│   ├── generar_todos_los_resumenes.py      # Resúmenes globales de métricas
│   ├── cargar_datos.py                     # Utilidades de carga y muestreo balanceado
│   │
│   ├── evaluacion_global.csv               # Métricas globales todos los modelos
│   ├── evaluacion_holdout.csv              # Métricas en holdout
│   ├── evaluacion_por_experimento_*.csv    # Métricas por experimento
│   ├── resumen_por_temperatura*.csv        # Métricas desagregadas por temperatura
│   │
│   ├── pls_voltaje_cv_warsaw.pkl           # Modelo PLS — dataset warsaw
│   ├── pls_voltaje_cv_balanceado.pkl       # Modelo PLS — dataset balanceado
│   ├── pls_voltaje_cv_warsaw_holdout.pkl   # Modelo PLS — warsaw holdout
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
├── simulator/
│   └── generar_datos_sinteticos_mcfc.py    # Generador de curvas sintéticas (Nernst)
│
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
experimentos       -- Condiciones operacionales por experimento (T, composición gaseosa)
mediciones         -- Curvas de polarización (j, V) por experimento
parametros_modelo  -- Parámetros ajustados del modelo Nernst por experimento
```

La conexión se gestiona con `psycopg2` mediante el diccionario `DB_CONFIG` definido en `config.py`.

En el entorno de Render (producción), la base de datos del laboratorio no es accesible. El dashboard opera automáticamente en **modo BD OFFLINE**, cargando predicciones directamente desde los archivos `.pkl`.

### Modelos de machine learning

Se entrenaron cinco familias de modelos en cuatro variantes cada una (warsaw / balanceado / warsaw_holdout / balanceado_holdout):

| Modelo | Descripción | R² test (balanceado) |
|---|---|---|
| **Nernst** | Semi-empírico — baseline físico | 0,993* |
| **PLS** | Partial Least Squares (n_components=9) | 0,969 |
| **KPLS** | Kernel PLS — kernel RBF | 0,983 |
| **GPR** | Gaussian Process Regression (kernel ARD) | 0,997 |
| **GPR Residual** | Híbrido: Nernst + GPR sobre residuos | 0,998 |

> *\* El R² de Nernst sobre el dataset balanceado no constituye un resultado independiente:
> los datos sintéticos fueron generados con el propio modelo de Nernst, por lo que su
> ajuste sobre ellos es esperado por construcción. La métrica de referencia honesta de
> Nernst es R² = 0,906 sobre el dataset experimental real (`warsaw_ut`).*

**Variables de entrada (9 features):**

| Variable | Descripción |
|---|---|
| `T` | Temperatura de operación (°C) |
| `H2a` | Fracción molar H₂ en ánodo |
| `H2Oa` | Fracción molar H₂O en ánodo |
| `CO2a` | Fracción molar CO₂ en ánodo |
| `O2c` | Fracción molar O₂ en cátodo |
| `CO2c` | Fracción molar CO₂ en cátodo |
| `N2c` | Fracción molar N₂ en cátodo |
| `j` | Densidad de corriente (A/cm²) |
| `r1` | Resistencia óhmica (Ω·cm²) |

**Variable objetivo:** Voltaje de celda `V` [V]

Las variantes **warsaw** usan el dataset experimental original (94,8% datos a 650 °C). Las variantes **balanceado** incorporan datos sintéticos para equilibrar la representación entre los cinco niveles de temperatura (400 puntos por temperatura, 2.000 en total).

### Generación de datos sintéticos

Para mitigar el severo desbalance de temperatura, se generaron **500 curvas de polarización sintéticas** (100 por temperatura) mediante el modelo de Nernst con perturbación controlada en `simulator/generar_datos_sinteticos_mcfc.py`. Un análisis de curvas de aprendizaje confirmó que 400 puntos por temperatura es suficiente para la convergencia de todos los modelos.

Un experimento holdout (excluido del entrenamiento) valida que los modelos balanceados generalizan coherentemente a temperaturas no vistas en datos reales, con R² ≥ 0,84 para GPR y GPR Residual.

### Optimizador

`models/optimizador_mcfc.py` encuentra el punto j* que maximiza la densidad de potencia P = j · V(j) usando `scipy.optimize.minimize_scalar` con método *bounded*. Para GPR y GPR Residual reporta además la región óptima de operación y la potencia mínima garantizada bajo incertidumbre ±2σ.

### Dashboard (Plotly Dash)

El dashboard está diseñado como **herramienta de investigación remota** para explorar el Digital Twin sin acceso al laboratorio. Sus funcionalidades principales:

- **Optimizador interactivo:** Ingresa condiciones operacionales y obtiene j* de máxima potencia con curvas de polarización predichas y bandas de incertidumbre ±2σ.
- **Comparación de modelos:** Tabla comparativa con j*, V*, P_max, región óptima y potencia garantizada para los cinco modelos.
- **Análisis ARD:** Panel de relevancia de variables del GPR Residual.
- **Zoom interactivo y exportación PNG** del gráfico de curvas de polarización.

> **Distinción importante — laboratorio vs. producción:**
>
> | Entorno | Acceso BD | Modelos | Uso previsto |
> |---|---|---|---|
> | **Laboratorio (local)** | Sí — PostgreSQL en vivo | Sí — `.pkl` locales | Desarrollo, reentrenamiento, DAQ |
> | **Render (remoto)** | No — BD OFFLINE | Sí — `.pkl` embebidos | Demostración e investigación remota |

### Despliegue en Render

```
# Procfile
web: gunicorn --chdir Dashboard dashboard:server
```

---

## Flujo de replicabilidad completo

```
1.  git clone https://github.com/crmunozb/mcfc_digital_twin-.git
2.  pip install -r requirements.txt
3.  createdb mcfc_digital_twin
4.  psql -d mcfc_digital_twin -f Database/schema.sql
5.  cp config.example.py config.py  →  editar credenciales
6.  python3 Database/load_data.py    →  carga Data_original_PGNN.xlsx
7.  python3 Dashboard/dashboard.py   →  dashboard en localhost:8050
```

**Para re-entrenar modelos** (opcional — los `.pkl` ya están incluidos en el entorno local):

```bash
# 1. (Solo para variantes _balanceado) generar e insertar los datos sintéticos en la BD
python3 simulator/generar_datos_sinteticos_mcfc.py --insertar

# 2. Entrenar los modelos. Cada script acepta --fuente para elegir el dataset:
#    --fuente warsaw_ut            → variante _warsaw  (solo datos reales)
#    --fuente warsaw_ut sintetico  → variante _balanceado (real + sintético, por defecto)
python3 models/entrenar_pls_cv.py
python3 models/entrenar_kpls_cv.py
python3 models/entrenar_gpr.py
python3 models/entrenar_gpr_residual.py
```

> **Importante:** las variantes `_balanceado` leen los datos sintéticos desde la base
> de datos, por lo que el paso 1 debe ejecutarse **antes** del re-entrenamiento. Si se
> entrena sin haber insertado los sintéticos, solo podrán generarse las variantes `_warsaw`.

---

## Dependencias principales

```
dash
plotly
scikit-learn==1.5.0
psycopg2-binary
scipy
numpy
pandas
gunicorn==26.0.0
joblib
```

Versión completa en `requirements.txt`.

---

## Contexto experimental

Los datos provienen de **110 experimentos** realizados en el laboratorio del Prof. Jarosław Milewski (Warsaw University of Technology), con una MCFC de electrolito Li₂CO₃/K₂CO₃ operando entre 550 y 650 °C. El dataset contiene **1.171 mediciones** de curvas de polarización bajo distintas composiciones de gas y temperaturas.

Este Digital Twin forma parte de un sistema mayor desarrollado en la Universidad de Concepción, que incluye una infraestructura de adquisición de datos y prototipo de integración con datalogger Novus FieldLogger mediante protocolo Modbus TCP.

---

## Citación

Si utiliza este trabajo en una publicación, por favor citar:

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
Departamento de Ingeniería Informática, Universidad de Concepción
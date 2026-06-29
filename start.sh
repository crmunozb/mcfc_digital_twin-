#!/usr/bin/env bash
#
# start.sh — Lanzador del Digital Twin MCFC
# =========================================
# Botón único para operar el sistema. Por defecto hace lo SEGURO:
# carga los modelos ya entrenados, los verifica y corre el optimizador.
# El reentrenamiento es una opción explícita porque sobrescribe los .pkl.
#
# Uso:
#   ./start.sh                 Verifica los 5 modelos + optimizador (sin BD)
#   ./start.sh --temp 550      Verifica a otra temperatura (550/575/600/625/650)
#   ./start.sh --dashboard     Lanza el dashboard interactivo (localhost:8050)
#   ./start.sh --retrain       Reentrena TODOS los modelos (requiere BD; ver aviso)
#   ./start.sh --help          Muestra esta ayuda
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
TEMP="650"
VARIANTE="warsaw"
MODE="verify"

# ── Parseo de argumentos ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --temp)       TEMP="$2"; shift 2 ;;
    --variante)   VARIANTE="$2"; shift 2 ;;
    --dashboard)  MODE="dashboard"; shift ;;
    --retrain)    MODE="retrain"; shift ;;
    --help|-h)    sed -n '3,20p' "$0" | sed 's/^#//'; exit 0 ;;
    *) echo "Argumento desconocido: $1 (usa --help)"; exit 1 ;;
  esac
done

# ── Chequeo rápido de versión de scikit-learn (debe ser 1.5.0) ───────────────
SKVER=$($PYTHON -c "import sklearn; print(sklearn.__version__)" 2>/dev/null || echo "no-instalado")
if [[ "$SKVER" != "1.5.0" ]]; then
  echo "⚠  AVISO: scikit-learn instalado = $SKVER, pero los modelos fueron"
  echo "   serializados con 1.5.0. Versiones distintas pueden alterar las"
  echo "   predicciones. Se recomienda:  pip install -r requirements.txt"
  echo ""
fi

case "$MODE" in
  verify)
    echo "▶ Verificando modelos y corriendo optimizador (T=${TEMP}°C, variante=${VARIANTE})…"
    echo ""
    exec "$PYTHON" run_modelos.py --temp "$TEMP" --variante "$VARIANTE"
    ;;

  dashboard)
    echo "▶ Lanzando dashboard interactivo en http://localhost:8050 …"
    echo "  (Ctrl+C para detener)"
    exec "$PYTHON" Dashboard/dashboard.py
    ;;

  retrain)
    echo "════════════════════════════════════════════════════════════════════"
    echo "  ⚠  REENTRENAMIENTO COMPLETO"
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Esto SOBRESCRIBE los 16 archivos .pkl en models/, que son los que"
    echo "  respaldan las tablas de la memoria. Requiere PostgreSQL en marcha"
    echo "  con los datos cargados (config.py configurado)."
    echo ""
    echo "  Pasos que ejecutará:"
    echo "    1. Insertar datos sintéticos en la BD  (--n_curvas 100)"
    echo "    2. Reentrenar PLS, KPLS, GPR y GPR Residual"
    echo ""
    read -r -p "  ¿Continuar? Escribe 'si' para confirmar: " RESP
    if [[ "$RESP" != "si" ]]; then
      echo "  Cancelado. No se modificó ningún modelo."
      exit 0
    fi
    echo ""
    echo "▶ [1/2] Generando e insertando datos sintéticos…"
    "$PYTHON" simulator/generar_datos_sinteticos_mcfc.py --n_curvas 100 --insertar
    echo ""
    echo "▶ [2/2] Reentrenando modelos…"
    "$PYTHON" models/entrenar_pls_cv.py
    "$PYTHON" models/entrenar_kpls_cv.py
    "$PYTHON" models/entrenar_gpr.py
    "$PYTHON" models/entrenar_gpr_residual.py
    echo ""
    echo "✓ Reentrenamiento completo. Verificando los modelos nuevos…"
    "$PYTHON" run_modelos.py --temp "$TEMP" --variante "$VARIANTE"
    ;;
esac
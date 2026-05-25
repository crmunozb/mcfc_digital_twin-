"""
probar_pkls.py
--------------
Prueba los modelos PLS y KPLS guardados en archivos .pkl.

Uso:
    cd ~/mcfc_digital_twin
    python3 modelos/probar_pkls.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATASET = os.path.join(ROOT_DIR, "Data", "Data_original_PGNN.xlsx")

MODELOS = {
    "PLS": os.path.join(BASE_DIR, "pls_voltaje_cv.pkl"),
    "KPLS": os.path.join(BASE_DIR, "kpls_voltaje_cv.pkl"),
}

FEATURES_DEFAULT = [
    "T",
    "H2a",
    "H2Oa",
    "CO2a",
    "O2c",
    "CO2c",
    "N2c",
    "i, A/cm²",
    "r_1",
]

TARGET = "Experiment"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def nrmse(y_true, y_pred, rango):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / rango


def predecir_desde_pkl(obj, X_df):
    """
    Intenta predecir con distintas estructuras posibles de .pkl:
    - Pipeline de scikit-learn directo
    - Diccionario con 'pipeline'
    - Diccionario con 'modelo'
    - Diccionario con 'pls' + scaler_X + scaler_y
    """

    # Caso 1: el .pkl es directamente un modelo/pipeline
    if hasattr(obj, "predict"):
        return obj.predict(X_df).ravel()

    # Caso 2: el .pkl es un diccionario
    if isinstance(obj, dict):
        features = obj.get("features", FEATURES_DEFAULT)
        X_use = X_df[features]

        # Pipeline guardado
        for key in ["pipeline", "modelo", "model", "regressor", "kpls", "pls_pipeline"]:
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key].predict(X_use).ravel()

        # PLS clásico guardado con scalers
        if "pls" in obj and "scaler_X" in obj and "scaler_y" in obj:
            X_sc = obj["scaler_X"].transform(X_use)
            y_pred_sc = obj["pls"].predict(X_sc).ravel()
            y_pred = obj["scaler_y"].inverse_transform(
                y_pred_sc.reshape(-1, 1)
            ).ravel()
            return y_pred

        # Otro nombre posible del regresor
        if "regresor" in obj and "scaler_X" in obj and "scaler_y" in obj:
            X_sc = obj["scaler_X"].transform(X_use)
            y_pred_sc = obj["regresor"].predict(X_sc).ravel()
            y_pred = obj["scaler_y"].inverse_transform(
                y_pred_sc.reshape(-1, 1)
            ).ravel()
            return y_pred

        raise ValueError(
            f"No se encontró una estructura de predicción reconocible. "
            f"Claves disponibles: {list(obj.keys())}"
        )

    raise TypeError(f"Tipo de objeto no soportado: {type(obj)}")


def main():
    print("Cargando dataset...")
    df = pd.read_excel(DATASET)

    required = FEATURES_DEFAULT + [TARGET]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas en el Excel: {missing}")

    df_clean = df[required].dropna()

    X = df_clean[FEATURES_DEFAULT]
    y = df_clean[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    rango = y.max() - y.min()

    print(f"Filas utilizadas: {len(df_clean)}")
    print(f"Train: {len(X_train)} filas")
    print(f"Test:  {len(X_test)} filas")

    for nombre, path in MODELOS.items():
        print("\n" + "=" * 70)
        print(f"Probando modelo: {nombre}")
        print(f"Archivo: {path}")

        if not os.path.exists(path):
            print("No existe este archivo.")
            continue

        obj = joblib.load(path)

        if isinstance(obj, dict):
            print("Claves del .pkl:", list(obj.keys()))

            if "features" in obj:
                print("Features guardadas:", obj["features"])

            if "best_params" in obj:
                print("Mejores parámetros:", obj["best_params"])

        y_pred_train = predecir_desde_pkl(obj, X_train)
        y_pred_test = predecir_desde_pkl(obj, X_test)

        r2_train = r2_score(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        nrmse_train = nrmse(y_train, y_pred_train, rango)

        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        nrmse_test = nrmse(y_test, y_pred_test, rango)

        print("\nMétricas:")
        print(f"Train | R²={r2_train:.4f} | MAE={mae_train:.4f} V | NRMSE={nrmse_train:.4f}")
        print(f"Test  | R²={r2_test:.4f} | MAE={mae_test:.4f} V | NRMSE={nrmse_test:.4f}")

        print("\nPrimeras 5 predicciones test:")
        for real, pred in zip(y_test[:5], y_pred_test[:5]):
            print(f"Real={real:.4f} V | Predicho={pred:.4f} V | Error={real-pred:+.4f} V")

        if np.any(np.isnan(y_pred_test)):
            print("Advertencia: hay predicciones NaN.")
        else:
            print("OK: predicciones válidas, sin NaN.")


if __name__ == "__main__":
    main()
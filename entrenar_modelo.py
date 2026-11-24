from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. RUTAS DEL PROYECTO
# ==========================================

PROYECTO = Path(__file__).resolve().parent.parent
RESULTADOS = PROYECTO / "results"

ARCHIVO_DATOS = RESULTADOS / "caracteristicas.csv"
ARCHIVO_MODELO = RESULTADOS / "modelo_random_forest.joblib"
ARCHIVO_PREDICCIONES = RESULTADOS / "predicciones_modelo.csv"
ARCHIVO_METRICAS = RESULTADOS / "metricas_modelo.txt"

RESULTADOS.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CARGAR LOS DATOS
# ==========================================

print("\nENTRENAMIENTO DEL MODELO")
print("=" * 45)

datos = pd.read_csv(ARCHIVO_DATOS)

OBJETIVO = "tumor_realzante_volumen_mm3"

if OBJETIVO not in datos.columns:
    raise ValueError("No se encontró la columna objetivo.")

print("Pacientes encontrados:", len(datos))


# ==========================================
# 3. SELECCIONAR LAS CARACTERÍSTICAS
# ==========================================

estadisticas = [
    "_media",
    "_desviacion",
    "_mediana",
    "_minimo",
    "_maximo"
]

columnas_entrada = [
    columna
    for columna in datos.columns
    if (
        ("_edema_" in columna or
         "_tumor_no_realzante_" in columna)
        and columna.endswith(tuple(estadisticas))
    )
]

if not columnas_entrada:
    raise ValueError("No se encontraron características de entrada.")

X = datos[columnas_entrada].copy()
y = datos[OBJETIVO].copy()

print("Características seleccionadas:", len(columnas_entrada))
print("Objetivo: volumen del tumor realzante (mm3)")


# ==========================================
# 4. DIVIDIR LOS DATOS
# ==========================================

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)

print("\nDatos para entrenamiento:", len(X_entrenamiento))
print("Datos para prueba:", len(X_prueba))


# ==========================================
# 5. CONSTRUIR EL MODELO
# ==========================================

modelo = Pipeline([
    ("imputacion", SimpleImputer(strategy="median")),
    ("random_forest", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_features=0.8,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])


# ==========================================
# 6. ENTRENAR
# ==========================================

print("\nEntrenando Random Forest...")

modelo.fit(X_entrenamiento, y_entrenamiento)

print("¡Entrenamiento completado!")


# ==========================================
# 7. REALIZAR PREDICCIONES
# ==========================================

predicciones = modelo.predict(X_prueba)

# Los volúmenes no pueden ser negativos.
predicciones = np.maximum(predicciones, 0)


# ==========================================
# 8. EVALUAR EL MODELO
# ==========================================

mae = mean_absolute_error(y_prueba, predicciones)

rmse = np.sqrt(
    mean_squared_error(y_prueba, predicciones)
)

r2 = r2_score(y_prueba, predicciones)


# Comparación con un modelo sencillo de referencia
modelo_referencia = DummyRegressor(strategy="median")

modelo_referencia.fit(
    X_entrenamiento,
    y_entrenamiento
)

predicciones_referencia = modelo_referencia.predict(X_prueba)

mae_referencia = mean_absolute_error(
    y_prueba,
    predicciones_referencia
)

print("\nRESULTADOS DE EVALUACIÓN")
print("=" * 45)

print(f"MAE: {mae:.2f} mm3")
print(f"RMSE: {rmse:.2f} mm3")
print(f"R2: {r2:.4f}")

print(f"\nMAE del modelo de referencia: {mae_referencia:.2f} mm3")


# ==========================================
# 9. GUARDAR PREDICCIONES
# ==========================================

tabla_predicciones = pd.DataFrame({
    "paciente": datos.loc[X_prueba.index, "paciente"].values,
    "volumen_real_mm3": y_prueba.values,
    "volumen_estimado_mm3": predicciones,
    "error_absoluto_mm3": np.abs(y_prueba.values - predicciones)
})

tabla_predicciones.to_csv(
    ARCHIVO_PREDICCIONES,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 10. GUARDAR EL MODELO Y LAS MÉTRICAS
# ==========================================

joblib.dump(modelo, ARCHIVO_MODELO)

with open(ARCHIVO_METRICAS, "w", encoding="utf-8") as archivo:
    archivo.write("EVALUACIÓN DEL MODELO RANDOM FOREST\n")
    archivo.write(f"Pacientes totales: {len(datos)}\n")
    archivo.write(f"Pacientes entrenamiento: {len(X_entrenamiento)}\n")
    archivo.write(f"Pacientes prueba: {len(X_prueba)}\n")
    archivo.write(f"Características: {len(columnas_entrada)}\n")
    archivo.write(f"MAE: {mae:.4f} mm3\n")
    archivo.write(f"RMSE: {rmse:.4f} mm3\n")
    archivo.write(f"R2: {r2:.4f}\n")
    archivo.write(f"MAE referencia: {mae_referencia:.4f} mm3\n")


# ==========================================
# 11. FINALIZAR
# ==========================================

print("\nARCHIVOS GUARDADOS")
print("=" * 45)

print("Modelo:", ARCHIVO_MODELO)
print("Predicciones:", ARCHIVO_PREDICCIONES)
print("Métricas:", ARCHIVO_METRICAS)

print("\n¡PROCESO FINALIZADO!")

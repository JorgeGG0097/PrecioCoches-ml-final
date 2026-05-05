"""
Compara en vivo los 5 modelos evaluados en el TFG:
  Random Forest, GBM (sklearn), XGBoost, LightGBM, CatBoost

Requiere: pip install xgboost lightgbm catboost
Ejecutar: python comparativa_modelos.py
"""
import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

RUTA_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS   = os.path.join(RUTA_SCRIPT, "..", "Cars_combinado_limpio.csv")
AÑO_ACTUAL   = 2026
RANDOM_STATE = 42

# ── Imports opcionales ────────────────────────────────────────────────────────
disponibles = {}
for lib, nombre in [("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost")]:
    try:
        __import__(lib)
        disponibles[nombre] = True
        print(f"[OK] {nombre}")
    except ImportError:
        disponibles[nombre] = False
        print(f"[NO] {nombre} no instalado  ->  pip install {lib}")

if disponibles.get("XGBoost"):
    from xgboost import XGBRegressor
if disponibles.get("LightGBM"):
    from lightgbm import LGBMRegressor
if disponibles.get("CatBoost"):
    from catboost import CatBoostRegressor

# ── Datos ─────────────────────────────────────────────────────────────────────
print("\nCargando datos...")
df = pd.read_csv(RUTA_DATOS, encoding="utf-8-sig")
df["antiguedad"] = AÑO_ACTUAL - df["año"]
df["km_por_año"] = (df["kilometraje_km"] / df["antiguedad"].clip(lower=1)).round(0).astype(int)

variables_numericas   = ["año", "potencia_cv", "kilometraje_km", "antiguedad", "km_por_año"]
variables_categoricas = ["marca", "modelo", "combustible", "transmision"]

X = df[variables_numericas + variables_categoricas]
y = df["precio_eur"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# ── Pipeline ──────────────────────────────────────────────────────────────────
def hacer_pipeline(estimador):
    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), variables_numericas),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), variables_categoricas),
    ])
    return Pipeline([("preprocesado", pre), ("modelo", estimador)])

def evaluar(nombre, pipeline):
    print(f"  Entrenando {nombre}...", end="", flush=True)
    pipeline.fit(X_train, y_train)

    r2_tr = r2_score(y_train, pipeline.predict(X_train))
    r2_te = r2_score(y_test,  pipeline.predict(X_test))
    mae   = mean_absolute_error(y_test, pipeline.predict(X_test))
    cv    = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)

    print(f" MAE={mae:,.0f} EUR  R2={r2_te:.4f}  CV={cv.mean():.4f}")
    return {"R2 test": round(r2_te, 4), "MAE (EUR)": int(mae),
            "Brecha": round(r2_tr - r2_te, 4), "CV R2": round(cv.mean(), 4)}

# ── Modelos ───────────────────────────────────────────────────────────────────
print("\nEvaluando modelos (puede tardar 20-30 minutos)...")
resultados = {}

resultados["Random Forest"] = evaluar("Random Forest", hacer_pipeline(
    RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=5,
                          random_state=RANDOM_STATE, n_jobs=-1)
))

resultados["GBM sklearn"] = evaluar("GBM sklearn", hacer_pipeline(
    HistGradientBoostingRegressor(
        max_iter=10000, learning_rate=0.05, max_depth=8, min_samples_leaf=20,
        early_stopping=True, n_iter_no_change=50, validation_fraction=0.1,
        tol=1e-4, random_state=RANDOM_STATE,
    )
))

if disponibles.get("XGBoost"):
    resultados["XGBoost"] = evaluar("XGBoost", hacer_pipeline(
        XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    ))

if disponibles.get("LightGBM"):
    resultados["LightGBM"] = evaluar("LightGBM", hacer_pipeline(
        LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8,
                      num_leaves=50, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    ))

if disponibles.get("CatBoost"):
    resultados["CatBoost"] = evaluar("CatBoost", hacer_pipeline(
        CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=8,
                          random_state=RANDOM_STATE, verbose=0)
    ))

# ── Tabla final ───────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("COMPARATIVA FINAL — Todos los modelos")
print("="*68)
print(f"  {'Modelo':<20} {'R2 test':>9} {'MAE (EUR)':>11} {'Brecha':>9} {'CV R2':>9}")
print("  " + "-"*62)

ordenados = sorted(resultados.items(), key=lambda x: x[1]["MAE (EUR)"])
for nombre, m in ordenados:
    marca = " <-- MEJOR" if nombre == ordenados[0][0] else ""
    print(f"  {nombre:<20} {m['R2 test']:>9.4f} {m['MAE (EUR)']:>11,} {m['Brecha']:>9.4f} {m['CV R2']:>9.4f}{marca}")

ganador = ordenados[0][0]
mae_min = ordenados[0][1]["MAE (EUR)"]
mae_max = ordenados[-1][1]["MAE (EUR)"]
print(f"\n  Mejor modelo : {ganador} ({mae_min:,} EUR MAE)")
print(f"  Mejora vs RF : {mae_max - mae_min:,} EUR menos de error medio")

"""
Entrena el modelo definitivo (HistGradientBoostingRegressor) y guarda todos
los artefactos necesarios para la app en esta misma carpeta.
Ejecutar desde el terminal: python entrenar_modelo.py
"""
import json
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

RUTA_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS   = os.path.join(RUTA_SCRIPT, "..", "Cars_combinado_limpio.csv")
AÑO_ACTUAL   = 2026
RANDOM_STATE = 42

_MARCA_MAP = {
    "Alfa":     "Alfa Romeo",
    "Mercedes": "Mercedes-Benz",
    "Land":     "Land Rover",
}

def _normalizar_marca(m):
    s = str(m)
    if s.lower().startswith("citro"):
        return "Citroen"
    return _MARCA_MAP.get(s, s)

# ── Carga ─────────────────────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv(RUTA_DATOS, encoding="utf-8-sig")
df["marca"] = df["marca"].apply(_normalizar_marca)
df["antiguedad"] = AÑO_ACTUAL - df["año"]
df["km_por_año"] = (df["kilometraje_km"] / df["antiguedad"].clip(lower=1)).round(0).astype(int)
print(f"  {len(df):,} registros cargados")
marcas_unicas = sorted(df["marca"].unique())
print(f"  {len(marcas_unicas)} marcas unicas tras normalizar: {marcas_unicas}")

variables_numericas   = ["año", "potencia_cv", "kilometraje_km", "antiguedad", "km_por_año"]
variables_categoricas = ["marca", "modelo", "combustible", "transmision", "etiqueta_ambiental", "tipo_venta"]

X = df[variables_numericas + variables_categoricas]
y = df["precio_eur"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
preprocesador = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), variables_numericas),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), variables_categoricas),
])

modelo = Pipeline([
    ("preprocesado", preprocesador),
    ("modelo", HistGradientBoostingRegressor(
        max_iter=10000,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        early_stopping=True,
        n_iter_no_change=50,
        validation_fraction=0.1,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )),
])

# ── Entrenamiento y evaluacion ────────────────────────────────────────────────
print("\nEntrenando modelo GBM...")
modelo.fit(X_train, y_train)

arboles = modelo.named_steps["modelo"].n_iter_
print(f"  Early stopping paro en {arboles} arboles")

r2_train  = r2_score(y_train, modelo.predict(X_train))
r2_test   = r2_score(y_test,  modelo.predict(X_test))
mae_test  = mean_absolute_error(y_test, modelo.predict(X_test))
brecha    = r2_train - r2_test

print(f"\n  R2 train : {r2_train:.4f}")
print(f"  R2 test  : {r2_test:.4f}")
print(f"  Brecha   : {brecha:.4f}  {'[OK]' if brecha < 0.05 else '[AVISO]'}")
print(f"  MAE test : {mae_test:,.0f} EUR")

print("\n  Cross-validation 5-fold...")
cv = cross_val_score(modelo, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
print(f"  R2 medio : {cv.mean():.4f} (+/- {cv.std():.4f})")

# ── Guardar artefactos en la carpeta definitivo/ ──────────────────────────────
print("\nGuardando artefactos...")

joblib.dump(modelo, os.path.join(RUTA_SCRIPT, "modelo_gbm.pkl"))

categorias = {col: sorted(df[col].dropna().unique().tolist()) for col in variables_categoricas}
with open(os.path.join(RUTA_SCRIPT, "categorias.json"), "w", encoding="utf-8") as f:
    json.dump(categorias, f, ensure_ascii=False, indent=2)

rangos = {
    col: {"min": int(df[col].min()), "max": int(df[col].max()), "median": int(df[col].median())}
    for col in ["año", "potencia_cv", "kilometraje_km"]
}
with open(os.path.join(RUTA_SCRIPT, "rangos_numericos.json"), "w", encoding="utf-8") as f:
    json.dump(rangos, f, ensure_ascii=False, indent=2)

with open(os.path.join(RUTA_SCRIPT, "features_info.json"), "w", encoding="utf-8") as f:
    json.dump({"variables_numericas": variables_numericas,
               "variables_categoricas": variables_categoricas,
               "año_actual": AÑO_ACTUAL,
               "mae_test": int(mae_test)}, f, ensure_ascii=False, indent=2)

print("Artefactos guardados en la carpeta definitivo/")
print(f"\nMAE final del modelo: {mae_test:,.0f} EUR")
print("Siguiente paso: streamlit run app.py")

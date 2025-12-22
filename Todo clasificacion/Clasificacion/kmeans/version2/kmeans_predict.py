import pandas as pd
import joblib, json
from pathlib import Path

# --- Rutas ---
MODEL_PATH = Path("/home/vale/Escritorio/Proyecto tesis/Clasificacion/kmeans/version2/kmeans_model.pkl")
SCALER_PATH = Path("/home/vale/Escritorio/Proyecto tesis/Clasificacion/kmeans/version2/scaler.pkl")
LABELS_MAP_PATH = Path("/home/vale/Escritorio/Proyecto tesis/Clasificacion/kmeans/version2/labels_map.json")
CSV_NEW = Path("/home/vale/Escritorio/Proyecto tesis/Clasificacion/kmeans/version2/nuevos_datos.xlsx")
XLSX_OUT = Path("/home/vale/Escritorio/Proyecto tesis/Clasificacion/kmeans/version2/predicciones_nuevos.xlsx")

# --- Cargar modelo y scaler ---
kmeans = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Leer nuevos datos ---
df_new = pd.read_csv(CSV_NEW)
for c in df_new.columns:
    df_new[c] = pd.to_numeric(df_new[c], errors="coerce")

# --- Seleccionar features y escalar ---
feats = ["tiempo_promedio", "nodos_promedio", "variables", "restricciones"]
X_new = df_new[feats].values
X_scaled = scaler.transform(X_new)

# --- Predecir clusters ---
df_new["cluster_id"] = kmeans.predict(X_scaled)

# --- Mapear etiquetas si existe labels_map.json ---
if LABELS_MAP_PATH.exists():
    with open(LABELS_MAP_PATH) as f:
        labels_map = {int(k): v for k, v in json.load(f).items()}
    df_new["dificultad_kmeans"] = df_new["cluster_id"].map(labels_map)
    print("Mapeo de etiquetas aplicado.")
else:
    print("No existe labels_map.json — solo se asignarán IDs de cluster.")

# --- Guardar resultados ---
df_new.to_excel(XLSX_OUT, index=False)
print(f"Predicciones guardadas en: {XLSX_OUT}")

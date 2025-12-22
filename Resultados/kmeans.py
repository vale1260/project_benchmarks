from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ==========================
# CONFIGURACIÓN
# ==========================
OVERRIDE_K = None  # poner 4 si quieres forzar k=4
MAX_K_ELBOW = 10
RANDOM_STATE = 42
N_INIT = 30

# Rutas (ajusta a tu caso)
CSV_IN = Path("/home/vale/Escritorio/Analisis y resultados tesis/nuevos_datos_kmeans_gptneo.ods")
OUT_DIR = CSV_IN.parent / "kmeans_gptneo"
XLSX_OUT = OUT_DIR / "clasificacion_kmeans_gptneo.xlsx"
PNG_PCA_OUT = OUT_DIR / "kmeans_clusters_gptneo.png"
PNG_ELBOW_OUT = OUT_DIR / "kmeans_elbow_gptneo.png"

MODEL_PATH = OUT_DIR / "kmeans_model.pkl"
SCALER_PATH = OUT_DIR / "scaler.pkl"
LABELS_MAP_PATH = OUT_DIR / "labels_map.json"

# ==========================
# UTILIDADES
# ==========================
def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def detect_columns(df: pd.DataFrame):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    name_col  = next((c for c in df.columns if c in ["nombre","problem","problema","archivo","file","nombre_problema"]), None)
    diff_col  = next((c for c in df.columns if c in ["dificultad","difficulty"]), None)
    time_col  = next((c for c in df.columns if c in ["tiempo_promedio","tiempo","avg_time","time"]), None)
    nodes_col = next((c for c in df.columns if c in ["nodos_promedio","nodo","nodos","avg_nodes","nodes"]), None)
    vars_col  = next((c for c in df.columns if c in ["variables","n_variables","num_variables"]), None)
    cons_col  = next((c for c in df.columns if c in ["restricciones","n_restricciones","constraints"]), None)

    missing_req = [k for k,v in [
        ("nombre", name_col),
        ("tiempo_promedio", time_col),
        ("nodos_promedio", nodes_col),
        ("variables", vars_col),
        ("restricciones", cons_col),
    ] if v is None]
    if missing_req:
        raise ValueError(f"Faltan columnas requeridas: {missing_req}. Columnas disponibles: {list(df.columns)}")
    return name_col, diff_col, time_col, nodes_col, vars_col, cons_col

def robust_high_fill(s: pd.Series):
    s_num = pd.to_numeric(s, errors="coerce")
    q95 = s_num.quantile(0.95)
    q75 = s_num.quantile(0.75)
    q25 = s_num.quantile(0.25)
    iqr = (q75 - q25) if pd.notna(q75) and pd.notna(q25) else 0
    base = q95 if pd.notna(q95) else s_num.max()
    if pd.isna(base):
        base = 1.0
    return float(base + 2.0 * (iqr if iqr > 0 else (s_num.std(ddof=0) if s_num.std(ddof=0) > 0 else base)))

def compute_elbow(Xs, max_k=10):
    ks = list(range(1, max_k+1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        km.fit(Xs)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(ks, inertias, marker='o')
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia (SSE)")
    plt.title("Método del Codo - KMeans")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PNG_ELBOW_OUT, bbox_inches="tight")
    plt.close()
    return ks, inertias

def choose_k_by_elbow(ks, inertias):
    if len(ks) < 3:
        return max(2, ks[0] if ks else 2)
    x1, y1 = float(ks[0]), float(inertias[0])
    x2, y2 = float(ks[-1]), float(inertias[-1])
    dx, dy = (x2 - x1), (y2 - y1)
    denom = (dx*dx + dy*dy) ** 0.5
    if denom == 0:
        return max(2, ks[0])
    max_dist = -1.0
    best_k = ks[0]
    for k, inertia in zip(ks, inertias):
        num = abs(dy*(k - x1) - dx*(inertia - y1))
        dist = num / denom
        if dist > max_dist:
            max_dist = dist
            best_k = k
    return max(2, int(best_k))

# ==========================
# MAIN
# ==========================
def main():
    if not CSV_IN.exists():
        print(f"No se encontró el archivo de entrada: {CSV_IN}")
        sys.exit(1)

    ensure_out_dir()

    # --- Lectura robusta ---
    print(f"Detectada extensión: {CSV_IN.suffix}")
    suffix = CSV_IN.suffix.lower()
    try:
        if suffix in [".xls", ".xlsx", ".ods"]:
            engine = "odf" if suffix == ".ods" else None
            if engine:
                df = pd.read_excel(CSV_IN, engine=engine)
            else:
                df = pd.read_excel(CSV_IN)
        elif suffix in [".csv", ".txt"]:
            encodings = ["utf-8", "latin-1", "cp1252"]
            df = None
            last_err = {}
            for enc in encodings:
                try:
                    df = pd.read_csv(CSV_IN, encoding=enc)
                    break
                except Exception as e:
                    last_err[enc] = str(e)
            if df is None:
                raise RuntimeError(f"No se pudo leer {CSV_IN} como CSV. Errores: {last_err}")
        else:
            try:
                df = pd.read_excel(CSV_IN)
            except Exception as e:
                try:
                    df = pd.read_csv(CSV_IN, encoding="latin-1")
                except Exception:
                    raise RuntimeError(f"Extensión '{suffix}' no reconocida y no se pudo leer el archivo: {e}")
    except Exception as e:
        print("Error al leer el archivo de entrada:", e)
        print("Asegúrate de tener instalado 'odfpy' si el archivo es .ods (pip install odfpy).")
        sys.exit(1)

    # --- Detección de columnas y limpieza ---
    name_col, diff_col, time_col, nodes_col, vars_col, cons_col = detect_columns(df)
    for col in [time_col, nodes_col, vars_col, cons_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    orig_missing_time  = df[time_col].isna().values
    orig_missing_nodes = df[nodes_col].isna().values

    time_fill  = robust_high_fill(df[time_col])
    nodes_fill = robust_high_fill(df[nodes_col])

    df_imp = df.copy()
    df_imp[time_col]  = df_imp[time_col].fillna(time_fill)
    df_imp[nodes_col] = df_imp[nodes_col].fillna(nodes_fill)
    for c in [vars_col, cons_col]:
        if df_imp[c].isna().any():
            df_imp[c] = df_imp[c].fillna(df_imp[c].median())

    feats = [time_col, nodes_col, vars_col, cons_col]
    X = df_imp[feats].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # --- Codo y selección k ---
    ks, inertias = compute_elbow(Xs, max_k=MAX_K_ELBOW)
    print(f"[Elbow] ks: {ks}")
    print(f"[Elbow] inertias: {[round(v,2) for v in inertias]}")
    print(f"Gráfico del codo guardado en: {PNG_ELBOW_OUT}")

    chosen_k = OVERRIDE_K if OVERRIDE_K is not None else choose_k_by_elbow(ks, inertias)
    print(f"k elegido para entrenamiento (método del codo / override): {chosen_k}")

    # --- Entrenar KMeans ---
    kmeans = KMeans(n_clusters=chosen_k, n_init=N_INIT, random_state=RANDOM_STATE)
    cluster_ids = kmeans.fit_predict(Xs)
    df_imp["cluster_id"] = cluster_ids

    # --- Silhouette (modelo final) ---
    sil_kmeans = None
    try:
        if len(set(cluster_ids)) >= 2 and Xs.shape[0] > 1:
            sil_kmeans = float(silhouette_score(Xs, cluster_ids))
            print(f"[Silhouette] silhouette_score del modelo final: {sil_kmeans:.4f}")
        else:
            print("[Silhouette] No se puede calcular silhouette: menos de 2 clusters o muestras insuficientes.")
    except Exception as e:
        print(f"[Silhouette] Error al calcular silhouette: {e}")

    # --- Adjusted Rand Score (ARS) contra etiquetas originales (si existen) ---
    ars_kmeans = None
    if diff_col and diff_col in df.columns and df[diff_col].notna().any():
        # Tomar solo filas con etiqueta original no nula
        mask = df[diff_col].notna().values
        if mask.sum() == 0:
            print("[ARS] No hay etiquetas originales válidas para calcular ARS.")
        else:
            # Codificar etiquetas originales (factorize) y predicciones correspondientes
            # factorize devuelve (-1) para NaN? No — usamos mask para excluir NaN.
            y_true_raw = df.loc[mask, diff_col].astype(str)
            y_true_enc, uniques = pd.factorize(y_true_raw.str.lower())
            y_pred_sub = df_imp.loc[mask, "cluster_id"].to_numpy()

            # Necesitamos al menos 2 etiquetas en y_true y en y_pred para que ARS sea informativo
            if len(np.unique(y_true_enc)) >= 2 and len(np.unique(y_pred_sub)) >= 2:
                try:
                    ars_kmeans = float(adjusted_rand_score(y_true_enc, y_pred_sub))
                    print(f"[ARS] Adjusted Rand Score (usar filas con etiqueta original): {ars_kmeans:.4f}")
                except Exception as e:
                    print(f"[ARS] Error calculando ARS: {e}")
            else:
                print("[ARS] No se calcula ARS: se requiere al menos 2 clases en etiquetas originales y 2 clusters en predicción.")

    else:
        print("[ARS] No se encontró columna de dificultad original con valores válidos; ARS no calculado.")

    # --- Si quieres mapear a 'difficulty' cuando k==4, mantenemos la función previa ---
    labels_map = None
    if chosen_k == 4:
        try:
            # Reusar la heurística previa: ordenar por suma de centroides y forzar 'unsolved'
            centroids = np.vstack([Xs[cluster_ids == c].mean(axis=0) for c in range(4)])
            scores = centroids.sum(axis=1)
            order_easy_to_hard = np.argsort(scores).tolist()

            missing_ratio = []
            for c in range(4):
                idx = (cluster_ids == c)
                miss_prop = ((orig_missing_time[idx].astype(int) + orig_missing_nodes[idx].astype(int)) >= 1).mean()
                missing_ratio.append(miss_prop)
            missing_ratio = np.array(missing_ratio)
            hardest = order_easy_to_hard[-1]

            z_hard_time = centroids[hardest, 0]
            z_hard_nodes = centroids[hardest, 1]
            if (missing_ratio[hardest] >= 0.5) or (z_hard_time >= 2.0) or (z_hard_nodes >= 2.0):
                force_unsolved = hardest
            else:
                force_unsolved = int(np.argmax(missing_ratio))

            lm = {int(force_unsolved): "unsolved"}
            remaining = [c for c in order_easy_to_hard if c != force_unsolved]
            if len(remaining) != 3:
                remaining = [c for c in range(4) if c != force_unsolved]
            labels_order = ["easy", "medium", "hard"]
            for cid, lab in zip(remaining, labels_order):
                lm[cid] = lab

            labels_map = lm
            df_imp["dificultad_kmeans"] = df_imp["cluster_id"].map(labels_map)
            # Guardar labels_map
            with open(LABELS_MAP_PATH, "w") as f:
                json.dump({int(k): v for k, v in labels_map.items()}, f)
            print(f"[Mapping] labels_map guardado en: {LABELS_MAP_PATH}")
        except Exception as e:
            print(f"[Mapping] Error generando labels_map para k==4: {e}")

    # --- Renombrar y guardar resultados ---
    rename_final = {
        name_col: "nombre",
        time_col: "tiempo_promedio",
        nodes_col: "nodos_promedio",
        vars_col: "variables",
        cons_col: "restricciones",
    }
    if diff_col:
        rename_final[diff_col] = "dificultad_original"

    result = df_imp.rename(columns=rename_final)
    if "dificultad_original" not in result.columns:
        result["dificultad_original"] = np.nan

    cols_final = [
        "nombre", "dificultad_original",
        "tiempo_promedio", "nodos_promedio", "variables", "restricciones",
        "cluster_id",
    ]
    if chosen_k == 4 and labels_map is not None:
        cols_final.insert(6, "dificultad_kmeans")

    result = result[cols_final]
    result.to_excel(XLSX_OUT, index=False)
    print(f"Excel guardado en: {XLSX_OUT}")

    # --- PCA y gráfico ---
    try:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pts = pca.fit_transform(Xs)
        df_plot = pd.DataFrame({"_pc1": pts[:,0], "_pc2": pts[:,1], "cluster_id": cluster_ids})
        plt.figure()
        if chosen_k == 4 and labels_map is not None and "dificultad_kmeans" in df_imp.columns:
            df_plot["dificultad_kmeans"] = df_imp["dificultad_kmeans"].values
            for lab in sorted(df_plot["dificultad_kmeans"].dropna().unique()):
                s = df_plot[df_plot["dificultad_kmeans"] == lab]
                plt.scatter(s["_pc1"], s["_pc2"], label=str(lab))
            plt.title(f"Clusters KMeans (k={chosen_k}) con etiquetas de dificultad")
            plt.legend()
        else:
            for cid in sorted(df_plot["cluster_id"].unique()):
                s = df_plot[df_plot["cluster_id"] == cid]
                plt.scatter(s["_pc1"], s["_pc2"], label=f"c{cid}")
            plt.title(f"Clusters KMeans (k={chosen_k})")
            plt.legend(title="cluster_id")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(PNG_PCA_OUT, bbox_inches="tight")
        plt.close()
        print(f"Gráfico PCA guardado en: {PNG_PCA_OUT}")
    except Exception as e:
        print(f"Error generando gráfico PCA: {e}")

    # --- Guardar artefactos ---
    try:
        joblib.dump(kmeans, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Modelo guardado en: {MODEL_PATH}")
        print(f"Scaler guardado en: {SCALER_PATH}")
    except Exception as e:
        print(f"Error guardando artefactos: {e}")

    # --- Resumen final ---
    print("----- Resumen -----")
    print(f"k usado: {chosen_k}")
    print(f"Silhouette (modelo final): {sil_kmeans}")
    print(f"Adjusted Rand Score (ARS, si calculado): {ars_kmeans}")
    if LABELS_MAP_PATH.exists():
        print(f"Labels map: {LABELS_MAP_PATH}")

if __name__ == "__main__":
    main()

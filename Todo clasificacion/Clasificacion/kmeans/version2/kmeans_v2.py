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
# Si deseas forzar k=4 para mantener el mapeo de dificultad, pon OVERRIDE_K=4.
# Si lo dejas en None, se elegirá k automáticamente por MÉTODO DEL CODO.
OVERRIDE_K = None  # Déjalo en None para elegir k automáticamente por MÉTODO DEL CODO

MAX_K_ELBOW = 10  # k máximo para graficar el codo
RANDOM_STATE = 42
N_INIT = 30

# Rutas de E/S (ajusta a tus rutas)
CSV_IN = Path("/home/vale/Escritorio/Proyecto tesis/Clasificacion/clasificacion_original.csv")
OUT_DIR = CSV_IN.parent / "kmeans"
XLSX_OUT = OUT_DIR / "clasificacion_kmeans.xlsx"
PNG_PCA_OUT = OUT_DIR / "kmeans_clusters.png"
PNG_ELBOW_OUT = OUT_DIR / "kmeans_elbow.png"

# Artefactos del modelo
MODEL_PATH = OUT_DIR / "kmeans_model.pkl"
SCALER_PATH = OUT_DIR / "scaler.pkl"
LABELS_MAP_PATH = OUT_DIR / "labels_map.json"

DESIRED_LABELS = ["easy", "medium", "hard", "unsolved"]

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
    """Devuelve lista de (k, inercia) y guarda gráfica del codo."""
    ks = list(range(1, max_k+1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        km.fit(Xs)
        inertias.append(km.inertia_)
    # Graficar
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
    """Elige k por método del codo usando la distancia máxima a la recta
    que une (k_min, inertia_max) con (k_max, inertia_min). Si hay pocos puntos
    o algo raro, devuelve 2 como valor mínimo razonable.
    """
    if len(ks) < 3:
        return max(2, ks[0] if ks else 2)

    # Puntos extremos
    x1, y1 = float(ks[0]), float(inertias[0])
    x2, y2 = float(ks[-1]), float(inertias[-1])

    # Vector de la línea base
    dx, dy = (x2 - x1), (y2 - y1)
    denom = (dx*dx + dy*dy) ** 0.5
    if denom == 0:
        # Inercia constante: devolver 2 por seguridad
        return max(2, ks[0])

    max_dist = -1.0
    best_k = ks[0]
    for k, inertia in zip(ks, inertias):
        # Distancia punto a línea (fórmula de área del paralelogramo / longitud base)
        num = abs(dy*(k - x1) - dx*(inertia - y1))
        dist = num / denom
        if dist > max_dist:
            max_dist = dist
            best_k = k

    # Asegurar al menos 2 clusters
    return max(2, int(best_k))


def best_k_by_silhouette(Xs, k_min=2, k_max=10):
    """Retorna k con mejor silhouette (promedio), evaluando en [k_min, k_max]."""
    best_k = None
    best_s = -1.0
    scores = {}
    for k in range(max(2, k_min), max(k_min, k_max)+1):
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels = km.fit_predict(Xs)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(Xs, labels)
        scores[k] = float(s)
        if s > best_s:
            best_s = s
            best_k = k
    return best_k, scores


def cluster_labels_k4(X_scaled, cluster_ids, orig_missing_time, orig_missing_nodes):
    """Mapea clusters a {easy, medium, hard, unsolved} cuando k=4.
    Criterio: orden por 'dureza' (suma de centroides) y fuerza 'unsolved' por missing/altos z.
    """
    k = len(np.unique(cluster_ids))
    assert k == 4, "Se espera k=4 para generar etiquetas de dificultad"

    centroids = np.vstack([X_scaled[cluster_ids == c].mean(axis=0) for c in range(k)])
    scores = centroids.sum(axis=1)
    order_easy_to_hard = np.argsort(scores).tolist()

    labels_map = {}

    missing_ratio = []
    for c in range(k):
        idx = (cluster_ids == c)
        miss_prop = (
            (orig_missing_time[idx].astype(int) + orig_missing_nodes[idx].astype(int)) >= 1
        ).mean()
        missing_ratio.append(miss_prop)
    missing_ratio = np.array(missing_ratio)

    hardest = order_easy_to_hard[-1]

    z_hard_time = centroids[hardest, 0]
    z_hard_nodes = centroids[hardest, 1]
    if (missing_ratio[hardest] >= 0.5) or (z_hard_time >= 2.0) or (z_hard_nodes >= 2.0):
        force_unsolved = hardest
    else:
        force_unsolved = int(np.argmax(missing_ratio))

    labels_map[force_unsolved] = "unsolved"

    remaining = [c for c in order_easy_to_hard if c != force_unsolved]
    if len(remaining) != 3:
        remaining = [c for c in range(k) if c != force_unsolved]

    labels_order = ["easy", "medium", "hard"]
    for cid, lab in zip(remaining, labels_order):
        labels_map[cid] = lab

    return labels_map, {
        "order_easy_to_hard": order_easy_to_hard,
        "centroid_scores": scores.tolist(),
        "missing_ratio": missing_ratio.tolist(),
        "forced_unsolved": int(force_unsolved),
    }

# ==========================
# MAIN
# ==========================

def main():
    if not CSV_IN.exists():
        print(f"No se encontró el archivo de entrada: {CSV_IN}")
        sys.exit(1)

    ensure_out_dir()

    # Leer datos
    if CSV_IN.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(CSV_IN)
    else:
        df = pd.read_csv(CSV_IN)

    name_col, diff_col, time_col, nodes_col, vars_col, cons_col = detect_columns(df)

    # Asegurar numéricos
    for col in [time_col, nodes_col, vars_col, cons_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Guardamos info de missing original para la heurística k=4
    orig_missing_time  = df[time_col].isna().values
    orig_missing_nodes = df[nodes_col].isna().values

    # Imputar valores faltantes de forma robusta
    time_fill  = robust_high_fill(df[time_col])
    nodes_fill = robust_high_fill(df[nodes_col])

    df_imp = df.copy()
    df_imp[time_col]  = df_imp[time_col].fillna(time_fill)
    df_imp[nodes_col] = df_imp[nodes_col].fillna(nodes_fill)
    for c in [vars_col, cons_col]:
        if df_imp[c].isna().any():
            df_imp[c] = df_imp[c].fillna(df_imp[c].median())

    # Matriz de features y escalado
    feats = [time_col, nodes_col, vars_col, cons_col]
    X = df_imp[feats].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Método del codo (guarda gráfica)
    ks, inertias = compute_elbow(Xs, max_k=MAX_K_ELBOW)
    print(f"[Elbow] ks: {ks}")
    print(f"[Elbow] inertias: {[round(v,2) for v in inertias]}")
    print(f"Gráfico del codo guardado en: {PNG_ELBOW_OUT}")

    # Selección de k
    chosen_k = OVERRIDE_K
    # Elegir k automáticamente por método del codo si no se fuerza
    if chosen_k is None:
        chosen_k = choose_k_by_elbow(ks, inertias)

    # (Opcional) calcular silueta para diagnóstico, no para decidir k
    _, sil_scores = best_k_by_silhouette(Xs, k_min=2, k_max=min(MAX_K_ELBOW, 12))
    print(f"[Silhouette] puntajes: {{ {', '.join([str(k)+': '+str(round(v,4)) for k,v in sil_scores.items()])} }}")
    print(f"k elegido para entrenamiento (método del codo): {chosen_k}")

    # Entrenar modelo final
    kmeans = KMeans(n_clusters=chosen_k, n_init=N_INIT, random_state=RANDOM_STATE)
    cluster_ids = kmeans.fit_predict(Xs)

    # Silhouette del modelo final
    sil_kmeans = None
    if len(set(cluster_ids)) >= 2:
        sil_kmeans = float(silhouette_score(Xs, cluster_ids))

    # Mapear etiquetas si k=4
    labels_map = None
    if chosen_k == 4:
        labels_map, debug_info = cluster_labels_k4(
            Xs, cluster_ids,
            orig_missing_time=orig_missing_time,
            orig_missing_nodes=orig_missing_nodes,
        )
        df_imp["cluster_id"] = cluster_ids
        df_imp["dificultad_kmeans"] = df_imp["cluster_id"].map(labels_map)
    else:
        df_imp["cluster_id"] = cluster_ids
        # Sin mapeo a dificultad si k != 4
        print("k != 4 → no se generan etiquetas de dificultad (solo cluster_id).")

    # ARS si existe dificultad original
    ars_kmeans = None
    if diff_col and diff_col in df.columns and df[diff_col].notna().any() and chosen_k == 4 and labels_map is not None:
        def _to_int_labels(series):
            mapping = {"easy":0, "medium":1, "hard":2, "unsolved":3}
            return series.astype(str).str.lower().map(mapping).fillna(-1).astype(int).to_numpy()
        y_true = _to_int_labels(df[diff_col])
        y_pred = _to_int_labels(df_imp["dificultad_kmeans"])
        if len(y_pred) == len(y_true) and len(set(y_pred)) >= 2 and (y_true >= 0).any():
            ars_kmeans = float(adjusted_rand_score(y_true, y_pred))

    # Renombrar columnas finales
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
        cols_final.insert(6, "dificultad_kmeans")  # antes de cluster_id

    result = result[cols_final]
    result.to_excel(XLSX_OUT, index=False)

    # PCA para gráfico
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pts = pca.fit_transform(Xs)
    df_plot = pd.DataFrame({
        "_pc1": pts[:, 0],
        "_pc2": pts[:, 1],
        "cluster_id": cluster_ids,
    })
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

    # Guardar artefactos
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    if chosen_k == 4 and labels_map is not None:
        with open(LABELS_MAP_PATH, "w") as f:
            json.dump({int(k): v for k, v in labels_map.items()}, f)

    # Logs
    print(f"[K-Means] k usado: {chosen_k}")
    print(f"[K-Means] Silhouette: {sil_kmeans}")
    if ars_kmeans is not None:
        print(f"[K-Means] Adjusted Rand Score (si aplica): {ars_kmeans}")
    print(f"Excel:  {XLSX_OUT}")
    print(f"Gráfico PCA: {PNG_PCA_OUT}")
    print(f"Gráfico Codo: {PNG_ELBOW_OUT}")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Scaler: {SCALER_PATH}")
    if (LABELS_MAP_PATH.exists()):
        print(f"labels_map: {LABELS_MAP_PATH}")


if __name__ == "__main__":
    main()

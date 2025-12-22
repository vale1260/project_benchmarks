from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score 
import matplotlib.pyplot as plt

DESIRED_LABELS = ["easy", "medium", "hard", "unsolved"]

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
        raise ValueError(f"Faltan columnas requeridas: {missing_req}. "
                         f"Columnas disponibles: {list(df.columns)}")
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

def cluster_labels_k4(X_scaled, cluster_ids, orig_missing_time, orig_missing_nodes):
    k = len(np.unique(cluster_ids))
    assert k == 4, "Se espera k=4"

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

    force_unsolved = None
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

    return labels_map, {"order_easy_to_hard": order_easy_to_hard,
                        "centroid_scores": scores.tolist(),
                        "missing_ratio": missing_ratio.tolist(),
                        "forced_unsolved": int(force_unsolved)}

def main():
    csv_in = Path("/home/vale/Escritorio/Proyecto tesis/Clasificacion/clasificacion_original.csv")
    if not csv_in.exists():
        print(f"No se encontró el archivo de entrada: {csv_in}")
        sys.exit(1)

    out_dir = csv_in.parent
    xlsx_out = out_dir / "kmeans/clasificacion_kmeans.xlsx"
    png_out  = out_dir / "kmeans/kmeans_clusters.png"

    if csv_in.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(csv_in)
    else:
        df = pd.read_csv(csv_in)

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

    kmeans = KMeans(n_clusters=4, n_init=30, random_state=42)
    cluster_ids = kmeans.fit_predict(Xs)

    labels_map, debug_info = cluster_labels_k4(
        Xs, cluster_ids,
        orig_missing_time=orig_missing_time,
        orig_missing_nodes=orig_missing_nodes
    )
    df_imp["cluster_id"] = cluster_ids
    df_imp["dificultad_kmeans"] = df_imp["cluster_id"].map(labels_map)

    sil_kmeans = None
    if len(set(cluster_ids)) >= 2:
        sil_kmeans = float(silhouette_score(Xs, cluster_ids))

    ars_kmeans = None
    if diff_col and diff_col in df.columns and df[diff_col].notna().any():
        def _to_int_labels(series):
            mapping = {"easy":0, "medium":1, "hard":2, "unsolved":3}
            return series.astype(str).str.lower().map(mapping).fillna(-1).astype(int).to_numpy()

        y_true = _to_int_labels(df[diff_col])
        y_pred = _to_int_labels(df_imp["dificultad_kmeans"])
        if len(y_pred) == len(y_true) and len(set(y_pred)) >= 2 and (y_true >= 0).any():
            ars_kmeans = float(adjusted_rand_score(y_true, y_pred))

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
        "nombre","dificultad_original",
        "tiempo_promedio","nodos_promedio","variables","restricciones",
        "dificultad_kmeans","cluster_id"
    ]
    result = result[cols_final]
    result.to_excel(xlsx_out, index=False)

    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(Xs)
    df_plot = pd.DataFrame({
        "_pc1": pts[:, 0],
        "_pc2": pts[:, 1],
        "dificultad_kmeans": df_imp["dificultad_kmeans"].values
    })

    plt.figure()
    for lab in sorted(df_plot["dificultad_kmeans"].unique()):
        s = df_plot[df_plot["dificultad_kmeans"] == lab]
        plt.scatter(s["_pc1"], s["_pc2"], label=lab)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters KMeans (k=4)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_out, bbox_inches="tight")
    plt.close()

    print(f"[K-Means] Silhouette: {sil_kmeans}")
    print(f"[K-Means] Adjusted Rand Score (si aplica): {ars_kmeans}")
    print(f"Excel:  {xlsx_out}")
    print(f"Gráfico:{png_out}")

if __name__ == "__main__":
    main()

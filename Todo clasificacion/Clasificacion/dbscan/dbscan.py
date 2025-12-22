from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score

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

def robust_high_fill(s: pd.Series) -> float:
    s_num = pd.to_numeric(s, errors="coerce")
    q95 = s_num.quantile(0.95)
    q75 = s_num.quantile(0.75)
    q25 = s_num.quantile(0.25)
    iqr = (q75 - q25) if pd.notna(q75) and pd.notna(q25) else 0
    base = q95 if pd.notna(q95) else s_num.max()
    if pd.isna(base):
        base = 1.0
    std = s_num.std(ddof=0)
    bump = iqr if iqr and iqr > 0 else (std if std and std > 0 else base)
    return float(base + 2.0 * bump)

def estimate_eps(Xs: np.ndarray, min_samples: int = 5,
                 percentiles = (70, 75, 80, 85, 90, 92, 94, 96, 98)):
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    kth = dists[:, -1]
    eps_candidates = [float(np.percentile(kth, p)) for p in percentiles]
    uniq = []
    seen = set()
    for e in eps_candidates:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq

def run_dbscan_autotune(Xs: np.ndarray, min_samples: int = 5, target_range=(3,4)):
    candidates = estimate_eps(Xs, min_samples=min_samples)
    best = None
    best_score = (-1, np.inf)  # (n_clus, -noise) — mayor es mejor

    for eps in candidates:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(Xs)
        n_clus = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int(np.sum(labels == -1))

        if target_range[0] <= n_clus <= target_range[1]:
            return model, labels, eps

        if n_clus <= 6:
            score = (n_clus, -noise)
            if score > best_score:
                best_score = score
                best = (model, labels, eps)

    if best is None:
        model = DBSCAN(eps=candidates[-1], min_samples=min_samples)
        labels = model.fit_predict(Xs)
        return model, labels, candidates[-1]
    return best

def map_dbscan_clusters(Xs: np.ndarray, labels: np.ndarray,
                        orig_missing_time: np.ndarray, orig_missing_nodes: np.ndarray):

    label_map = { -1: "unsolved" }
    clusters = sorted([c for c in np.unique(labels) if c != -1])
    if not clusters:
        return label_map, {"centroid_scores": [], "missing_ratio": [], "forced_unsolved": -1}

    centroids = []
    miss_ratio = []
    for c in clusters:
        idx = (labels == c)
        centroids.append(Xs[idx].mean(axis=0))
        miss_ratio.append(((orig_missing_time[idx].astype(int) + orig_missing_nodes[idx].astype(int)) >= 1).mean())
    centroids = np.vstack(centroids)  
    miss_ratio = np.array(miss_ratio)

    scores = centroids.sum(axis=1)  
    order_easy_to_hard = np.argsort(scores).tolist()  
    hardest_idx = order_easy_to_hard[-1]              
    hardest_cid = clusters[hardest_idx]

    z_hard_time, z_hard_nodes = centroids[hardest_idx, 0], centroids[hardest_idx, 1]
    force_unsolved = None
    if (miss_ratio[hardest_idx] >= 0.50) or (z_hard_time >= 2.0) or (z_hard_nodes >= 2.0):
        force_unsolved = hardest_cid
    else:
        force_unsolved = clusters[int(np.argmax(miss_ratio))]

    label_map[force_unsolved] = "unsolved"

    remaining_pairs = [(clusters[i], scores[i]) for i in order_easy_to_hard if clusters[i] != force_unsolved]
    
    remaining_pairs.sort(key=lambda x: x[1])
    rank_labels = ["easy", "medium", "hard"]
    for (cid, _), lab in zip(remaining_pairs, rank_labels):
        label_map[cid] = lab

    for cid in clusters:
        if cid not in label_map:
            label_map[cid] = "hard"

    debug = {
        "centroid_scores": scores.tolist(),
        "missing_ratio": miss_ratio.tolist(),
        "forced_unsolved": int(force_unsolved),
        "cluster_ids": clusters,
    }
    return label_map, debug

def main():
    csv_in = Path("/home/vale/Escritorio/Proyecto tesis/Clasificacion/clasificacion_original.csv")
    if not csv_in.exists():
        print(f"No se encontró el archivo de entrada: {csv_in}")
        sys.exit(1)

    out_dir = csv_in.parent
    xlsx_out = out_dir / "dbscan/clasificacion_dbscan.xlsx"
    png_out  = out_dir / "dbscan/dbscan_clusters.png"

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

    model, labels, used_eps = run_dbscan_autotune(Xs, min_samples=5, target_range=(3,4))
    df_imp["cluster_id"] = labels

    label_map, debug_info = map_dbscan_clusters(
        Xs, labels,
        orig_missing_time=orig_missing_time,
        orig_missing_nodes=orig_missing_nodes
    )
    df_imp["dificultad_dbscan"] = df_imp["cluster_id"].map(lambda c: label_map.get(int(c), "unsolved"))

    sil_dbscan = None
    if len(set(labels)) >= 2:
        try:
            sil_dbscan = float(silhouette_score(Xs, labels))
        except Exception:
            sil_dbscan = None

    ars_dbscan = None
    if diff_col and diff_col in df.columns and df[diff_col].notna().any():
        def _to_int_labels(series):
            mapping = {"easy":0, "medium":1, "hard":2, "unsolved":3}
            return series.astype(str).str.lower().map(mapping).fillna(-1).astype(int).to_numpy()

        y_true = _to_int_labels(df[diff_col])
        y_pred = _to_int_labels(df_imp["dificultad_dbscan"])
        if len(y_pred) == len(y_true) and len(set(y_pred)) >= 2 and (y_true >= 0).any():
            ars_dbscan = float(adjusted_rand_score(y_true, y_pred))

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
        "dificultad_dbscan","cluster_id"
    ]
    result = result[cols_final]
    result.to_excel(xlsx_out, index=False)

    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(Xs)
    df_plot = pd.DataFrame({
        "_pc1": pts[:, 0],
        "_pc2": pts[:, 1],
        "dificultad_dbscan": df_imp["dificultad_dbscan"].values
    })

    plt.figure()
    for lab in sorted(df_plot["dificultad_dbscan"].unique()):
        s = df_plot[df_plot["dificultad_dbscan"] == lab]
        plt.scatter(s["_pc1"], s["_pc2"], label=lab)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Clusters DBSCAN (eps≈{used_eps:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_out, bbox_inches="tight")
    plt.close()

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"[DBSCAN] Silhouette: {sil_dbscan}")
    print(f"[DBSCAN] Adjusted Rand Score (si aplica): {ars_dbscan}")
    print(f"Excel:  {xlsx_out}")
    print(f"Gráfico:{png_out}")

if __name__ == "__main__":
    main()

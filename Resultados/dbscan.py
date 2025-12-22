from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ==========================
# RUTAS / CONFIG
# ==========================
CSV_IN = Path("/home/vale/Escritorio/Analisis y resultados tesis/nuevos_datos_dbscan_gptneo.ods")
OUT_DIR = CSV_IN.parent / "dbscan_gptneo"
XLSX_OUT = OUT_DIR / "clasificacion_dbscan_gptneo.xlsx"
PNG_OUT  = OUT_DIR / "dbscan_clusters_gptneo.png"
MODEL_PATH = OUT_DIR / "dbscan_model.pkl"
SCALER_PATH = OUT_DIR / "scaler.pkl"
LABELS_MAP_PATH = OUT_DIR / "labels_map.json"

RANDOM_STATE = 42

# ==========================
# UTILIDADES (idénticas a las previas)
# ==========================
DESIRED_LABELS = ["easy", "medium", "hard", "unsolved"]

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
                 percentiles=(70, 75, 80, 85, 90, 92, 94, 96, 98)):
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
    best_score = (-1, np.inf)  # (n_clus, -noise)
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

# ==========================
# MAIN
# ==========================
def main():
    if not CSV_IN.exists():
        print(f"No se encontró el archivo de entrada: {CSV_IN}")
        sys.exit(1)

    ensure_out_dir()

    # ---- Lectura robusta (.ods/.xlsx/.csv) ----
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
        print("Si es .ods instala 'odfpy' (pip install odfpy) y vuelve a intentarlo.")
        sys.exit(1)

    # ---- Detectar columnas ----
    name_col, diff_col, time_col, nodes_col, vars_col, cons_col = detect_columns(df)
    for col in [time_col, nodes_col, vars_col, cons_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    orig_missing_time  = df[time_col].isna().values
    orig_missing_nodes = df[nodes_col].isna().values

    # ---- Imputación robusta ----
    time_fill  = robust_high_fill(df[time_col])
    nodes_fill = robust_high_fill(df[nodes_col])
    df_imp = df.copy()
    df_imp[time_col]  = df_imp[time_col].fillna(time_fill)
    df_imp[nodes_col] = df_imp[nodes_col].fillna(nodes_fill)
    for c in [vars_col, cons_col]:
        if df_imp[c].isna().any():
            df_imp[c] = df_imp[c].fillna(df_imp[c].median())

    # ---- Escalado ----
    feats = [time_col, nodes_col, vars_col, cons_col]
    X = df_imp[feats].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ---- Ejecutar DBSCAN (auto-tune eps) ----
    model, labels, used_eps = run_dbscan_autotune(Xs, min_samples=5, target_range=(3,4))
    df_imp["cluster_id"] = labels

    # ---- Mapear clusters a etiquetas human-friendly ----
    label_map, debug_info = map_dbscan_clusters(
        Xs, labels,
        orig_missing_time=orig_missing_time,
        orig_missing_nodes=orig_missing_nodes
    )
    label_map[-1] = "unsolved"
    df_imp["dificultad_dbscan"] = df_imp["cluster_id"].map(lambda c: label_map.get(int(c), "unsolved"))

    # ---- Silhouette ----
    sil_dbscan = None
    try:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            sil_dbscan = float(silhouette_score(Xs, labels))
    except Exception:
        sil_dbscan = None

    # ---- Adjusted Rand Score (ARS) - robust ----
    ars_dbscan = None
    if diff_col and diff_col in df.columns and df[diff_col].notna().any():
        y_raw = df[diff_col].astype(str).str.strip().str.lower()
        mask = df[diff_col].notna()
        n_with = int(mask.sum())
        print(f"[ARS] Filas con etiqueta original no-nula: {n_with}")

        # Intentar mapear etiquetas estándar
        standard_map = {"easy":0, "medium":1, "hard":2, "unsolved":3}
        if set(y_raw.unique()) & set(standard_map.keys()):
            y_true_map = y_raw.map(standard_map).fillna(-1).astype(int).to_numpy()
            y_pred_map = df_imp["dificultad_dbscan"].astype(str).str.lower().map(standard_map).fillna(-1).astype(int).to_numpy()
            valid_mask = y_true_map >= 0
            if valid_mask.sum() > 0 and len(np.unique(y_pred_map[valid_mask])) >= 2 and len(np.unique(y_true_map[valid_mask])) >= 2:
                try:
                    ars_dbscan = float(adjusted_rand_score(y_true_map[valid_mask], y_pred_map[valid_mask]))
                except Exception as e:
                    print(f"[ARS] Error calculando ARS con mapeo estándar: {e}")
            else:
                print("[ARS] Mapeo estándar no cumple condiciones (>=2 clases); se intentará factorize fallback.")

        # Fallback: factorize etiquetas originales y comparar con cluster_id
        if ars_dbscan is None:
            y_true_raw = df.loc[mask, diff_col].astype(str).str.strip().str.lower()
            if len(y_true_raw) > 0:
                y_true_enc, uniques = pd.factorize(y_true_raw)
                y_pred_sub = df_imp.loc[mask, "cluster_id"].to_numpy()
                if len(np.unique(y_true_enc)) >= 2 and len(np.unique(y_pred_sub)) >= 2:
                    try:
                        ars_dbscan = float(adjusted_rand_score(y_true_enc, y_pred_sub))
                    except Exception as e:
                        print(f"[ARS] Error calculando ARS con factorize fallback: {e}")
                else:
                    print("[ARS] No se calculó ARS: se requieren >=2 clases en etiquetas originales y >=2 clusters predichos (en el subconjunto con etiqueta).")
            else:
                print("[ARS] No hay filas con etiqueta original para fallback.")
    else:
        print("[ARS] No se encontró columna de dificultad original con valores válidos; ARS no calculado.")

    # ---- Guardar resultados y artefactos ----
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
        "dificultad_dbscan", "cluster_id"
    ]
    result = result[cols_final]
    result.to_excel(XLSX_OUT, index=False)

    # PCA y gráfico
    try:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pts = pca.fit_transform(Xs)
        df_plot = pd.DataFrame({"_pc1": pts[:,0], "_pc2": pts[:,1], "dificultad_dbscan": df_imp["dificultad_dbscan"].values})
        plt.figure()
        for lab in sorted(df_plot["dificultad_dbscan"].unique()):
            s = df_plot[df_plot["dificultad_dbscan"] == lab]
            plt.scatter(s["_pc1"], s["_pc2"], label=str(lab))
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title(f"Clusters DBSCAN (eps≈{used_eps:.4f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PNG_OUT, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error generando gráfico PCA: {e}")

    # Guardar modelo, scaler y labels_map
    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        with open(LABELS_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in label_map.items()}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error guardando artefactos: {e}")

    # ---- Resumen y prints ----
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print("----- Resumen DBSCAN -----")
    print(f"eps usado: {used_eps}")
    print(f"N clusters (sin ruido): {n_clusters}")
    print(f"N ruido (label -1): {n_noise}")
    print(f"Silhouette: {sil_dbscan}")
    print(f"Adjusted Rand Score (si aplicable): {ars_dbscan}")
    print(f"Excel guardado en: {XLSX_OUT}")
    print(f"Gráfico guardado en: {PNG_OUT}")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Scaler guardado en: {SCALER_PATH}")
    print(f"Labels map guardado en: {LABELS_MAP_PATH}")

if __name__ == "__main__":
    main()

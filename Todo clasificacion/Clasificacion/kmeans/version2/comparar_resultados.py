from pathlib import Path
import pandas as pd
import numpy as np

# ==========================
# RUTAS (ajústalas si hace falta)
# ==========================
BASE = Path("/home/vale/Escritorio/Proyecto tesis/Clasificacion")
OLD_XLSX = BASE / "kmeans/clasificacion_kmeans.xlsx"            # del entrenamiento
NEW_XLSX = BASE / "kmeans/version2/predicciones_nuevos.xlsx"    # de kmeans_predict.py
OUT_XLSX = BASE / "kmeans/version2/comparacion_resultados.xlsx" # salida con resúmenes

# ==========================
# UTILIDADES
# ==========================

def normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def pick_label_column(df: pd.DataFrame):
    """Devuelve (col_label, tipo), donde tipo es 'dificultad' o 'cluster'.
    Prioriza 'dificultad_kmeans' si existe; si no, usa 'cluster_id'.
    """
    if "dificultad_kmeans" in df.columns:
        return "dificultad_kmeans", "dificultad"
    if "cluster_id" in df.columns:
        return "cluster_id", "cluster"
    return None, None


# ==========================
# MAIN
# ==========================

def main():
    if not OLD_XLSX.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrenamiento: {OLD_XLSX}")
    if not NEW_XLSX.exists():
        raise FileNotFoundError(f"No se encontró el archivo de nuevos datos: {NEW_XLSX}")

    df_old = pd.read_excel(OLD_XLSX)
    df_new = pd.read_excel(NEW_XLSX)

    df_old = normcols(df_old)
    df_new = normcols(df_new)

    # Columnas clave
    name_old = next((c for c in ["nombre","problem","problema","archivo","file","nombre_problema"] if c in df_old.columns), None)
    name_new = next((c for c in ["nombre","problem","problema","archivo","file","nombre_problema"] if c in df_new.columns), None)

    lab_old, kind_old = pick_label_column(df_old)
    lab_new, kind_new = pick_label_column(df_new)

    if lab_old is None:
        raise ValueError("El archivo de entrenamiento no tiene ni 'dificultad_kmeans' ni 'cluster_id'.")
    if lab_new is None:
        raise ValueError("El archivo de nuevos datos no tiene ni 'dificultad_kmeans' ni 'cluster_id'.")

    # =============================
    # Distribuciones por conjunto
    # =============================
    dist_old = df_old[lab_old].value_counts(dropna=False).sort_index()
    dist_new = df_new[lab_new].value_counts(dropna=False).sort_index()

    resumen_dist = pd.DataFrame({
        "categoria": sorted(set(dist_old.index).union(set(dist_new.index)), key=lambda x: str(x)),
    })
    resumen_dist["conteo_old"] = resumen_dist["categoria"].map(dist_old).fillna(0).astype(int)
    resumen_dist["conteo_new"] = resumen_dist["categoria"].map(dist_new).fillna(0).astype(int)

    # =============================
    # Comparación por nombre (si es posible)
    # =============================
    comparacion = None
    crosstab = None
    if name_old and name_new and name_old in df_old.columns and name_new in df_new.columns:
        merged = pd.merge(
            df_old[[name_old, lab_old]].rename(columns={name_old: "nombre", lab_old: "label_old"}),
            df_new[[name_new, lab_new]].rename(columns={name_new: "nombre", lab_new: "label_new"}),
            on="nombre", how="inner"
        )
        if not merged.empty:
            merged["cambio"] = np.where(merged["label_old"].astype(str) == merged["label_new"].astype(str), "=", "≠")
            comparacion = merged.sort_values(["cambio", "nombre"])  # primero iguales, luego distintos
            try:
                crosstab = pd.crosstab(merged["label_old"], merged["label_new"], dropna=False)
            except Exception:
                crosstab = None

    # =============================
    # Guardar en Excel
    # =============================
    with pd.ExcelWriter(OUT_XLSX) as writer:
        resumen_dist.to_excel(writer, sheet_name="distribuciones", index=False)
        if comparacion is not None:
            comparacion.to_excel(writer, sheet_name="comparacion_por_nombre", index=False)
        if crosstab is not None:
            crosstab.to_excel(writer, sheet_name="crosstab_old_vs_new")
        # Guardar vistas originales por si se quiere inspeccionar
        df_old.to_excel(writer, sheet_name="_old_raw", index=False)
        df_new.to_excel(writer, sheet_name="_new_raw", index=False)

    # =============================
    # Prints de consola
    # =============================
    print("Resumen de distribuciones (también en hoja 'distribuciones'):\n")
    print(resumen_dist.to_string(index=False))

    if comparacion is not None:
        cambios = (comparacion["cambio"] == "≠").sum()
        total = len(comparacion)
        print(f"\nCoincidencias por nombre: {total} (cambios: {cambios})")
        if crosstab is not None:
            print("\nCrosstab (old vs new):\n")
            print(crosstab)
    else:
        print("\nNo hay columna de 'nombre' común entre ambos archivos; se omite comparación fila a fila.")

    print(f"\nArchivo de salida: {OUT_XLSX}")


if __name__ == "__main__":
    main()

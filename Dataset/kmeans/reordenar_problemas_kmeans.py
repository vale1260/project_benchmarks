import argparse
import pandas as pd
from pathlib import Path
import shutil
import sys
import csv
from datetime import datetime
import re
import unicodedata

VALID_LABELS = {"easy", "medium", "hard", "unsolved"}

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def stem_from_name(name: str) -> str:
    p = Path(str(name).strip())
    return normalize_text(p.stem)

def label_normalize(lbl):
    m = normalize_text(lbl)
    aliases = {
        "facil": "easy", "ez": "easy",
        "medio": "medium", "med": "medium", "avg": "medium",
        "dificil": "hard", "hd": "hard",
        "no_resuelto": "unsolved", "sin_resolver": "unsolved", "nr": "unsolved",
    }
    return aliases.get(m, m)

def ensure_dirs(base: Path):
    for d in VALID_LABELS:
        (base / d).mkdir(parents=True, exist_ok=True)

def scan_sources(root: Path, extensions):
    index = {}
    for ext in extensions:
        for p in root.rglob(f"*{ext}"):
            if p.is_file():
                st = stem_from_name(p.name)
                index.setdefault(st, p.resolve())
    return index

def unique_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def main():
    ap = argparse.ArgumentParser(description="Reordenar problemas según Excel (K-Means).")
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default=None, help="Nombre o índice de hoja (opcional).")
    ap.add_argument("--name-col", default="nombre")
    ap.add_argument("--label-col", default="dificultad_kmeans")
    ap.add_argument("--origen", required=True)
    ap.add_argument("--destino", required=True)
    ap.add_argument("--exts", default=".bch", help='Lista separada por comas, p.ej. ".bch,.dat"')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    excel_path = Path(args.excel).expanduser().resolve()
    src_root = Path(args.origen).expanduser().resolve()
    dst_root = Path(args.destino).expanduser().resolve()

    if not excel_path.exists():
        print(f"[ERROR] No existe el Excel: {excel_path}", file=sys.stderr)
        sys.exit(1)
    if not src_root.exists():
        print(f"[ERROR] No existe la carpeta de origen: {src_root}", file=sys.stderr)
        sys.exit(1)

    dst_root.mkdir(parents=True, exist_ok=True)
    ensure_dirs(dst_root)

    try:
        if args.sheet is None:
            df = pd.read_excel(excel_path)
        else:
            try:
                sheet_spec = int(args.sheet)
            except:
                sheet_spec = args.sheet
            df = pd.read_excel(excel_path, sheet_name=sheet_spec)
    except Exception as e:
        print(f"[ERROR] No pude leer el Excel: {e}", file=sys.stderr)
        sys.exit(1)

    cols_map = {normalize_text(c): c for c in df.columns}
    name_key = normalize_text(args.name_col)
    label_key = normalize_text(args.label_col)

    if name_key not in cols_map or label_key not in cols_map:
        print("[ERROR] No se encontraron las columnas requeridas en el Excel.", file=sys.stderr)
        print(f"  Esperadas (puedes cambiarlas con --name-col/--label-col): '{args.name_col}', '{args.label_col}'", file=sys.stderr)
        print(f"  Columnas disponibles: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    name_col = cols_map[name_key]
    label_col = cols_map[label_key]

    extensions = [e.strip() for e in args.exts.split(",") if e.strip()]
    for e in list(extensions):
        if not e.startswith("."):
            extensions[extensions.index(e)] = "." + e

    idx = scan_sources(src_root, extensions)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = dst_root / f"reordenar_log_{timestamp}.txt"
    report_csv = dst_root / f"reordenar_report_{timestamp}.csv"
    missing_csv = dst_root / f"missing_{timestamp}.csv"
    unknown_csv = dst_root / f"unknown_labels_{timestamp}.csv"

    total = 0
    copied = 0
    skipped_unknown_label = 0
    not_found = 0
    collisions = 0

    per_label = {k: 0 for k in VALID_LABELS}
    missing_rows = []
    unknown_rows = []

    with open(log_path, "w", encoding="utf-8") as flog, open(report_csv, "w", newline="", encoding="utf-8") as frep:
        writer = csv.writer(frep)
        writer.writerow(["archivo_excel", "stem_excel", "label_excel", "ruta_origen", "ruta_destino", "accion", "nota"])

        for i, row in df.iterrows():
            raw_name = row[name_col]
            raw_label = row[label_col]
            total += 1

            nm = str(raw_name).strip()
            if not nm or str(nm).lower() in {"nan", "none"}:
                msg = f"[Fila {i}] Nombre vacío/NaN -> skip."
                print(msg, file=flog)
                writer.writerow([nm, "", raw_label, "", "", "skip", "nombre vacío/NaN"])
                continue

            lbl = label_normalize(raw_label)
            if lbl not in VALID_LABELS:
                skipped_unknown_label += 1
                unknown_rows.append((nm, raw_label))
                msg = f"[Fila {i}] Etiqueta desconocida '{raw_label}' (-> '{lbl}') para '{nm}' -> skip."
                print(msg, file=flog)
                writer.writerow([nm, "", raw_label, "", "", "skip", "etiqueta desconocida"])
                continue

            stem = stem_from_name(nm)
            if stem not in idx:
                not_found += 1
                missing_rows.append(nm)
                msg = f"[Fila {i}] No se encontró archivo para stem='{stem}' en {src_root} -> skip."
                print(msg, file=flog)
                writer.writerow([nm, stem, lbl, "", "", "skip", "no encontrado en origen"])
                continue

            src_path = idx[stem]
            dst_dir = dst_root / lbl
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / src_path.name
            final_dst = unique_path(dst_path)
            note = ""
            if final_dst != dst_path:
                collisions += 1
                note = "colisión de nombre -> renombrado"

            action = "would-copy" if args.dry_run else "copied"
            if args.dry_run:
                print(f"[DRY-RUN] {action}: '{src_path}' -> '{final_dst}'", file=flog)
            else:
                try:
                    shutil.copy2(str(src_path), str(final_dst))
                    copied += 1
                    per_label[lbl] += 1
                except Exception as e:
                    action = "error"
                    note = f"ERROR: {e}"
                    print(f"[ERROR] No se pudo copiar '{src_path}' -> '{final_dst}': {e}", file=flog)

            writer.writerow([nm, stem, lbl, str(src_path), str(final_dst), action, note])

    if missing_rows:
        pd.DataFrame({"no_encontrado_en_origen": missing_rows}).to_csv(missing_csv, index=False)
    if unknown_rows:
        pd.DataFrame(unknown_rows, columns=["archivo_excel","label_original"]).to_csv(unknown_csv, index=False)

    print("--------------------------------------------------")
    print("Resumen de la operación")
    print("--------------------------------------------------")
    print(f"Total filas procesadas: {total}")
    print(f"Copiados: {copied}")
    print(f"Etiquetas desconocidas (omitidos): {skipped_unknown_label}")
    print(f"No encontrados en origen (omitidos): {not_found}")
    print(f"Colisiones de nombre (renombrados): {collisions}")
    print("Por etiqueta destino:")
    for k, v in per_label.items():
        print(f"  - {k}: {v}")
    print(f"\nLog detallado: {log_path}")
    print(f"Reporte CSV:   {report_csv}")
    if missing_rows:
        print(f"CSV de no encontrados: {missing_csv}")
    if unknown_rows:
        print(f"CSV de labels desconocidos: {unknown_csv}")

    matched = total - not_found - skipped_unknown_label
    print("\nDiagnóstico rápido:")
    print(f"  - Matching efectivo: {matched}/{total} ({(matched/total*100 if total else 0):.1f}%)")
    if not_found > 0:
        print("  - Revisa 'missing_*.csv' para ver ejemplos de nombres que no calzaron.")
    if skipped_unknown_label > 0:
        print("  - Revisa 'unknown_labels_*.csv' para labels fuera de easy/medium/hard/unsolved.")

if __name__ == "__main__":
    main()

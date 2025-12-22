from pathlib import Path
import pandas as pd
import json
import sys
import unicodedata
import difflib
from typing import Optional, Dict, Tuple, List

# ========= CONFIG =========
EXCEL_PATH = Path("/home/vale/Escritorio/Proyecto tesis/Clasificacion/kmeans/clasificacion_kmeans.xlsx")
ROOT_OPTIM = Path("/home/vale/Escritorio/ibex-lib/benchs/optim")
OUTPUT_JSONL = Path("dataset_nuevo_kmeans.jsonl")

# Solo KMeans
NAME_CANDIDATES = ["nombre", "problem", "problema"]
NEW_DIFF_CANDIDATES = ["dificultad_kmeans"]

# Extensiones permitidas
ALLOWED_SUFFIXES = {".bch", ""}  # acepta .bch y sin extensión
# Rutas preferidas que pisan duplicados
PREFERRED_DIR_HINTS = ["benchs_victor_fixxed"]

# ========= HELPERS =========
def normalize_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    # quitar separadores/ruta y signos comunes
    for ch in [" ", "_", "-", ".", "\t", "\n", "\r", "/", "\\", "(", ")", "[", "]"]:
        s = s.replace(ch, "")
    return s

def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    name_col = next((c for c in cols if c.lower() in NAME_CANDIDATES), None)
    if name_col is None:
        low = {c.lower(): c for c in cols}
        for c in NAME_CANDIDATES:
            if c in low:
                name_col = low[c]; break
    if name_col is None:
        raise ValueError(f"No se encontró columna de nombre (candidatas: {NAME_CANDIDATES}). Columnas: {cols}")

    diff_col = next((c for c in cols if c.lower() in NEW_DIFF_CANDIDATES), None)
    if diff_col is None:
        low = {c.lower(): c for c in cols}
        for c in NEW_DIFF_CANDIDATES:
            if c in low:
                diff_col = low[c]; break
    if diff_col is None:
        raise ValueError(f"No se encontró columna 'dificultad_kmeans'. Columnas: {cols}")

    return name_col, diff_col

def looks_like_bch(p: Path) -> bool:
    """Heurística simple para aceptar archivos sin extensión que parecen .bch."""
    try:
        head = p.read_text(encoding="utf-8", errors="ignore")[:4000]
    except Exception:
        try:
            head = p.read_text(encoding="latin-1", errors="ignore")[:4000]
        except Exception:
            return False
    h = head.lower()
    tokens = ["variables", "constraints", "minimize", "maximize", "end"]
    return sum(t in h for t in tokens) >= 2

def index_bch_files(root: Path) -> Dict[str, Path]:
    """
    Recorre TODO 'root':
      - incluye *.bch (case-insensitive)
      - incluye archivos sin extensión que 'parezcan' .bch
    Registra DOS claves por archivo:
      - stem normalizado (p.stem)           -> "3pk"
      - ruta relativa sin extensión normal. -> "easy/3pk" -> "easy3pk"
    Luego prioriza rutas que contengan PREFERRED_DIR_HINTS (p. ej. benchs_victor_fixxed).
    """
    if not root.exists():
        raise FileNotFoundError(f"No existe la raíz: {root}")

    idx: Dict[str, Path] = {}

    # 1) Primer pase: todo el árbol
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in ALLOWED_SUFFIXES:
            continue
        if suf == "" and not looks_like_bch(p):
            continue

        stem_key = normalize_key(p.stem)
        rel_key  = normalize_key(str(p.relative_to(root).with_suffix("")))
        idx.setdefault(stem_key, p)
        idx.setdefault(rel_key, p)

    # 2) Priorizar los "fixxed"
    if PREFERRED_DIR_HINTS:
        hints = [h.lower() for h in PREFERRED_DIR_HINTS]
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf not in ALLOWED_SUFFIXES:
                continue
            if suf == "" and not looks_like_bch(p):
                continue
            p_low = str(p).lower()
            if any(h in p_low for h in hints):
                stem_key = normalize_key(p.stem)
                rel_key  = normalize_key(str(p.relative_to(root).with_suffix("")))
                idx[stem_key] = p
                idx[rel_key]  = p

    return idx

def find_file_for_problem(stem: str, index: Dict[str, Path]) -> Optional[Path]:
    """
    Cascada: exacta -> quitar '.bch' -> substring -> fuzzy (difflib).
    """
    key = normalize_key(stem)
    if key in index:
        return index[key]
    if key.endswith("bch"):
        key2 = key[:-3]
        if key2 in index:
            return index[key2]

    # substring
    for k, p in index.items():
        if key and (key in k or k in key):
            return p

    # fuzzy
    matches = difflib.get_close_matches(key, list(index.keys()), n=1, cutoff=0.8)
    if matches:
        return index[matches[0]]
    return None

# ========= MAIN =========
def main():
    if not EXCEL_PATH.exists():
        print(f"No se encontró el Excel: {EXCEL_PATH}", file=sys.stderr)
        sys.exit(1)
    if not ROOT_OPTIM.exists():
        print(f"No se encontró la carpeta raíz: {ROOT_OPTIM}", file=sys.stderr)
        sys.exit(1)

    print(f"Leyendo Excel: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    name_col, newdiff_col = detect_columns(df)

    work = df[[name_col, newdiff_col]].dropna().copy()

    print(f"Indexando problemas en: {ROOT_OPTIM} (incluye .bch y sin extensión)")
    file_index = index_bch_files(ROOT_OPTIM)
    if not file_index:
        print("No se encontraron archivos válidos en la ruta indicada.", file=sys.stderr)
        sys.exit(1)

    not_found = []
    suggestions = []
    written = 0

    with OUTPUT_JSONL.open("w", encoding="utf-8") as out:
        for _, row in work.iterrows():
            raw_name = str(row[name_col]).strip()
            new_diff = str(row[newdiff_col]).strip().lower()
            if not raw_name:
                continue

            path = find_file_for_problem(raw_name, file_index)
            if path is None or not path.exists():
                not_found.append(raw_name)
                # sugerencia para depurar
                key = normalize_key(raw_name)
                sm = difflib.get_close_matches(key, list(file_index.keys()), n=1, cutoff=0.6)
                if sm:
                    try:
                        relp = str(file_index[sm[0]].relative_to(ROOT_OPTIM))
                    except Exception:
                        relp = str(file_index[sm[0]])
                    suggestions.append((raw_name, sm[0], relp))
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")

            record = {"text": text, "difficulty": new_diff}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"JSONL generado: {OUTPUT_JSONL}  (líneas escritas: {written})")
    if not_found:
        print(f"No se encontraron {len(not_found)} archivos para los nombres del Excel. Ejemplos:", file=sys.stderr)
        for name in not_found[:10]:
            print(f"   - {name}", file=sys.stderr)
        if len(not_found) > 10:
            print(f"   ... y {len(not_found) - 10} más", file=sys.stderr)

    if suggestions:
        print("\nSugerencias de coincidencias cercanas (Excel → índice):", file=sys.stderr)
        for raw, key_match, path_rel in suggestions[:10]:
            print(f"   - '{raw}' ~ '{key_match}'  ->  {path_rel}", file=sys.stderr)
        if len(suggestions) > 10:
            print(f"   ... y {len(suggestions) - 10} más", file=sys.stderr)

if __name__ == "__main__":
    main()
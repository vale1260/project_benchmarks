from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import hashlib
from pathlib import Path
import json
import re
from typing import List, Dict, Optional, Tuple

# --- CONFIGURACIÓN ---
class Config:
    ORIGINAL_PATH = Path("/home/colossus/project_benchmarks/Dataset/kmeans")
    GENERATED_PATH = Path("generated_problems")
    CACHE_PATH = Path("data/errors_cache.jsonl")

    GENERATION_CONFIG = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "do_sample": True,
        "repetition_penalty": 1.1
    }

# --- INICIALIZACIÓN DEL MODELO ---
def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        **Config.GENERATION_CONFIG
    )
    return tokenizer, model, generator

def check_model_loading():
    model_path = Path("./models/fine_tuned_model")
    if not model_path.exists():
        print("ERROR: No se encuentra el modelo fine-tuned en la ruta especificada")
        print("Por favor, verifica que el modelo este en: ./models/fine_tuned_model/")
        return False
    else:
        print("Modelo fine-tuned encontrado correctamente")
        return True

# --- SISTEMA CACHE PARA EVITAR DUPLICADOS ---
def load_cache() -> List[Dict]:
    cache = []
    if Config.CACHE_PATH.exists():
        with open(Config.CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    cache.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"Cache cargado: {len(cache)} problemas en historial")
    return cache

def save_cache(cache_data: List[Dict]) -> None:
    Config.CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(Config.CACHE_PATH, "w", encoding="utf-8") as f:
        for item in cache_data:
            f.write(json.dumps(item) + "\n")
    print("Cache actualizado")

def normalize_problem_text(text: str) -> str:
    text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def is_problem_unique(problem_text: str, cache: List[Dict]) -> bool:
    normalized_new = normalize_problem_text(problem_text)
    for item in cache:
        if 'problem' in item:
            normalized_existing = normalize_problem_text(item['problem'])
            if normalized_new == normalized_existing:
                print("Problema duplicado detectado en cache")
                return False
    print("Problema unico")
    return True

def validate_problem_structure(problem_text: str) -> bool:
    """Validación mínima: secciones básicas + objetivo real y >=1 restricción."""
    lines = [l for l in problem_text.strip().split('\n') if l.strip()]

    # Rechazar cualquier 'Maximize'
    if any('maximize' in l.lower() for l in lines):
        print("Salida inválida: se encontró 'Maximize' (solo se permite 'Minimize').")
        return False

    has_vars = any(l.strip().lower() == 'variables' for l in lines)
    has_min  = any(l.strip().lower() == 'minimize' for l in lines)
    has_cons = any(l.strip().lower() == 'constraints' for l in lines)
    has_end  = any(l.strip().lower() == 'end' for l in lines)
    if not (has_vars and has_min and has_cons and has_end):
        print("Estructura incompleta (Variables/Minimize/Constraints/end).")
        return False

    txt = "\n".join(lines)
    # Minimize encabezado seguido de una línea con algún xN y ';'
    if not re.search(r'(?im)^\s*minimize\s*$\s*^.*\bx\d+\b.*;\s*$', txt):
        print("No se detectó una función objetivo válida tras 'Minimize'.")
        return False

    # Al menos una variable tipo "x1 in [a, b];"
    if not any(re.search(r'\bx\d+\s+in\s*\[', l, flags=re.I) and l.strip().endswith(';') for l in lines):
        print("No hay variables válidas.")
        return False

    # Al menos 1 restricción con operador y ';'
    if not any(any(op in l for op in ['<=', '>=', '=']) and l.strip().endswith(';') for l in lines):
        print("No hay restricciones válidas.")
        return False

    print("Estructura del problema valida")
    return True

def cut_at_first_end(text: str) -> str:
    pattern = re.compile(r"(.*?\bend\b)", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

def generate_with_model(difficulty: str) -> str:
    """Genera un problema usando SOLO el modelo fine-tuned (sin imponer conteos)."""
    # Prompt neutro: da ejemplo y reglas generales, sin números fijos
    prompt = f"""Generate a {difficulty} linear optimization problem in EXACT format.

Example (MINIMIZATION):
Variables
x1 in [0, 10];
x2 in [-5, 15];
x3 in [1, 20];

Minimize
2.5*x1 + 3.1*x2 - 1.7*x3;

Constraints
x1 + 2*x2 <= 15;
x1 - x3 >= 3;
x2 == 2.5;

end

Requirements:
- Use only this structure and headings: Variables / Minimize / Constraints / end
- Variables must have bounds [lower, upper] and end with ';'
- The objective must be a single linear expression on the NEXT line after 'Minimize' and end with ';'
- Each constraint must end with ';'
- End with exactly 'end'

Now generate a NEW unique {difficulty} problem:
"""
    tokenizer, model, generator = initialize_model()
    response = generator(
        prompt,
        max_new_tokens=Config.GENERATION_CONFIG["max_new_tokens"],
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = response[0]['generated_text']
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    return cut_at_first_end(generated_text)

def postprocess_problem(raw_problem: str) -> str:
    """Limpia formato; conserva objetivo en misma línea; no inventa contenido."""
    text = raw_problem.strip()
    text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    out = []
    section = None
    objective_captured = False

    def push_objective_from_tail(tail: str):
        nonlocal objective_captured
        expr = tail.strip(" :")
        if expr:
            expr = expr.rstrip(';') + ';'
            out.append(expr)
            objective_captured = True

    for line in lines:
        low = line.lower()

        if 'variables' in low and '[' not in low:
            section = "Variables"
            if not out or out[-1] != "Variables":
                out.append("Variables")
            continue

        if 'minimize' in low:
            section = "Minimize"
            if not out or out[-1] != "Minimize":
                out.append("Minimize")
            tail = re.split(r'minimize', line, flags=re.I, maxsplit=1)[1]
            if tail.strip():
                push_objective_from_tail(tail)
            continue

        if 'maximize' in low:
            # No admitimos maximización: lo dejamos pasar para que la validación falle.
            section = None
            continue

        if 'constraints' in low and '[' not in low:
            section = "Constraints"
            if not out or out[-1] != "Constraints":
                out.append("Constraints")
            continue

        if low == 'end':
            out.append('end')
            break

        if section == "Minimize" and not objective_captured:
            line = line.rstrip(';') + ';'
            out.append(line)
            objective_captured = True
        elif section in ("Variables", "Constraints"):
            if not line.endswith(';') and line.lower() != 'end':
                line += ';'
            out.append(line)
        else:
            pass

    if not out or out[-1].lower() != 'end':
        out.append('end')

    cleaned = []
    for ln in out:
        if cleaned and cleaned[-1] == ln and ln in ("Variables", "Minimize", "Constraints"):
            continue
        cleaned.append(ln)

    return "\n".join(cleaned)

def save_problem(problem_text: str, difficulty: str = "easy") -> Path:
    Config.GENERATED_PATH.mkdir(exist_ok=True, parents=True)
    difficulty_path = Config.GENERATED_PATH / difficulty
    difficulty_path.mkdir(exist_ok=True, parents=True)
    h = hashlib.sha256(problem_text.encode("utf-8")).hexdigest()[:10]
    filename = difficulty_path / f"problem_{difficulty}_{h}.bch"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(problem_text)
    return filename

def generate_new_problem(difficulty: str = "easy") -> Tuple[Optional[str], Optional[Path]]:
    print(f"\n Generando problema (tema/dificultad): {difficulty}")
    cache = load_cache()
    max_attempts = 15
    for attempt in range(max_attempts):
        print(f" Intento {attempt + 1}/{max_attempts}")
        generated = generate_with_model(difficulty)
        generated = postprocess_problem(generated)
        if validate_problem_structure(generated) and is_problem_unique(generated, cache):
            print("Problema valido")
            cache.append({"difficulty": difficulty, "problem": generated})
            save_cache(cache)
            saved_path = save_problem(generated, difficulty)
            return generated, saved_path
        else:
            print("Problema invalido o duplicado, reintentando...")
    print("No se pudo generar un problema valido")
    return None, None

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    if not check_model_loading():
        exit(1)
    Config.MODEL_PATH = "./models/fine_tuned_model"
    difficulty = input("Selecciona etiqueta (easy/medium/hard u otra): ").strip().lower() or "easy"
    problem, saved_path = generate_new_problem(difficulty)
    if problem:
        print("\n" + "="*50)
        print("PROBLEMA GENERADO EXITOSAMENTE")
        print("="*50)
        print(problem)
        print(f"Guardado en: {saved_path}")
        print(f"Cache actualizado: {Config.CACHE_PATH}")
    else:
        print("No se pudo generar un problema valido. Intenta nuevamente...")

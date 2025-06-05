from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import hashlib
from pathlib import Path
import json
import re
import random

# --- CONFIGURACIÓN ---
ORIGINAL_PATH = Path("/home/colossus/ibex-lib-master/benchs/optim")
GENERATED_PATH = Path("generated_problems")
GENERATED_PATH.mkdir(exist_ok=True, parents=True)

CACHE_PATH = Path("data/errors_cache.jsonl")
CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)

MODEL_PATH = "./models/fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    pad_token_id=tokenizer.eos_token_id
)

# --- FUNCIONES AUXILIARES ---

def load_cache():
    cache = []
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    cache.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cache

def save_cache(cache_data):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for item in cache_data:
            f.write(json.dumps(item) + "\n")
    print("Cache guardada.")

def is_problem_unique(problem_text, cache):
    normalized_new = normalize_problem_text(problem_text)
    for item in cache:
        if 'problem' in item:
            normalized_existing = normalize_problem_text(item['problem'])
            if normalized_new == normalized_existing:
                return False
    return True

def normalize_problem_text(text):
    text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def validate_format(problem_text):
    required_sections = ["variables", "constraints", "end"]
    lower_text = problem_text.lower()

    for section in required_sections:
        if section not in lower_text:
            print(f"Sección '{section}' faltante")
            return False

    if not any(opt in lower_text for opt in ["minimize", "maximize"]):
        print("Sección 'minimize' o 'maximize' faltante")
        return False

    # Corrección de la expresión regular
    var_section = re.search(r'variables(.*?)(minimize|maximize)', problem_text, re.DOTALL | re.IGNORECASE)
    if var_section:
        var_lines = var_section.group(1).strip().split('\n')
        for line in var_lines:
            if not re.match(r'^\s*x\d+\s+in\s+\[\s*\d*\.?\d+\s*,\s*\d*\.?\d+\s*\];?\s*$', line.strip()):
                print(f"Formato de variable incorrecto: {line.strip()}")
                return False
    else:
        print("No se pudo extraer la sección de variables")
        return False

    return True

def generate_structured_problem(difficulty):
    config = {
        "easy": {"num_vars": 3, "num_constraints": 2},
        "medium": {"num_vars": 5, "num_constraints": 4},
        "hard": {"num_vars": 7, "num_constraints": 6}
    }[difficulty]

    num_vars = config['num_vars']
    num_constraints = config['num_constraints']

    variables = []
    for i in range(1, num_vars + 1):
        var_type = random.choice(['int', 'float'])
        if var_type == 'int':
            lower = random.randint(0, 10)
            upper = random.randint(lower + 1, 20)
        else:
            lower = round(random.uniform(0, 5), 2)
            upper = round(random.uniform(lower + 0.1, 10), 2)
        variables.append(f"x{i} in [{lower}, {upper}];")
    var_text = "\n".join(variables)

    opt_type = random.choice(["minimize", "maximize"])

    obj_vars = [f"x{i}" for i in range(1, num_vars + 1)]
    coeffs = [round(random.uniform(0.5, 5), 2) for _ in range(num_vars)]
    subset_size = random.randint(2, num_vars)
    subset = random.sample(list(zip(coeffs, obj_vars)), subset_size)
    terms = [f"{random.choice(['+', '-'])} {c}*{v}" for c, v in subset]
    objective = " ".join(terms).lstrip("+ ").strip()

    constraints = []
    for _ in range(num_constraints):
        coeffs = [round(random.uniform(0.1, 3), 2) for _ in range(num_vars)]
        constr_vars = [f"{random.choice(['+', '-'])} {c}*{v}" for c, v in zip(coeffs, obj_vars)]
        lhs = " ".join(constr_vars).lstrip("+ ").strip()
        rhs = round(random.uniform(5, 20), 2)
        operator = random.choice(["<=", ">=", "="])
        constraints.append(f"{lhs} {operator} {rhs};")

    problem = f"""variables
{var_text}

{opt_type}
{objective};

constraints
{"\n".join(constraints)}

end"""

    return problem

def save_problem(problem_text, difficulty="easy"):
    difficulty_path = GENERATED_PATH / difficulty
    difficulty_path.mkdir(exist_ok=True, parents=True)
    h = hashlib.sha256(problem_text.encode("utf-8")).hexdigest()[:10]
    filename = difficulty_path / f"problem_{difficulty}_{h}.bch"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(problem_text)
    print(f"Problema guardado en: {filename}")
    return filename

def generate_new_problem(difficulty="easy"):
    print(f"Generando problema para dificultad: {difficulty}")
    cache = load_cache()

    max_attempts = 5
    for attempt in range(max_attempts):
        print(f"Intento {attempt + 1}")
        generated = generate_structured_problem(difficulty)

        if validate_format(generated) and is_problem_unique(generated, cache):
            print("Problema válido y único generado")
            cache.append({"difficulty": difficulty, "problem": generated})
            save_cache(cache)
            saved_path = save_problem(generated, difficulty)
            return generated, saved_path
        else:
            print("Problema inválido o duplicado, intentando de nuevo...")

    print("No se pudo generar un problema único. Intenta más tarde.")
    return None, None

# --- EJECUCIÓN ---
if __name__ == "__main__":
    difficulty = input("Selecciona dificultad (easy/medium/hard): ").strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        print("Dificultad inválida. Usando 'easy' por defecto.")
        difficulty = "easy"

    result, saved_path = generate_new_problem(difficulty)

    if result:
        print("\n=== Problema generado ===\n")
        print(result)
        print(f"\nGuardado en: {saved_path}")
    else:
        print("\nNo se pudo generar un problema válido. Intenta nuevamente.")

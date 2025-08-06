from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import hashlib
from pathlib import Path
import json
import re
import random
from typing import List, Dict, Optional, Tuple

# --- CONFIGURACIÓN ---
class Config:
    ORIGINAL_PATH = Path("/home/colossus/ibex-lib-master/benchs/optim")
    GENERATED_PATH = Path("generated_problems")
    CACHE_PATH = Path("data/errors_cache.jsonl")
    MODEL_PATH = "./models/fine_tuned_model"
    
    DIFFICULTY_SETTINGS = {
        "easy": {"num_vars": (2, 4), "num_constraints": (1, 3), "var_range": (0, 10)},
        "medium": {"num_vars": (4, 6), "num_constraints": (3, 5), "var_range": (-5, 15)},
        "hard": {"num_vars": (6, 8), "num_constraints": (5, 8), "var_range": (-10, 20)}
    }
    
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

tokenizer, model, generator = initialize_model()

# --- FUNCIONES AUXILIARES ---
def load_cache() -> List[Dict]:
    cache = []
    if Config.CACHE_PATH.exists():
        with open(Config.CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    cache.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cache

def save_cache(cache_data: List[Dict]) -> None:
    Config.CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(Config.CACHE_PATH, "w", encoding="utf-8") as f:
        for item in cache_data:
            f.write(json.dumps(item) + "\n")

def normalize_problem_text(text: str) -> str:
    text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def is_problem_unique(problem_text: str, cache: List[Dict]) -> bool:
    normalized_new = normalize_problem_text(problem_text)
    for item in cache:
        if 'problem' in item:
            normalized_existing = normalize_problem_text(item['problem'])
            if normalized_new == normalized_existing:
                return False
    return True

def validate_problem_structure(problem_text: str) -> bool:
    text_lower = problem_text.lower()
    required_elements = [
        ("variables", r"variables\s"),
        ("objective", r"(minimize|maximize)"),
        ("constraints", r"constraints\s"),
        ("domain", r"in\s*\[")
    ]
    for name, pattern in required_elements:
        if not re.search(pattern, text_lower):
            print(f"Elemento faltante: {name}")
            return False
    return True

def cut_at_first_end(text: str) -> str:
    pattern = re.compile(r"(.*?\bend\b)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else text.strip()

def generate_with_model(difficulty: str) -> str:
    prompt = f"""Generate a {difficulty} optimization problem with the following characteristics:
- Variables: {Config.DIFFICULTY_SETTINGS[difficulty]['num_vars'][0]} to {Config.DIFFICULTY_SETTINGS[difficulty]['num_vars'][1]} variables
- Constraints: {Config.DIFFICULTY_SETTINGS[difficulty]['num_constraints'][0]} to {Config.DIFFICULTY_SETTINGS[difficulty]['num_constraints'][1]} constraints
- Variable ranges: between {Config.DIFFICULTY_SETTINGS[difficulty]['var_range'][0]} and {Config.DIFFICULTY_SETTINGS[difficulty]['var_range'][1]}

Format:
Variables
x1 in [lower, upper];
x2 in [lower, upper];
...

Minimize/Maximize
objective_function;

Constraints
constraint1;
constraint2;
...
end
"""
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
    problem_text = raw_problem
    problem_text = re.sub(r"(Variables|Constraints|Minimize|Maximize):", r"\1", problem_text, flags=re.IGNORECASE)
    problem_text = re.sub(r'\n{3,}', '\n\n', problem_text)
    if not problem_text.strip().lower().endswith("end"):
        problem_text = problem_text.strip() + "\nend"
    return problem_text

def save_problem(problem_text: str, difficulty: str = "easy") -> Path:
    Config.GENERATED_PATH.mkdir(exist_ok=True, parents=True)
    difficulty_path = Config.GENERATED_PATH / difficulty
    difficulty_path.mkdir(exist_ok=True, parents=True)
    
    h = hashlib.sha256(problem_text.encode("utf-8")).hexdigest()[:10]
    filename = difficulty_path / f"problem_{difficulty}_{h}.bch"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(problem_text)
    
    return filename

def generate_structured_fallback(difficulty: str) -> str:
    settings = Config.DIFFICULTY_SETTINGS[difficulty]
    num_vars = random.randint(*settings['num_vars'])
    num_constraints = random.randint(*settings['num_constraints'])
    var_range = settings['var_range']
    
    variables = []
    for i in range(1, num_vars + 1):
        lower = random.randint(var_range[0], var_range[1] - 1)
        upper = random.randint(lower + 1, var_range[1])
        variables.append(f"x{i} in [{lower}, {upper}];")

    opt_type = random.choice(["Minimize", "Maximize"])
    objective_terms = [f"{random.choice(['+', '-'])} {round(random.uniform(0.5, 5), 2)}*x{i}" for i in range(1, num_vars + 1) if random.random() > 0.5]
    if not objective_terms:
        objective_terms.append(f"+ {round(random.uniform(0.5, 5), 2)}*x1")
    objective = " ".join(objective_terms).lstrip("+ ").strip()

    constraints = []
    for _ in range(num_constraints):
        terms = [f"{random.choice(['+', '-'])} {round(random.uniform(0.1, 3), 2)}*x{i}" for i in range(1, num_vars + 1) if random.random() > 0.3]
        if not terms:
            terms.append(f"+ {round(random.uniform(0.1, 3), 2)}*x1")
        lhs = " ".join(terms).lstrip("+ ").strip()
        rhs = round(random.uniform(5, 20), 2)
        op = random.choice(["<=", ">=", "=="])
        constraints.append(f"{lhs} {op} {rhs};")

    return f"""Variables
{"\n".join(variables)}

{opt_type}
{objective};

Constraints
{"\n".join(constraints)}
end"""

def generate_new_problem(difficulty: str = "easy") -> Tuple[Optional[str], Optional[Path]]:
    print(f"Generando problema de dificultad: {difficulty}")
    cache = load_cache()
    
    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"Intento {attempt + 1}/{max_attempts}")
        if random.random() < 0.7:
            generated = generate_with_model(difficulty)
            generated = postprocess_problem(generated)
        else:
            generated = generate_structured_fallback(difficulty)

        if validate_problem_structure(generated) and is_problem_unique(generated, cache):
            print("Problema válido y único generado")
            cache.append({"difficulty": difficulty, "problem": generated})
            save_cache(cache)
            saved_path = save_problem(generated, difficulty)
            return generated, saved_path
        else:
            print("Problema inválido o duplicado, reintentando...")

    print("No se pudo generar un problema válido después de varios intentos")
    return None, None

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    difficulty = input("Selecciona dificultad (easy/medium/hard): ").strip().lower()
    if difficulty not in Config.DIFFICULTY_SETTINGS:
        print("Dificultad inválida. Usando 'easy' por defecto.")
        difficulty = "easy"

    problem, saved_path = generate_new_problem(difficulty)

    if problem:
        print("\n=== Problema generado ===\n")
        print(problem)
        print(f"\nGuardado en: {saved_path}")
    else:
        print("\nNo se pudo generar un problema válido. Intenta nuevamente.")

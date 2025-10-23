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
    """Carga el cache de problemas generados anteriormente."""
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
    """Guarda el cache de problemas generados."""
    Config.CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(Config.CACHE_PATH, "w", encoding="utf-8") as f:
        for item in cache_data:
            f.write(json.dumps(item) + "\n")
    print("Cache actualizado")

def normalize_problem_text(text: str) -> str:
    """Normaliza el texto del problema para comparación."""
    text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def is_problem_unique(problem_text: str, cache: List[Dict]) -> bool:
    """Verifica si el problema generado es único comparando con el cache."""
    normalized_new = normalize_problem_text(problem_text)
    
    for item in cache:
        if 'problem' in item:
            normalized_existing = normalize_problem_text(item['problem'])
            if normalized_new == normalized_existing:
                print("Problema duplicado detectado en cache")
                return False
    
    print("Problema único - no está en cache")
    return True

def validate_problem_structure(problem_text: str) -> bool:
    """Valida la estructura del problema de forma rigurosa."""
    lines = problem_text.strip().split('\n')
    
    # Verificar estructura básica
    has_variables = any('variables' in line.lower() for line in lines)
    has_objective = any(word in line.lower() for line in lines for word in ['minimize', 'maximize'])  # ← AMBOS
    has_constraints = any('constraints' in line.lower() for line in lines)
    has_end = any('end' in line.lower() for line in lines)
    
    if not all([has_variables, has_objective, has_constraints, has_end]):
        print("Estructura básica incompleta")
        return False
    
    # Verificar formato de variables
    variable_lines = [line for line in lines if ' in [' in line and '];' in line]
    if len(variable_lines) < 2:
        print("Muy pocas variables definidas")
        return False
    
    # Verificar que hay restricciones
    constraint_lines = [line for line in lines if any(op in line for op in ['<=', '>=', '==', '=']) and ';' in line]
    if len(constraint_lines) < 1:
        print("No hay restricciones válidas")
        return False
    
    print("Estructura del problema válida")
    return True

def cut_at_first_end(text: str) -> str:
    """Corta el texto justo después de la primera aparición de 'end'."""
    pattern = re.compile(r"(.*?\bend\b)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()

def generate_with_model(difficulty: str) -> str:
    """Genera un problema usando SOLO el modelo fine-tuned."""
    settings = Config.DIFFICULTY_SETTINGS[difficulty]
    
    # PROMPT MODIFICADO - AHORA INCLUYE MINIMIZE Y MAXIMIZE
    prompt = f"""Generate a {difficulty} optimization problem in EXACT format:

Example 1 - MINIMIZATION:
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
4*x1 + 3*x3 <= 30;

end

Example 2 - MAXIMIZATION:
Variables
y1 in [0, 8];
y2 in [2, 12];
y3 in [-3, 10];

Maximize
1.5*y1 + 2.8*y2 + 0.9*y3;

Constraints
y1 + y2 <= 10;
2*y1 - y3 >= 1;
y2 <= 8;
y1 + y2 + y3 <= 15;

end

Requirements:
- Exactly {settings['num_vars'][0]}-{settings['num_vars'][1]} variables
- Exactly {settings['num_constraints'][0]}-{settings['num_constraints'][1]} constraints
- Variables must have bounds [lower, upper]
- Use coefficients with decimals like 2.5, 1.7, 0.9, etc.
- Objective can be EITHER Minimize OR Maximize  # ← NUEVA INSTRUCCIÓN
- Each constraint must end with ;
- Objective must end with ;
- End with exactly 'end'

Now generate a new unique {difficulty} problem (can be minimization or maximization):
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

    generated_text = cut_at_first_end(generated_text)
    return generated_text

def postprocess_problem(raw_problem: str) -> str:
    """Post-procesa el problema generado para forzar formato correcto."""
    problem_text = raw_problem.strip()
    
    # Eliminar comentarios y texto extra
    problem_text = re.sub(r'#.*$', '', problem_text, flags=re.MULTILINE)
    problem_text = re.sub(r'//.*$', '', problem_text, flags=re.MULTILINE)
    
    # Forzar formato de secciones
    sections = ["Variables", "Minimize", "Maximize", "Constraints", "end"]
    current_section = None
    output_lines = []
    
    for line in problem_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Detectar secciones (AHORA INCLUYE MAXIMIZE)
        lower_line = line.lower()
        if 'variable' in lower_line and '[' not in line:
            current_section = "Variables"
            output_lines.append("Variables")
        elif 'minimize' in lower_line:
            current_section = "Minimize"
            output_lines.append("Minimize")
        elif 'maximize' in lower_line:  # ← NUEVA DETECCIÓN
            current_section = "Maximize"
            output_lines.append("Maximize")
        elif 'constraint' in lower_line:
            current_section = "Constraints"
            output_lines.append("Constraints")
        elif 'end' in lower_line:
            output_lines.append("end")
            break
        else:
            # Agregar línea según la sección actual
            if current_section and line:
                output_lines.append(line)
    
    # Unir y asegurar formato final
    problem_text = '\n'.join(output_lines)
    
    # Asegurar que termina con end
    if not problem_text.strip().lower().endswith('end'):
        problem_text = problem_text.strip() + '\nend'
    
    return problem_text

def save_problem(problem_text: str, difficulty: str = "easy") -> Path:
    """Guarda el problema generado en un archivo."""
    Config.GENERATED_PATH.mkdir(exist_ok=True, parents=True)
    difficulty_path = Config.GENERATED_PATH / difficulty
    difficulty_path.mkdir(exist_ok=True, parents=True)
    
    h = hashlib.sha256(problem_text.encode("utf-8")).hexdigest()[:10]
    filename = difficulty_path / f"problem_{difficulty}_{h}.bch"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(problem_text)
    
    return filename

def generate_new_problem(difficulty: str = "easy") -> Tuple[Optional[str], Optional[Path]]:
    """Genera un nuevo problema de optimización usando SOLO el modelo entrenado."""
    print(f"\n Generando problema de dificultad: {difficulty}")
    cache = load_cache()
    
    max_attempts = 15
    for attempt in range(max_attempts):
        print(f" Intento {attempt + 1}/{max_attempts}")
        
        # Generar con modelo entrenado
        generated = generate_with_model(difficulty)
        generated = postprocess_problem(generated)
        
        # Verificar unicidad y estructura
        if validate_problem_structure(generated) and is_problem_unique(generated, cache):
            print("Problema válido y único generado con el modelo entrenado")
            
            # Actualizar cache
            cache.append({"difficulty": difficulty, "problem": generated})
            save_cache(cache)
            
            # Guardar archivo
            saved_path = save_problem(generated, difficulty)
            return generated, saved_path
        else:
            print("Problema inválido o duplicado, reintentando...")
    
    print("No se pudo generar un problema válido después de todos los intentos")
    return None, None

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # Verificar modelo
    if not check_model_loading():
        exit(1)
    
    # Configuración
    Config.MODEL_PATH = "./models/fine_tuned_model"
    
    # Seleccionar dificultad
    difficulty = input("Selecciona dificultad (easy/medium/hard): ").strip().lower()
    if difficulty not in Config.DIFFICULTY_SETTINGS:
        print("Dificultad inválida. Usando 'easy' por defecto.")
        difficulty = "easy"

    # Generar problema
    problem, saved_path = generate_new_problem(difficulty)

    if problem:
        print("\n" + "="*50)
        print("PROBLEMA GENERADO EXITOSAMENTE")
        print("="*50)
        print(problem)
        print(f"\n Guardado en: {saved_path}")
        print(f" Cache actualizado: data/errors_cache.jsonl")
        
    else:
        print("\n No se pudo generar un problema válido. Intenta nuevamente.")

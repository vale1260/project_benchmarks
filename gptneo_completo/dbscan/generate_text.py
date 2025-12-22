from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import hashlib
from pathlib import Path
import json
import re
from typing import List, Dict, Optional, Tuple
import random

# --- CONFIGURACIÓN PARA GPT-NEO ---
class Config:
    MODEL_PATH = "./models/fine_tuned_gpt_neo"
    GENERATED_PATH = Path("generated_problems")
    CACHE_PATH = Path("data/errors_cache.jsonl")

    # CONFIGURACIÓN MÁS PERMISIVA TEMPORALMENTE
    GENERATION_CONFIG = {
        "temperature": 0.9,
        "top_p": 0.92,
        "top_k": 60,
        "max_new_tokens": 400,
        "do_sample": True,
        "repetition_penalty": 1.3,
        "num_beams": 1,
    }

# --- INICIALIZACIÓN DEL MODELO GPT-NEO ---
def initialize_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
        
        # Para GPT-Neo, asegurarnos de tener el pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_PATH,
            torch_dtype=torch.float32,
        )
        
        # Usar CPU para GPT-Neo
        device = torch.device("cpu")
        model.to(device)
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            **Config.GENERATION_CONFIG
        )
        print("Modelo inicializado correctamente")
        return tokenizer, model, generator
    except Exception as e:
        print(f"Error inicializando modelo: {e}")
        return None, None, None

def check_model_loading():
    model_path = Path(Config.MODEL_PATH)
    if not model_path.exists():
        print(f"ERROR: No se encuentra el modelo fine-tuned en la ruta: {Config.MODEL_PATH}")
        return False
    else:
        print("Modelo GPT-Neo fine-tuned encontrado correctamente")
        return True

# --- SISTEMA CACHE ---
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
    if not problem_text:
        return False
        
    normalized_new = normalize_problem_text(problem_text)
    for item in cache:
        if 'problem' in item:
            normalized_existing = normalize_problem_text(item['problem'])
            if normalized_new == normalized_existing:
                print("✗ Problema duplicado detectado en cache")
                return False
    print("Problema único")
    return True

def validate_problem_structure(problem_text: str) -> bool:
    """Validación más flexible para debugging"""
    if not problem_text or len(problem_text.strip()) < 50:
        print("Texto demasiado corto o vacío")
        return False
        
    print(f"Validando problema de {len(problem_text)} caracteres...")
    lines = [l.strip() for l in problem_text.split('\n') if l.strip()]
    
    if len(lines) < 6:
        print(f"Muy pocas líneas: {len(lines)}")
        return False

    # Verificar secciones principales (case insensitive)
    sections_found = {
        'variables': any('variables' in l.lower() for l in lines),
        'minimize': any('minimize' in l.lower() for l in lines),
        'constraints': any('constraints' in l.lower() for l in lines),
        'end': any('end' == l.lower() for l in lines)
    }
    
    print(f"Secciones encontradas: {sections_found}")
    
    if not all(sections_found.values()):
        missing = [k for k,v in sections_found.items() if not v]
        print(f"Faltan secciones: {missing}")
        return False

    # Buscar variables (patrón más flexible)
    variables_found = any(
        re.search(r'\b\w+\s+in\s*\[', l, re.IGNORECASE) and ';' in l 
        for l in lines
    )
    if not variables_found:
        print("No se encontraron variables válidas")
        return False

    # Buscar función objetivo
    objective_found = False
    for i, line in enumerate(lines):
        if 'minimize' in line.lower():
            # Buscar en la línea actual o siguiente
            if any(char in line for char in ['*', '+', '-']) and ';' in line:
                objective_found = True
                break
            elif i + 1 < len(lines) and any(char in lines[i+1] for char in ['*', '+', '-']) and ';' in lines[i+1]:
                objective_found = True
                break
    
    if not objective_found:
        print("No se encontró función objetivo válida")
        return False

    # Buscar al menos una restricción
    constraints_found = any(
        any(op in l for op in ['<=', '>=', '=']) and ';' in l
        for l in lines
    )
    if not constraints_found:
        print("No se encontraron restricciones válidas")
        return False

    print("Estructura del problema válida")
    return True

def cut_at_first_end(text: str) -> str:
    if not text:
        return ""
    pattern = re.compile(r"(.*?\bend\b)", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

def generate_with_model(difficulty: str) -> str:
    """Genera con prompts diversos"""
    
    prompts = [
        # Prompt más simple y directo
        f"""Generate a {difficulty} linear optimization problem:

Variables
x1 in [0, 10];
x2 in [0, 8];

Minimize
3*x1 + 2*x2;

Constraints
x1 + x2 <= 12;
x1 >= 1;

end

Now create a different one:""",

        # Prompt con nombres de variables diferentes
        f"""Create a linear programming problem:

Variables
a in [1, 15];
b in [2, 10];

Minimize
4*a + 5*b;

Constraints
2*a + b <= 20;
a >= 3;

end

New {difficulty} problem:""",

        # Prompt minimalista
        f"""Linear optimization problem:

Variables
var1 in [_,_];
var2 in [_,_];

Minimize
___;

Constraints
___;

end

Fill with values:""",

        # Prompt técnico
        f"""Compose a {difficulty} linear minimization:

Variables
x in [lower1, upper1];
y in [lower2, upper2];

Minimize
coefficient1*x + coefficient2*y;

Constraints
constraint1;
constraint2;

end

Generate:"""
    ]
    
    prompt = random.choice(prompts)
    print(f"Usando prompt tipo {prompts.index(prompt) + 1}")
    
    tokenizer, model, generator = initialize_model()
    if generator is None:
        return ""
    
    try:
        print("Generando...")
        response = generator(
            prompt,
            max_new_tokens=350,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        generated_text = response[0]['generated_text'].strip()
        print(f"Texto generado ({len(generated_text)} chars)")
        
        result = cut_at_first_end(generated_text)
        print(f"Recortado a ({len(result)} chars)")
        
        return result
        
    except Exception as e:
        print(f"Error en generación: {e}")
        return ""

def postprocess_problem(raw_problem: str) -> str:
    """Post-procesamiento más robusto"""
    if not raw_problem:
        return ""
        
    print("Post-procesando...")
    
    # Limpiar comentarios
    text = re.sub(r'#.*$', '', raw_problem, flags=re.MULTILINE)
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    if not lines:
        return ""
    
    # Reconstruir con formato consistente
    sections = []
    current_section = []
    section_headers = ['variables', 'minimize', 'constraints']
    
    for line in lines:
        lower_line = line.lower()
        
        # Detectar inicio de sección
        if any(header in lower_line for header in section_headers):
            if current_section:
                sections.append("\n".join(current_section))
            current_section = [line]
        elif lower_line == 'end':
            if current_section:
                sections.append("\n".join(current_section))
            sections.append('end')
            break
        else:
            # Asegurar que las líneas de código terminen con ;
            if (any(char in line for char in ['[', '*', '=', '<', '>']) and 
                not line.endswith(';') and 
                not any(word in lower_line for word in section_headers + ['end'])):
                line += ';'
            current_section.append(line)
    
    # Asegurar que terminamos con 'end'
    if current_section and (not sections or sections[-1].lower() != 'end'):
        sections.append("\n".join(current_section))
        if sections[-1].lower() != 'end':
            sections.append('end')
    
    result = "\n".join(sections)
    print(f"Post-procesado: {len(result)} chars")
    return result

def save_problem(problem_text: str, difficulty: str = "easy") -> Path:
    Config.GENERATED_PATH.mkdir(exist_ok=True, parents=True)
    difficulty_path = Config.GENERATED_PATH / difficulty
    difficulty_path.mkdir(exist_ok=True, parents=True)
    h = hashlib.sha256(problem_text.encode("utf-8")).hexdigest()[:10]
    filename = difficulty_path / f"problem_{difficulty}_{h}.bch"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(problem_text)
    print(f"Guardado en: {filename}")
    return filename

def generate_new_problem(difficulty: str = "easy") -> Tuple[Optional[str], Optional[Path]]:
    print(f"\nGenerando problema: {difficulty}")
    cache = load_cache()
    max_attempts = 10
    
    for attempt in range(max_attempts):
        print(f"\nIntento {attempt + 1}/{max_attempts}")
        generated = generate_with_model(difficulty)
        
        if not generated:
            print("Generación falló")
            continue
            
        processed = postprocess_problem(generated)
        
        if not processed:
            print(" Post-procesamiento falló")
            continue
        
        print("Validando...")
        is_valid = validate_problem_structure(processed)
        is_unique = is_problem_unique(processed, cache)
        
        if is_valid and is_unique:
            print("¡Problema válido y único!")
            cache.append({"difficulty": difficulty, "problem": processed})
            save_cache(cache)
            saved_path = save_problem(processed, difficulty)
            return processed, saved_path
        else:
            print("Problema inválido o duplicado, reintentando...")
    
    print(f"No se pudo generar un problema válido después de {max_attempts} intentos")
    return None, None

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print("=" * 60)
    print("GENERADOR DE PROBLEMAS DE OPTIMIZACIÓN")
    print("=" * 60)
    
    if not check_model_loading():
        exit(1)
    
    difficulty = input("Selecciona dificultad (easy/medium/unsolved): ").strip().lower() or "easy"
    problem, saved_path = generate_new_problem(difficulty)
    
    if problem:
        print("\n" + "="*60)
        print("PROBLEMA GENERADO EXITOSAMENTE")
        print("="*60)
        print(problem)
        print(f"Guardado en: {saved_path}")
    else:
        print("\n No se pudo generar un problema válido")

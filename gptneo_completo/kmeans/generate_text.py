from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import hashlib
from pathlib import Path
import json
import re
from typing import List, Dict, Optional, Tuple

# --- CONFIGURACIÓN MEJORADA ---
class Config:
    MODEL_PATH = "./models/fine_tuned_gpt_neo"
    GENERATED_PATH = Path("generated_problems")
    CACHE_PATH = Path("data/errors_cache.jsonl")

    # CONFIGURACIÓN MÁS ESTRICTA para GPT-Neo
    GENERATION_CONFIG = {
        "temperature": 0.3,           # 🔽 MÁS BAJO = más determinista
        "top_p": 0.85,
        "top_k": 40,                  # ✅ Añadir top_k
        "max_new_tokens": 300,        # 🔽 Reducido para más enfoque
        "do_sample": True,
        "repetition_penalty": 1.3,    # 🔽 Más alto para menos repeticiones
        "num_beams": 1,               # Búsqueda greedy para más consistencia
    }

# --- INICIALIZACIÓN ---
def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        **Config.GENERATION_CONFIG
    )
    return tokenizer, model, generator

def check_model_loading():
    model_path = Path(Config.MODEL_PATH)
    if not model_path.exists():
        print("ERROR: No se encuentra el modelo fine-tuned")
        return False
    print("✅ Modelo GPT-Neo fine-tuned encontrado")
    return True

# --- CACHE (igual) ---
def load_cache() -> List[Dict]:
    cache = []
    if Config.CACHE_PATH.exists():
        with open(Config.CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    cache.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"📊 Cache cargado: {len(cache)} problemas")
    return cache

def save_cache(cache_data: List[Dict]) -> None:
    Config.CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(Config.CACHE_PATH, "w", encoding="utf-8") as f:
        for item in cache_data:
            f.write(json.dumps(item) + "\n")

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
                return False
    return True

# --- VALIDACIÓN MÁS FLEXIBLE ---
def validate_problem_structure(problem_text: str) -> bool:
    """Validación más flexible para GPT-Neo"""
    lines = [l.strip() for l in problem_text.split('\n') if l.strip()]
    
    # Verificar secciones principales (case insensitive)
    sections_found = {
        'variables': any('variables' in l.lower() for l in lines),
        'minimize': any('minimize' in l.lower() for l in lines),
        'constraints': any('constraints' in l.lower() for l in lines),
        'end': any(l.lower() == 'end' for l in lines)
    }
    
    if not all(sections_found.values()):
        print(f"❌ Faltan secciones: {[k for k,v in sections_found.items() if not v]}")
        return False

    # Buscar variables (patrón más flexible)
    variables_found = any(
        re.search(r'x\d+\s+in\s*\[', l, re.IGNORECASE) and ';' in l 
        for l in lines
    )
    if not variables_found:
        print("❌ No se encontraron variables válidas")
        return False

    # Buscar al menos una restricción
    constraints_found = any(
        any(op in l for op in ['<=', '>=', '==', '=']) and ';' in l
        for l in lines
    )
    if not constraints_found:
        print("❌ No se encontraron restricciones válidas")
        return False

    print("✅ Estructura válida")
    return True

def cut_at_first_end(text: str) -> str:
    pattern = re.compile(r"(.*?\bend\b)", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

# --- GENERACIÓN MEJORADA ---
def generate_with_model(difficulty: str) -> str:
    """Generación con prompt más específico"""
    prompt = f"""Generate a {difficulty} linear optimization problem.

FORMAT:
Variables
x1 in [lower, upper];
x2 in [lower, upper];

Minimize
linear_expression;

Constraints
constraint1;
constraint2;

end

Example:
Variables
x1 in [0, 10];
x2 in [1, 5];

Minimize
3*x1 + 2*x2;

Constraints
x1 + x2 <= 8;
x1 >= 2;

end

Generate a new {difficulty} problem:
"""
    
    tokenizer, model, generator = initialize_model()
    
    try:
        response = generator(
            prompt,
            max_new_tokens=Config.GENERATION_CONFIG["max_new_tokens"],
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False  # ✅ Solo el texto generado, no el prompt
        )
        
        generated_text = response[0]['generated_text'].strip()
        return cut_at_first_end(generated_text)
        
    except Exception as e:
        print(f"❌ Error en generación: {e}")
        return ""

def postprocess_problem(raw_problem: str) -> str:
    """Post-procesamiento más robusto"""
    if not raw_problem:
        return ""
        
    # Limpiar comentarios y espacios
    text = re.sub(r'#.*$', '', raw_problem, flags=re.MULTILINE)
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    if not lines:
        return ""
        
    # Reconstruir con formato consistente
    sections = []
    current_section = []
    
    for line in lines:
        lower_line = line.lower()
        
        if any(section in lower_line for section in ['variables', 'minimize', 'constraints']):
            if current_section:
                sections.append("\n".join(current_section))
            current_section = [line]
        elif lower_line == 'end':
            if current_section:
                sections.append("\n".join(current_section))
            sections.append('end')
            break
        else:
            # Asegurar que las líneas terminen con ;
            if not line.endswith(';') and not any(word in lower_line for word in ['variables', 'minimize', 'constraints', 'end']):
                line += ';'
            current_section.append(line)
    
    if current_section and 'end' not in sections:
        sections.append("\n".join(current_section))
        sections.append('end')
    
    return "\n".join(sections)

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
    print(f"\n🎯 Generando problema: {difficulty}")
    cache = load_cache()
    
    for attempt in range(15):
        print(f"🔄 Intento {attempt + 1}/15")
        generated = generate_with_model(difficulty)
        
        if not generated:
            print("❌ Generación falló, reintentando...")
            continue
            
        processed = postprocess_problem(generated)
        
        if validate_problem_structure(processed) and is_problem_unique(processed, cache):
            print("✅ Problema válido y único generado!")
            cache.append({"difficulty": difficulty, "problem": processed})
            save_cache(cache)
            saved_path = save_problem(processed, difficulty)
            return processed, saved_path
        else:
            print("❌ Problema inválido o duplicado, reintentando...")
    
    print("❌ No se pudo generar un problema válido después de 15 intentos")
    return None, None

# --- EJECUCIÓN ---
if __name__ == "__main__":
    if not check_model_loading():
        exit(1)
        
    difficulty = input("Selecciona dificultad (easy/medium/hard): ").strip().lower() or "easy"
    problem, saved_path = generate_new_problem(difficulty)
    
    if problem:
        print("\n" + "="*50)
        print("🎉 PROBLEMA GENERADO EXITOSAMENTE")
        print("="*50)
        print(problem)
        print(f"\n💾 Guardado en: {saved_path}")
    else:
        print("\n😞 No se pudo generar un problema válido")

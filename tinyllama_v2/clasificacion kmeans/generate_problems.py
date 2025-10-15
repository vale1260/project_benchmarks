import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime
import glob
import re

class ProblemGenerator:
    def __init__(self):
        self._models = {}
        self._tokenizers = {}
        self.model_paths = {}
        self.difficulties = ['easy', 'medium', 'hard', 'unsolved']
        self.base_output_dir = "problemas_generados_tinyllama"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*70)
        print(" GENERADOR CON TINYLLAMA-1.1B")
        print("="*70)
        print(f"  Dispositivo: {self.device}")
        
        os.makedirs(self.base_output_dir, exist_ok=True)
        for difficulty in self.difficulties:
            os.makedirs(os.path.join(self.base_output_dir, difficulty), exist_ok=True)
            model_path = f'./models/tinyllama-{difficulty}'
            if os.path.exists(model_path):
                self.model_paths[difficulty] = model_path
                print(f" Modelo encontrado: {difficulty}")
            else:
                print(f"  No encontrado: {difficulty}")
    
    def _load_model(self, difficulty):
        """Carga lazy del modelo solo cuando se necesita"""
        if difficulty not in self._models:
            print(f" Cargando modelo '{difficulty}'...")
            try:
                model_path = self.model_paths[difficulty]
                
                # Cargar tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                # Configurar padding token
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token:
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
                # Cargar modelo
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                model.to(self.device)
                model.eval()
                
                self._tokenizers[difficulty] = tokenizer
                self._models[difficulty] = model
                print(f" Modelo '{difficulty}' cargado")
            except Exception as e:
                print(f" Error cargando modelo {difficulty}: {e}")
                raise e
        return self._models[difficulty], self._tokenizers[difficulty]

    def get_next_filename(self, difficulty):
        difficulty_dir = os.path.join(self.base_output_dir, difficulty)
        existing_files = glob.glob(os.path.join(difficulty_dir, "problema_*.bch"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        next_number = len(existing_files) + 1
        filename = f"problema_{difficulty}_{next_number:03d}_{timestamp}.bch"
        return os.path.join(difficulty_dir, filename)
    
    def get_prompt_consistent_with_training(self, difficulty):
        """Prompt idéntico al usado en entrenamiento"""
        return f"### Dificultad: {difficulty}\n### Problema:\n"
    
    def generate_problem(self, difficulty, num_problems=1):
        if difficulty not in self.model_paths:
            raise ValueError(f"Dificultad {difficulty} no disponible")
        
        model, tokenizer = self._load_model(difficulty)
        problems = []
        
        print(f" Generando {num_problems} problema(s) con TinyLlama-1.1B...")
        
        for i in range(num_problems):
            prompt_text = self.get_prompt_consistent_with_training(difficulty)
            
            # Codificar prompt
            input_ids = tokenizer.encode(
                prompt_text, 
                return_tensors='pt',
                add_special_tokens=True
            ).to(self.device)
            
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            # Parámetros optimizados para TinyLlama
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=1024,
                    min_length=150,
                    num_return_sequences=1,
                    temperature=0.8,  # Balance entre creatividad y coherencia
                    top_k=50,
                    top_p=0.92,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                )
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extraer y limpiar
            problem_text = self._extract_and_clean(generated_text, prompt_text)
            
            problems.append(problem_text)
            print(f"   ✓ Problema {i+1}/{num_problems} generado")
        
        return problems
    
    def _extract_and_clean(self, generated_text, prompt_text):
        """Extrae y limpia el contenido generado"""
        content = generated_text
        
        # Buscar después del prompt
        if '### Problema:\n' in content:
            content = content.split('### Problema:\n', 1)[1]
        elif 'Problema:\n' in content:
            content = content.split('Problema:\n', 1)[1]
        
        # Remover marcadores de fin
        content = content.split('### FIN')[0]
        content = content.split('###')[0]
        
        # Limpiar repeticiones
        if 'Dificultad:' in content:
            content = content.split('Dificultad:')[0]
        
        content = content.strip()
        
        return self.validate_and_clean_problem(content)
    
    def validate_and_clean_problem(self, problem_text):
        """Validación y limpieza del problema"""
        required = ['variables', 'minimize', 'constraints']
        has_structure = all(kw in problem_text.lower() for kw in required[:2])
        has_variable = bool(re.search(r'\w+\s+in\s+\[', problem_text))
        is_long = len(problem_text) > 50
        
        if not (has_structure and has_variable and is_long):
            print("  Usando problema de respaldo")
            return self.get_fallback_problem()
        
        # Limpiar líneas
        lines = problem_text.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                cleaned_lines.append(line_stripped)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append('')
                prev_empty = True
        
        problem_text = '\n'.join(cleaned_lines)
        
        # Asegurar 'end'
        if not problem_text.strip().endswith('end'):
            if '\nend' in problem_text:
                parts = re.split(r'\nend\b', problem_text)
                problem_text = parts[0] + '\n\nend'
            else:
                problem_text = problem_text.rstrip() + '\n\nend'
        
        return problem_text
    
    def get_fallback_problem(self):
        return """variables

x1 in [0,1];
x2 in [0,1];

minimize

3*x1 + 2*x2;

constraints

x1 + x2 <= 1;
x1 >= 0;
x2 >= 0;

end"""
    
    def save_as_bch(self, problem_text, difficulty):
        filepath = self.get_next_filename(difficulty)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(problem_text)
        
        filename_only = os.path.basename(filepath)
        print(f" Guardado: {filename_only}")
        
        # Estadísticas
        lines = problem_text.split('\n')
        variables = len([l for l in lines if re.search(r'\w+\s+in\s+\[', l)])
        constraints = len([l for l in lines if any(op in l for op in ['<=', '>=', '=='])])
        
        print(f" Variables: {variables} | Restricciones: {constraints} | {len(problem_text)} chars")
        
        return filepath
    
    def show_existing_problems(self, difficulty):
        difficulty_dir = os.path.join(self.base_output_dir, difficulty)
        existing_files = glob.glob(os.path.join(difficulty_dir, "problema_*.bch"))
        
        if existing_files:
            print(f"\n Problemas existentes en '{difficulty}': {len(existing_files)}")
            for i, filepath in enumerate(sorted(existing_files)[-5:], 1):
                print(f"   {i}. {os.path.basename(filepath)}")
        else:
            print(f"\n No hay problemas previos en '{difficulty}'")
    
    def analyze_quality(self, problem_text):
        """Analiza la calidad del problema"""
        if problem_text == self.get_fallback_problem():
            return False, 0, "FALLBACK"
        
        score = 0
        issues = []
        
        if 'variables' in problem_text.lower():
            score += 15
        else:
            issues.append("sin 'variables'")
        
        if any(kw in problem_text.lower() for kw in ['minimize', 'maximize']):
            score += 15
        else:
            issues.append("sin objetivo")
        
        if 'constraints' in problem_text.lower():
            score += 10
        else:
            issues.append("sin 'constraints'")
        
        num_vars = len(re.findall(r'\w+\s+in\s+\[', problem_text))
        score += min(num_vars * 10, 30)
        
        num_constraints = len(re.findall(r'[<>=]{1,2}', problem_text))
        score += min(num_constraints * 5, 20)
        
        if problem_text.strip().endswith('end'):
            score += 10
        
        is_valid = score >= 60
        status = "VÁLIDO" if is_valid else f"INCOMPLETO"
        
        return is_valid, score, status
    
    def interactive_generation(self):
        print("\n" + "="*70)
        print(" GENERADOR DE PROBLEMAS - TINYLLAMA-1.1B")
        print("="*70)
        
        available = list(self.model_paths.keys())
        if not available:
            print("\n No hay modelos TinyLlama entrenados disponibles")
            print("   Ejecuta 'python fine_tune_tinyllama.py' primero")
            return
        
        print("\n Dificultades disponibles:")
        for i, diff in enumerate(available, 1):
            print(f"   {i}. {diff}")
        
        while True:
            try:
                choice = int(input(f"\n Selecciona dificultad (1-{len(available)}): "))
                if 1 <= choice <= len(available):
                    difficulty = available[choice - 1]
                    break
                print("  Número fuera de rango")
            except ValueError:
                print(" Ingresa un número válido")
        
        self.show_existing_problems(difficulty)
        
        while True:
            try:
                cantidad = int(input(f"\n ¿Cuántos problemas generar? "))
                if cantidad > 0:
                    break
                print("  Debe ser mayor a 0")
            except ValueError:
                print(" Ingresa un número válido")
        
        print(f"\n{'='*70}")
        print(f"Generando {cantidad} problema(s) con TinyLlama...")
        print("="*70)
        
        try:
            problems = self.generate_problem(difficulty, cantidad)
            
            valid_count = 0
            total_score = 0
            
            for i, problem in enumerate(problems, 1):
                print(f"\n{'─'*70}")
                print(f"PROBLEMA {i}/{cantidad}")
                print("─"*70)
                
                self.save_as_bch(problem, difficulty)
                is_valid, score, status = self.analyze_quality(problem)
                total_score += score
                
                if is_valid:
                    valid_count += 1
                    print(f" {status} - Calidad: {score}/100")
                else:
                    print(f"  {status} - Calidad: {score}/100")
                
                # Preview
                print(f"\n PREVIEW:")
                for line in problem.split('\n')[:12]:
                    print(f"   {line}")
                if len(problem.split('\n')) > 12:
                    print("   ...")
            
            print("\n" + "="*70)
            print(" RESUMEN")
            print("="*70)
            print(f" Válidos: {valid_count}/{cantidad} ({valid_count/cantidad*100:.1f}%)")
            if cantidad > 0:
                print(f" Calidad promedio: {total_score/cantidad:.1f}/100")
            print(f" Ubicación: {os.path.join(self.base_output_dir, difficulty)}/")
            print("="*70)
            
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    try:
        generator = ProblemGenerator()
        
        if not generator.model_paths:
            print("\n No se encontraron modelos TinyLlama entrenados")
            print("   Ejecuta 'python fine_tune_tin

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import os
import torch

def load_optim_problems_by_difficulty():
    """Carga los problemas organizados por dificultad"""
    problems_by_difficulty = {
        "easy": [],
        "medium": [], 
        "hard": [],
        "unsolved": []
    }
    
    base_path = "/home/colossus/project_benchmarks/Dataset/kmeans"
    
    for difficulty in problems_by_difficulty.keys():
        difficulty_path = os.path.join(base_path, difficulty)
        if not os.path.exists(difficulty_path):
            print(f"Advertencia: No se encontro la carpeta {difficulty}")
            continue
            
        for filename in os.listdir(difficulty_path):
            if filename.endswith(".bch"):
                with open(os.path.join(difficulty_path, filename), "r", encoding="utf-8") as file:
                    content = file.read().strip()
                    if content:
                        problems_by_difficulty[difficulty].append(content)
        
        print(f"Cargados {len(problems_by_difficulty[difficulty])} problemas de {difficulty}")
    
    return problems_by_difficulty

def train_for_difficulty(difficulty, texts, model_name="microsoft/phi-1_5"):
    """Entrena un modelo específico para una dificultad"""
    print(f"\n=== Entrenando modelo para dificultad: {difficulty} ===")
    
    if len(texts) == 0:
        print(f"Saltando {difficulty}: No hay problemas")
        return None
    
    # Tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Dividir datos 70-30
    train_texts, val_texts = train_test_split(texts, test_size=0.3, random_state=42)
    print(f"Entrenamiento: {len(train_texts)}, Validacion: {len(val_texts)}")
    
    # Tokenizar
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding="max_length",
        max_length=128
    )
    train_encodings["labels"] = train_encodings["input_ids"].copy()
    
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding="max_length",
        max_length=128
    )
    val_encodings["labels"] = val_encodings["input_ids"].copy()
    
    # Crear datasets
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    
    # Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir=f"./models/fine_tuned_{difficulty}",
        per_device_train_batch_size=1,
        num_train_epochs=2,
        logging_dir=f"./logs/{difficulty}",
        save_total_limit=2,
        no_cuda=True,
        gradient_accumulation_steps=1,
        optim="adafactor",
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
    )

    # Entrenar
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    
    # Guardar modelo
    model.save_pretrained(f"./models/fine_tuned_{difficulty}")
    tokenizer.save_pretrained(f"./models/fine_tuned_{difficulty}")
    
    print(f"Modelo para {difficulty} guardado en: ./models/fine_tuned_{difficulty}")
    return f"./models/fine_tuned_{difficulty}"

def main():
    # Cargar problemas organizados por dificultad
    problems_by_difficulty = load_optim_problems_by_difficulty()
    
    # Entrenar modelos separados
    trained_models = {}
    for difficulty, texts in problems_by_difficulty.items():
        model_path = train_for_difficulty(difficulty, texts)
        if model_path:
            trained_models[difficulty] = model_path
    
    # Guardar metadata de modelos entrenados
    import json
    with open("./models/trained_models_metadata.json", "w") as f:
        json.dump(trained_models, f, indent=2)
    
    print(f"\nEntrenamiento completado. Modelos entrenados: {list(trained_models.keys())}")

if __name__ == "__main__":
    main()

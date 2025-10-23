from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import os
import torch

def load_optim_problems():
    """Carga SOLO los problemas de la carpeta optim/"""
    texts = []
    base_path = "/home/colossus/project_benchmarks/Dataset/dbscan"
    
    for difficulty in ["easy", "medium", "hard", "unsolved"]:
        difficulty_path = os.path.join(base_path, difficulty)
        if not os.path.exists(difficulty_path):
            continue
        for filename in os.listdir(difficulty_path):
            if filename.endswith(".bch"):
                with open(os.path.join(difficulty_path, filename), "r", encoding="utf-8") as file:
                    content = file.read().strip()
                    if content:
                        texts.append(content)
    
    print(f"Cargados {len(texts)} problemas de {base_path}")
    return texts

def main():
    # Modelo y tokenizer
    model_name = "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Cargar SOLO problemas optim/
    texts = load_optim_problems()
    
    if len(texts) == 0:
        print("ERROR: No se encontraron problemas en optim/")
        return
    
    # Dividir 90-10
    train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
    print(f"Entrenamiento: {len(train_texts)}, Validación: {len(val_texts)}")
    
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
    
    from datasets import Dataset
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    
    # Entrenamiento
    training_args = TrainingArguments(
        output_dir="./models/fine_tuned_model",
        per_device_train_batch_size=1,
        num_train_epochs=2,
        logging_dir="./logs",
        save_total_limit=2,
        no_cuda=True,
        gradient_accumulation_steps=1,
        optim="adafactor",
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("./models/fine_tuned_model")
    tokenizer.save_pretrained("./models/fine_tuned_model")
    
    print("Entrenamiento completado usando SOLO problemas de optim/")

if __name__ == "__main__":
    main()

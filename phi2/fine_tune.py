from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import os
import torch

def load_dataset_from_folder(folder_path):
    texts = []
    for difficulty in ["easy", "medium", "hard"]:
        difficulty_path = os.path.join(folder_path, difficulty)
        if not os.path.exists(difficulty_path):
            continue
        for filename in os.listdir(difficulty_path):
            if filename.endswith(".bch"):
                with open(os.path.join(difficulty_path, filename), "r", encoding="utf-8") as file:
                    texts.append(file.read())
    return texts

def prepare_dataset(texts, tokenizer):
    # DIVIDIR en entrenamiento y validación (90-10)
    train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
    
    print(f"Ejemplos de entrenamiento: {len(train_texts)}")
    print(f"Ejemplos de validación: {len(val_texts)}")
    
    # Tokenizar entrenamiento
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    train_encodings["labels"] = train_encodings["input_ids"].copy()
    
    # Tokenizar validación
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    val_encodings["labels"] = val_encodings["input_ids"].copy()
    
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    
    return train_dataset, val_dataset

def main():
    model_name = "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)

    base_path = "/home/colossus/ibex-lib-master/benchs/optim"
    texts = load_dataset_from_folder(base_path)
    
    train_dataset, val_dataset = prepare_dataset(texts, tokenizer)

    # CONFIGURACIÓN CORREGIDA - sin evaluation_strategy
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
        
        # EVALUACIÓN CORREGIDA:
        eval_strategy="epoch",  # ← CAMBIADO de evaluation_strategy
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Dataset de validación
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("./models/fine_tuned_model")
    tokenizer.save_pretrained("./models/fine_tuned_model")
    
    print("Entrenamiento completado con división 90-10")

if __name__ == "__main__":
    main()

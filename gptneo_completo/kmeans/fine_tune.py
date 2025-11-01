from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import os
import torch

def load_optim_problems():
    """Carga SOLO los problemas de la carpeta optim/"""
    texts = []
    base_path = "/home/colossus/project_benchmarks/Dataset/kmeans"
    
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
    # ✅ USAR GPT-Neo 1.3B - mismo tamaño que Phi-1.5
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT-Neo necesita pad_token explícito
    tokenizer.pad_token = tokenizer.eos_token
    
    # Cargar para CPU (sin problemas de memoria)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    
    # Forzar CPU
    device = torch.device("cpu")
    model.to(device)
    print(f"Modelo cargado en: {device}")
    print(f"Usando: GPT-Neo 1.3B (mismo tamaño que Phi-1.5)")
    
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
        max_length=128,
        return_tensors="pt"
    )
    train_encodings["labels"] = train_encodings["input_ids"].clone()
    
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    val_encodings["labels"] = val_encodings["input_ids"].clone()
    
    from datasets import Dataset
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    
    # Configuración para CPU
    training_args = TrainingArguments(
        output_dir="./models/fine_tuned_gpt_neo",
        per_device_train_batch_size=1,
        num_train_epochs=2,
        logging_dir="./logs",
        save_total_limit=1,
        learning_rate=3e-5,
        gradient_accumulation_steps=16,
        optim="adafactor",
        report_to="none",
        eval_strategy="no",
        save_strategy="epoch",
        fp16=False,
        no_cuda=True,  # Forzar CPU
        dataloader_pin_memory=False,
        logging_steps=5,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    
    # Guardar modelo
    model.save_pretrained("./models/fine_tuned_gpt_neo")
    tokenizer.save_pretrained("./models/fine_tuned_gpt_neo")
    
    print("✅ ¡Entrenamiento completado!")
    print("📁 Modelo guardado en: ./models/fine_tuned_gpt_neo/")

if __name__ == "__main__":
    main()

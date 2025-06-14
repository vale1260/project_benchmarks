from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
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
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
    encodings["labels"] = encodings["input_ids"].copy()
    return Dataset.from_dict(encodings)

def main():
    model_name = "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    base_path = "/home/colossus/ibex-lib-master/benchs/optim"
    texts = load_dataset_from_folder(base_path)
    dataset = prepare_dataset(texts, tokenizer)

    training_args = TrainingArguments(
        output_dir="./models/fine_tuned_model",
        per_device_train_batch_size=1,          # Reducido al mínimo
        num_train_epochs=2,
        logging_dir="./logs",
        save_total_limit=2,
        no_cuda=True,                           # Fuerza CPU
        gradient_accumulation_steps=1,
        optim="adafactor",                      # Optimización más liviana
        report_to="none"                        # Evita errores si no usas wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("./models/fine_tuned_model")
    tokenizer.save_pretrained("./models/fine_tuned_model")

if __name__ == "__main__":
    main()


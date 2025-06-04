from datasets import Dataset
from transformers import AutoTokenizer
import json
from pathlib import Path

# Chequear si el modelo fine-tuneado existe
local_model_path = Path("./models/fine_tuned_model")
config_path = local_model_path / "config.json"

if config_path.exists():
    print("Usando modelo fine-tuneado local.")
    MODEL_NAME = str(local_model_path)
else:
    print("Modelo fine-tuneado no encontrado, usando modelo base Mistral.")
    MODEL_NAME = "microsoft/phi-1_5"

DATASET_PATH = Path("datasets/dataset.jsonl")
OUTPUT_PATH = Path("datasets/tokenized_dataset")

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding=False, truncation=False)

def main():
    print("Cargando dataset...")
    data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    dataset = Dataset.from_list(data)

    print(f"Cargando tokenizer desde {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizando...")
    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=False)

    print(f"Guardando en: {OUTPUT_PATH}")
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    tokenized.save_to_disk(OUTPUT_PATH)

if __name__ == "__main__":
    main()

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import hashlib

DATA_PATH = Path("/home/colossus/ibex-lib-master/benchs/optim")
GENERATED_PATH = Path("generated_problems")
GENERATED_PATH.mkdir(exist_ok=True, parents=True)

CACHE_PATH = Path("data/errors_cache.jsonl")
CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)

MODEL_PATH = "./models/fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True
)

def load_cache():
    cache = []
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                cache.append(json.loads(line))
    return cache

def save_cache(cache_data):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for item in cache_data:
            f.write(json.dumps(item) + "\n")
    print("Cache guardada.")

def generate_new_problem(prompt):
    cache = load_cache()
    for entry in cache:
        if entry.get("prompt") == prompt:
            print("Usando cache.")
            return entry["generated_text"]

    output = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )[0]["generated_text"]

    generated = output[len(prompt):].strip()
    cache.append({"prompt": prompt, "generated_text": generated})
    save_cache(cache)

    return generated

if __name__ == "__main__":
    prompt = input("Ingresa prompt: ")
    result = generate_new_problem(prompt)
    print(result)

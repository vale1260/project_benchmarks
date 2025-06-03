import json
from pathlib import Path

ORIGINAL_PATH = Path("/home/colossus/ibex-lib-master/benchs/optim")
GENERATED_PATH = Path("generated_problems")
OUTPUT_PATH = Path("datasets/dataset.jsonl")

def collect_problems():
    data = []
    for difficulty in ["easy", "medium", "hard"]:
        for folder in [ORIGINAL_PATH / difficulty, GENERATED_PATH / difficulty]:
            if folder.exists():
                for file in folder.glob("*.bch"):
                    with open(file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            data.append({"text": content, "difficulty": difficulty})
    return data

def main():
    data = collect_problems()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Dataset JSONL guardado en {OUTPUT_PATH} con {len(data)} problemas")

if __name__ == "__main__":
    main()

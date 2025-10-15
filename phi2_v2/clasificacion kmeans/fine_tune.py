import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import os

class FilteredMathProblemDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]['text']
        difficulty = self.examples[idx]['difficulty']
        
        formatted_text = f"### Dificultad: {difficulty}\n### Problema:\n{text}\n### FIN"
        
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def train_models(jsonl_file):
    print("Modelo: microsoft/phi-1_5")
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5', trust_remote_code=True)
    
    # Configurar padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Cargar ejemplos
    all_examples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_examples.append(json.loads(line))
    
    print(f"Total de ejemplos cargados: {len(all_examples)}")
    
    difficulties = ['easy', 'medium', 'hard', 'unsolved']
    
    for difficulty in difficulties:
        print(f"\n{'='*60}")
        print(f"Entrenando modelo para dificultad: {difficulty}")
        print(f"{'='*60}")
        
        filtered_examples = [ex for ex in all_examples if ex['difficulty'] == difficulty]
        
        if len(filtered_examples) == 0:
            print(f"No hay ejemplos para: {difficulty}")
            continue
        
        print(f"Ejemplos: {len(filtered_examples)}")
        
        if len(filtered_examples) < 2:
            print(f"Insuficientes datos")
            continue
        
        filtered_dataset = FilteredMathProblemDataset(filtered_examples, tokenizer, max_length=1024)
        
        train_size = int(0.85 * len(filtered_dataset))
        val_size = len(filtered_dataset) - train_size
        
        if val_size == 0:
            val_size = 1
            train_size = len(filtered_dataset) - 1
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            filtered_dataset, [train_size, val_size]
        )
        
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        
        # Cargar modelo
        print(f"Cargando modelo base...")
        model = AutoModelForCausalLM.from_pretrained(
            'microsoft/phi-1_5',
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        model.resize_token_embeddings(len(tokenizer))
        
        training_args = TrainingArguments(
            output_dir=f'./models/phi-{difficulty}',
            overwrite_output_dir=True,
            num_train_epochs=10,
            per_device_train_batch_size=1,  # Phi necesita batch pequeño
            per_device_eval_batch_size=1,
            learning_rate=2e-5,  # LR más bajo para Phi
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=f'./logs/phi-{difficulty}',
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=8,  # Simular batch más grande
            gradient_checkpointing=True,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print(f"Iniciando entrenamiento...")
        trainer.train()
        
        trainer.save_model()
        tokenizer.save_pretrained(f'./models/phi-{difficulty}')
        
        print(f"Modelo guardado")
        
        metrics = trainer.evaluate()
        print(f"Loss final: {metrics['eval_loss']:.4f}")
        
        # Liberar memoria
        del model
        del trainer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    train_models('dataset_nuevo_kmeans.jsonl')

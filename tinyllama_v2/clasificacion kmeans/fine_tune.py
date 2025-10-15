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
        
        # Formato consistente para entrenamiento
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
    print("="*70)
    print("FINE-TUNING CON TINYLLAMA-1.1B")
    print("="*70)
    
    # Cargar tokenizer de TinyLlama
    print("\n Cargando tokenizer de TinyLlama...")
    tokenizer = AutoTokenizer.from_pretrained(
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        trust_remote_code=True
    )
    
    # Configurar padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print(f"Tokenizer cargado")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Pad token: {tokenizer.pad_token}")
    
    # Cargar todos los ejemplos
    all_examples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_examples.append(json.loads(line))
    
    print(f"\n Total de ejemplos cargados: {len(all_examples)}")
    
    difficulties = ['easy', 'medium', 'hard', 'unsolved']
    
    for difficulty in difficulties:
        print(f"\n{'='*70}")
        print(f" Entrenando modelo para dificultad: {difficulty.upper()}")
        print(f"{'='*70}")
        
        # Filtrar ejemplos por dificultad
        filtered_examples = [ex for ex in all_examples if ex['difficulty'] == difficulty]
        
        if len(filtered_examples) == 0:
            print(f" No hay ejemplos para la dificultad: {difficulty}")
            continue
        
        print(f"✓ Ejemplos encontrados: {len(filtered_examples)}")
        
        if len(filtered_examples) < 10:
            print(f"  ADVERTENCIA: Pocos ejemplos ({len(filtered_examples)})")
            print(f"   Se recomienda al menos 50-100 ejemplos para buen rendimiento")
        
        if len(filtered_examples) < 2:
            print(f" Insuficientes datos para dividir en train/val")
            continue
        
        # Crear dataset
        filtered_dataset = FilteredMathProblemDataset(
            filtered_examples, 
            tokenizer, 
            max_length=1024
        )
        
        # División train/validation
        train_size = int(0.85 * len(filtered_dataset))
        val_size = len(filtered_dataset) - train_size
        
        if val_size == 0:
            val_size = 1
            train_size = len(filtered_dataset) - 1
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            filtered_dataset, [train_size, val_size]
        )
        
        print(f" Train: {len(train_dataset)} | Validación: {len(val_dataset)}")
        
        # Cargar modelo base TinyLlama
        print(f"\n Cargando modelo base TinyLlama-1.1B...")
        model = AutoModelForCausalLM.from_pretrained(
            'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Redimensionar embeddings si es necesario
        model.resize_token_embeddings(len(tokenizer))
        print(f" Modelo cargado")
        
        # Configuración de entrenamiento optimizada para TinyLlama
        training_args = TrainingArguments(
            output_dir=f'./models/tinyllama-{difficulty}',
            overwrite_output_dir=True,
            num_train_epochs=10,
            per_device_train_batch_size=1,  # Batch pequeño por memoria
            per_device_eval_batch_size=1,
            learning_rate=2e-5,  # LR óptimo para TinyLlama
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=f'./logs/tinyllama-{difficulty}',
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # Precisión mixta si hay GPU
            gradient_accumulation_steps=8,  # Simular batch más grande
            gradient_checkpointing=True,  # Ahorra memoria
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
        
        print(f"\n Iniciando entrenamiento...")
        print(f"   Épocas: {training_args.num_train_epochs}")
        print(f"   Learning rate: {training_args.learning_rate}")
        print(f"   Batch size efectivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        
        trainer.train()
        
        # Guardar modelo y tokenizer
        print(f"\n Guardando modelo...")
        trainer.save_model()
        tokenizer.save_pretrained(f'./models/tinyllama-{difficulty}')
        
        print(f" Modelo para {difficulty} guardado en: ./models/tinyllama-{difficulty}/")
        
        # Mostrar métricas finales
        metrics = trainer.evaluate()
        print(f"\n Métricas finales:")
        print(f"   Loss de validación: {metrics['eval_loss']:.4f}")
        
        # Liberar memoria
        print(f"  Liberando memoria...")
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f" Listo para siguiente dificultad")

if __name__ == "__main__":
    print("\n INICIANDO FINE-TUNING DE TINYLLAMA")
    print("="*70)
    
    # Crear directorios necesarios
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    dataset_file = 'dataset_nuevo_kmeans.jsonl'
    
    if not os.path.exists(dataset_file):
        print(f" ERROR: No se encuentra el archivo {dataset_file}")
        print("   Por favor, verifica que el archivo existe en el directorio actual.")
    else:
        print(f" Usando dataset: {dataset_file}\n")
        train_models(dataset_file)
        
        print("\n" + "="*70)
        print(" FINE-TUNING COMPLETADO")
        print("="*70)
        print("Los modelos se guardaron en:")
        print("  - ./models/tinyllama-easy/")
        print("  - ./models/tinyllama-medium/")
        print("  - ./models/tinyllama-hard/")
        print("  - ./models/tinyllama-unsolved/")
        print("="*70)


# LLM Benchmarks Package 

Este paquete contiene las carpetas y scripts listos para trabajar con modelos **Mistral**, **LLaMA2** y **DeepSeek** para:

* Fine-tuning
* Generación de nuevos problemas
* Preparación y tokenización de datasets
* Uso de sistema de cache para prompts generados

---

## Estructura del paquete

```
/mistral/
/llama2/
/deepseek/
    ├── data/
    │   ├── easy/
    │   ├── medium/
    │   ├── hard/
    │   └── errors_cache.jsonl
    ├── models/
    │   └── fine_tuned_model/
    ├── datasets/
    │   ├── dataset.jsonl
    │   └── tokenized_dataset/
    ├── fine_tune.py
    ├── generate_text.py
    ├── prepare_dataset.py
    └── tokenize_dataset.py
```

---

## Instrucciones de uso

### 1 Instala las dependencias

Ejecuta:
```bash
pip install -r requirements.txt
```

---

### 2 Crear el dataset combinado (JSONL)

Ejecuta:
```bash
python prepare_dataset.py
```
Esto generará `datasets/dataset.jsonl` combinando problemas originales y generados.

---

### 3 Tokenizar el dataset

Ejecuta:
```bash
python tokenize_dataset.py
```
Esto creará `datasets/tokenized_dataset/` usando el tokenizer del modelo correspondiente.

---

### 4 Fine-tuning del modelo

Ejecuta:
```bash
python fine_tune.py
```
Esto entrenará el modelo y guardará el resultado en `models/fine_tuned_model/`.

---

### 5 Generar nuevos problemas

Ejecuta:
```bash
python generate_text.py
```
Ingresa un prompt cuando se te pida.
El sistema verificará el cache (`errors_cache.jsonl`) y si es nuevo, generará un problema y lo guardará en `generated_problems/`.

---

## Notas importantes

- Cada script está configurado para trabajar con rutas locales (`./data`), así evitas problemas con rutas absolutas.
- El uso de `.jsonl` permite trabajar con datasets grandes de forma eficiente.
- Los modelos cargados son:
    - Mistral → `mistralai/Mistral-7B-v0.1`
    - LLaMA2 → `meta-llama/Llama-2-7b-hf`
    - DeepSeek → `deepseek-ai/deepseek-llm-7b-base`

---

## Automatización con script bash

El script `run_all.sh` se ejecutará por carpeta de modelo (por ejemplo, solo `/mistral/`), y correrá en orden:
1. `prepare_dataset.py`
2. `tokenize_dataset.py`
3. `fine_tune.py`
4. `generate_text.py` (con un prompt predefinido o pasado como argumento)

Esto permite automatizar todo el flujo para cada modelo de forma aislada.

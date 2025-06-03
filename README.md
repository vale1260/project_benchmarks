
# LLM Benchmarks Package 

Este paquete contiene las carpetas y scripts listos para trabajar con modelos **Tinyllama**, **Phi2** y **DeepSeek** para:
* Fine-tuning
* Generación de nuevos problemas
* Preparación y tokenización de datasets
* Uso de sistema de cache para prompts generados

---

## Estructura del paquete

```
/tinyllama/
/phi2/
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

### 2 Prepara los datos

Copia tus archivos `.bch` dentro de:
```
/mistral/data/easy/
/llama2/data/easy/
/deepseek/data/easy/
```
y lo mismo para `medium/` y `hard/`.
Cada modelo trabajará con su propio conjunto aislado.

---

### 3 Crear el dataset combinado (JSONL)

Ejecuta:
```bash
python3 prepare_dataset.py
```
Esto generará `datasets/dataset.jsonl` combinando problemas originales y generados.

---

### 4 Tokenizar el dataset

Ejecuta:
```bash
python3 tokenize_dataset.py
```
Esto creará `datasets/tokenized_dataset/` usando el tokenizer del modelo correspondiente.

---

### 5 Fine-tuning del modelo

Ejecuta:
```bash
python3 fine_tune.py
```
Esto entrenará el modelo y guardará el resultado en `models/fine_tuned_model/`.

---

### 6 Generar nuevos problemas

Ejecuta:
```bash
python3 generate_text.py
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

El script `run_all.sh` (que te puedo preparar) se ejecutará por carpeta de modelo (por ejemplo, solo `/mistral/`), y correrá en orden:
1. `prepare_dataset.py`
2. `tokenize_dataset.py`
3. `fine_tune.py`
4. `generate_text.py` (con un prompt predefinido o pasado como argumento)

Esto permite automatizar todo el flujo para cada modelo de forma aislada.

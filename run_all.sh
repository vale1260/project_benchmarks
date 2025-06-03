#!/bin/bash

# Script para correr todo el pipeline en una carpeta de modelo
# Uso: ./run_all.sh [modelo] [prompt]
# Ejemplo: ./run_all.sh mistral "Genera un problema de ejemplo"

MODEL_DIR=$1
PROMPT=$2

if [ -z "$MODEL_DIR" ]; then
    echo "❌ Debes especificar la carpeta del modelo (mistral, llama2, deepseek)"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ La carpeta $MODEL_DIR no existe"
    exit 1
fi

cd $MODEL_DIR

echo "🚀 Ejecutando prepare_dataset.py..."
python prepare_dataset.py

echo "🚀 Ejecutando tokenize_dataset.py..."
python tokenize_dataset.py

echo "🚀 Ejecutando fine_tune.py..."
python fine_tune.py

if [ -z "$PROMPT" ]; then
    echo "ℹ️ No se pasó prompt, saltando generación de texto."
else
    echo "🚀 Ejecutando generate_text.py con prompt: $PROMPT"
    echo $PROMPT | python generate_text.py
fi

echo "✅ Proceso completado en carpeta $MODEL_DIR"

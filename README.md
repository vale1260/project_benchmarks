
# LLM Benchmarks Package 

Este paquete contiene las carpetas y scripts listos para trabajar con modelos **Tinyllama** y **Phi1.5** para:
* Fine-tuning
* Generación de nuevos problemas

---

## Estructura del paquete

```
/tinyllama/
/phi1.5/
    ├── fine_tune.py
    ├── generate_text.py
```

---

## Instrucciones de uso

### 1 Instala las dependencias

Ejecuta:
```bash
pip install -r requirements.txt
```

---

### 2 Fine-tuning del modelo

Ejecuta:
```bash
python3 fine_tune.py
```
Esto entrenará el modelo y guardará el resultado.

---

### 3 Generar nuevos problemas

Ejecuta:
```bash
python3 generate_text.py
```
Ingresa la dificultad cuando se te pida.
El sistema verificará el cache y si es nuevo, generará un problema y lo guardará.

---

## Notas importantes

- Cada script está configurado para trabajar con rutas locales (`./data`), así evitas problemas con rutas absolutas.
- Los modelos cargados son:
    - Phi2 → `microsoft/phi-1_5`
    - TinyLlama → `TinyLlama/TinyLlama-1.1B-Chat-v1.0`


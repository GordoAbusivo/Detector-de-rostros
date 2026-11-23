# Detector de rostros — ESCOM

Reconocimiento facial de 29 personas del grupo. El repo compara **tres enfoques sobre exactamente las mismas fotos de prueba**:

| | Enfoque | Qué es |
|---|---|---|
| A | **ArcFace + Ridge / SVM / KNN** | Una CNN de reconocimiento facial ya entrenada (ArcFace `antelopev2`, ResNet-100) convierte cada cara en un vector de 512 números; un clasificador sencillo decide la persona. |
| B | **CNN ResNet con fine-tuning** (`--modo finetune`) | ResNet18/50 preentrenada en ImageNet y reentrenada con nuestras fotos. |
| C | **CNN pequeña desde cero** (`--modo scratch`) | Línea base: muestra cuánto ayuda el preentrenamiento. |

> Ojo: **A también es una red neuronal convolucional**. La diferencia es que ArcFace se entrenó con millones de rostros
> para separar identidades; B y C aprenden solo con nuestras fotos. Por eso se comparan, no se asume cuál gana.

## Datos

Fotos en Hugging Face: <https://huggingface.co/escom-identificador-rostros>. **Ni fotos, ni vectores, ni modelos entrenados
(`.joblib`, `.pt`) se suben a este repo** (datos biométricos; ver `.gitignore`).

## Probar el modelo ya entrenado (Docker)

Requiere el archivo privado `ridge_antelope.joblib` (te lo pasa el equipo).

```bash
mkdir -p models && cp /ruta/a/ridge_antelope.joblib models/
export HOST_UID=$(id -u) HOST_GID=$(id -g)     # Linux/Mac (en Windows omitir)
docker compose up --build                      # abrir http://localhost:7860
```

La primera vez descarga ~260 MB de modelos (necesita internet). Confianza: alta ≥ 0.60, media 0.50–0.60, baja < 0.50.
Siempre responde una de las 29 personas (no existe "desconocido"). Si falla algo: puerto ocupado → `PUERTO=8000 docker compose up`;
"no configuration file provided" → la terminal no está dentro de la carpeta del repo.

## Reproducir la comparación

Entorno: `python 3.12`, `pip install -r requirements.txt` (y `-r requirements-cnn.txt` para la CNN; con GPU, por ejemplo Colab).

```bash
# 0. Pipeline de datos (ya existente): limpiar, recortar, alinear y extraer vectores
python src/extraer_vectores.py          # vectors/antelope.npy + metadatos.csv
python src/dividir_sets.py              # vectors/division_sets.csv  (60/20/20 por SETS completos)

# A. ArcFace + clasificadores (se reentrenan SOLO con train; el .joblib final usó todas las fotos)
python src/evaluar_embeddings.py

# B / C. CNN, misma partición
python src/preparar_particiones.py --val-de-train
python src/entrenar_cnn.py --modo finetune --arq resnet50 --epocas 15
python src/entrenar_cnn.py --modo finetune --arq resnet18 --epocas 15
python src/entrenar_cnn.py --modo scratch --epocas 60 --tam 112

# Tabla final
python src/comparar_modelos.py          # resultados/comparacion.md
```

La división es **por sets** (una sesión de fotos nunca queda repartida entre train y prueba); así se evita que el modelo
"reconozca la ropa y el fondo" en vez de la cara. Cada modelo guarda `resultados/<modelo>/metricas.json`
(accuracy, F1 macro, F1 ponderado) y, solo en local, matriz de confusión, métricas por persona y lista de errores.

## Resultados

_Pendiente: correr los comandos de arriba y pegar aquí `resultados/comparacion.md`._

## Estructura

```
src/                  scripts del pipeline, predicción, front (app.py) y comparación
  entrenar_cnn.py     CNN (B y C)
  evaluar_embeddings.py, preparar_particiones.py, comparar_modelos.py
Dockerfile, docker-compose.yml   probar el modelo en CPU
requirements.txt, requirements-cnn.txt
```

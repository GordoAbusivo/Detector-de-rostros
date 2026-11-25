# Detector de rostros — ESCOM

Reconocimiento facial de 29 personas del grupo. El repo compara **cuatro enfoques** con la misma regla de división de datos:

| | Enfoque | Qué es |
|---|---|---|
| A | **ArcFace + Ridge / SVM / KNN** | Una CNN de reconocimiento facial ya entrenada (ArcFace `antelopev2`, ResNet-100) convierte cada cara en un vector de 512 números; un clasificador sencillo decide la persona. |
| B | **CNN ResNet con fine-tuning, PyTorch** (`--modo finetune`) | ResNet18/50 preentrenada en ImageNet y reentrenada con nuestras fotos. |
| C | **CNN pequeña desde cero, PyTorch** (`--modo scratch`) | Línea base: muestra cuánto ayuda el preentrenamiento. |
| D | **CNN en TensorFlow / Keras sobre la cara recortada** (Kaggle) | Detecta y recorta la cara con MediaPipe (o Haar), y entrena una CNN propia o una preentrenada (MobileNetV2, ResNet50V2, EfficientNetB0). |

> Ojo: **A también es una red neuronal convolucional**. La diferencia es que ArcFace se entrenó con millones de rostros
> para separar identidades; B, C y D aprenden solo con nuestras fotos. Por eso se comparan, no se asume cuál gana.

## Datos

Fotos en Hugging Face: <https://huggingface.co/escom-identificador-rostros> (dataset privado
`escom-identificador-rostros/identificador-rostros-escom`). **Ni fotos, ni caras recortadas, ni vectores, ni modelos entrenados
(`.joblib`, `.pt`, `.keras`) se suben a este repo** (datos biométricos; ver `.gitignore`).

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

## Enfoque D: CNN en Keras sobre la cara (Kaggle)

Notebook listo para correr: [`kaggle/entrenar_cnn_keras_kaggle.ipynb`](kaggle/entrenar_cnn_keras_kaggle.ipynb).

### Por qué recortar la cara

Las fotos del dataset son de **cuerpo entero o 3/4 de cuerpo**: la cara ocupa una fracción pequeña. Si la red recibe la foto completa, aprende
ropa y fondo (que cambian de un set a otro) en lugar de rasgos faciales. Por eso el flujo tiene dos etapas: **(1)** un detector clásico localiza y
recorta la cara, **(2)** la CNN solo clasifica ese recorte. La red nunca tiene que "buscar" la cara, solo reconocerla.

### Decisiones

| Duda | Elección | Motivo |
|---|---|---|
| MediaPipe u OpenCV | **MediaPipe** (BlazeFace *full range*) primero, **Haar de OpenCV** de respaldo (`--detector auto`) | Haar solo ve caras casi de frente y suele fallar con caras chicas o giradas, justo el caso de fotos de cuerpo entero. MediaPipe (modelo *full range*, pensado para caras pequeñas o lejanas) ya se usa en el resto del repo (`recortar.py`). Haar no descarga nada y sirve si MediaPipe no se instala. |
| PyTorch o Keras | **Keras** para este enfoque; PyTorch (B y C) se mantiene | Para un modelo de clasificación de este tamaño ambos rinden igual. Keras es más corto de escribir, trae EarlyStopping y guardado listos, y los modelos preentrenados se descargan en una línea. Se dejan los dos para poder comparar. |
| Qué tamaño de cara | 224 px guardados; la red entra a 128 (scratch) o 224 (preentrenada) | Se guarda grande una sola vez y cada modelo la reduce según necesite. |
| Cómo se divide | **Por sets** (60/20/20 por persona, `dividir_sets.py`) | Una sesión (misma ropa y fondo) nunca queda repartida entre train y test; si no, el accuracy sale inflado. |

### Pasos en Kaggle

1. Crea un notebook nuevo y **File → Import Notebook** con `kaggle/entrenar_cnn_keras_kaggle.ipynb`.
2. En *Session options*: **Accelerator = GPU**, **Internet = On**.
3. En *Add-ons → Secrets*: crea `HF_TOKEN` (token de Hugging Face con **lectura** al dataset) y actívalo para el notebook.
4. *Run All*. El notebook clona este repo, recorta las caras (reanudable), entrena y muestra curvas y métricas.
5. Mantén el notebook **privado** (contiene caras y nombres en sus salidas).

Los mismos pasos desde terminal (con GPU, o en CPU para pruebas pequeñas), con `pip install -r requirements-keras.txt`:

```bash
export HF_TOKEN=hf_xxx                                    # solo si bajas de Hugging Face
python src/kaggle_preparar_caras.py                       # data/caras/ + data/caras_manifest.csv
python src/entrenar_cnn_keras.py --modo scratch --tam 128 --epocas 60
python src/entrenar_cnn_keras.py --modo finetune --arq mobilenetv2 --tam 224 --epocas 25
python src/predecir_keras.py foto.jpg --modelo models/keras_mobilenetv2.keras --panel paneles
python src/comparar_modelos.py                            # resultados/comparacion.md
```

`kaggle_preparar_caras.py --origen carpeta/` recorta fotos locales con estructura `persona/set/foto.jpg` en vez de bajarlas de Hugging Face.
Para probar rápido: `--personas a b --max-por-persona 60`.

### Qué hace cada script nuevo

| Script | Función |
|---|---|
| `src/caras.py` | Detector de caras (MediaPipe / Haar / auto) y recorte cuadrado con margen. Lo comparten entrenamiento y predicción. |
| `src/kaggle_preparar_caras.py` | Baja el dataset persona por persona, recorta, borra las originales y escribe el manifiesto. Reanudable. |
| `src/entrenar_cnn_keras.py` | Entrena y evalúa (scratch o fine-tuning). Normalización y aumento de datos van **dentro** del modelo. |
| `src/predecir_keras.py` | Foto nueva → detecta la cara → top-3 personas con confianza (y panel opcional entrada / cara / predicción). |

### Notas de diseño

- **Pixeles 0-255 de entrada**: la normalización está dentro del modelo, así que al predecir basta con pasarle la cara recortada.
- **Aumento de datos suave** (volteo horizontal, giro ±5%, zoom, traslación, brillo y contraste). Sin volteo vertical ni giros grandes: nadie sale de cabeza.
- **Pooling global en vez de `Flatten` + `Dense` enorme**: la CNN desde cero queda en ≈1 M de parámetros (una capa densa sobre el mapa aplanado puede pasar de 16 M de parámetros y tiende a memorizar).
- **Pesos por clase y *label smoothing* 0.1** para compensar que unas personas tengan más fotos.
- **Fine-tuning en dos fases**: primero solo la cabeza nueva con la base congelada (BatchNorm en modo inferencia), luego toda la red con *learning rate* 10 veces menor.
- **Si hay varias caras en la foto se toma la más grande** (la persona retratada; las del fondo son más chicas). Las fotos sin cara válida (de espalda, cara menor a `--min-cara` px) se omiten y quedan listadas en `data/caras/_manifest/*_omitidas.csv`.
- Revisa siempre la muestra de recortes que muestra el notebook antes de entrenar: un detector que se equivoca de cara contamina el entrenamiento.

### Alcance de la comparación

La regla de división es la misma que en A–C (`dividir_sets.asignar_split`), pero el enfoque D parte de las fotos de Hugging Face en las que se detectó una cara, así
que el tamaño de cada set —y con él la elección de los sets de prueba— puede diferir un poco del flujo local. Compara los resultados como **misma regla de
división**, no como **exactamente las mismas fotos de prueba**; para lo segundo, usa el flujo local de la sección siguiente para A–C.

## Reproducir A, B y C (flujo local)

Entorno: `python 3.12`, `pip install -r requirements.txt` (y `-r requirements-cnn.txt` para la CNN de PyTorch; con GPU, por ejemplo Colab).

```bash
# 0. Pipeline de datos (ya existente): limpiar, recortar, alinear y extraer vectores
python src/extraer_vectores.py          # vectors/antelope.npy + metadatos.csv
python src/dividir_sets.py              # vectors/division_sets.csv  (60/20/20 por SETS completos)

# A. ArcFace + clasificadores (se reentrenan SOLO con train; el .joblib final usó todas las fotos)
python src/evaluar_embeddings.py

# B / C. CNN de PyTorch, misma partición
python src/preparar_particiones.py --val-de-train
python src/entrenar_cnn.py --modo finetune --arq resnet50 --epocas 15
python src/entrenar_cnn.py --modo finetune --arq resnet18 --epocas 15
python src/entrenar_cnn.py --modo scratch --epocas 60 --tam 112

# Tabla final (junta también los resultados de Keras)
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
  entrenar_cnn.py     CNN de PyTorch (B y C)
  caras.py, kaggle_preparar_caras.py, entrenar_cnn_keras.py, predecir_keras.py     CNN de Keras (D)
  evaluar_embeddings.py, preparar_particiones.py, comparar_modelos.py
kaggle/               notebook para entrenar el enfoque D en Kaggle
Dockerfile, docker-compose.yml   probar el modelo en CPU
requirements.txt, requirements-cnn.txt, requirements-keras.txt
```

## Privacidad

Las fotos son datos biométricos de compañeros. Dataset y notebooks de Kaggle **privados**, sin subir fotos, recortes, vectores ni modelos al repo
(`.gitignore` ya los excluye), y borrar todo al terminar el curso. Si subes un notebook al repo, limpia antes sus salidas.

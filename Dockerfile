# Imagen para PROBAR el modelo ya entrenado con el front (src/app.py). No sirve para entrenar ni repetir el pipeline
# (eso se hace en un entorno propio, ver README).
# Lleva SOLO codigo y librerias: el modelo (models/*_antelope.joblib) NO va dentro, se monta desde tu equipo
# (ver docker-compose.yml y el README).
FROM python:3.12-slim

# libgl1, libglib2.0-0 (OpenCV) y libegl1 + libgles2 (mediapipe): sin ellos mediapipe no arranca en la imagen minima
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HOME=/tmp

# Mismo requirements.txt del proyecto, pero para predecir en CPU: sin librerias CUDA, con onnxruntime a secas
# y sin lo que solo hacen falta para entrenar (descargar el dataset, notebooks).
COPY requirements.txt .
RUN grep -v -e '^nvidia-' -e '^ipykernel' -e '^huggingface_hub' requirements.txt \
      | sed 's/^onnxruntime-gpu==/onnxruntime==/' > req.txt \
 && pip install -r req.txt && rm req.txt

COPY src ./src

ENTRYPOINT ["python"]
CMD ["src/app.py", "--host", "0.0.0.0", "--sin-navegador"]

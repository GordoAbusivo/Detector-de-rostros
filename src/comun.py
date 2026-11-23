"""
Utilidades comunes del proyecto. Antes estaban repetidas en cada script (EXTENSIONES, soporte .heic,
cargar_rgb, descarga de modelos, borde negro, ...); ahora viven aqui y los demas scripts las importan:

    from comun import RAIZ, EXTENSIONES, cargar_rgb, listar_imagenes

Contiene:
  - Rutas estandar del proyecto (RAIZ, DATASETS, CLEAN, FACE, MUESTRA, ARCFACE, REPORTS, VECTORS, MODELOS, PARTICIONES, RESULTADOS, ...).
  - guardar_resultados(...): guarda las metricas de un modelo en resultados/<nombre>/ (mismo formato para todos).
  - EXTENSIONES de imagen validas, UMBRAL_TAMANO_ROSTRO y listar_imagenes(carpeta).
  - cargar_rgb(ruta): lee una imagen respetando la orientacion EXIF (fotos de celular giradas) y .heic.
  - asegurar_modelo(ruta, url): descarga un modelo de MediaPipe si no existe; asegurar_arcface(): los ArcFace ONNX.
  - con_borde(img): rellena con borde negro del 50% (los recortes ajustados fallan en el Landmarker).
  - preprocesar_cara(cara_rgb): normalizacion de pixeles de ArcFace, (x-127.5)/127.5, formato (1,3,112,112).
  - detector_rostros(...) y landmarker(...): crean los detectores de MediaPipe (mediapipe se importa solo al usarlos).

Los scripts se ejecutan como `python src/algo.py`, asi que src/ esta en sys.path y `import comun` funciona.
"""
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # soporte para .heic / .HEIC (se registra al importar este modulo)
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("AVISO: pillow-heif no instalado; los .heic se reportaran como 'no se pudo leer'.")

# ---------------------------------------------------------------- rutas (sin importar desde donde se corra)
RAIZ = Path(__file__).resolve().parent.parent          # carpeta del proyecto
DATASETS = RAIZ / "datasets"                           # fotos ORIGINALES (nunca se modifican)
DATA = RAIZ / "data"
CLEAN = DATA / "clean"
FACE = DATA / "face"
MUESTRA = DATA / "muestra"
ARCFACE = DATA / "arcface"
PRUEBAS_UNICAS = DATA / "PRUEBAS_UNICAS"
REPORTS = RAIZ / "reports"
VECTORS = RAIZ / "vectors"
MODELOS = RAIZ / "models"
PARTICIONES = DATA / "particiones"                     # train/val/test con carpetas por persona (para la CNN)
RESULTADOS = RAIZ / "resultados"                       # metricas de cada modelo (para compararlos)

MODELO_DETECTOR = MODELOS / "modelDetectorRostros.tflite"
MODELO_LANDMARKER = MODELOS / "face_landmarker.task"
MODELO_ARCFACE = MODELOS / "model_antelope.onnx"       # modelo principal (antelopev2)
URL_DETECTOR = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite")
URL_LANDMARKER = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")

URL_INSIGHTFACE = "https://github.com/deepinsight/insightface/releases/download/v0.7/{}.zip"
PAQUETES_ARCFACE = {   # nombre con el que se guarda: (paquete de InsightFace, modelo de reconocimiento dentro del zip)
    "model_bufalo":   ("buffalo_l",  "w600k_r50.onnx"),
    "model_antelope": ("antelopev2", "glintr100.onnx"),
}

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
UMBRAL_TAMANO_ROSTRO = 80        # ancho y alto minimos (px) de un rostro recortado (ver 7.5 de la guia)


# ---------------------------------------------------------------- imagenes
def listar_imagenes(carpeta, ordenar=True):
    """Todas las imagenes (por extension) dentro de `carpeta`, en cualquier subcarpeta."""
    imagenes = (p for p in Path(carpeta).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES)
    return sorted(imagenes) if ordenar else list(imagenes)


def cargar_rgb(ruta) -> np.ndarray:
    img = Image.open(ruta)
    img = ImageOps.exif_transpose(img)  # respeta la orientacion del celular
    return np.ascontiguousarray(img.convert("RGB"))


def con_borde(img: np.ndarray) -> np.ndarray:
    """Rellena con borde negro del 50% del lado mayor (los recortes ajustados al rostro fallan en el Landmarker)."""
    k = int(max(img.shape[:2]) * 0.5)
    return np.ascontiguousarray(np.pad(img, ((k, k), (k, k), (0, 0)), mode="constant"))


def preprocesar_cara(cara_rgb: np.ndarray) -> np.ndarray:
    """Cara alineada 112x112 RGB uint8 -> (1, 3, 112, 112) float32 en [-1, 1], como pide ArcFace."""
    x = (cara_rgb.astype(np.float32) - 127.5) / 127.5
    return x.transpose(2, 0, 1)[None]


# ---------------------------------------------------------------- modelos de MediaPipe
def asegurar_modelo(ruta: Path, url: str) -> Path:
    """Descarga el modelo a `ruta` si todavia no existe."""
    ruta = Path(ruta)
    if not ruta.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando modelo a {ruta} ...")
        urllib.request.urlretrieve(url, ruta)
    return ruta


def asegurar_arcface(nombres=None):
    """Descarga (solo si no existen) los ArcFace ONNX y guarda unicamente el modelo de reconocimiento.
    nombres: por ejemplo ["model_antelope"]; por defecto todos."""
    MODELOS.mkdir(parents=True, exist_ok=True)
    for nombre in (nombres or PAQUETES_ARCFACE):
        paquete, onnx = PAQUETES_ARCFACE[nombre]
        destino = MODELOS / f"{nombre}.onnx"
        if destino.exists():
            continue
        print(f"Descargando {paquete} ...")
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / f"{paquete}.zip"
            urllib.request.urlretrieve(URL_INSIGHTFACE.format(paquete), zpath)
            with zipfile.ZipFile(zpath) as z:
                miembro = next(n for n in z.namelist() if n.endswith(onnx))
                with z.open(miembro) as origen, open(destino, "wb") as salida:
                    shutil.copyfileobj(origen, salida)
        print(f"  guardado en {destino} ({destino.stat().st_size / 1e6:.0f} MB)")


def detector_rostros(modelo, confianza=0.5):
    """FaceDetector de MediaPipe (usar con `with`). Detecta cuantos rostros hay y su caja."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opciones = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(modelo)),
        min_detection_confidence=confianza,
    )
    return vision.FaceDetector.create_from_options(opciones)


def landmarker(modelo, num_faces=1, confianza=0.2, matrices=False):
    """FaceLandmarker de MediaPipe (usar con `with`). matrices=True agrega la matriz de pose (yaw/pitch/roll)."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    extra = {"output_facial_transformation_matrixes": True} if matrices else {}
    opciones = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(modelo)),
        num_faces=num_faces,
        min_face_detection_confidence=confianza,
        min_face_presence_confidence=confianza,
        **extra,
    )
    return vision.FaceLandmarker.create_from_options(opciones)


# ---------------------------------------------------------------- resultados (mismo formato para todos los modelos)
def guardar_resultados(nombre, reales, predichos, clases, rutas):
    """Guarda en resultados/<nombre>/ las metricas de un modelo evaluado en PRUEBA.

    reales / predichos: listas con el NOMBRE de la persona; clases: lista de personas; rutas: ruta de cada foto.
      metricas.json         accuracy, F1 macro, F1 ponderado, n  -> SOLO numeros agregados (se puede subir a GitHub)
      por_clase.csv         precision / recall / F1 por persona   \
      matriz_confusion.csv  filas = real, columnas = predicho      } llevan nombres de personas: estan en .gitignore
      errores.csv           fotos mal clasificadas               /
    """
    import json

    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    carpeta = RESULTADOS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    clases = list(clases)
    metricas = {
        "modelo": nombre,
        "n_prueba": len(reales),
        "n_personas": len(clases),
        "accuracy": round(float(accuracy_score(reales, predichos)), 4),
        "f1_macro": round(float(f1_score(reales, predichos, labels=clases, average="macro", zero_division=0)), 4),
        "f1_ponderado": round(float(f1_score(reales, predichos, labels=clases, average="weighted", zero_division=0)), 4),
    }
    (carpeta / "metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    informe = classification_report(reales, predichos, labels=clases, output_dict=True, zero_division=0)
    pd.DataFrame(informe).T.loc[clases].to_csv(carpeta / "por_clase.csv", encoding="utf-8")
    pd.DataFrame(confusion_matrix(reales, predichos, labels=clases), index=clases, columns=clases) \
        .to_csv(carpeta / "matriz_confusion.csv", encoding="utf-8")
    errores = [(str(r), a, b) for r, a, b in zip(rutas, reales, predichos) if a != b]
    pd.DataFrame(errores, columns=["ruta", "real", "predicho"]).to_csv(carpeta / "errores.csv", index=False, encoding="utf-8")
    print(f"[{nombre}] prueba n={metricas['n_prueba']}  accuracy={metricas['accuracy']:.4f}  "
          f"F1 macro={metricas['f1_macro']:.4f}  errores={len(errores)}  -> {carpeta}")
    return metricas

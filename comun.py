"""Utilidades compartidas por los tres equipos.

Aquí viven las funciones que tienen que ser IGUALES en preprocesamiento,
entrenamiento y demo (detección y recorte de rostros, características HOG y
guardado de métricas). Si alguien las cambia, avisen al resto del equipo.
"""
import csv
import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
PARTICIONES = DATOS / "particiones"
RESULTADOS = RAIZ / "resultados"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- imágenes
def leer_imagen(ruta):
    """Lee una imagen BGR. Funciona con rutas con acentos o espacios (Windows)."""
    import cv2

    datos = np.fromfile(str(ruta), dtype=np.uint8)
    if datos.size == 0:
        return None
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def guardar_imagen(ruta, imagen):
    import cv2

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(ruta.suffix or ".jpg", imagen)
    if ok:
        buffer.tofile(str(ruta))
    return ok


def listar_imagenes(carpeta):
    """Imágenes de una carpeta y de TODAS sus subcarpetas (persona/set_01/001.jpg)."""
    return sorted(p for p in Path(carpeta).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES)


# ------------------------------------------------------ detección de rostros
_haar = None


def detectar_rostros_haar(imagen):
    """Devuelve una lista de cajas (x, y, ancho, alto) usando Haar de OpenCV."""
    global _haar
    import cv2

    if _haar is None:
        ruta = Path(getattr(getattr(cv2, "data", None), "haarcascades", "")) / "haarcascade_frontalface_default.xml"
        if not ruta.exists():
            raise RuntimeError(
                "Tu instalación de OpenCV no incluye el detector Haar. Instala OpenCV 4.x "
                "(pip install \"opencv-python<5\") o usa --detector mediapipe."
            )
        _haar = cv2.CascadeClassifier(str(ruta))
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    cajas = _haar.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return [tuple(int(v) for v in c) for c in cajas]


def crear_detector(nombre="haar"):
    """Devuelve una función imagen -> lista de cajas.

    Opciones: 'haar', 'mediapipe' o 'auto' (MediaPipe si está disponible; si no, Haar).
    Con fotos de cuerpo completo, MediaPipe detecta mejor rostros pequeños y con accesorios.
    """
    if nombre == "haar":
        return detectar_rostros_haar

    import cv2

    try:
        import mediapipe as mp

        detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    except (ImportError, AttributeError) as error:
        if nombre == "mediapipe":
            raise RuntimeError(f"MediaPipe no está disponible ({error}). Instálalo con: pip install mediapipe") from error
        print("AVISO: MediaPipe no está disponible; se usa Haar (falla más con disfraces y cuerpo completo).")
        return detectar_rostros_haar

    def detectar(imagen):
        alto, ancho = imagen.shape[:2]
        resultado = detector.process(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        cajas = []
        for d in resultado.detections or []:
            b = d.location_data.relative_bounding_box
            cajas.append((int(b.xmin * ancho), int(b.ymin * alto), int(b.width * ancho), int(b.height * alto)))
        return cajas

    return detectar


def recortar_con_margen(imagen, caja, margen=0.2):
    """Recorta un cuadrado alrededor del rostro, con un margen relativo."""
    x, y, w, h = caja
    cx, cy = x + w / 2, y + h / 2
    lado = max(w, h) * (1 + 2 * margen)
    x0, y0 = int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2))
    x1, y1 = int(min(imagen.shape[1], cx + lado / 2)), int(min(imagen.shape[0], cy + lado / 2))
    return imagen[y0:y1, x0:x1]


# --------------------------------------------------- características clásicas
def caracteristicas_hog(imagen, tam=128):
    """Vector HOG de una imagen BGR (rostro ya recortado)."""
    import cv2
    from skimage.feature import hog

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, (tam, tam))
    return hog(gris, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")


# ------------------------------------------------------------ particiones
def listar_split(split):
    """Devuelve (rutas, etiquetas) de datos/particiones/<split>/<persona>/*."""
    base = PARTICIONES / split
    if not base.exists():
        raise FileNotFoundError(f"No existe {base}. ¿Ya corrieron preprocesamiento/particionar.py?")
    rutas, etiquetas = [], []
    for carpeta in sorted(p for p in base.iterdir() if p.is_dir()):
        for archivo in listar_imagenes(carpeta):
            rutas.append(archivo)
            etiquetas.append(carpeta.name)
    return rutas, etiquetas


# ---------------------------------------------------------------- métricas
def guardar_resultados(nombre, y_real, y_pred, clases, rutas=None):
    """Guarda métricas, matriz de confusión y predicciones en resultados/<nombre>/.

    El equipo de reporte lee estos archivos, así que ambos modelos deben usar
    esta misma función.
    """
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, precision_recall_fscore_support)

    carpeta = RESULTADOS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    precision, recall, f1, _ = precision_recall_fscore_support(y_real, y_pred, average="macro", zero_division=0)
    metricas = {
        "modelo": nombre,
        "n_prueba": len(y_real),
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    (carpeta / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    matriz = confusion_matrix(y_real, y_pred, labels=clases)
    np.savetxt(carpeta / "matriz_confusion.csv", matriz, fmt="%d", delimiter=",",
               header=",".join(clases), comments="")

    (carpeta / "reporte_por_clase.txt").write_text(
        classification_report(y_real, y_pred, labels=clases, zero_division=0), encoding="utf-8")

    with open(carpeta / "predicciones.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["archivo", "real", "predicho"])
        for i, (r, p) in enumerate(zip(y_real, y_pred)):
            escritor.writerow([str(rutas[i]) if rutas is not None else i, r, p])

    print(f"\n[{nombre}] accuracy={metricas['accuracy']:.4f}  f1_macro={metricas['f1_macro']:.4f}")
    print(f"Resultados guardados en {carpeta}")
    return metricas

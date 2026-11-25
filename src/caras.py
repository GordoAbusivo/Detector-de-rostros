"""
Deteccion y recorte de caras para la CNN de Keras. Lo usan kaggle_preparar_caras.py (entrenamiento) y
predecir_keras.py (inferencia), asi la red ve EXACTAMENTE el mismo tipo de recorte al entrenar y al predecir.

    from caras import DetectorCaras, recortar_cuadrado

    detector = DetectorCaras("auto")                 # "mediapipe", "haar" o "auto"
    caja = detector.mejor_caja(rgb)                  # (x, y, w, h, score, nombre_detector) o None
    cara = recortar_cuadrado(rgb, caja, margen=0.5, tam=224)   # array uint8 (224, 224, 3)

Detectores (por que dos):
  - MediaPipe (BlazeFace "full range"): pensado para caras pequenas o lejanas. Es el recomendado para las fotos del
    dataset, que son mayormente de cuerpo entero / 3/4 de cuerpo.
  - Haar (OpenCV): viene con cv2, no descarga nada, pero solo ve caras casi de frente y falla mas con caras pequenas
    o giradas, y da mas falsos positivos.
  - "auto": prueba MediaPipe y, si no encuentra nada, intenta con Haar. Si MediaPipe no esta instalado, usa solo Haar.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from comun import MODELO_DETECTOR, URL_DETECTOR, asegurar_modelo, detector_rostros

LADO_MAX_HAAR = 1024      # las fotos grandes se reducen a este lado mayor antes de buscar con Haar (mas rapido)


class DetectorCaras:
    def __init__(self, backend="auto", confianza=0.5, min_cara=60):
        """min_cara: lado minimo (px, en la foto original) de una cara para tomarla en cuenta."""
        if backend not in ("auto", "mediapipe", "haar"):
            raise ValueError(f"detector desconocido: {backend}")
        self.min_cara = min_cara
        self.mp = None
        self.haar = None
        if backend in ("auto", "mediapipe"):
            try:
                import mediapipe  # noqa: F401
                self.mp = detector_rostros(asegurar_modelo(MODELO_DETECTOR, URL_DETECTOR), confianza)
            except Exception as e:
                if backend == "mediapipe":
                    raise
                print(f"AVISO: MediaPipe no disponible ({type(e).__name__}); se usa solo Haar.")
        if backend in ("auto", "haar") or self.mp is None:
            import cv2
            ruta = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self.haar = cv2.CascadeClassifier(str(ruta))
            if self.haar.empty():
                raise RuntimeError(f"No se pudo cargar {ruta}")

    # ------------------------------------------------------------ detectores individuales
    def _cajas_mediapipe(self, rgb):
        import mediapipe as mp
        res = self.mp.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        return [(d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height,
                 float(d.categories[0].score), "mediapipe") for d in res.detections]

    def _cajas_haar(self, rgb):
        import cv2
        gris = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
        k = min(1.0, LADO_MAX_HAAR / max(gris.shape))
        if k < 1.0:
            gris = cv2.resize(gris, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        minimo = max(20, int(self.min_cara * k))
        cajas = self.haar.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(minimo, minimo))
        return [(x / k, y / k, w / k, h / k, 1.0, "haar") for x, y, w, h in cajas]

    # ------------------------------------------------------------ interfaz
    def cajas(self, rgb):
        """Todas las caras encontradas (x, y, w, h, score, detector), sin filtrar por tamano."""
        encontradas = self._cajas_mediapipe(rgb) if self.mp is not None else []
        if not any(self._valida(c) for c in encontradas) and self.haar is not None:
            encontradas = encontradas + self._cajas_haar(rgb)
        return encontradas

    def _valida(self, caja):
        return min(caja[2], caja[3]) >= self.min_cara

    def mejor_caja(self, rgb):
        """La cara mas grande de la foto (la persona retratada; las de fondo suelen ser mas chicas) o None."""
        validas = [c for c in self.cajas(rgb) if self._valida(c)]
        return max(validas, key=lambda c: c[2] * c[3]) if validas else None

    def n_caras(self, rgb):
        return sum(self._valida(c) for c in self.cajas(rgb))

    def cerrar(self):
        if self.mp is not None:
            self.mp.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cerrar()


def recortar_cuadrado(rgb, caja, margen=0.5, tam=224):
    """Recorte cuadrado centrado en la cara, con `margen` extra (0.5 = lado 1.5 veces el de la cara: incluye pelo y
    orejas). Si se sale de la foto se desplaza hacia adentro. Devuelve un array uint8 (tam, tam, 3)."""
    alto, ancho = rgb.shape[:2]
    x, y, w, h = caja[:4]
    lado = min(max(w, h) * (1 + margen), alto, ancho)
    izq = min(max(x + w / 2 - lado / 2, 0), ancho - lado)
    arriba = min(max(y + h / 2 - lado / 2, 0), alto - lado)
    izq, arriba, lado = int(round(izq)), int(round(arriba)), int(round(lado))
    recorte = Image.fromarray(rgb[arriba:arriba + lado, izq:izq + lado])
    return np.asarray(recorte.resize((tam, tam), Image.Resampling.LANCZOS))

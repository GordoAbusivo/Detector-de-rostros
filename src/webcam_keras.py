"""
Reconocimiento en vivo con la webcam (o con un video) usando la CNN de Keras: detecta las caras de cada cuadro, las recorta
igual que en el entrenamiento (caras.py) y dibuja un recuadro con el nombre y la confianza.

Se corre en TU computadora (Kaggle no tiene acceso a la camara). Pasos:
  1. Descarga de Kaggle (pestana Output) models/keras_<nombre>.keras y models/keras_<nombre>_clases.json y ponlos en models/.
  2. pip install -r requirements-keras.txt   y   pip install opencv-python   (la version con ventanas; la "headless" no abre ventanas)
  3. python src/webcam_keras.py --modelo models/keras_mobilenetv2.keras

Teclas: q o Esc = salir, s = guardar una captura del cuadro actual (capturas/, quedan solo en tu equipo).
Colores: verde = confianza alta (>=70%), naranja = media (>=40%), rojo = baja. Siempre elige a una de las personas del
grupo (no existe "desconocido"): una confianza baja significa que la red duda.

Uso:
    python src/webcam_keras.py --modelo models/keras_mobilenetv2.keras
    python src/webcam_keras.py --modelo models/keras_mobilenetv2.keras --camara 1          # webcam externa
    python src/webcam_keras.py --modelo models/keras_mobilenetv2.keras --video prueba.mp4  # en vez de camara
    python src/webcam_keras.py --modelo ... --video in.mp4 --salida out.mp4 --sin-ventana  # sin pantalla
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2                  # noqa: E402
import numpy as np          # noqa: E402

from caras import DetectorCaras, recortar_cuadrado   # noqa: E402
from comun import MODELOS, RAIZ                       # noqa: E402
from predecir_keras import CONFIANZA_ALTA, CONFIANZA_MEDIA, cargar_modelo   # noqa: E402


def color_bgr(conf):
    return (0, 160, 0) if conf >= CONFIANZA_ALTA else (0, 140, 255) if conf >= CONFIANZA_MEDIA else (0, 0, 220)


def dibujar(cuadro_bgr, caja, texto, conf):
    x, y, w, h = (int(v) for v in caja[:4])
    color = color_bgr(conf)
    cv2.rectangle(cuadro_bgr, (x, y), (x + w, y + h), color, 2)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    arriba = max(y - th - 10, 0)
    cv2.rectangle(cuadro_bgr, (x, arriba), (x + tw + 8, arriba + th + 10), color, -1)
    cv2.putText(cuadro_bgr, texto, (x + 4, arriba + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def procesar_cuadro(rgb, detector, modelo, clases, tam, margen, max_caras):
    """Devuelve [(caja, nombre, confianza)] para las caras mas grandes del cuadro."""
    cajas = sorted((c for c in detector.cajas(rgb) if detector._valida(c)), key=lambda c: c[2] * c[3], reverse=True)[:max_caras]
    if not cajas:
        return []
    caras = np.stack([recortar_cuadrado(rgb, c, margen, tam) for c in cajas]).astype("float32")
    probas = modelo.predict(caras, verbose=0)
    return [(c, clases[int(p.argmax())], float(p.max())) for c, p in zip(cajas, probas)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=str(MODELOS / "keras_mobilenetv2.keras"))
    ap.add_argument("--camara", type=int, default=0, help="indice de la camara (0 = la de la laptop, 1 = externa)")
    ap.add_argument("--video", help="ruta de un video; si se da, se usa en vez de la camara")
    ap.add_argument("--detector", choices=["auto", "mediapipe", "haar"], default="auto")
    ap.add_argument("--margen", type=float, default=0.5, help="el MISMO margen con el que se recortaron las caras de entrenamiento")
    ap.add_argument("--max-caras", type=int, default=3, help="cuantas caras (las mas grandes) reconocer por cuadro")
    ap.add_argument("--no-espejo", action="store_true", help="no voltear la imagen de la camara (por defecto se ve como espejo)")
    ap.add_argument("--salida", help="guardar el resultado en este video (.mp4)")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir ventana (para servidores o pruebas)")
    ap.add_argument("--max-cuadros", type=int, default=None, help="parar tras N cuadros")
    a = ap.parse_args()

    modelo, clases = cargar_modelo(a.modelo)
    tam = modelo.input_shape[1]
    es_camara = a.video is None
    cap = cv2.VideoCapture(a.camara if es_camara else a.video)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: no se pudo abrir {'la camara ' + str(a.camara) if es_camara else a.video}. "
                         "Prueba otro indice con --camara 1, y revisa que ninguna otra app la este usando.")
    espejo = es_camara and not a.no_espejo
    escritor = None
    carpeta_capturas = RAIZ / "capturas"
    t_prev, fps, n = time.time(), 0.0, 0

    with DetectorCaras(a.detector) as detector:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if espejo:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for caja, nombre, conf in procesar_cuadro(rgb, detector, modelo, clases, tam, a.margen, a.max_caras):
                dibujar(bgr, caja, f"{nombre} {conf * 100:.0f}%", conf)

            ahora = time.time()
            fps = 0.9 * fps + 0.1 / max(ahora - t_prev, 1e-6) if n else 0.0
            t_prev, n = ahora, n + 1
            cv2.putText(bgr, f"{fps:.0f} fps  (q = salir, s = captura)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            if a.salida:
                if escritor is None:
                    alto, ancho = bgr.shape[:2]
                    escritor = cv2.VideoWriter(a.salida, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 20, (ancho, alto))
                escritor.write(bgr)
            if not a.sin_ventana:
                try:
                    cv2.imshow("Detector de rostros ESCOM", bgr)
                except cv2.error:
                    raise SystemExit("ERROR: tu OpenCV no abre ventanas (version 'headless'). Instala la normal: "
                                     "pip uninstall -y opencv-python-headless && pip install opencv-python")
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord("s"):
                    carpeta_capturas.mkdir(exist_ok=True)
                    ruta = carpeta_capturas / f"captura_{int(time.time())}.jpg"
                    cv2.imwrite(str(ruta), bgr)
                    print("Captura guardada en", ruta)
            if a.max_cuadros and n >= a.max_cuadros:
                break

    cap.release()
    if escritor is not None:
        escritor.release()
        print("Video guardado en", a.salida)
    if not a.sin_ventana:
        cv2.destroyAllWindows()
    print(f"{n} cuadros procesados.")


if __name__ == "__main__":
    main()

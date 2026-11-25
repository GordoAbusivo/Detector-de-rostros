"""
Identifica a la persona de una o varias fotos con la CNN de Keras: detecta la cara, la recorta igual que en el
entrenamiento (caras.py) y la clasifica. Muestra las 3 personas mas probables y la confianza.

Siempre responde una de las personas del grupo (no existe "desconocido"): una confianza baja avisa que la red duda.

Uso:
    python src/predecir_keras.py foto1.jpg foto2.jpg
    python src/predecir_keras.py fotos_prueba/*.jpg --modelo models/keras_mobilenetv2.keras --panel salida_paneles
    python src/predecir_keras.py cara_ya_recortada.jpg --sin-detector      # la foto ya ES una cara
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np          # noqa: E402
from PIL import Image       # noqa: E402

from caras import DetectorCaras, recortar_cuadrado   # noqa: E402
from comun import MODELOS, cargar_rgb                 # noqa: E402

CONFIANZA_ALTA, CONFIANZA_MEDIA = 0.70, 0.40


def cargar_modelo(ruta):
    import keras
    ruta = Path(ruta)
    modelo = keras.models.load_model(ruta)
    clases = json.loads((ruta.parent / f"{ruta.stem}_clases.json").read_text(encoding="utf-8"))
    return modelo, clases


def preparar_cara(rgb, detector, tam, margen):
    """Devuelve (cara_uint8, caja). Sin detector (None) la foto completa se ajusta al cuadrado central."""
    if detector is None:
        lado = min(rgb.shape[:2])
        caja = ((rgb.shape[1] - lado) / 2, (rgb.shape[0] - lado) / 2, lado, lado)
        return recortar_cuadrado(rgb, caja, 0.0, tam), caja
    caja = detector.mejor_caja(rgb)
    if caja is None:
        return None, None
    return recortar_cuadrado(rgb, caja, margen, tam), caja


def predecir(modelo, clases, cara, top=3):
    p = modelo.predict(cara[None].astype("float32"), verbose=0)[0]
    orden = np.argsort(p)[::-1][:top]
    return [(clases[i], float(p[i])) for i in orden]


def panel(original, cara, resultado, ruta_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nombre, conf = resultado[0]
    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
    ejes[0].imshow(original)
    ejes[0].set_title("1. Entrada")
    ejes[1].imshow(cara)
    ejes[1].set_title(f"2. Cara que ve la red ({cara.shape[1]}x{cara.shape[0]})")
    color = "darkgreen" if conf >= CONFIANZA_ALTA else "darkorange"
    texto = "3. Prediccion\n\n" + "\n".join(f"{n}: {c * 100:.1f}%" for n, c in resultado)
    ejes[2].text(0.05, 0.5, texto, fontsize=13, color=color, va="center", fontweight="bold")
    for e in ejes:
        e.axis("off")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fotos", nargs="+")
    ap.add_argument("--modelo", default=str(MODELOS / "keras_scratch.keras"))
    ap.add_argument("--detector", choices=["auto", "mediapipe", "haar"], default="auto")
    ap.add_argument("--sin-detector", action="store_true", help="la foto ya es una cara: solo se ajusta a cuadrado")
    ap.add_argument("--margen", type=float, default=0.5, help="el MISMO margen con el que se recortaron las caras de entrenamiento")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--panel", help="carpeta donde guardar un panel PNG (entrada | cara | prediccion) por foto")
    a = ap.parse_args()

    modelo, clases = cargar_modelo(a.modelo)
    tam = modelo.input_shape[1]
    detector = None if a.sin_detector else DetectorCaras(a.detector)
    if a.panel:
        Path(a.panel).mkdir(parents=True, exist_ok=True)

    for ruta in a.fotos:
        try:
            rgb = cargar_rgb(ruta)
        except Exception as e:
            print(f"{ruta}: no se pudo leer ({e})")
            continue
        cara, _ = preparar_cara(rgb, detector, tam, a.margen)
        if cara is None:
            print(f"{ruta}: no se detecto ninguna cara (prueba otra foto, de frente y con la cara visible)")
            continue
        resultado = predecir(modelo, clases, cara, a.top)
        nombre, conf = resultado[0]
        nivel = "alta" if conf >= CONFIANZA_ALTA else "media" if conf >= CONFIANZA_MEDIA else "baja"
        print(f"{ruta}: {nombre}  (confianza {conf * 100:.1f}%, {nivel})   otras: " +
              ", ".join(f"{n} {c * 100:.1f}%" for n, c in resultado[1:]))
        if a.panel:
            panel(Image.fromarray(rgb), cara, resultado, Path(a.panel) / f"{Path(ruta).stem}_panel.png")

    if detector is not None:
        detector.cerrar()


if __name__ == "__main__":
    main()

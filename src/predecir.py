"""
Pipeline completo para probar el modelo con una foto nueva (cualquier tamano, sin recortar).

  foto -> Face Landmarker (5 puntos) -> alinear a 112x112 -> normalizar pixeles
       -> ArcFace antelope (ONNX) -> vector de 512 -> L2 -> clasificador (Ridge / SVM / KNN) -> persona

Siempre responde una de las personas con las que se entreno el modelo (no hay opcion "desconocido").
Imprime UNA linea por foto: la persona, la confianza y el modelo usado. La confianza es el parecido
(coseno) con el centro de la persona elegida: alta >= 0.60, media 0.50-0.60, baja < 0.50.

Uso (desde cualquier carpeta, con el entorno activado):
    python src/predecir.py foto.jpg                          # ridge (por defecto)
    python src/predecir.py foto.jpg --modelo svm             # usa models/svm_antelope.joblib
    python src/predecir.py foto.jpg --modelo knn
    python src/predecir.py foto.jpg --modelo ruta/otro.joblib   # cualquier paquete .joblib
    python src/predecir.py foto1.jpg foto2.jpg --guardar-alineada

--modelo NOMBRE carga models/NOMBRE_antelope.joblib. Ese archivo es un PAQUETE (dict con "modelo",
"clases" y "centroides") que se exporta desde analisis_embeddings.ipynb; los .joblib sueltos
(ridge.joblib, svm.joblib, knn.joblib) NO sirven aqui porque no traen las clases ni los centroides.

Requiere: models/<modelo>_antelope.joblib, models/model_antelope.onnx, models/face_landmarker.task
"""
import argparse
import threading
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from PIL import Image

from alinear import cinco_puntos, alinear
from comun import (MODELO_ARCFACE, MODELO_LANDMARKER, MODELOS, URL_LANDMARKER, asegurar_arcface, asegurar_modelo,
                   cargar_rgb, con_borde, landmarker, preprocesar_cara)

# Niveles de confianza = parecido (coseno) con el centro de la persona elegida. Calibrados con vectors/ (22/11/2026):
# la mediana con el centro propio es ~0.81; 96.7% de las imagenes propias pasa de 0.60 y solo 0.01% de los
# centros AJENOS llega a 0.60 (0.4% llega a 0.50). Fotos nuevas (disfraz, cubrebocas) suelen salir mas bajas.
UMBRAL_ALTA, UMBRAL_MEDIA = 0.60, 0.50


def nivel_confianza(parecido: float) -> str:
    return "alta" if parecido >= UMBRAL_ALTA else "media" if parecido >= UMBRAL_MEDIA else "baja"


def rostro_mas_grande(res, ancho, alto):
    """De todos los rostros detectados, el de mayor area."""
    def area(lm):
        xs = np.array([p.x for p in lm]) * ancho; ys = np.array([p.y for p in lm]) * alto
        return (xs.max() - xs.min()) * (ys.max() - ys.min())
    return max(res.face_landmarks, key=area)


class Predictor:
    """Carga el clasificador de UN modelo (ridge, svm, knn...). El ArcFace y el detector de rostros pesan mucho y NO
    dependen del clasificador, asi que se cargan una sola vez y se COMPARTEN entre todos los Predictor.
    Lo usan la terminal (main) y el front (app.py)."""

    _sesion = None                      # ArcFace ONNX compartido
    _entrada = None
    _detector = None                    # Face Landmarker compartido
    _cerrojo = threading.Lock()         # el detector no es seguro con varios hilos (front)

    def __init__(self, modelo="ridge"):
        ruta = Path(modelo) if modelo.endswith(".joblib") \
            else MODELOS / f"{modelo}_antelope.joblib"
        if not ruta.is_file():
            raise FileNotFoundError(f"no existe {ruta}. Copia ahi el archivo del modelo (ridge_antelope.joblib) o exportalo con src/exportar_modelos.py.")
        paquete = joblib.load(ruta)
        if not isinstance(paquete, dict) or not {"modelo", "clases", "centroides"} <= set(paquete):
            raise ValueError(f"{ruta.name} es un modelo suelto, no un paquete con "
                             "'modelo', 'clases' y 'centroides'. Exportalo con la celda del notebook.")
        self.clasificador = paquete["modelo"]
        self.clases = list(paquete["clases"])
        self.cent = np.asarray(paquete["centroides"])
        self.nombre = ruta.stem.removesuffix("_antelope")
        with Predictor._cerrojo:
            if Predictor._sesion is None:
                asegurar_arcface(["model_antelope"])                # solo se descargan la primera vez
                asegurar_modelo(MODELO_LANDMARKER, URL_LANDMARKER)
                Predictor._sesion = ort.InferenceSession(str(MODELO_ARCFACE), providers=["CPUExecutionProvider"])
                Predictor._entrada = Predictor._sesion.get_inputs()[0].name
                Predictor._detector = landmarker(MODELO_LANDMARKER, num_faces=5, confianza=0.2)

    @classmethod
    def cerrar(cls):
        """Libera el detector compartido (llamar al terminar el programa)."""
        with cls._cerrojo:
            if cls._detector is not None:
                cls._detector.close()
                cls._detector = cls._sesion = cls._entrada = None

    def predecir_rgb(self, rgb):
        """rgb: array uint8 (alto, ancho, 3 o 4). Devuelve un dict, o None si no se detecta ningun rostro:
        persona, confianza (0-1), nivel (alta/media/baja), n_rostros, cara (112x112 alineada) y modelo."""
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        with Predictor._cerrojo:
            res = Predictor._detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:                      # reintento con borde (rostros muy recortados)
                rgb = con_borde(rgb)
                res = Predictor._detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:
                return None
            alto, ancho = rgb.shape[:2]
            cara = alinear(rgb, cinco_puntos(rostro_mas_grande(res, ancho, alto), ancho, alto))
            v = Predictor._sesion.run(None, {Predictor._entrada: preprocesar_cara(cara)})[0]
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        idx = self.clases.index(self.clasificador.predict(v)[0])
        puntaje = float(v[0] @ self.cent[idx])              # parecido con el centro de la persona elegida
        return {"persona": self.clases[idx], "confianza": puntaje, "nivel": nivel_confianza(puntaje),
                "n_rostros": len(res.face_landmarks), "cara": cara, "modelo": self.nombre}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fotos", nargs="+")
    ap.add_argument("--guardar-alineada", action="store_true", help="guarda la cara alineada junto a la foto")
    ap.add_argument("--modelo", default="ridge",
                    help="ridge | svm | knn (carga models/<nombre>_antelope.joblib) o la ruta a un paquete .joblib")
    args = ap.parse_args()

    try:
        predictor = Predictor(args.modelo)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"ERROR: {e}")

    try:
        for foto in args.fotos:
            nombre = Path(foto).name
            r = predictor.predecir_rgb(cargar_rgb(Path(foto)))
            if r is None:
                print(f"{nombre}: no se detecto ningun rostro."); continue
            if r["n_rostros"] > 1:
                print(f"{nombre}: aviso, {r['n_rostros']} rostros detectados; se usa el mas grande.")
            if args.guardar_alineada:
                destino = Path(foto).with_name(Path(foto).stem + "_alineada.png")
                Image.fromarray(r["cara"]).save(destino); print("Cara alineada guardada en", destino)
            print(f"{nombre}: {r['persona']} | confianza {r['confianza']:.2f} ({r['nivel']}) [{r['modelo']}]")
    finally:
        Predictor.cerrar()


if __name__ == "__main__":
    main()

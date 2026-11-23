"""
PASO 1 para ArcFace: alinea cada imagen de data/muestra/ a 112x112 (plantilla estandar de ArcFace/InsightFace)
y la guarda en data/arcface/ con la misma estructura persona/set_XX/archivo (en PNG, sin perdida).

Como funciona:
  1. Face Landmarker (MediaPipe) saca 5 puntos: ojo izq, ojo der, punta de nariz, comisura izq y der de la boca
     ("izq/der" = como se ve en la imagen).
  2. Se calcula la transformacion de similitud (rotacion + escala + traslacion) que lleva esos 5 puntos a la
     plantilla de ArcFace para 112x112.
  3. Se aplica con warpAffine -> imagen 112x112 con el rostro derecho, centrado y a escala estandar.

data/muestra solo se usa para saber QUE imagenes alinear. La alineacion se hace sobre la imagen ORIGINAL de
data/clean/ (misma persona/set/nombre), porque los recortes ajustados no tienen contexto alrededor del rostro y
al rotarlos dejan esquinas negras. Si en el original no se detecta rostro, se usa el recorte con borde negro.
Si no se detectan landmarks, la imagen se omite y se anota en reports/alinear_omitidas.csv (solo si hubo).

Uso:
    python src/alinear.py
    python src/alinear.py --limite 40 --salida /ruta/de/prueba

Salida:
    data/arcface/persona/set_XX/archivo.png   (112x112, RGB; para el modelo: (x-127.5)/127.5)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from comun import (ARCFACE, CLEAN, EXTENSIONES, MODELO_LANDMARKER, MUESTRA, REPORTS, cargar_rgb, con_borde,
                   landmarker, listar_imagenes)

# Plantilla ArcFace 112x112 (InsightFace): ojo izq, ojo der, nariz, boca izq, boca der
PLANTILLA = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# Indices del mallado de MediaPipe (478 puntos). "izq" = lado izquierdo de la imagen
OJO_IZQ = (33, 133)    # esquinas externa e interna
OJO_DER = (362, 263)
NARIZ = 1
BOCA_IZQ, BOCA_DER = 61, 291


def cinco_puntos(lm, ancho, alto) -> np.ndarray:
    p = lambda i: np.array([lm[i].x * ancho, lm[i].y * alto], dtype=np.float32)
    return np.stack([
        (p(OJO_IZQ[0]) + p(OJO_IZQ[1])) / 2,
        (p(OJO_DER[0]) + p(OJO_DER[1])) / 2,
        p(NARIZ), p(BOCA_IZQ), p(BOCA_DER),
    ])


def alinear(rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
    M, _ = cv2.estimateAffinePartial2D(pts, PLANTILLA, method=cv2.LMEDS)
    escala = float(np.hypot(M[0, 0], M[1, 0]))
    if escala < 0.5:  # imagen muy grande: reducir antes con INTER_AREA para evitar aliasing
        f = max(2 * escala, 1e-3)
        rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        M, _ = cv2.estimateAffinePartial2D(pts * f, PLANTILLA, method=cv2.LMEDS)
    return cv2.warpAffine(rgb, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default=str(MUESTRA))
    ap.add_argument("--originales", default=str(CLEAN))
    ap.add_argument("--salida", default=str(ARCFACE))
    ap.add_argument("--modelo", default=str(MODELO_LANDMARKER))
    ap.add_argument("--limite", type=int, default=0, help="procesar solo N imagenes (prueba)")
    args = ap.parse_args()

    entrada, salida, originales = Path(args.entrada), Path(args.salida), Path(args.originales)
    if salida.exists() and any(salida.iterdir()):
        raise SystemExit(f"ERROR: {salida} ya tiene contenido; vacialo antes de volver a correr.")

    imagenes = listar_imagenes(entrada)
    if args.limite:
        imagenes = imagenes[:: max(1, len(imagenes) // args.limite)][: args.limite]
    print(f"Imagenes a alinear: {len(imagenes)}")
    if not imagenes:
        raise SystemExit(f"ERROR: no se encontraron imagenes en {entrada.resolve()}")

    def buscar_original(rel: Path):
        """persona/set/archivo.ext en data/clean (los .heic se recortaron como .jpg, asi que se busca por nombre)."""
        for cand in (originales / rel.parent).glob(rel.stem + ".*"):
            if cand.suffix.lower() in EXTENSIONES:
                return cand
        return None

    guardadas, omitidas, usos_recorte = 0, [], 0
    with landmarker(args.modelo, num_faces=1, confianza=0.2) as lm:
        for i, ruta in enumerate(imagenes, 1):
            rel = ruta.relative_to(entrada)
            try:
                def detectar(img):
                    return lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img)).face_landmarks

                rgb, marcas = None, None
                orig = buscar_original(rel)
                if orig is not None:                 # 1o: imagen completa (hay contexto alrededor del rostro)
                    rgb = cargar_rgb(orig)
                    marcas = detectar(rgb)
                if not marcas:                       # 2o: el recorte con borde negro
                    usos_recorte += 1
                    rgb = con_borde(cargar_rgb(ruta))
                    marcas = detectar(rgb)
                if not marcas:
                    omitidas.append((rel.as_posix(), "sin landmarks"))
                    continue
                alto, ancho = rgb.shape[:2]
                cara = alinear(rgb, cinco_puntos(marcas[0], ancho, alto))
                destino = (salida / rel).with_suffix(".png")
                destino.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(destino), cv2.cvtColor(cara, cv2.COLOR_RGB2BGR))
                guardadas += 1
            except Exception as e:
                omitidas.append((rel.as_posix(), f"error: {e}"))
            if i % 500 == 0:
                print(f"  {i}/{len(imagenes)}")

    print(f"\nAlineadas y guardadas en {salida}: {guardadas}")
    print(f"Omitidas: {len(omitidas)} | Alineadas desde el recorte (original no sirvio): {usos_recorte}")
    if omitidas:
        reporte = REPORTS / "alinear_omitidas.csv"
        reporte.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(omitidas, columns=["ruta", "motivo"]).to_csv(reporte, index=False, encoding="utf-8")
        print(f"CSV de omitidas guardado en {reporte}")


if __name__ == "__main__":
    main()

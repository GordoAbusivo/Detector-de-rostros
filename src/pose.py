"""
Estima la orientacion de la cabeza (yaw / pitch / roll, en grados) de cada imagen de data/face/
con MediaPipe Face Landmarker y la guarda en reports/pose_face.csv.

  yaw   > 0  : el rostro mira hacia la DERECHA de la imagen ; < 0 : hacia la IZQUIERDA
  pitch > 0  : el rostro mira hacia ABAJO                    ; < 0 : hacia ARRIBA
  roll       : inclinacion lateral de la cabeza (sentido no verificado visualmente)
  (signos verificados mirando imagenes reales; con |yaw| muy alto el pitch es menos confiable)

Si el Landmarker no encuentra rostro en el recorte, yaw, pitch y roll quedan vacios (NaN).

Uso:
    python src/pose.py
    python src/pose.py --limite 50      # solo para probar
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import mediapipe as mp

from comun import (FACE, MODELO_LANDMARKER, REPORTS, URL_LANDMARKER, asegurar_modelo, cargar_rgb, con_borde,
                   landmarker, listar_imagenes)


def euler_desde_matriz(m: np.ndarray):
    """m: matriz 4x4 de transformacion facial. Devuelve (yaw, pitch, roll) en grados."""
    r = m[:3, :3]
    r = r / np.linalg.norm(r, axis=0)  # quita la escala
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default=str(FACE))
    ap.add_argument("--modelo", default=str(MODELO_LANDMARKER))
    ap.add_argument("--salida", default=str(REPORTS / "pose_face.csv"))
    ap.add_argument("--limite", type=int, default=0, help="procesar solo N imagenes (prueba)")
    args = ap.parse_args()

    entrada = Path(args.entrada)
    modelo = asegurar_modelo(Path(args.modelo), URL_LANDMARKER)
    imagenes = listar_imagenes(entrada)
    if args.limite:
        imagenes = imagenes[:: max(1, len(imagenes) // args.limite)][: args.limite]
    print(f"Imagenes a procesar: {len(imagenes)}")
    if not imagenes:
        raise SystemExit(f"ERROR: no se encontraron imagenes en {entrada.resolve()}")

    filas = []
    with landmarker(modelo, num_faces=1, confianza=0.2, matrices=True) as lm:
        for i, ruta in enumerate(imagenes, 1):
            rel = ruta.relative_to(entrada)
            fila = {"persona": rel.parts[0], "archivo": ruta.name, "ruta": rel.as_posix(),
                    "yaw": np.nan, "pitch": np.nan, "roll": np.nan}
            try:
                rgb = cargar_rgb(ruta)
                # los recortes ajustados al rostro fallan mucho en el Landmarker: se rellena con borde negro
                rgb = con_borde(rgb)
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if res.facial_transformation_matrixes:
                    fila["yaw"], fila["pitch"], fila["roll"] = euler_desde_matriz(
                        np.array(res.facial_transformation_matrixes[0]))
            except Exception as e:
                print(f"  error en {ruta}: {e}")
            filas.append(fila)
            if i % 500 == 0:
                print(f"  {i}/{len(imagenes)}")

    df = pd.DataFrame(filas).round(2)
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False, encoding="utf-8")
    print(f"Sin landmarks: {int(df['yaw'].isna().sum())} de {len(df)}")
    print(f"CSV guardado en {salida}")


if __name__ == "__main__":
    main()

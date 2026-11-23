"""
Paso 1: detectar imagenes sin rostro
Imagenes de datasets/ en las que MediaPipe NO detecta ningun rostro (detector de rango completo, ~5 m).

Salida: reports/imagenesSinRostro.csv (no se crea si todas las imagenes tienen rostro).

Uso: python src/SinRostro.py
"""
import argparse
import csv
from pathlib import Path

import mediapipe as mp

from comun import (DATASETS, MODELO_DETECTOR, REPORTS, URL_DETECTOR, asegurar_modelo, cargar_rgb,
                   detector_rostros, listar_imagenes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=str(DATASETS))
    ap.add_argument("--modelo", default=str(MODELO_DETECTOR))
    ap.add_argument("--confianza", type=float, default=0.5)
    ap.add_argument("--salida", default=str(REPORTS / "imagenesSinRostro.csv"))
    args = ap.parse_args()

    raiz = Path(args.datasets)
    modelo = asegurar_modelo(Path(args.modelo), URL_DETECTOR)

    imagenes = listar_imagenes(raiz)
    print(f"Imagenes a revisar: {len(imagenes)}")
    if not imagenes:
        raise SystemExit(f"ERROR: no se encontraron imagenes en {raiz.resolve()}")

    sin_rostro = []
    with detector_rostros(modelo, args.confianza) as detector:
        for i, ruta in enumerate(imagenes, 1):
            persona = ruta.relative_to(raiz).parts[0]
            try:
                rgb = cargar_rgb(ruta)
                resultado = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if len(resultado.detections) == 0:
                    sin_rostro.append((persona, ruta.name, str(ruta), "sin rostro"))
            except Exception as e:
                sin_rostro.append((persona, ruta.name, str(ruta), f"no se pudo leer: {e}"))
            if i % 500 == 0:
                print(f"  {i}/{len(imagenes)} revisadas, {len(sin_rostro)} sin rostro")

    print(f"\nImagenes SIN rostro detectado: {len(sin_rostro)} de {len(imagenes)}")

    if len(sin_rostro) == 0:
        print("Todas las imagenes tienen rostro; no se genera CSV.")
        return

    for persona, archivo, ruta, motivo in sin_rostro:
        print(f"  {ruta}  [{motivo}]")

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "archivo", "ruta", "motivo"])
        w.writerows(sin_rostro)
    print(f"\nCSV guardado en {salida}")

if __name__ == "__main__":
    main()

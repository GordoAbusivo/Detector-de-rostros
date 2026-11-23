"""
Imagenes de datasets/ en las que MediaPipe detecta MAS DE 1 rostro (2 o mas).

Uso (desde la raiz del proyecto):
    python src/dobleRostro.py
"""
import argparse
import csv
from pathlib import Path

import mediapipe as mp

from comun import DATASETS, MODELO_DETECTOR, REPORTS, cargar_rgb, detector_rostros, listar_imagenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=str(DATASETS))
    ap.add_argument("--modelo", default=str(MODELO_DETECTOR))
    ap.add_argument("--confianza", type=float, default=0.5)
    ap.add_argument("--salida", default=str(REPORTS / "imagenesMasDeUnRostro.csv"))
    args = ap.parse_args()

    raiz = Path(args.datasets)
    modelo = Path(args.modelo)

    imagenes = listar_imagenes(raiz)
    print(f"Imagenes a revisar: {len(imagenes)}")
    if not imagenes:
        raise SystemExit(f"ERROR: no se encontraron imagenes en {raiz.resolve()}")

    multiples = []   # (persona, archivo, ruta, num_rostros)
    con_error = []   # (persona, archivo, ruta, error)
    with detector_rostros(modelo, args.confianza) as detector:
        for i, ruta in enumerate(imagenes, 1):
            persona = ruta.relative_to(raiz).parts[0]
            try:
                rgb = cargar_rgb(ruta)
                resultado = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                n = len(resultado.detections)
                if n > 1:
                    multiples.append((persona, ruta.name, str(ruta), n))
            except Exception as e:
                con_error.append((persona, ruta.name, str(ruta), f"no se pudo leer: {e}"))
            if i % 500 == 0:
                print(f"  {i}/{len(imagenes)} revisadas, {len(multiples)} con mas de 1 rostro")

    if con_error:
        print(f"\nAVISO: {len(con_error)} imagenes no se pudieron leer (no se evaluaron):")
        for _, _, ruta, motivo in con_error:
            print(f"  {ruta}  [{motivo}]")

    print(f"\nImagenes con MAS DE 1 rostro: {len(multiples)} de {len(imagenes)}")

    if len(multiples) == 0:
        print("Ninguna imagen tiene mas de 1 rostro; no se genera CSV.")
        return

    for persona, archivo, ruta, n in multiples:
        print(f"  {ruta}  [{n} rostros]")

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "archivo", "ruta", "num_rostros"])
        w.writerows(multiples)
    print(f"\nCSV guardado en {salida}")


if __name__ == "__main__":
    main()

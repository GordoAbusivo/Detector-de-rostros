"""
Imagenes de datasets/ con nitidez menor a un umbral (por defecto 50).

Nitidez = varianza del Laplaciano sobre la imagen en escala de grises
(a mayor valor, mas nitida; borrosa = valor bajo).

Uso:
    python src/nitidez.py
    python src/nitidez.py --umbral 50
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from comun import DATASETS, REPORTS, cargar_rgb, listar_imagenes


def nitidez(rgb: np.ndarray) -> float:
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=str(DATASETS))
    ap.add_argument("--umbral", type=float, default=50.0)
    ap.add_argument("--salida", default=str(REPORTS / "imagenesNitidezBaja.csv"))
    args = ap.parse_args()

    raiz = Path(args.datasets)
    imagenes = listar_imagenes(raiz)
    print(f"Imagenes a revisar: {len(imagenes)}")
    if not imagenes:
        raise SystemExit(f"ERROR: no se encontraron imagenes en {raiz.resolve()}")

    borrosas = []    # (persona, archivo, ruta, nitidez)
    con_error = []   # (persona, archivo, ruta, error)
    for i, ruta in enumerate(imagenes, 1):
        persona = ruta.relative_to(raiz).parts[0]
        try:
            valor = nitidez(cargar_rgb(ruta))
            if valor < args.umbral:
                borrosas.append((persona, ruta.name, str(ruta), round(valor, 2)))
        except Exception as e:
            con_error.append((persona, ruta.name, str(ruta), f"no se pudo leer: {e}"))
        if i % 500 == 0:
            print(f"  {i}/{len(imagenes)} revisadas, {len(borrosas)} con nitidez < {args.umbral:g}")

    if con_error:
        print(f"\nAVISO: {len(con_error)} imagenes no se pudieron leer (no se evaluaron):")
        for _, _, ruta, motivo in con_error:
            print(f"  {ruta}  [{motivo}]")

    print(f"\nImagenes con nitidez < {args.umbral:g}: {len(borrosas)} de {len(imagenes)}")

    if len(borrosas) == 0:
        print("Ninguna imagen esta por debajo del umbral; no se genera CSV.")
        return

    borrosas.sort(key=lambda r: r[3])  # las mas borrosas primero
    for persona, archivo, ruta, valor in borrosas:
        print(f"  {ruta}  [nitidez {valor}]")

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "archivo", "ruta", "nitidez"])
        w.writerows(borrosas)
    print(f"\nCSV guardado en {salida}")


if __name__ == "__main__":
    main()

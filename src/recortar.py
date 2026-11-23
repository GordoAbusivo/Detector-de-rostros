"""
Recorta SOLO el rostro de cada imagen de data/clean/ y lo guarda en data/face/
con la misma estructura persona/set/archivo.

- Usa el detector de MediaPipe (mismo modelo que SinRostro.py) solo para obtener la caja del rostro.
- Si una imagen falla por cualquier motivo, se omite y se anota en
  reports/recortesOmitidos.csv (solo se crea si hubo omitidas).
- Los .heic se guardan como .jpg.
- Los originales no se tocan.

Uso:
    python src/recortar.py
    python src/recortar.py --margen 0.1     # 10% extra alrededor del rostro (por defecto 0 = caja ajustada)
"""
import argparse
import csv
from pathlib import Path

from PIL import Image
import mediapipe as mp

from comun import (CLEAN, FACE, MODELO_DETECTOR, REPORTS, URL_DETECTOR, asegurar_modelo, cargar_rgb,
                   detector_rostros, listar_imagenes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default=str(CLEAN))
    ap.add_argument("--salida", default=str(FACE))
    ap.add_argument("--modelo", default=str(MODELO_DETECTOR))
    ap.add_argument("--confianza", type=float, default=0.5)
    ap.add_argument("--margen", type=float, default=0.0,
                    help="fraccion extra alrededor de la caja del rostro (0.1 = 10%%)")
    args = ap.parse_args()

    entrada, salida = Path(args.entrada), Path(args.salida)
    modelo = asegurar_modelo(Path(args.modelo), URL_DETECTOR)

    if salida.exists() and any(salida.iterdir()):
        raise SystemExit(f"ERROR: {salida} ya tiene contenido; vacialo antes de volver a correr.")

    imagenes = listar_imagenes(entrada)
    print(f"Imagenes a recortar: {len(imagenes)}")
    if not imagenes:
        raise SystemExit(f"ERROR: no se encontraron imagenes en {entrada.resolve()}")

    guardadas = 0
    omitidas = []  # (persona, archivo, ruta, motivo)
    with detector_rostros(modelo, args.confianza) as detector:
        for i, ruta in enumerate(imagenes, 1):
            persona = ruta.relative_to(entrada).parts[0]
            try:
                rgb = cargar_rgb(ruta)
                alto_img, ancho_img = rgb.shape[:2]
                det = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).detections
                # data/clean ya no tiene imagenes sin rostro ni con rostros multiples: solo se toma el unico rostro
                caja = det[0].bounding_box  # origin_x, origin_y, width, height en pixeles
                mx, my = caja.width * args.margen, caja.height * args.margen
                x1 = max(0, int(round(caja.origin_x - mx)))
                y1 = max(0, int(round(caja.origin_y - my)))
                x2 = min(ancho_img, int(round(caja.origin_x + caja.width + mx)))
                y2 = min(alto_img, int(round(caja.origin_y + caja.height + my)))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    omitidas.append((persona, ruta.name, str(ruta), "caja de rostro invalida"))
                    continue

                destino = salida / ruta.relative_to(entrada)
                if destino.suffix.lower() == ".heic":
                    destino = destino.with_suffix(".jpg")
                destino.parent.mkdir(parents=True, exist_ok=True)
                recorte = Image.fromarray(rgb[y1:y2, x1:x2])
                if destino.suffix.lower() in {".jpg", ".jpeg"}:
                    recorte.save(destino, quality=95)
                else:
                    recorte.save(destino)
                guardadas += 1
            except Exception as e:
                omitidas.append((persona, ruta.name, str(ruta), f"error: {e}"))
            if i % 500 == 0:
                print(f"  {i}/{len(imagenes)} procesadas, {guardadas} guardadas, {len(omitidas)} omitidas")

    print(f"\nRecortes guardados en {salida}: {guardadas}")
    print(f"Omitidas: {len(omitidas)}")

    if omitidas:
        for _, _, ruta, motivo in omitidas:
            print(f"  {ruta}  [{motivo}]")
        reporte = REPORTS / "recortesOmitidos.csv"
        reporte.parent.mkdir(parents=True, exist_ok=True)
        with open(reporte, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["persona", "archivo", "ruta", "motivo"])
            w.writerows(omitidas)
        print(f"CSV de omitidas guardado en {reporte}")


if __name__ == "__main__":
    main()

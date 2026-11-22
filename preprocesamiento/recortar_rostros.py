"""Detecta y recorta el rostro de cada foto de datos/crudo -> datos/recortes.

Uso (desde la raíz del repo):
    python preprocesamiento/recortar_rostros.py
    python preprocesamiento/recortar_rostros.py --detector mediapipe --tam 224 --margen 0.2
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import (DATOS, RESULTADOS, crear_detector, guardar_imagen,  # noqa: E402
                   leer_imagen, listar_imagenes, recortar_con_margen)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrada", type=Path, default=DATOS / "crudo")
    ap.add_argument("--salida", type=Path, default=DATOS / "recortes")
    ap.add_argument("--detector", choices=["haar", "mediapipe"], default="haar")
    ap.add_argument("--tam", type=int, default=224, help="lado final del recorte en px")
    ap.add_argument("--margen", type=float, default=0.2, help="margen alrededor del rostro (fracción)")
    args = ap.parse_args()

    import cv2

    if not args.entrada.exists():
        sys.exit(f"No existe {args.entrada}. Pongan las fotos en datos/crudo/<persona>/")

    detectar = crear_detector(args.detector)
    bitacora = []
    resumen = {}

    for carpeta in sorted(p for p in args.entrada.iterdir() if p.is_dir()):
        conteo = {"ok": 0, "varios_rostros": 0, "sin_rostro": 0, "ilegible": 0}
        for ruta in listar_imagenes(carpeta):
            img = leer_imagen(ruta)
            if img is None:
                estado = "ilegible"
            else:
                cajas = detectar(img)
                if not cajas:
                    estado = "sin_rostro"
                else:
                    estado = "ok" if len(cajas) == 1 else "varios_rostros"
                    caja = max(cajas, key=lambda c: c[2] * c[3])  # el rostro más grande
                    recorte = recortar_con_margen(img, caja, args.margen)
                    recorte = cv2.resize(recorte, (args.tam, args.tam))
                    guardar_imagen(args.salida / carpeta.name / (ruta.stem + ".jpg"), recorte)
            conteo[estado] += 1
            if estado != "ok":
                bitacora.append((str(ruta), estado))
        resumen[carpeta.name] = conteo

    print(f"{'persona':<25}{'ok':>6}{'varios':>8}{'sin_rostro':>12}{'ilegible':>10}")
    for persona, c in resumen.items():
        print(f"{persona:<25}{c['ok']:>6}{c['varios_rostros']:>8}{c['sin_rostro']:>12}{c['ilegible']:>10}")

    RESULTADOS.mkdir(exist_ok=True)
    salida = RESULTADOS / "bitacora_recorte.csv"
    with open(salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["archivo", "estado"])
        escritor.writerows(bitacora)
    print(f"\nFotos con problemas (sin rostro, varios rostros, ilegibles): {len(bitacora)} -> {salida}")


if __name__ == "__main__":
    main()

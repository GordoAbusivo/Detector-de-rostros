"""Audita datos/crudo: fotos por persona, imágenes ilegibles y duplicados exactos.

Uso (desde la raíz del repo):
    python preprocesamiento/revisar_dataset.py
    python preprocesamiento/revisar_dataset.py --entrada datos/crudo
"""
import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import DATOS, RESULTADOS, leer_imagen, listar_imagenes  # noqa: E402

MIN_RECOMENDADO = 15


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrada", type=Path, default=DATOS / "crudo")
    args = ap.parse_args()

    if not args.entrada.exists():
        sys.exit(f"No existe {args.entrada}. Pongan las fotos en datos/crudo/<persona>/")

    filas = []
    hashes = defaultdict(list)
    for carpeta in sorted(p for p in args.entrada.iterdir() if p.is_dir()):
        total, ilegibles, tamanos = 0, 0, []
        for ruta in listar_imagenes(carpeta):
            total += 1
            hashes[hashlib.md5(ruta.read_bytes()).hexdigest()].append(ruta)
            img = leer_imagen(ruta)
            if img is None:
                ilegibles += 1
                print(f"  ILEGIBLE: {ruta}")
            else:
                tamanos.append(min(img.shape[:2]))
        filas.append({
            "persona": carpeta.name,
            "fotos": total,
            "ilegibles": ilegibles,
            "lado_min_px": min(tamanos) if tamanos else 0,
        })

    duplicados = [rutas for rutas in hashes.values() if len(rutas) > 1]
    for rutas in duplicados:
        print("  DUPLICADAS:", " == ".join(str(r) for r in rutas))
    dup_por_persona = defaultdict(int)
    for rutas in duplicados:
        for r in rutas[1:]:
            dup_por_persona[r.relative_to(args.entrada).parts[0]] += 1
    for fila in filas:
        fila["duplicadas"] = dup_por_persona[fila["persona"]]

    print(f"\n{'persona':<25}{'fotos':>7}{'ilegibles':>11}{'duplicadas':>12}{'lado_min':>10}")
    for f in filas:
        aviso = "  <- pocas fotos" if f["fotos"] < MIN_RECOMENDADO else ""
        print(f"{f['persona']:<25}{f['fotos']:>7}{f['ilegibles']:>11}{f['duplicadas']:>12}{f['lado_min_px']:>10}{aviso}")

    if filas:
        cantidades = [f["fotos"] for f in filas]
        print(f"\nPersonas: {len(filas)} | Fotos: {sum(cantidades)} | "
              f"min={min(cantidades)} max={max(cantidades)} (si la diferencia es grande, el dataset está desbalanceado)")

    RESULTADOS.mkdir(exist_ok=True)
    salida = RESULTADOS / "auditoria_dataset.csv"
    with open(salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["persona", "fotos", "ilegibles", "duplicadas", "lado_min_px"])
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()

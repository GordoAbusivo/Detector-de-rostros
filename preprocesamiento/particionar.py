"""Divide datos/recortes en train / val / test, por persona.

Uso (desde la raíz del repo):
    python preprocesamiento/particionar.py --bloque 3
    python preprocesamiento/particionar.py --train 0.7 --val 0.15 --semilla 42 --forzar

--bloque N: mantiene juntas N fotos consecutivas (por nombre de archivo) en la misma
partición. Si las fotos se tomaron en ráfaga, evita que fotos casi idénticas queden
en train y en test (fuga de datos que infla el accuracy).
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import DATOS, listar_imagenes  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrada", type=Path, default=DATOS / "recortes")
    ap.add_argument("--salida", type=Path, default=DATOS / "particiones")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--bloque", type=int, default=1, help="fotos consecutivas que se mantienen juntas")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--forzar", action="store_true", help="borra la salida si ya existe")
    args = ap.parse_args()

    if not args.entrada.exists():
        sys.exit(f"No existe {args.entrada}. Corran primero recortar_rostros.py")
    if args.salida.exists():
        if not args.forzar:
            sys.exit(f"{args.salida} ya existe. Usen --forzar para rehacerla.")
        shutil.rmtree(args.salida)

    rng = random.Random(args.semilla)
    totales = {"train": 0, "val": 0, "test": 0}

    print(f"{'persona':<25}{'train':>7}{'val':>6}{'test':>6}")
    for carpeta in sorted(p for p in args.entrada.iterdir() if p.is_dir()):
        archivos = listar_imagenes(carpeta)
        bloques = [archivos[i:i + args.bloque] for i in range(0, len(archivos), args.bloque)]
        n = len(bloques)
        if n < 3:
            print(f"{carpeta.name:<25} AVISO: solo {n} bloque(s); se omite (necesita >= 3)")
            continue

        rng.shuffle(bloques)
        n_val = max(1, round(n * args.val))
        n_test = max(1, round(n * (1 - args.train - args.val)))
        n_train = n - n_val - n_test
        reparto = {
            "train": bloques[:n_train],
            "val": bloques[n_train:n_train + n_val],
            "test": bloques[n_train + n_val:],
        }

        cuentas = {}
        for split, lista in reparto.items():
            destino = args.salida / split / carpeta.name
            destino.mkdir(parents=True, exist_ok=True)
            cuenta = 0
            for bloque in lista:
                for archivo in bloque:
                    shutil.copy2(archivo, destino / archivo.name)
                    cuenta += 1
            cuentas[split] = cuenta
            totales[split] += cuenta
        print(f"{carpeta.name:<25}{cuentas['train']:>7}{cuentas['val']:>6}{cuentas['test']:>6}")

    print(f"{'TOTAL':<25}{totales['train']:>7}{totales['val']:>6}{totales['test']:>6}")
    print(f"\nListo: {args.salida}")


if __name__ == "__main__":
    main()

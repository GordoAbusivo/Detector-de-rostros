"""Divide datos/recortes en train / val / test, por persona.

Dos modos (--modo):
  set   : reparte SETS COMPLETOS (set_01, set_02, ...) entre train/val/test. Es lo
          más limpio: las fotos de un mismo set son ráfagas casi idénticas, y así
          nunca quedan fotos parecidas en train y test. (Se usa solo si hay subcarpetas.)
  foto  : reparte fotos sueltas; con --bloque N mantiene juntas N fotos consecutivas.
  auto  : (por defecto) 'set' si las fotos están en subcarpetas, si no 'foto'.

Uso (desde la raíz del repo):
    python preprocesamiento/particionar.py
    python preprocesamiento/particionar.py --modo foto --bloque 3 --forzar
"""
import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import DATOS, listar_imagenes  # noqa: E402


def agrupar(carpeta, archivos, modo, bloque):
    """Devuelve una lista de grupos; cada grupo es una lista de (archivo, nombre_destino)."""
    por_set = defaultdict(list)
    for archivo in archivos:
        rel = archivo.relative_to(carpeta)
        por_set[rel.parent.as_posix()].append((archivo, rel))
    hay_sets = any(k != "." for k in por_set)

    if modo == "set" or (modo == "auto" and hay_sets):
        if not hay_sets:
            return None
        grupos = []
        for clave, items in sorted(por_set.items()):
            grupos.append([(a, rel.as_posix().replace("/", "_")) for a, rel in items])
        return grupos

    planas = [(a, rel.as_posix().replace("/", "_")) for a, rel in
              sorted((a, rel) for items in por_set.values() for a, rel in items)]
    return [planas[i:i + bloque] for i in range(0, len(planas), bloque)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", type=Path, default=DATOS / "recortes")
    ap.add_argument("--salida", type=Path, default=DATOS / "particiones")
    ap.add_argument("--modo", choices=["auto", "set", "foto"], default="auto")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--bloque", type=int, default=1, help="(modo foto) fotos consecutivas que se mantienen juntas")
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

    print(f"{'persona':<25}{'grupos':>7}{'train':>8}{'val':>6}{'test':>6}")
    for carpeta in sorted(p for p in args.entrada.iterdir() if p.is_dir()):
        grupos = agrupar(carpeta, listar_imagenes(carpeta), args.modo, args.bloque)
        if grupos is None:
            sys.exit(f"{carpeta.name}: --modo set pero no hay subcarpetas de sets.")
        n = len(grupos)
        if n < 3:
            print(f"{carpeta.name:<25} AVISO: solo {n} grupo(s); se omite (necesita >= 3)")
            continue

        rng.shuffle(grupos)
        n_val = max(1, round(n * args.val))
        n_test = max(1, round(n * (1 - args.train - args.val)))
        n_train = n - n_val - n_test
        reparto = {
            "train": grupos[:n_train],
            "val": grupos[n_train:n_train + n_val],
            "test": grupos[n_train + n_val:],
        }

        cuentas = {}
        for split, lista in reparto.items():
            destino = args.salida / split / carpeta.name
            destino.mkdir(parents=True, exist_ok=True)
            cuenta = 0
            for grupo in lista:
                for archivo, nombre in grupo:
                    shutil.copy2(archivo, destino / nombre)
                    cuenta += 1
            cuentas[split] = cuenta
            totales[split] += cuenta
        print(f"{carpeta.name:<25}{n:>7}{cuentas['train']:>8}{cuentas['val']:>6}{cuentas['test']:>6}")

    print(f"{'TOTAL':<25}{'':>7}{totales['train']:>8}{totales['val']:>6}{totales['test']:>6}")
    print(f"\nListo: {args.salida}")


if __name__ == "__main__":
    main()

"""
Excluye de la muestra de trabajo las imagenes ATIPICAS: las que se parecen poco al centro de su persona
(similitud coseno < 0.4), por ejemplo alineaciones fallidas, muy borrosas o de otra persona. Procedimiento del
Anexo B de GUIA_PIPELINE.txt; antes vivia en una celda de analisis_embeddings.ipynb.

Que hace (solo con --ejecutar; sin esa opcion es una SIMULACION que no toca nada):
  1. Guarda una copia de vectors/antelope.npy, bufalo.npy, metadatos.csv y reports/muestra_balanceada.csv en
     respaldo_atipicas_<fecha-hora>/ (carpeta ignorada por git).
  2. Borra la imagen de data/muestra/ y la alineada de data/arcface/.
  3. Quita esas MISMAS filas de antelope.npy, bufalo.npy, metadatos.csv y muestra_balanceada.csv.
  4. Anota lo excluido en reports/imagenes_excluidas_atipicas.csv y lo que se conservo en
     reports/atipicas_conservadas.txt (una ruta por linea).
data/clean, data/face y datasets/ NO se tocan: la imagen sigue existiendo, solo sale de la muestra.

Al recalcular los centros pueden aparecer atipicas nuevas: repite hasta que no haya mas.
Las imagenes de reports/atipicas_conservadas.txt se conservan siempre (decisiones ya tomadas).

Uso:
    python src/limpiar_atipicas.py                                   # lista las atipicas (simulacion)
    python src/limpiar_atipicas.py --conservar persona/set_01/foto.jpg   # ademas conserva esas
    python src/limpiar_atipicas.py --ejecutar
Desde un notebook (galeria numerada con '#'):
    from limpiar_atipicas import limpiar_atipicas, cargar_vectores
    limpiar_atipicas(atipicas, conservar_ids=[412, 1380], ejecutar=True)
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from comun import ARCFACE, MUESTRA, RAIZ, REPORTS, VECTORS

UMBRAL_ATIPICA = 0.4
CONSERVADAS = REPORTS / "atipicas_conservadas.txt"
LOG = REPORTS / "imagenes_excluidas_atipicas.csv"
COLUMNAS_LOG = ["persona", "archivo", "ruta", "set", "orientacion", "sim_centroide"]


def cargar_vectores():
    """(X, datos): vectores de antelope normalizados (L2) y metadatos con la similitud al centro de su persona."""
    X = normalize(np.load(VECTORS / "antelope.npy"), norm="l2")
    datos = pd.read_csv(VECTORS / "metadatos.csv")
    datos["sim_centroide"] = similitud_al_centro(X, datos)
    return X, datos


def similitud_al_centro(X, datos):
    centroides = {p: normalize(X[(datos["persona"] == p).to_numpy()].mean(axis=0, keepdims=True))[0]
                  for p in datos["persona"].unique()}
    return np.array([X[i] @ centroides[p] for i, p in enumerate(datos["persona"])], dtype=np.float32)


def buscar_atipicas(datos, umbral=UMBRAL_ATIPICA):
    """Imagenes con similitud al centro menor al umbral, de la mas atipica a la menos. El indice es la fila."""
    return datos[datos["sim_centroide"] < umbral].sort_values("sim_centroide")


def leer_conservadas():
    return {l.strip() for l in CONSERVADAS.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if CONSERVADAS.exists() else set()


def limpiar_atipicas(atipicas, conservar_ids=(), conservar_rutas=(), ejecutar=False, usar_archivo=False):
    """Borra de la muestra las `atipicas` (DataFrame de buscar_atipicas) salvo las que se quieran conservar.

    conservar_ids    : numeros '#' (indice de `atipicas`), como los que muestra la galeria del notebook.
    conservar_rutas  : rutas 'persona/set_01/foto.jpg'.
    usar_archivo     : tambien conserva las rutas de reports/atipicas_conservadas.txt.
    ejecutar=False   : simulacion (no toca nada). Devuelve un dict con lo que se borro / se borraria.
    """
    if atipicas is None:
        raise RuntimeError("Ejecuta primero la galeria de atipicas: sus numeros '#' se usan aqui.")
    desconocidos = [i for i in conservar_ids if i not in atipicas.index]
    if desconocidos:
        raise ValueError(f"Estos numeros no estan en la galeria: {desconocidos}")
    rutas_conservar = set(conservar_rutas)
    if usar_archivo:
        rutas_conservar |= leer_conservadas()
    if len(conservar_ids):
        rutas_conservar |= set(atipicas.loc[list(conservar_ids), "ruta"])
    conservadas = atipicas[atipicas["ruta"].isin(rutas_conservar)]
    a_borrar = atipicas[~atipicas["ruta"].isin(rutas_conservar)]

    print(f"En la galeria: {len(atipicas)} | se CONSERVAN: {len(conservadas)} | se BORRAN: {len(a_borrar)}\n")
    print("Se conservan:")
    print(conservadas[["persona", "ruta", "orientacion", "sim_centroide"]].round(2).to_string()
          if len(conservadas) else "  (ninguna)")
    print("\nSe borran, por persona:")
    print(a_borrar["persona"].value_counts().to_string() if len(a_borrar) else "  (ninguna)")
    resultado = {"borradas": len(a_borrar), "conservadas": len(conservadas), "ejecutado": False}
    if not ejecutar:
        print("\n[SIMULACION] No se borro nada. Usa --ejecutar (o ejecutar=True) para borrar de verdad.")
        return resultado
    if not len(a_borrar):
        print("\nNo hay nada que borrar.")
        return resultado

    # --- comprobaciones: vectores, metadatos y la lista de atipicas deben estar alineados fila a fila
    A, B = np.load(VECTORS / "antelope.npy"), np.load(VECTORS / "bufalo.npy")
    meta = pd.read_csv(VECTORS / "metadatos.csv")
    muestra_csv = pd.read_csv(REPORTS / "muestra_balanceada.csv")
    assert len(A) == len(B) == len(meta), "antelope.npy, bufalo.npy y metadatos.csv no tienen las mismas filas"
    assert (meta.loc[a_borrar.index, "ruta"].values == a_borrar["ruta"].values).all(), \
        "los numeros '#' ya no corresponden a vectors/ (cambio algo): vuelve a calcular las atipicas"

    # --- respaldo rapido de los 4 archivos que se modifican (~50 MB)
    resp = RAIZ / f"respaldo_atipicas_{datetime.now():%Y%m%d_%H%M%S}"
    resp.mkdir(parents=True)
    for f in [VECTORS / "antelope.npy", VECTORS / "bufalo.npy", VECTORS / "metadatos.csv",
              REPORTS / "muestra_balanceada.csv"]:
        shutil.copy2(f, resp / f.name)

    # --- 1) imagenes de la muestra y las alineadas (data/face, data/clean y datasets NO se tocan)
    n_m = n_a = 0
    for ruta in a_borrar["ruta"]:
        pm, pa = MUESTRA / ruta, ARCFACE / Path(ruta).with_suffix(".png")
        n_m += pm.exists()
        n_a += pa.exists()
        pm.unlink(missing_ok=True)
        pa.unlink(missing_ok=True)

    # --- 2) vectores y metadatos: se quitan las MISMAS filas en los tres
    mantener = ~meta.index.isin(a_borrar.index)
    np.save(VECTORS / "antelope.npy", A[mantener])
    np.save(VECTORS / "bufalo.npy", B[mantener])
    meta[mantener].to_csv(VECTORS / "metadatos.csv", index=False)

    # --- 3) muestra_balanceada.csv
    muestra_csv[~muestra_csv["ruta"].isin(a_borrar["ruta"])].to_csv(REPORTS / "muestra_balanceada.csv", index=False)

    # --- 4) registros acumulativos
    a_borrar[COLUMNAS_LOG].to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    if len(conservadas):
        CONSERVADAS.write_text("\n".join(sorted(leer_conservadas() | set(conservadas["ruta"]))) + "\n", encoding="utf-8")

    print(f"\nBorradas: {len(a_borrar)} (muestra: {n_m} archivos, arcface: {n_a} archivos)")
    print(f"Quedan {int(mantener.sum())} imagenes | respaldo en {resp}")
    resultado.update(ejecutado=True, respaldo=resp, quedan=int(mantener.sum()))
    return resultado


def main():
    ap = argparse.ArgumentParser(description="Excluye de la muestra las imagenes atipicas (similitud al centro < umbral).")
    ap.add_argument("--umbral", type=float, default=UMBRAL_ATIPICA)
    ap.add_argument("--conservar", nargs="*", default=[], metavar="RUTA",
                    help="rutas persona/set_XX/foto.jpg que NO se deben borrar")
    ap.add_argument("--ejecutar", action="store_true", help="borrar de verdad (sin esto es una simulacion)")
    args = ap.parse_args()

    _, datos = cargar_vectores()
    atipicas = buscar_atipicas(datos, args.umbral)
    print(f"{len(atipicas)} imagenes con similitud < {args.umbral}\n")
    if not len(atipicas):
        return
    resultado = limpiar_atipicas(atipicas, conservar_rutas=args.conservar, ejecutar=args.ejecutar, usar_archivo=True)
    if resultado["ejecutado"]:
        _, datos = cargar_vectores()
        quedan = int((datos["sim_centroide"] < args.umbral).sum())
        print(f"Con los centros recalculados hay {quedan} con similitud < {args.umbral} "
              "(las de atipicas_conservadas.txt se conservan). Vuelve a correr este script para revisarlas.")


if __name__ == "__main__":
    main()

"""
Arma data/particiones/{train,val,test}/<persona>/*.png para la CNN, con LA MISMA division por sets que usan
Ridge/SVM/KNN (vectors/division_sets.csv). Asi la comparacion entre modelos es justa: mismas fotos de prueba.

Usa las caras ALINEADAS de data/arcface/ (112x112), las mismas que ve ArcFace.

Entrada : vectors/metadatos.csv + vectors/division_sets.csv + data/arcface/
Salida  : data/particiones/ (copias; no toca los originales)

Una persona con pocos sets puede quedar SIN validacion (ver dividir_sets.py) y la CNN necesita al menos una foto de
cada persona en train, val y test. Si pasa, el script lo avisa y se detiene; con --val-de-train mueve a validacion el
set mas chico de train de esa persona (se mueve un set completo, nunca fotos sueltas de una misma sesion).

Uso:
    python src/preparar_particiones.py
    python src/preparar_particiones.py --val-de-train
"""
import argparse
import shutil
import sys

import pandas as pd

from comun import PARTICIONES, VECTORS
from extraer_vectores import ruta_arcface


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-de-train", action="store_true", help="dar validacion a quien no la tenga, usando un set de train")
    a = ap.parse_args()

    datos = pd.read_csv(VECTORS / "metadatos.csv")
    division = pd.read_csv(VECTORS / "division_sets.csv")
    if not (division["ruta"].values == datos["ruta"].values).all():
        sys.exit("ERROR: division_sets.csv no coincide con metadatos.csv. Vuelve a correr src/dividir_sets.py.")
    datos["split"] = division["split"].values

    sin = sorted(p for p in datos["persona"].unique() if not ((datos["persona"] == p) & (datos["split"] == "val")).any())
    if sin:
        if not a.val_de_train:
            sys.exit(f"ERROR: {len(sin)} persona(s) sin fotos de validacion: {', '.join(sin)}.\n"
                     "Corre de nuevo con --val-de-train (mueve un set de train a validacion).")
        for p in sin:
            g = datos[(datos["persona"] == p) & (datos["split"] == "train")]
            tam = g.groupby("set").size().sort_values()
            if len(tam) < 2:
                sys.exit(f"ERROR: {p} tiene menos de 2 sets en train; no se puede sacar validacion sin dejarla sin entrenamiento.")
            datos.loc[(datos["persona"] == p) & (datos["set"] == tam.index[0]), "split"] = "val"
            print(f"  {p}: el set {tam.index[0]} ({tam.iloc[0]} fotos) pasa de train a val")

    if PARTICIONES.exists():
        shutil.rmtree(PARTICIONES)
    for fila in datos.itertuples():
        origen = ruta_arcface(fila.ruta)
        destino = PARTICIONES / fila.split / fila.persona / f"{fila.set}__{origen.name}"   # el set en el nombre evita choques
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origen, destino)

    print(pd.crosstab(datos["persona"], datos["split"]).reindex(columns=["train", "val", "test"], fill_value=0).to_string())
    print("\nTotal:", datos["split"].value_counts().to_dict(), "\nListo en", PARTICIONES)


if __name__ == "__main__":
    main()

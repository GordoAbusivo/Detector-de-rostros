"""
Divide la muestra en entrenamiento / validacion / prueba por SETS completos (60 / 20 / 20) y guarda
vectors/division_sets.csv (ruta, split). Antes vivia en las celdas 12 y 14 de analisis_embeddings.ipynb.

Un set es una sesion de fotos (misma ropa, fondo y luz). Para cada persona se elige la combinacion de sets
cuyo total de imagenes se acerca mas al 20% (prueba) y luego al 20% (validacion); el resto es entrenamiento.
Asi una sesion nunca queda repartida entre entrenamiento y prueba (ver 7.17 y 7.18 de GUIA_PIPELINE.txt).
La eleccion es determinista (sin azar): ante un empate gana la primera combinacion.

Entrada : vectors/metadatos.csv (persona, ruta, set, ...), el MISMO que usan los vectores
Salida  : vectors/division_sets.csv

Uso:
    python src/dividir_sets.py
Desde un notebook:  from dividir_sets import asignar_split; datos["split"] = asignar_split(datos)
"""
from itertools import combinations

import numpy as np
import pandas as pd

from comun import VECTORS

FRAC_VAL, FRAC_TEST = 0.20, 0.20
MIN_SETS_PARA_VALIDACION = 4        # con menos sets en el resto no se saca validacion


def mejor_combinacion(tam, frac, total, dejar):
    """Sets cuya suma de imagenes mas se acerca a frac*total, dejando al menos `dejar` sets fuera."""
    mejor, mejor_err = (), np.inf
    for k in range(1, len(tam) - dejar + 1):
        for c in combinations(tam.index, k):
            err = abs(tam[list(c)].sum() - frac * total)
            if err < mejor_err:
                mejor, mejor_err = c, err
    return mejor


def asignar_split(datos: pd.DataFrame) -> pd.Series:
    """Serie 'train' / 'val' / 'test' alineada con `datos` (necesita las columnas persona y set)."""
    split = pd.Series("train", index=datos.index)
    for persona, g in datos.groupby("persona"):
        total = len(g)
        tam = g.groupby("set").size()
        test = mejor_combinacion(tam, FRAC_TEST, total, dejar=2)
        split[(datos["persona"] == persona) & datos["set"].isin(test)] = "test"
        resto = tam.drop(list(test))
        if len(resto) >= MIN_SETS_PARA_VALIDACION:
            val = mejor_combinacion(resto, FRAC_VAL, total, dejar=2)
            split[(datos["persona"] == persona) & datos["set"].isin(val)] = "val"
    return split


def guardar_division(datos: pd.DataFrame):
    """Escribe vectors/division_sets.csv (ruta, split) a partir de un DataFrame con la columna split."""
    VECTORS.mkdir(exist_ok=True)
    salida = VECTORS / "division_sets.csv"
    datos[["ruta", "split"]].to_csv(salida, index=False)
    return salida


def main():
    datos = pd.read_csv(VECTORS / "metadatos.csv")
    datos["split"] = asignar_split(datos)
    tabla = pd.crosstab(datos["persona"], datos["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    print(tabla.to_string())
    print("\nTotal:", datos["split"].value_counts().to_dict())
    print("CSV guardado en", guardar_division(datos))


if __name__ == "__main__":
    main()

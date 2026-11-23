"""
Evalua ArcFace + clasificador (Ridge, SVM, KNN) en la MISMA particion de prueba que la CNN y guarda las metricas en
resultados/<nombre>/ con guardar_resultados() (mismo formato que entrenar_cnn.py).

IMPORTANTE: models/ridge_antelope.joblib se entreno con TODAS las fotos (entrenado_con = "todo"), asi que NO se puede
evaluar con la particion de prueba (ya las vio). Aqui cada clasificador se entrena de nuevo SOLO con train (y val se
usa solo para reportar) y se mide en test.

Entrada : vectors/antelope.npy + vectors/metadatos.csv + vectors/division_sets.csv
Salida  : resultados/arcface_ridge, resultados/arcface_svm, resultados/arcface_knn

Uso:
    python src/evaluar_embeddings.py
    python src/evaluar_embeddings.py --modelos ridge svm
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from comun import VECTORS, guardar_resultados
from exportar_modelos import crear_modelos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", choices=list(crear_modelos()), default=list(crear_modelos()))
    a = ap.parse_args()

    X = normalize(np.load(VECTORS / "antelope.npy"), norm="l2")
    datos = pd.read_csv(VECTORS / "metadatos.csv")
    division = pd.read_csv(VECTORS / "division_sets.csv")
    assert (division["ruta"].values == datos["ruta"].values).all(), "division_sets.csv no coincide con metadatos.csv"
    y = datos["persona"].to_numpy()
    split = division["split"].to_numpy()
    clases = sorted(np.unique(y))
    train, test = split == "train", split == "test"
    print(f"personas={len(clases)} train={train.sum()} val={(split == 'val').sum()} test={test.sum()}")

    modelos = crear_modelos()
    for nombre in a.modelos:
        pred = modelos[nombre].fit(X[train], y[train]).predict(X[test])
        guardar_resultados(f"arcface_{nombre}", list(y[test]), list(pred), clases, datos.loc[test, "ruta"].tolist())


if __name__ == "__main__":
    main()

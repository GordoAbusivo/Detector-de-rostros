"""
Entrena y exporta los clasificadores (Ridge, SVM, KNN) como PAQUETES que usa src/predecir.py:
    models/ridge_antelope.joblib, models/svm_antelope.joblib, models/knn_antelope.joblib
Cada paquete es un dict con "modelo", "clases", "centroides" (el centro de cada persona), con que se entreno y la
version de scikit-learn. Antes vivia en celdas de analisis_embeddings.ipynb (y en el Anexo A de la guia).

Entrada: vectors/antelope.npy + vectors/metadatos.csv (y vectors/division_sets.csv con --solo-train)

Uso:
    python src/exportar_modelos.py                       # entrena con TODAS las imagenes (modelo final)
    python src/exportar_modelos.py --solo-train          # solo con las de entrenamiento, como se evaluo
    python src/exportar_modelos.py --modelos ridge       # solo uno
Desde un notebook:
    from exportar_modelos import crear_modelos, exportar
    ridge = crear_modelos()["ridge"]; exportar()
"""
import argparse

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

from comun import MODELOS, VECTORS


def crear_modelos():
    """Los tres clasificadores con los hiperparametros del proyecto (una sola definicion para notebook y script)."""
    return {
        "ridge": RidgeClassifier(alpha=1.0),
        "svm": SVC(kernel="linear", C=1.0),
        "knn": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
    }


def exportar(entrenar_con_todo=True, nombres=None):
    """Entrena y guarda los paquetes en models/. Devuelve {nombre: ruta}."""
    X = normalize(np.load(VECTORS / "antelope.npy"), norm="l2")
    datos = pd.read_csv(VECTORS / "metadatos.csv")
    y = datos["persona"].to_numpy()
    clases = sorted(np.unique(y))
    centroides = np.stack([normalize(X[y == p].mean(axis=0, keepdims=True))[0] for p in clases])
    if entrenar_con_todo:
        X_fit, y_fit = X, y
    else:
        division = pd.read_csv(VECTORS / "division_sets.csv")
        assert (division["ruta"].values == datos["ruta"].values).all(), "division_sets.csv no coincide con metadatos.csv"
        train = (division["split"] == "train").to_numpy()
        X_fit, y_fit = X[train], y[train]

    MODELOS.mkdir(exist_ok=True)
    rutas = {}
    modelos = crear_modelos()
    for nombre in (nombres or modelos):
        modelo = modelos[nombre].fit(X_fit, y_fit)
        rutas[nombre] = MODELOS / f"{nombre}_antelope.joblib"
        joblib.dump({"modelo": modelo, "clases": [str(c) for c in clases], "centroides": centroides,
                     "entrenado_con": "todo" if entrenar_con_todo else "train",
                     "n_imagenes": len(X_fit), "sklearn": sklearn.__version__}, rutas[nombre])
        print(f"{rutas[nombre].name} guardado | {len(clases)} personas | {len(X_fit)} imagenes")
    return rutas


def main():
    ap = argparse.ArgumentParser(description="Exporta los clasificadores como paquetes para predecir.py.")
    ap.add_argument("--solo-train", action="store_true", help="entrenar solo con las imagenes de train")
    ap.add_argument("--modelos", nargs="+", choices=list(crear_modelos()), default=None)
    args = ap.parse_args()
    exportar(entrenar_con_todo=not args.solo_train, nombres=args.modelos)


if __name__ == "__main__":
    main()

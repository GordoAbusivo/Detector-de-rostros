"""Modelo A: rostros recortados + características HOG + clasificador clásico.

Prueba varios clasificadores, elige el mejor según accuracy en VALIDACIÓN y
lo evalúa UNA vez en PRUEBA.

Uso (desde la raíz del repo):
    python entrenamiento/modelo_clasico/entrenar_clasico.py
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comun import (RESULTADOS, caracteristicas_hog, guardar_resultados,  # noqa: E402
                   leer_imagen, listar_split)


def cargar(split, tam):
    rutas, etiquetas = listar_split(split)
    X, y, usadas = [], [], []
    for ruta, etiqueta in zip(rutas, etiquetas):
        img = leer_imagen(ruta)
        if img is None:
            print(f"  (se omite imagen ilegible: {ruta})")
            continue
        X.append(caracteristicas_hog(img, tam))
        y.append(etiqueta)
        usadas.append(ruta)
    return np.array(X), np.array(y), usadas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tam", type=int, default=128, help="lado de la imagen para HOG")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    print("Extrayendo características HOG...")
    X_tr, y_tr, _ = cargar("train", args.tam)
    X_va, y_va, _ = cargar("val", args.tam)
    X_te, y_te, rutas_te = cargar("test", args.tam)
    clases = sorted(set(y_tr))
    print(f"train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}  personas={len(clases)}  dim={X_tr.shape[1]}")

    candidatos = {
        "svm_lineal": SVC(kernel="linear", C=1.0),
        "svm_rbf": SVC(kernel="rbf", C=10, gamma="scale"),
        "knn": KNeighborsClassifier(n_neighbors=3, metric="cosine"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=args.semilla, n_jobs=-1),
    }

    mejor_nombre, mejor_acc, mejor_modelo = None, -1.0, None
    print(f"\n{'algoritmo':<16}{'acc. validación':>16}")
    for nombre, modelo in candidatos.items():
        modelo.fit(X_tr, y_tr)
        acc = float(np.mean(modelo.predict(X_va) == y_va))
        print(f"{nombre:<16}{acc:>16.4f}")
        if acc > mejor_acc:
            mejor_nombre, mejor_acc, mejor_modelo = nombre, acc, modelo

    print(f"\nMejor en validación: {mejor_nombre} ({mejor_acc:.4f}). Evaluando en prueba...")
    y_pred = mejor_modelo.predict(X_te)
    guardar_resultados(f"modelo_clasico_{mejor_nombre}", y_te, y_pred, clases, rutas_te)

    destino = RESULTADOS / f"modelo_clasico_{mejor_nombre}" / "mejor_modelo"
    destino.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": mejor_modelo, "clases": clases, "tam": args.tam}, destino / "modelo.joblib")
    print(f"Modelo guardado en {destino / 'modelo.joblib'} (no se sube a GitHub)")


if __name__ == "__main__":
    main()

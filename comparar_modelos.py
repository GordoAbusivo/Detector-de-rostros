"""Compara los modelos: lee resultados/*/metricas.json y genera una tabla y una gráfica.

Uso (desde la raíz del repo):
    python reporte/comparar_modelos.py
Salida: resultados/comparacion.md y resultados/comparacion.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import RESULTADOS  # noqa: E402

METRICAS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
ETIQUETAS = ["Accuracy", "Precisión", "Recall", "F1"]


def main():
    resultados = []
    for archivo in sorted(RESULTADOS.glob("*/metricas.json")):
        resultados.append(json.loads(archivo.read_text(encoding="utf-8")))
    if not resultados:
        sys.exit("No hay resultados todavía. Corran primero los scripts de entrenamiento.")

    lineas = ["| Modelo | n (prueba) | " + " | ".join(ETIQUETAS) + " |",
              "|---|---|" + "---|" * len(ETIQUETAS)]
    for r in resultados:
        lineas.append(f"| {r['modelo']} | {r['n_prueba']} | " + " | ".join(f"{r[m]:.3f}" for m in METRICAS) + " |")
    (RESULTADOS / "comparacion.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    ancho = 0.8 / len(resultados)
    plt.figure(figsize=(8, 4.5))
    for i, r in enumerate(resultados):
        posiciones = [j + i * ancho for j in range(len(METRICAS))]
        plt.bar(posiciones, [r[m] for m in METRICAS], width=ancho, label=r["modelo"])
    plt.xticks([j + 0.4 - ancho / 2 for j in range(len(METRICAS))], ETIQUETAS)
    plt.ylim(0, 1.05)
    plt.ylabel("Valor")
    plt.title("Comparación de modelos en el conjunto de prueba")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTADOS / "comparacion.png", dpi=150)

    print("\n".join(lineas))
    print(f"\nGuardado en {RESULTADOS / 'comparacion.md'} y {RESULTADOS / 'comparacion.png'}")


if __name__ == "__main__":
    main()

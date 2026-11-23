"""
Junta resultados/*/metricas.json (ArcFace+Ridge/SVM/KNN y las CNN) en una sola tabla ordenada por F1 macro.
Guarda resultados/comparacion.md (se puede subir a GitHub: solo numeros agregados) y la imprime.

Uso:  python src/comparar_modelos.py
"""
import json

from comun import RESULTADOS


def main():
    filas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTADOS.glob("*/metricas.json"))]
    if not filas:
        raise SystemExit("No hay resultados/*/metricas.json. Corre evaluar_embeddings.py y/o entrenar_cnn.py primero.")
    filas.sort(key=lambda m: m["f1_macro"], reverse=True)
    tabla = ["| Modelo | Accuracy | F1 macro | F1 ponderado | Fotos de prueba |", "|---|---|---|---|---|"]
    tabla += [f"| {m['modelo']} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} | {m['f1_ponderado']:.4f} | {m['n_prueba']} |"
              for m in filas]
    texto = "\n".join(tabla)
    (RESULTADOS / "comparacion.md").write_text(texto + "\n", encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()

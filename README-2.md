# Equipo de Entrenamiento (4 personas)

**Meta:** entrenar dos modelos sobre **las mismas particiones** y dejar sus resultados en `resultados/`.

| Integrante | Modelo | Nombre |
|---|---|---|
| 1 y 2 | **A:** rostros recortados + algoritmo clásico (HOG + SVM / KNN / Random Forest) | |
| 3 y 4 | **B:** transformer (ViT con fine-tuning) | |

Todos los comandos se corren desde la **raíz del repo**.

## Reglas para que la comparación sea justa
1. Ambos modelos usan `datos/particiones/` tal cual (no re-partan por su cuenta).
2. **Entrenen y ajusten con `train` y `val`. `test` se evalúa una sola vez, al final.** Si ajustan viendo `test`, los resultados no valen.
3. Guarden los resultados con `comun.guardar_resultados(...)` (los scripts ya lo hacen): así el equipo de reporte los lee sin sorpresas.
4. Semilla 42.

## Modelo A — clásico (personas 1 y 2)
```bash
python entrenamiento/modelo_clasico/entrenar_clasico.py
```
Extrae HOG, prueba SVM lineal, SVM RBF, KNN y Random Forest, elige el mejor por accuracy en `val` y lo evalúa en `test`.

Ideas de mejora (repártanse):
- Persona 1: afinar hiperparámetros (`GridSearchCV` en C, gamma, k) y probar PCA antes del clasificador.
- Persona 2: probar otras características (LBP, o embeddings de una red preentrenada como FaceNet) y comparar contra HOG.

## Modelo B — transformer (personas 3 y 4)
```bash
python entrenamiento/transformer/entrenar_vit.py --epocas 10
```
Fine-tuning de `google/vit-base-patch16-224-in21k`. Entrenar un transformer desde cero con tan pocas fotos no funciona; por eso se parte de uno preentrenado.

- **Sin GPU:** usen Google Colab o Kaggle (GPU gratis). Suban el repo y `datos/particiones/` (por Drive), instalen `requirements.txt` y corran el script.
- Persona 3: correr el entrenamiento base y registrar curvas de pérdida/accuracy por época.
- Persona 4: probar variaciones (más épocas, `--lr`, congelar capas, más aumentos de datos) y reportar cuál ayuda.

## Entregable
Carpetas `resultados/modelo_clasico_<algoritmo>/` y `resultados/vit/` con `metricas.json`, `matriz_confusion.csv`, `reporte_por_clase.txt` y `predicciones.csv`.

# Equipo de Reporte (3 personas)

**Meta:** entregar el reporte completo, la comparación de modelos y la demo/presentación. **Pueden empezar hoy**, sin esperar a los otros equipos.

| Integrante | Rol | Nombre |
|---|---|---|
| 1 | Redacción: introducción, marco teórico, metodología | |
| 2 | Resultados y métricas: comparación, gráficas, análisis de errores, conclusiones | |
| 3 | Demo, presentación y revisión final | |

El esqueleto del reporte está en [`estructura_reporte.md`](estructura_reporte.md).

## Persona 1 — Redacción
- Introducción y objetivo, marco teórico (detección de rostros, HOG, SVM/KNN/RF, ViT, métricas) y metodología.
- Pedir a preprocesamiento su sección de dataset; pedir a entrenamiento la descripción de cada modelo y sus hiperparámetros.
- Llevar las referencias (formato que pida el profesor).

## Persona 2 — Resultados y métricas
```bash
python reporte/comparar_modelos.py
```
- Genera `resultados/comparacion.md` (tabla) y `resultados/comparacion.png` (gráfica) a partir de `resultados/*/metricas.json`.
- Armar las matrices de confusión, ver qué personas se confunden más y mirar las fotos mal clasificadas (`predicciones.csv`).
- Escribir resultados, discusión y conclusiones: ¿qué modelo ganó?, ¿por qué?, ¿qué limitaciones tiene el dataset?

## Persona 3 — Demo, presentación y revisión final
```bash
python reporte/demo_webcam.py --modelo resultados/modelo_clasico_<algoritmo>/mejor_modelo/modelo.joblib
```
- La demo usa la cámara y el Modelo A. Pruébenla con el modelo real en cuanto entrenamiento lo tenga.
- Preparar las diapositivas y el guion de la exposición.
- Revisión final: ortografía, formato, figuras numeradas, referencias, que los números del texto coincidan con las tablas.

## Mientras esperan resultados
Escriban todo lo que no dependa de ellos: introducción, marco teórico, metodología, y dejen las tablas y gráficas con marcadores `[PENDIENTE]`.

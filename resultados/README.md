# Resultados

Aquí guardan los scripts de entrenamiento sus resultados (se llenan solos):

```
resultados/
├── auditoria_dataset.csv        # preprocesamiento/revisar_dataset.py
├── bitacora_recorte.csv         # preprocesamiento/recortar_rostros.py
├── modelo_clasico_<algoritmo>/  # entrenamiento/modelo_clasico/entrenar_clasico.py
├── vit/                         # entrenamiento/transformer/entrenar_vit.py
├── comparacion.md / .png        # reporte/comparar_modelos.py
```

Cada carpeta de modelo contiene `metricas.json`, `matriz_confusion.csv`, `reporte_por_clase.txt` y `predicciones.csv`.
Estos archivos **sí se suben a GitHub** (son livianos) para que el equipo de reporte los use.

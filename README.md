# Detector de rostros del grupo

Proyecto de Machine Learning (ESCOM-IPN). Objetivo: reconocer los rostros de los integrantes del grupo y comparar dos enfoques:

- **Modelo A:** rostros recortados + algoritmo clásico (HOG + SVM / KNN).
- **Modelo B:** transformer (ViT) con fine-tuning.

## Equipos (11 personas)

| Equipo | Carpeta | Personas | Responsabilidad |
|---|---|---|---|
| Preprocesamiento | `preprocesamiento/` | 4 | Limpiar dataset, recortar rostros, particionar train/val/test |
| Entrenamiento | `entrenamiento/` | 4 | Modelo A (2 personas) y Modelo B (2 personas) |
| Reporte | `reporte/` | 3 | Reporte, métricas comparativas, demo y presentación |

Rellenen los nombres en el README de cada carpeta.

## Estructura

```
proyecto-rostros/
├── datos/                 # NO se sube al repo (ver datos/README.md)
├── preprocesamiento/
├── entrenamiento/
├── reporte/
├── requirements.txt
├── CONTRIBUIR.md
└── .gitignore
```

## Flujo general

1. Preprocesamiento deja listo `datos/particiones/{train,val,test}/<persona>/*.jpg`.
2. Entrenamiento entrena ambos modelos sobre **las mismas particiones** y guarda predicciones en `resultados/`.
3. Reporte toma `resultados/` y compara métricas.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Fechas de control (completar)

- [ ] Dataset limpio y particionado: ____
- [ ] Primeros resultados de ambos modelos: ____
- [ ] Comparación final: ____
- [ ] Reporte completo: ____
- [ ] Entrega: ____

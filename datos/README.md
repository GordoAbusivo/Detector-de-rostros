# Datos

Esta carpeta **no se sube a GitHub**: son fotos de personas reales. Compártanla por Drive y cada quien la descomprime aquí.

## Estructura esperada

```
datos/
├── crudo/            # fotos originales: crudo/<nombre_persona>/foto1.jpg
├── recortes/         # salida de recortar_rostros.py: recortes/<persona>/*.jpg
└── particiones/      # salida de particionar.py
    ├── train/<persona>/*.jpg
    ├── val/<persona>/*.jpg
    └── test/<persona>/*.jpg
```

## Convenciones
- Una carpeta por persona, con un nombre corto y sin espacios ni acentos (`ana_lopez`, no `Ana López`).
- El nombre de la carpeta es la etiqueta (la clase) que predicen los modelos.
- Para trabajar con una muestra pequeña mientras el dataset no está listo, copien 2-3 personas a `crudo/`.

## Privacidad
Las fotos son de compañeros. Úsenlas solo para este proyecto y no las publiquen ni las suban a servicios públicos.

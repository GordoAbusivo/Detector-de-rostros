"""
Lee el ancho y alto (px) de cada recorte de data/face (corrigiendo la rotacion EXIF) y los guarda en
reports/tam_images_face.csv con persona, archivo, ruta, ancho, alto.

Uso: python src/tamxImagen.py
"""
import pandas as pd
from PIL import Image

from comun import FACE, REPORTS, listar_imagenes

filas = []
for ruta in listar_imagenes(FACE, ordenar=False):  # sin ordenar: mismo orden de filas que siempre
    with Image.open(ruta) as im:          # solo lee el encabezado, es rápido
        ancho, alto = im.size
        if im.getexif().get(274, 1) in (5, 6, 7, 8):   # foto girada por el celular
            ancho, alto = alto, ancho
    rel = ruta.relative_to(FACE)
    filas.append((rel.parts[0], ruta.name, rel.as_posix(), ancho, alto))  # rel = persona/set_XX/archivo

if not filas:
    raise SystemExit(f"ERROR: no se encontraron imagenes en {FACE}")

df = pd.DataFrame(filas, columns=["persona", "archivo", "ruta", "ancho", "alto"])
print(len(df), "imágenes")

salida = REPORTS / "tam_images_face.csv"
salida.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(salida, index=False, encoding="utf-8")
print(f"CSV guardado en {salida}")

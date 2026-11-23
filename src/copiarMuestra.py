"""
Copia a data/muestra/ las imagenes listadas en reports/muestra_balanceada.csv
(desde data/face/, misma estructura persona/set_XX/archivo). No toca los originales.

Uso: python src/copiarMuestra.py
"""
import shutil

import pandas as pd

from comun import FACE as ORIGEN, MUESTRA as DESTINO, REPORTS

if DESTINO.exists() and any(DESTINO.iterdir()):
    raise SystemExit(f"ERROR: {DESTINO} ya tiene contenido; vacialo antes de volver a correr.")

rutas = pd.read_csv(REPORTS / "muestra_balanceada.csv")["ruta"]
for r in rutas:
    destino = DESTINO / r
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ORIGEN / r, destino)

print(f"Copiadas {len(rutas)} imagenes a {DESTINO}")

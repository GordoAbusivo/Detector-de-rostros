"""
Copia a data/clean/ las imagenes de datasets/ (misma estructura persona/set/archivo),
EXCLUYENDO:
  - imagenes sin rostro (reports/imagenesSinRostro.csv)
  - imagenes con mas de 1 rostro (reports/imagenesMasDeUnRostro.csv)
Las imagenes con nitidez baja se conservan, salvo las que ya salen en "sin rostro".
Los originales en datasets/ no se tocan (se COPIAN, no se mueven).

Uso: python src/crearClean.py
"""
import csv
import shutil

from comun import CLEAN, DATASETS, REPORTS, listar_imagenes



def rutas(nombre):
    with open(REPORTS / nombre, encoding="utf-8") as f:
        return {r["ruta"] for r in csv.DictReader(f)}


excluir = rutas("imagenesSinRostro.csv") | rutas("imagenesMasDeUnRostro.csv")

if CLEAN.exists() and any(CLEAN.iterdir()):
    raise SystemExit(f"ERROR: {CLEAN} ya tiene contenido; vacialo antes de volver a correr.")

copiadas = excluidas = 0
for ruta in listar_imagenes(DATASETS):
    if str(ruta) in excluir:
        excluidas += 1
        continue
    destino = CLEAN / ruta.relative_to(DATASETS)
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ruta, destino)
    copiadas += 1

print(f"Copiadas a {CLEAN}: {copiadas}")
print(f"Excluidas: {excluidas}")

"""
Compara reports/imagenesSinRostro.csv con reports/imagenesNitidezBaja.csv.
Genera reports/comparacionSinRostroVsNitidez.csv con la columna 'grupo':
  ambos | solo_sin_rostro | solo_nitidez_baja
Uso: python src/comparacion.py
"""
import csv

from comun import REPORTS as REP


def leer(nombre):
    with open(REP / nombre, encoding="utf-8") as f:
        return {r["ruta"]: r for r in csv.DictReader(f)}


sin_rostro = leer("imagenesSinRostro.csv")
borrosas = leer("imagenesNitidezBaja.csv")
A, B = set(sin_rostro), set(borrosas)

filas = []
for ruta in sorted(A | B):
    grupo = "ambos" if ruta in A and ruta in B else ("solo_sin_rostro" if ruta in A else "solo_nitidez_baja")
    base = sin_rostro.get(ruta) or borrosas[ruta]
    filas.append((base["persona"], base["archivo"], ruta, grupo, borrosas[ruta]["nitidez"] if ruta in B else ""))

print(f"Sin rostro: {len(A)} | Nitidez baja: {len(B)}")
print(f"En AMBOS: {len(A & B)}")
print(f"SOLO sin rostro: {len(A - B)}")
print(f"SOLO nitidez baja: {len(B - A)}")

salida = REP / "comparacionSinRostroVsNitidez.csv"
with open(salida, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["persona", "archivo", "ruta", "grupo", "nitidez"])
    w.writerows(filas)
print(f"CSV guardado en {salida}")

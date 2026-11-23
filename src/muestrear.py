"""
Muestreo balanceado por clase con cuotas de orientacion de la cabeza.

Para cada persona toma hasta N imagenes (por defecto 300) de data/face/:
  1. Solo imagenes que pasaron el umbral de tamano (ancho y alto >= --umbral) y con pose valida.
  2. Orientacion en 9 grupos, igual que preprocesamiento.ipynb: frente; 4 puras (izquierda / derecha /
     arriba / abajo: SOLO un eje supera --grados) y 4 diagonales (izquierda_arriba / derecha_arriba /
     izquierda_abajo / derecha_abajo: los DOS ejes superan --grados).
  3. Cuotas por orientacion (constante PESOS_DEFECTO dentro de este archivo): 30% frente, 10% cada una de
     las puras y 7.5% cada una de las diagonales. Si una orientacion no tiene suficientes imagenes en esa
     persona, se toman todas y el sobrante se reparte entre las demas orientaciones con disponibilidad
     (las diagonales hacia abajo son muy escasas: mediana de 6 imagenes por persona).
  4. Dentro de cada (persona x orientacion) NO se elige al azar: se reparte por turnos entre los distintos
     set_XX (para variar fondo, ropa y luz) y dentro de cada set se prefieren las imagenes mas nitidas y de
     mayor tamano.
  5. Si la persona tiene menos de N imagenes utilizables, se toman todas.

Entradas : reports/tam_images_face.csv, reports/pose_face.csv, data/face/
Salida   : reports/muestra_balanceada.csv   (no copia ni mueve imagenes)

Uso:
    python src/muestrear.py
    python src/muestrear.py --n 200 --umbral 80 --grados 15
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from comun import FACE, REPORTS, UMBRAL_TAMANO_ROSTRO

ORIENTACIONES = ["frente", "izquierda", "derecha", "arriba", "abajo",
                 "izquierda_arriba", "derecha_arriba", "izquierda_abajo", "derecha_abajo"]
PESOS_DEFECTO = {"frente": 0.30,
                 "izquierda": 0.10, "derecha": 0.10, "arriba": 0.10, "abajo": 0.10,
                 "izquierda_arriba": 0.075, "derecha_arriba": 0.075,
                 "izquierda_abajo": 0.075, "derecha_abajo": 0.075}


def orientacion(yaw: float, pitch: float, u: float) -> str:
    """Misma regla que la funcion categoria() de preprocesamiento.ipynb (yaw<0 izquierda, pitch<0 arriba)."""
    h = "izquierda" if yaw < -u else "derecha" if yaw > u else ""
    v = "arriba" if pitch < -u else "abajo" if pitch > u else ""
    return "frente" if not h and not v else "_".join(x for x in (h, v) if x)


def nitidez_normalizada(ruta: Path) -> float:
    """Varianza del Laplaciano a un tamano fijo (128x128) para que sea comparable entre recortes."""
    img = ImageOps.exif_transpose(Image.open(ruta)).convert("L")
    gris = np.asarray(img.resize((128, 128), Image.LANCZOS))
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def repartir_cuotas(n: int, disponibles: dict, pesos: dict) -> dict:
    """Reparte n entre orientaciones segun pesos, sin pasar de lo disponible; el sobrante se redistribuye."""
    cuota = {o: 0 for o in disponibles}
    restante = min(n, sum(disponibles.values()))
    while restante > 0:
        activas = [o for o in disponibles if cuota[o] < disponibles[o]]
        if not activas:
            break
        suma = sum(pesos[o] for o in activas)
        asignado = 0
        for o in activas:
            extra = int(restante * pesos[o] / suma)
            extra = min(extra, disponibles[o] - cuota[o])
            cuota[o] += extra
            asignado += extra
        if asignado == 0:  # residuo por redondeo: de a uno, a la orientacion de mayor peso con espacio
            for o in sorted(activas, key=lambda x: -pesos[x]):
                if restante - asignado <= 0:
                    break
                cuota[o] += 1
                asignado += 1
        restante -= asignado
    return cuota


def elegir_por_sets(celda: pd.DataFrame, k: int) -> pd.DataFrame:
    """Toma k imagenes por turnos entre sets; en cada set, primero las de mejor puntaje."""
    if k >= len(celda):
        return celda
    por_set = {s: g.sort_values("puntaje", ascending=False) for s, g in celda.groupby("set")}
    # sets con mejor imagen disponible primero, para que el desempate sea estable
    orden_sets = sorted(por_set, key=lambda s: -por_set[s]["puntaje"].iloc[0])
    elegidas, ronda = [], 0
    while len(elegidas) < k:
        hubo = False
        for s in orden_sets:
            g = por_set[s]
            if ronda < len(g) and len(elegidas) < k:
                elegidas.append(g.iloc[ronda])
                hubo = True
        if not hubo:
            break
        ronda += 1
    return pd.DataFrame(elegidas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="imagenes por persona")
    ap.add_argument("--umbral", type=int, default=UMBRAL_TAMANO_ROSTRO, help="ancho y alto minimos (px)")
    ap.add_argument("--grados", type=float, default=15.0, help="tolerancia (grados) para 'frente'")
    ap.add_argument("--salida", default=str(REPORTS / "muestra_balanceada.csv"))
    args = ap.parse_args()

    tam = pd.read_csv(REPORTS / "tam_images_face.csv")
    pose = pd.read_csv(REPORTS / "pose_face.csv")[["ruta", "yaw", "pitch", "roll"]]
    d = tam.merge(pose, on="ruta")
    d = d[(d["ancho"] >= args.umbral) & (d["alto"] >= args.umbral) & d["yaw"].notna()].copy()
    d["orientacion"] = [orientacion(y, p, args.grados) for y, p in zip(d["yaw"], d["pitch"])]
    d["set"] = d["ruta"].str.split("/").str[1]
    print(f"Imagenes utilizables (umbral {args.umbral} px y con pose): {len(d)}")

    print("Calculando nitidez de cada recorte ...")
    carpeta = FACE
    d["nitidez"] = [nitidez_normalizada(carpeta / r) for r in d["ruta"]]

    # puntaje: nitidez (70%) + tamano del rostro (30%), como percentil dentro de cada persona
    d["puntaje"] = (0.7 * d.groupby("persona")["nitidez"].rank(pct=True)
                    + 0.3 * d.groupby("persona")["ancho"].rank(pct=True))

    elegidas = []
    for persona, g in d.groupby("persona"):
        disp = {o: int((g["orientacion"] == o).sum()) for o in ORIENTACIONES}
        cuotas = repartir_cuotas(args.n, disp, PESOS_DEFECTO)
        for o in ORIENTACIONES:
            if cuotas[o] > 0:
                elegidas.append(elegir_por_sets(g[g["orientacion"] == o], cuotas[o]))

    m = pd.concat(elegidas)
    cols = ["persona", "archivo", "ruta", "set", "orientacion", "yaw", "pitch", "roll", "ancho", "alto", "nitidez"]
    m = m[cols].sort_values(["persona", "ruta"]).round(2)
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(salida, index=False, encoding="utf-8")

    resumen = pd.crosstab(m["persona"], m["orientacion"]).reindex(columns=ORIENTACIONES, fill_value=0)
    resumen["total"] = resumen.sum(axis=1)
    resumen["sets"] = m.groupby("persona")["set"].nunique()
    print("\nImagenes elegidas por persona y orientacion:")
    print(resumen.sort_values("total").to_string())
    print(f"\nTotal: {len(m)} imagenes | CSV guardado en {salida}")


if __name__ == "__main__":
    main()

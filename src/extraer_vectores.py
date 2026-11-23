"""
Paso 15: convierte cada imagen alineada de la muestra (data/arcface/, 112x112) en un vector de 512 numeros
con los modelos ArcFace ONNX y los guarda en vectors/. Antes vivia en las celdas 12-16 de preprocesamiento.ipynb.

  reports/muestra_balanceada.csv + data/arcface/*.png
      -> vectors/antelope.npy   (N, 512) float32   (antelopev2, ResNet-100: ES EL MODELO PRINCIPAL)
      -> vectors/bufalo.npy     (N, 512) float32   (buffalo_l, ResNet-50: alternativa mas ligera)
      -> vectors/metadatos.csv  persona, archivo, ruta, set, orientacion (MISMO ORDEN que los .npy)

Usa la GPU si esta disponible (CUDA) y si no cae a CPU con un aviso (en CPU tarda horas). Si faltan los modelos
ONNX los descarga de InsightFace. SOBREESCRIBE los archivos de vectors/ cada vez que se corre.

Uso (desde cualquier carpeta, con el entorno activado):
    python src/extraer_vectores.py
    python src/extraer_vectores.py --cpu                # forzar CPU
    python src/extraer_vectores.py --modelos antelope   # solo uno

Desde un notebook:  from extraer_vectores import extraer_todo; extraer_todo()
"""
import argparse
import ctypes
import site
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from comun import ARCFACE, MODELOS, REPORTS, VECTORS, PAQUETES_ARCFACE, asegurar_arcface, preprocesar_cara

COLUMNAS_METADATOS = ["persona", "archivo", "ruta", "set", "orientacion"]
LOTE = 64


def ruta_arcface(ruta_csv) -> Path:
    """'persona/set_01/foto.jpg' (del CSV) -> data/arcface/persona/set_01/foto.png"""
    return ARCFACE / Path(ruta_csv).with_suffix(".png")


def cargar_cuda():
    """Carga las librerias CUDA del entorno (hace falta ANTES de crear las sesiones con GPU)."""
    nv = Path(site.getsitepackages()[0]) / "nvidia"
    for patron in ["cu13/lib/libcudart.so.13*", "cu13/lib/libcublasLt.so.13*", "cu13/lib/libcublas.so.13*",
                   "*/lib/libcurand.so.10*", "cu13/lib/libcufft.so.12*", "cudnn/lib/libcudnn.so.9*"]:
        for f in sorted(nv.glob(patron))[:1]:
            ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)


def crear_sesiones(nombres=None, usar_gpu=True):
    """{nombre_modelo: InferenceSession}. GPU primero; si falla usa CPU (funciona, pero lento)."""
    import onnxruntime as ort
    ort.set_default_logger_severity(3)       # oculta avisos de forma de salida (la salida si es (N, 512))
    if usar_gpu:
        try:
            cargar_cuda()
        except Exception as e:
            print(f"AVISO: no se pudieron cargar las librerias CUDA ({e}); se usara CPU.")
    proveedores = ["CUDAExecutionProvider", "CPUExecutionProvider"] if usar_gpu else ["CPUExecutionProvider"]
    sesiones = {}
    for n in (nombres or PAQUETES_ARCFACE):
        sesiones[n] = ort.InferenceSession(str(MODELOS / f"{n}.onnx"), providers=proveedores)
        print(f"{n} -> {sesiones[n].get_providers()[0]}")
    return sesiones


def extraer(sesion, rutas_csv, lote=LOTE) -> np.ndarray:
    """Vectores (N, 512) float32 de las imagenes de data/arcface indicadas por `rutas_csv`, en ese orden."""
    entrada = sesion.get_inputs()[0].name
    salida = []
    for i in range(0, len(rutas_csv), lote):
        x = np.concatenate([preprocesar_cara(np.asarray(Image.open(ruta_arcface(r)).convert("RGB")))
                            for r in rutas_csv[i:i + lote]])
        salida.append(sesion.run(None, {entrada: x})[0])
        print(f"  {min(i + lote, len(rutas_csv))}/{len(rutas_csv)}", end="\r")
    print()
    return np.vstack(salida).astype(np.float32)


def extraer_todo(modelos=("antelope", "bufalo"), usar_gpu=True):
    """Extrae los vectores de la muestra y los guarda en vectors/ junto con metadatos.csv."""
    muestra = pd.read_csv(REPORTS / "muestra_balanceada.csv")
    faltan = [r for r in muestra["ruta"] if not ruta_arcface(r).exists()]
    if faltan:
        raise SystemExit(f"ERROR: faltan {len(faltan)} imagenes alineadas en {ARCFACE} (ej. {faltan[0]}). "
                         "Corre antes src/alinear.py (y revisa que reports/muestra_balanceada.csv este al dia).")
    asegurar_arcface()
    VECTORS.mkdir(exist_ok=True)
    muestra[COLUMNAS_METADATOS].to_csv(VECTORS / "metadatos.csv", index=False)
    sesiones = crear_sesiones([f"model_{m}" for m in modelos], usar_gpu)
    rutas_csv = muestra["ruta"].tolist()
    vectores = {}
    for nombre, sesion in sesiones.items():
        clave = nombre.replace("model_", "")          # "bufalo", "antelope"
        vectores[clave] = extraer(sesion, rutas_csv)
        np.save(VECTORS / f"{clave}.npy", vectores[clave])
        print(f"{clave}: {vectores[clave].shape}")
    return vectores


def main():
    ap = argparse.ArgumentParser(description="Extrae los vectores ArcFace de la muestra alineada.")
    ap.add_argument("--cpu", action="store_true", help="forzar CPU (lento)")
    ap.add_argument("--modelos", nargs="+", default=["antelope", "bufalo"], choices=["antelope", "bufalo"])
    args = ap.parse_args()
    extraer_todo(args.modelos, usar_gpu=not args.cpu)


if __name__ == "__main__":
    main()

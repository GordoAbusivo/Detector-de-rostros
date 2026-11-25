"""
Baja el dataset de Hugging Face (persona por persona), recorta la CARA de cada foto y guarda solo los recortes.
Pensado para Kaggle: las fotos originales (10-20 GB) nunca se acumulan en disco, se borran al terminar cada persona.

Salida (dentro de --salida, por defecto data/caras/):
    <persona>/<set>__<archivo>.jpg      cara cuadrada de --tam x --tam
    _manifest/<persona>.csv             una fila por cara guardada (persona, set, archivo, ruta, detector, lado_cara, ...)
    _manifest/<persona>_omitidas.csv    fotos sin cara valida (de espalda, cara muy chica, error) y el motivo
    ../caras_manifest.csv               union de todos los manifiestos (lo lee entrenar_cnn_keras.py)

Es REANUDABLE: si la sesion se corta, vuelve a correr el mismo comando y se saltan las personas ya terminadas.
Las fotos de una misma sesion (set) conservan su set en el manifiesto: la division train/val/test se hace por set.

Necesita un token de Hugging Face con acceso de lectura al dataset privado (variable de entorno HF_TOKEN; en Kaggle,
guardalo en Add-ons > Secrets, nunca lo escribas en el notebook).

Uso:
    python src/kaggle_preparar_caras.py                                   # baja de Hugging Face y recorta
    python src/kaggle_preparar_caras.py --origen data/originales          # carpeta local persona/set/foto.jpg
    python src/kaggle_preparar_caras.py --personas garcia_juan perez_juan # solo algunas personas
    python src/kaggle_preparar_caras.py --detector haar --margen 0.4 --tam 160
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from caras import DetectorCaras, recortar_cuadrado
from comun import DATA, EXTENSIONES, cargar_rgb

REPO_HF = "escom-identificador-rostros/identificador-rostros-escom"
CAMPOS = ["persona", "set", "archivo", "ruta", "detector", "score", "lado_cara", "n_caras"]


def agrupar_por_persona(rutas):
    """rutas: iterable de rutas relativas tipo [data/]persona/set/archivo.jpg -> {persona: [(set, ruta), ...]}."""
    grupos = defaultdict(list)
    for r in rutas:
        partes = Path(r).parts
        if len(partes) < 3 or Path(r).suffix.lower() not in EXTENSIONES:
            continue
        grupos[partes[-3]].append((partes[-2], r))
    return grupos


def procesar_persona(persona, items, salida, detector, tam, margen):
    """items: [(set, ruta_local_de_la_foto)]. Devuelve (filas_ok, filas_omitidas)."""
    destino = salida / persona
    destino.mkdir(parents=True, exist_ok=True)
    ok, omitidas = [], []
    for set_id, ruta in tqdm(sorted(items), desc=persona, leave=False):
        ruta = Path(ruta)
        try:
            rgb = cargar_rgb(ruta)
            todas = [c for c in detector.cajas(rgb) if detector._valida(c)]
            if not todas:
                omitidas.append((persona, set_id, ruta.name, "sin cara valida"))
                continue
            caja = max(todas, key=lambda c: c[2] * c[3])
            cara = recortar_cuadrado(rgb, caja, margen, tam)
            nombre = f"{set_id}__{ruta.stem}.jpg"
            Image.fromarray(cara).save(destino / nombre, quality=95)
            ok.append((persona, set_id, ruta.name, f"{persona}/{nombre}", caja[5], round(caja[4], 3),
                       int(min(caja[2], caja[3])), len(todas)))
        except Exception as e:
            omitidas.append((persona, set_id, ruta.name, f"error: {type(e).__name__}: {e}"))
    return ok, omitidas


def escribir_csv(ruta, cabecera, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cabecera)
        w.writerows(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", help="carpeta local con persona/set/foto.jpg; si se omite, se baja de Hugging Face")
    ap.add_argument("--repo", default=REPO_HF, help="dataset de Hugging Face (repo_type=dataset)")
    ap.add_argument("--version", default="main", help="rama o tag del dataset (por ejemplo v1.0)")
    ap.add_argument("--salida", default=str(DATA / "caras"))
    ap.add_argument("--tmp", default=None, help="carpeta temporal para las fotos originales (en Kaggle: /kaggle/temp)")
    ap.add_argument("--personas", nargs="+", help="procesar solo estas personas")
    ap.add_argument("--tam", type=int, default=224, help="lado (px) de la cara guardada")
    ap.add_argument("--margen", type=float, default=0.5, help="espacio extra alrededor de la cara (0.5 = lado x1.5)")
    ap.add_argument("--detector", choices=["auto", "mediapipe", "haar"], default="auto")
    ap.add_argument("--min-cara", type=int, default=60, help="lado minimo (px) de la cara en la foto original")
    ap.add_argument("--confianza", type=float, default=0.5, help="confianza minima de MediaPipe")
    ap.add_argument("--max-por-persona", type=int, default=None, help="solo las primeras N fotos (pruebas rapidas)")
    ap.add_argument("--rehacer", action="store_true", help="ignorar lo ya procesado y empezar de cero")
    a = ap.parse_args()

    salida = Path(a.salida)
    manifiestos = salida / "_manifest"
    if a.rehacer and salida.exists():
        shutil.rmtree(salida)
    manifiestos.mkdir(parents=True, exist_ok=True)

    # ---- 1) que fotos hay y de quien
    if a.origen:
        base = Path(a.origen)
        grupos = agrupar_por_persona(p.relative_to(base) for p in base.rglob("*") if p.is_file())
        local = lambda r: base / r                                                    # noqa: E731
        descargar = None
    else:
        from huggingface_hub import HfApi, snapshot_download
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            sys.exit("ERROR: falta HF_TOKEN (token de Hugging Face con acceso de lectura al dataset).\n"
                     "En Kaggle: Add-ons > Secrets, y en el notebook os.environ['HF_TOKEN'] = secreto.")
        archivos = HfApi(token=token).list_repo_files(a.repo, repo_type="dataset", revision=a.version)
        grupos = agrupar_por_persona(archivos)
        tmp_base = Path(a.tmp or tempfile.gettempdir())

        def descargar(persona, rutas):
            carpeta = tmp_base / f"fotos_{persona}"
            # Un patron por carpeta de set (no uno por foto): con ~30 mil archivos en el repo, miles de patrones tardan mucho.
            patrones = sorted({Path(r).parent.as_posix() + "/*" for r in rutas})
            snapshot_download(a.repo, repo_type="dataset", revision=a.version, token=token,
                              local_dir=carpeta, allow_patterns=patrones)
            return carpeta

    if a.personas:
        faltan = sorted(set(a.personas) - set(grupos))
        if faltan:
            sys.exit(f"ERROR: no hay fotos de: {', '.join(faltan)}. Personas disponibles: {', '.join(sorted(grupos))}")
        grupos = {p: grupos[p] for p in a.personas}
    if not grupos:
        sys.exit("ERROR: no se encontro ninguna foto con estructura [data/]persona/set/archivo.jpg")
    print(f"{len(grupos)} personas, {sum(len(v) for v in grupos.values())} fotos. Detector: {a.detector}")

    # ---- 2) persona por persona (reanudable)
    with DetectorCaras(a.detector, a.confianza, a.min_cara) as detector:
        for persona in sorted(grupos):
            if (manifiestos / f"{persona}.csv").exists():
                print(f"  {persona}: ya procesada, se salta")
                continue
            items = sorted(grupos[persona], key=lambda t: t[1])[:a.max_por_persona]
            carpeta_tmp = None
            if descargar:
                carpeta_tmp = descargar(persona, [r for _, r in items])
                items = [(s, carpeta_tmp / r) for s, r in items]
            else:
                items = [(s, local(r)) for s, r in items]
            ok, omitidas = procesar_persona(persona, items, salida, detector, a.tam, a.margen)
            if carpeta_tmp:
                shutil.rmtree(carpeta_tmp, ignore_errors=True)       # fuera las fotos originales
            escribir_csv(manifiestos / f"{persona}_omitidas.csv", ["persona", "set", "archivo", "motivo"], omitidas)
            escribir_csv(manifiestos / f"{persona}.csv", CAMPOS, ok)  # este archivo marca la persona como terminada
            print(f"  {persona}: {len(ok)} caras guardadas, {len(omitidas)} omitidas")

    # ---- 3) manifiesto global
    partes = [pd.read_csv(p) for p in sorted(manifiestos.glob("*.csv")) if not p.stem.endswith("_omitidas")]
    manifiesto = pd.concat(partes, ignore_index=True)
    destino = salida.parent / "caras_manifest.csv"
    manifiesto.to_csv(destino, index=False)
    omitidas = sum(len(pd.read_csv(p)) for p in manifiestos.glob("*_omitidas.csv"))
    print(f"\nTotal: {len(manifiesto)} caras de {manifiesto['persona'].nunique()} personas "
          f"({omitidas} fotos omitidas). Manifiesto: {destino}")
    por_det = manifiesto["detector"].value_counts().to_dict()
    print("Caras por detector:", por_det)


if __name__ == "__main__":
    main()

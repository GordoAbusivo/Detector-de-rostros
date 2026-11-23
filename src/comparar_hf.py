"""
Compara las imagenes de un dataset externo (p. ej. el de Hugging Face) contra datasets/
para saber cuales YA TIENES (duplicadas) y cuales son NUEVAS, sin importar el nombre del
archivo, el tamano ni la recompresion.

Por que no basta un hash exacto (MD5/SHA): las fotos del dataset de Hugging Face pasaron por
redimensionar.py (max 768x1536, JPEG calidad 90, rotacion EXIF aplicada, renombradas 001.jpg...),
asi que sus bytes ya no coinciden con tus originales. Se compara el CONTENIDO VISUAL:
  1) dHash de 256 bits (16x16): preselecciona candidatas por distancia de Hamming.
  2) Error medio absoluto (MAD, escala 0-255) entre miniaturas de 128x128 en gris: confirma.

Ojo: dentro de un mismo set hay fotogramas de video/rafagas casi identicos (en datasets/ hay
~1,400 pares asi, 86 de ellos byte a byte iguales). Por eso hay una zona gris:
  DUPLICADA  misma foto con certeza (hamming <= --max-hamming y MAD <= --mad-dup)
  CASI_IGUAL no se puede distinguir entre la misma foto reencodeada y un fotograma casi igual
             (MAD <= --mad-casi). Por defecto NO conviene importarla: revisar a mano si hace falta
  NUEVA      sin pareja en datasets/  -> candidata a importar

NO borra ni mueve nada: solo escribe CSVs en reports/.

Uso:
  python src/comparar_hf.py --hf hf_data
  python src/comparar_hf.py --autotest 500      # calibra umbrales con tus propias fotos (no necesita HF)
"""
import argparse
import csv
import os
import pickle
import random
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from comun import RAIZ, listar_imagenes

try:
    from tqdm import tqdm
except ImportError:  # sin barra de progreso
    def tqdm(it, **_):
        return it

MAX_HAMMING = 12      # de 256 bits (calibrado con --autotest)
MAD_DUP = 0.8         # error medio (0-255) a 128x128: misma foto con certeza
MAD_CASI = 2.5        # por encima de esto se considera otra foto
MAX_HAMMING_CASI = 24
# Segunda via a CASI_IGUAL: hash casi identico pero con mas ruido (fotos de baja resolucion/poca luz).
# Se prefiere marcar de mas (el usuario revisa) que dejar pasar un duplicado como NUEVA.
HAMMING_ESTRECHO = 6
MAD_ESTRECHO = 5.0
LADO = 128            # lado de la miniatura de confirmacion
CHUNK = 256


# ------------------------------------------------------------------ huellas
def huella(ruta):
    """(hash 256 bits como 4 uint64, miniatura LADOxLADO gris, ancho, alto) o ("ERROR", msg)."""
    try:
        with Image.open(ruta) as im:
            # SIN im.draft(): decodificar reducido anade error propio (medido: p99 0.65 vs 1.35)
            im = ImageOps.exif_transpose(im)     # misma orientacion que redimensionar.py
            ancho, alto = im.size
            g = im.convert("L")
            a = np.asarray(g.resize((17, 16), Image.BOX), dtype=np.int16)
            bits = (a[:, 1:] > a[:, :-1]).astype(np.uint8).ravel()          # 256 bits
            h = np.packbits(bits).view(np.uint64)                           # 4 x uint64
            m = np.asarray(g.resize((LADO, LADO), Image.BOX), dtype=np.uint8).ravel()
        return h, m, ancho, alto
    except Exception as e:
        return ("ERROR", str(e))


def _trabajo(ruta):
    return str(ruta), huella(ruta)


def norm_persona(nombre):
    s = unicodedata.normalize("NFKD", nombre)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def persona_set(ruta, base):
    partes = ruta.relative_to(base).parts
    if len(partes) >= 3:
        return partes[-3], partes[-2]
    if len(partes) == 2:
        return partes[0], ""
    return "", ""


def calcular(carpeta, etiqueta, workers, usar_cache=True):
    """Devuelve (rutas, H (n,4) uint64, M (n,1024) uint8, errores). Usa cache por (tamano, mtime)."""
    carpeta = Path(carpeta)
    archivos = listar_imagenes(carpeta)
    cache_f = RAIZ / "reports" / f".cache_huellas_{etiqueta}.pkl"
    cache = {}
    if usar_cache and cache_f.exists():
        try:
            cache = pickle.loads(cache_f.read_bytes())
        except Exception:
            cache = {}
    nuevos, firmas = [], {}
    for p in archivos:
        st = p.stat()
        firmas[str(p)] = (st.st_size, st.st_mtime_ns)
        c = cache.get(str(p))
        if not c or c["firma"] != firmas[str(p)]:
            nuevos.append(p)
    if nuevos:
        with Pool(workers) as pool:
            for ruta, res in tqdm(pool.imap_unordered(_trabajo, nuevos, chunksize=32),
                                  total=len(nuevos), desc=f"huellas {etiqueta}"):
                cache[ruta] = {"firma": firmas[ruta], "res": res}
        if usar_cache:
            cache_f.parent.mkdir(exist_ok=True)
            cache_f.write_bytes(pickle.dumps(cache))
    rutas, H, M, errores = [], [], [], []
    for p in archivos:
        res = cache[str(p)]["res"]
        if isinstance(res[0], str):          # ("ERROR", mensaje)
            errores.append((str(p), res[1]))
            continue
        rutas.append(p)
        H.append(res[0])
        M.append(res[1])
    return rutas, np.array(H, dtype=np.uint64).reshape(-1, 4), np.array(M, dtype=np.uint8), errores


# ------------------------------------------------------------------ comparacion
def mejor_pareja(HA, HB):
    """Para cada fila de HA, indice y distancia de Hamming de la fila mas cercana de HB."""
    idx = np.zeros(len(HA), dtype=np.int64)
    dist = np.zeros(len(HA), dtype=np.int32)
    for i in range(0, len(HA), CHUNK):
        x = HA[i:i + CHUNK, None, :] ^ HB[None, :, :]
        d = np.bitwise_count(x).sum(-1)
        idx[i:i + CHUNK] = d.argmin(1)
        dist[i:i + CHUNK] = d.min(1)
    return idx, dist


def mad(MA, MB):
    """Error medio absoluto (0-255) fila a fila entre miniaturas (n, LADO*LADO)."""
    out = np.empty(len(MA), dtype=np.float32)
    for i in range(0, len(MA), 512):
        out[i:i + 512] = np.abs(MA[i:i + 512].astype(np.int16) - MB[i:i + 512].astype(np.int16)).mean(1)
    return out


# ------------------------------------------------------------------ autoprueba
def autotest(n, workers, semilla=0):
    """Simula el pipeline de redimensionar.py sobre n fotos locales y mide cuanto se separan
    'la misma foto reencodeada' de 'la foto distinta mas parecida' (fotogramas casi iguales)."""
    print("[autotest] huellas de datasets/ ...")
    rutas, H, M, _ = calcular(RAIZ / "datasets", "local", workers)
    rnd = random.Random(semilla)
    elegidas = rnd.sample(range(len(rutas)), min(n, len(rutas)))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sims = []
        for k, i in enumerate(elegidas):
            try:
                with Image.open(rutas[i]) as im:
                    im = ImageOps.exif_transpose(im)
                    prop = min(768 / im.width, 1536 / im.height)
                    if prop < 1:
                        im = im.resize((int(im.width * prop), int(im.height * prop)), Image.LANCZOS)
                    destino = tmp / f"{k:05d}.jpg"
                    im.convert("RGB").save(destino, "JPEG", quality=90)
                sims.append((i, destino))
            except Exception:
                pass
        res = [huella(d) for _, d in sims]
    ok = [(i, r) for (i, _), r in zip(sims, res) if not isinstance(r[0], str)]
    HS = np.array([r[0] for _, r in ok], dtype=np.uint64).reshape(-1, 4)
    MS = np.array([r[1] for _, r in ok], dtype=np.uint8)
    orig = np.array([i for i, _ in ok])
    d_true = np.bitwise_count(HS ^ H[orig]).sum(-1)          # contra su propio original
    m_true = mad(MS, M[orig])
    d_false = np.zeros(len(ok), dtype=np.int32)              # contra la OTRA foto mas cercana
    m_false = np.zeros(len(ok), dtype=np.float32)
    for k in range(0, len(ok), 128):
        d = np.bitwise_count(HS[k:k + 128, None, :] ^ H[None, :, :]).sum(-1).astype(np.int32)
        for j in range(d.shape[0]):
            d[j, orig[k + j]] = 999                          # excluir al propio original
        mejor = d.argmin(1)
        d_false[k:k + 128] = d.min(1)
        m_false[k:k + 128] = mad(MS[k:k + 128], M[mejor])
    pct = lambda a, q: float(np.percentile(a, q))
    print(f"\n[autotest] {len(ok)} fotos simuladas (resize 768x1536 + JPEG q90) vs sus originales")
    print(f"  MISMA foto : hamming max={d_true.max()} p99={pct(d_true,99):.0f} | "
          f"MAD max={m_true.max():.2f} p99={pct(m_true,99):.2f} mediana={pct(m_true,50):.2f}")
    print(f"  OTRA foto  : hamming min={d_false.min()} p5={pct(d_false,5):.0f} | "
          f"MAD min={m_false.min():.2f} p5={pct(m_false,5):.2f} mediana={pct(m_false,50):.2f}")
    det = (d_true <= MAX_HAMMING) & (m_true <= MAD_DUP)
    es_casi = lambda d, m: ((d <= MAX_HAMMING_CASI) & (m <= MAD_CASI)) | ((d <= HAMMING_ESTRECHO) & (m <= MAD_ESTRECHO))
    casi = es_casi(d_true, m_true) & ~det
    fp = (d_false <= MAX_HAMMING) & (m_false <= MAD_DUP)
    fp_casi = es_casi(d_false, m_false) & ~fp
    print(f"  Con max-hamming={MAX_HAMMING}, mad-dup={MAD_DUP}, mad-casi={MAD_CASI}:")
    print(f"    misma foto  -> DUPLICADA {det.sum()}/{len(ok)}  CASI_IGUAL {casi.sum()}  NUEVA(perdida) {len(ok)-det.sum()-casi.sum()}")
    print(f"    otra foto   -> DUPLICADA {fp.sum()} (copias casi exactas ya presentes dentro de datasets/)  "
          f"CASI_IGUAL {fp_casi.sum()}  NUEVA {len(ok)-fp.sum()-fp_casi.sum()}")


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", default=str(RAIZ / "hf_data"), help="carpeta del dataset descargado (persona/set/archivo)")
    ap.add_argument("--local", default=str(RAIZ / "datasets"), help="carpeta propia a comparar (default: datasets/)")
    ap.add_argument("--salida", default=str(RAIZ / "reports"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--max-hamming", type=int, default=MAX_HAMMING)
    ap.add_argument("--mad-dup", type=float, default=MAD_DUP)
    ap.add_argument("--mad-casi", type=float, default=MAD_CASI)
    ap.add_argument("--autotest", type=int, metavar="N", help="solo calibrar umbrales con N fotos propias")
    ap.add_argument("--sin-cache", action="store_true")
    a = ap.parse_args()

    if a.autotest:
        return autotest(a.autotest, a.workers)

    hf_dir, loc_dir, out = Path(a.hf), Path(a.local), Path(a.salida)
    if not hf_dir.is_dir():
        sys.exit(f"ERROR: no existe {hf_dir}. Descarga primero el dataset (ver GUIA_HUGGINGFACE.md).")
    out.mkdir(exist_ok=True)

    r_loc, H_loc, M_loc, err_l = calcular(loc_dir, "local", a.workers, not a.sin_cache)
    r_hf, H_hf, M_hf, err_h = calcular(hf_dir, "hf", a.workers, not a.sin_cache)
    print(f"local: {len(r_loc)} imagenes | hf: {len(r_hf)} imagenes | con error: {len(err_l)} local, {len(err_h)} hf")
    for ruta, msg in (err_l + err_h)[:10]:
        print(f"  no se pudo leer {ruta}: {msg}")

    idx, dist = mejor_pareja(H_hf, H_loc)
    err = mad(M_hf, M_loc[idx])

    filas = []
    for i, p in enumerate(r_hf):
        per_hf, set_hf = persona_set(p, hf_dir)
        q = r_loc[idx[i]]
        per_loc, set_loc = persona_set(q, loc_dir)
        if dist[i] <= a.max_hamming and err[i] <= a.mad_dup:
            estado = "DUPLICADA"
        elif (dist[i] <= MAX_HAMMING_CASI and err[i] <= a.mad_casi) or \
                (dist[i] <= HAMMING_ESTRECHO and err[i] <= MAD_ESTRECHO):
            estado = "CASI_IGUAL"
        else:
            estado = "NUEVA"
        hay = estado != "NUEVA"
        filas.append({
            "persona_hf": per_hf, "set_hf": set_hf, "archivo_hf": str(p.relative_to(hf_dir)),
            "persona_local": per_loc if hay else "",
            "set_local": set_loc if hay else "",
            "archivo_local": str(q.relative_to(loc_dir)) if hay else "",
            "hamming": int(dist[i]), "mad": round(float(err[i]), 3), "estado": estado,
            "misma_persona": (norm_persona(per_hf) == norm_persona(per_loc)) if hay else "",
        })

    campos = list(filas[0].keys()) if filas else []
    for nombre, sel in (("hf_vs_local_todas.csv", filas),
                        ("hf_nuevas.csv", [f for f in filas if f["estado"] == "NUEVA"]),
                        ("hf_duplicadas.csv", [f for f in filas if f["estado"] == "DUPLICADA"]),
                        ("hf_casi_iguales_revisar.csv", [f for f in filas if f["estado"] == "CASI_IGUAL"])):
        with open(out / nombre, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(sel)

    # ---- mapa de sets: a que set propio corresponde cada set de HF (importante: la division
    #      train/val/test es POR SET; una misma sesion repartida en dos sets filtraria fotos)
    mapa = defaultdict(lambda: {"Counter": Counter(), "destino": Counter()})
    for f in filas:
        m = mapa[(f["persona_hf"], f["set_hf"])]
        m["Counter"][f["estado"]] += 1
        if f["estado"] != "NUEVA":
            m["destino"][(f["persona_local"], f["set_local"])] += 1
    with open(out / "hf_mapa_sets.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["persona_hf", "set_hf", "total", "duplicadas", "casi_iguales", "nuevas",
                    "pct_ya_tengo", "set_local_principal", "fotos_en_ese_set", "recomendacion"])
        for (per, st), m in sorted(mapa.items()):
            c = m["Counter"]; tot = sum(c.values()); ya = c["DUPLICADA"] + c["CASI_IGUAL"]
            (pl, sl), n = m["destino"].most_common(1)[0] if m["destino"] else (("", ""), 0)
            if ya / tot >= 0.9:
                rec = "SESION YA EXISTENTE: no importar"
            elif ya / tot >= 0.2:
                rec = "MEZCLA: sesion parcialmente repetida; importar solo NUEVAS dentro de " + (f"{pl}/{sl}" if sl else "su set")
            else:
                rec = "SESION NUEVA: importar como set nuevo"
            w.writerow([per, st, tot, c["DUPLICADA"], c["CASI_IGUAL"], c["NUEVA"],
                        round(100 * ya / tot, 1), f"{pl}/{sl}" if sl else "", n, rec])

    por = defaultdict(Counter)
    for f in filas:
        por[f["persona_hf"]][f["estado"]] += 1
    loc_personas = {norm_persona(persona_set(p, loc_dir)[0]) for p in r_loc}
    print(f"\n{'persona (HF)':30s} {'total':>6s} {'DUP':>6s} {'CASI':>6s} {'NUEVA':>6s}  en datasets/?")
    for per in sorted(por):
        c = por[per]
        print(f"{per:30s} {sum(c.values()):6d} {c['DUPLICADA']:6d} {c['CASI_IGUAL']:6d} {c['NUEVA']:6d}  "
              f"{'si' if norm_persona(per) in loc_personas else 'NO (persona nueva)'}")
    tot = Counter(f["estado"] for f in filas)
    print(f"\nTOTAL hf={len(filas)}  DUPLICADA={tot['DUPLICADA']}  CASI_IGUAL={tot['CASI_IGUAL']}  NUEVA={tot['NUEVA']}")
    cruz = [f for f in filas if f["estado"] == "DUPLICADA" and f["misma_persona"] is False]
    if cruz:
        print(f"ATENCION: {len(cruz)} duplicadas aparecen con OTRO nombre de persona (revisar hf_duplicadas.csv)")
    print(f"CSVs en {out}/: hf_vs_local_todas, hf_duplicadas, hf_casi_iguales_revisar, hf_nuevas, hf_mapa_sets")


if __name__ == "__main__":
    main()

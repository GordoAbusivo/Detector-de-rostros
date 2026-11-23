"""
Importa a datasets/ las fotos del dataset de Hugging Face que NO tenias (resultado de comparar_hf.py).

Reglas:
  - Se importan las fotos NUEVA y (por decision del usuario) las CASI_IGUAL. Las DUPLICADA no.
  - Persona: si ya existe en datasets/ (sin importar mayusculas) se usa SU carpeta; si no, la de HF.
  - Set (la division train/val/test es POR SET, una sesion no debe quedar repartida):
      * si el set de HF tiene fotos que ya tenias, las nuevas van al MISMO set tuyo (es la misma sesion);
      * si no tiene ninguna coincidencia y la persona ya existia -> set nuevo "set_hf_NN";
      * persona nueva -> se conservan los nombres de set de HF (set_01...).
  - Archivo destino: hf_<set_hf>_<nombre original>.jpg  (nunca pisa nada; trazable).
  - Fotos byte-a-byte iguales entre las que se importan: se importa una sola. Si la misma foto aparece
    con personas distintas NO se importa ninguna (etiqueta dudosa) y se lista en hf_conflictos.csv.
  - Solo COPIA. No borra ni modifica nada. Escribe reports/hf_importadas.csv (para poder revertir).

Uso:
  python src/importar_hf.py                 # simulacion (no copia nada)
  python src/importar_hf.py --ejecutar      # copia y verifica cada archivo con SHA-256
  python src/importar_hf.py --sin-casi      # no importar las CASI_IGUAL
"""
import argparse
import csv
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from comun import RAIZ



def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def numero_set(nombre):
    d = "".join(c for c in nombre if c.isdigit())
    return d.zfill(2) if d else nombre


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", default=str(RAIZ / "hf_data"))
    ap.add_argument("--datasets", default=str(RAIZ / "datasets"))
    ap.add_argument("--reports", default=str(RAIZ / "reports"))
    ap.add_argument("--ejecutar", action="store_true")
    ap.add_argument("--sin-casi", action="store_true")
    a = ap.parse_args()
    hf, ds, rep = Path(a.hf), Path(a.datasets), Path(a.reports)

    with open(rep / "hf_vs_local_todas.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    existentes = {p.name.lower(): p.name for p in ds.iterdir() if p.is_dir()}

    # ---- 1) a que set va cada set de HF (a partir de las coincidencias)
    votos = defaultdict(Counter)
    for r in filas:
        if r["estado"] in ("DUPLICADA", "CASI_IGUAL") and r["misma_persona"] == "True":
            peso = 1 if r["estado"] == "DUPLICADA" else 0.01     # las DUPLICADA mandan; las CASI solo desempatan
            votos[(r["persona_hf"], r["set_hf"])][(r["persona_local"], r["set_local"])] += peso
    n_dup = Counter(); n_casi = Counter()
    for r in filas:
        k = (r["persona_hf"], r["set_hf"])
        if r["estado"] == "DUPLICADA": n_dup[k] += 1
        if r["estado"] == "CASI_IGUAL": n_casi[k] += 1

    def destino_set(persona_hf, set_hf):
        k = (persona_hf, set_hf)
        pl = existentes.get(persona_hf.lower())
        if pl is None:
            return persona_hf, set_hf, "persona nueva"
        # sesion existente: al menos 1 DUPLICADA (misma foto exacta) o >=3 CASI
        if votos[k] and (n_dup[k] >= 1 or n_casi[k] >= 3):
            (_, sl), _ = votos[k].most_common(1)[0]
            return pl, sl, "mismo set"
        return pl, f"set_hf_{numero_set(set_hf)}", "set nuevo"

    # ---- 2) plan de copia
    incluir = {"NUEVA"} | (set() if a.sin_casi else {"CASI_IGUAL"})
    plan = []
    for r in filas:
        if r["estado"] not in incluir:
            continue
        pers, st, motivo = destino_set(r["persona_hf"], r["set_hf"])
        origen = hf / r["archivo_hf"]
        nombre = f"hf_{r['set_hf']}_{Path(r['archivo_hf']).name}"
        plan.append({"origen": origen, "persona": pers, "set": st, "motivo": motivo, "estado": r["estado"],
                     "persona_hf": r["persona_hf"], "set_hf": r["set_hf"], "archivo_hf": r["archivo_hf"],
                     "destino": ds / pers / st / nombre})
    faltan = [p for p in plan if not p["origen"].is_file()]
    if faltan:
        sys.exit(f"ERROR: faltan {len(faltan)} archivos de HF en {hf} (ej. {faltan[0]['origen']}). "
                 "Termina la descarga antes de importar.")
    choque = [p for p in plan if p["destino"].exists()]
    if choque:
        sys.exit(f"ERROR: {len(choque)} destinos ya existen (ej. {choque[0]['destino']}); no se pisa nada.")

    # ---- 3) duplicados exactos (SHA-256) dentro de lo que se va a importar
    print(f"calculando SHA-256 de {len(plan)} archivos...")
    for p in plan:
        p["sha"] = sha256(p["origen"])
    por_sha = defaultdict(list)
    for p in plan:
        por_sha[p["sha"]].append(p)
    omitir, conflictos = [], []
    for sha, grupo in por_sha.items():
        if len(grupo) == 1:
            continue
        if len({g["persona"] for g in grupo}) > 1:
            conflictos.extend(grupo)
        else:
            omitir.extend(sorted(grupo, key=lambda g: str(g["destino"]))[1:])
    ids_omitir = {id(p) for p in omitir} | {id(p) for p in conflictos}
    final = [p for p in plan if id(p) not in ids_omitir]

    # ---- 4) resumen
    print(f"\nA importar: {len(final)} fotos  (NUEVA {sum(p['estado']=='NUEVA' for p in final)}, "
          f"CASI_IGUAL {sum(p['estado']=='CASI_IGUAL' for p in final)})")
    print(f"Omitidas por ser byte-identicas a otra que ya se importa: {len(omitir)}")
    print(f"Conflictos (misma foto con personas distintas, no se importan): {len(conflictos)}")
    print(f"\n{'persona (destino)':32s} {'fotos':>6s} {'sets':>5s}  motivo")
    por = defaultdict(list)
    for p in final:
        por[p["persona"]].append(p)
    for pers in sorted(por, key=str.lower):
        ps = por[pers]
        motivos = Counter(p["motivo"] for p in ps)
        print(f"{pers:32s} {len(ps):6d} {len({p['set'] for p in ps}):5d}  "
              + ", ".join(f"{k}: {v}" for k, v in motivos.items()))
    nuevas_personas = sorted({p["persona"] for p in final if p["motivo"] == "persona nueva"})
    print(f"\nPersonas NUEVAS ({len(nuevas_personas)}): {', '.join(nuevas_personas)}")
    sets_nuevos = sorted({(p["persona"], p["set"]) for p in final if p["motivo"] == "set nuevo"})
    print(f"Sets nuevos en personas existentes ({len(sets_nuevos)}): "
          + (", ".join(f"{a}/{b}" for a, b in sets_nuevos) or "ninguno"))

    if conflictos:
        with open(rep / "hf_conflictos.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["archivo_hf", "persona_hf", "sha256"])
            for p in conflictos:
                w.writerow([p["archivo_hf"], p["persona_hf"], p["sha"]])

    if not a.ejecutar:
        print("\n[SIMULACION] No se copio nada. Usa --ejecutar para importar.")
        return

    # ---- 5) copiar y verificar
    manifiesto = rep / "hf_importadas.csv"
    with open(manifiesto, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "set", "archivo", "estado", "motivo", "archivo_hf", "sha256"])
        malos = 0
        for i, p in enumerate(final, 1):
            p["destino"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p["origen"], p["destino"])
            if sha256(p["destino"]) != p["sha"]:
                malos += 1
                print(f"  ERROR de verificacion: {p['destino']}")
            w.writerow([p["persona"], p["set"], p["destino"].name, p["estado"], p["motivo"],
                        p["archivo_hf"], p["sha"]])
            if i % 5000 == 0:
                print(f"  {i}/{len(final)} copiadas")
    print(f"\nCopiadas y verificadas: {len(final) - malos}/{len(final)}   errores: {malos}")
    print(f"Manifiesto (para revertir): {manifiesto}")
    if malos:
        sys.exit(1)


if __name__ == "__main__":
    main()

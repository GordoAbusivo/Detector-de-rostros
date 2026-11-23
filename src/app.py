"""
Front para probar el modelo: arrastra una foto y muestra la persona y su confianza.

Sin dependencias extra (solo la libreria estandar de Python + lo que ya usa predecir.py) y SIN internet: es un
servidor local que entrega una sola pagina (src/app.html) sin ningun recurso externo. Usa exactamente la misma
logica que `python src/predecir.py` (clase Predictor): detectar rostro -> alinear a 112x112 -> ArcFace antelope ->
clasificador -> persona. La confianza es el parecido (coseno) con el centro de la persona elegida:
alta >= 0.60, media 0.50-0.60, baja < 0.50. Siempre responde una de las personas del modelo (no hay "desconocido").

Uso (desde cualquier carpeta, con el entorno activado):
    python src/app.py                    # abre http://127.0.0.1:7860 en el navegador
    python src/app.py --modelo svm
    python src/app.py --puerto 8000 --sin-navegador
    python src/app.py --host 0.0.0.0     # SOLO dentro de Docker (ver docker-compose.yml): alli el puerto se publica
                                         # unicamente en 127.0.0.1 del anfitrion; no lo uses asi en una red abierta

Privacidad y seguridad:
  - Por defecto solo escucha en 127.0.0.1 (tu computadora, no la red). Se rechazan peticiones con otro Host (DNS rebinding)
    y las que no vienen de la propia pagina (cabecera X-Reco).
  - Las fotos se procesan en memoria: no se guardan en disco ni se registran en ningun log.
  - Limite de 25 MB por foto. Solo se pueden usar los modelos de models/*_antelope.joblib.
Requiere los paquetes models/<modelo>_antelope.joblib (se exportan con src/exportar_modelos.py).
"""
import argparse
import base64
import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

from comun import MODELOS, cargar_rgb
from predecir import UMBRAL_ALTA, UMBRAL_MEDIA, Predictor

HTML = Path(__file__).with_name("app.html")
MAX_BYTES = 25 * 1024 * 1024
SEGURIDAD = {
    "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                               "img-src 'self' data: blob:; connect-src 'self'; base-uri 'none'; "
                               "form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def modelos_disponibles():
    return sorted(p.stem.removesuffix("_antelope") for p in MODELOS.glob("*_antelope.joblib"))


class Estado:
    """Modelos cargados (cada uno se carga la primera vez que se pide) y datos del servidor."""
    puerto = 7860
    modelo_defecto = "ridge"
    _predictores = {}
    _cerrojo = threading.Lock()

    @classmethod
    def predictor(cls, nombre):
        with cls._cerrojo:
            if nombre not in cls._predictores:
                cls._predictores[nombre] = Predictor(nombre)
            return cls._predictores[nombre]

    @classmethod
    def cerrar(cls):
        Predictor.cerrar()


def png_base64(cara_rgb) -> str:
    buf = io.BytesIO()
    Image.fromarray(cara_rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class Manejador(BaseHTTPRequestHandler):
    server_version = "RecoFacial"
    sys_version = ""

    def log_message(self, *args):          # sin registro de peticiones: no queda rastro de las fotos
        pass

    # ---------------------------------------------------------------- utilidades
    def _responder(self, codigo, cuerpo: bytes, tipo):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        for k, v in SEGURIDAD.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(cuerpo)

    def _json(self, codigo, datos):
        self._responder(codigo, json.dumps(datos, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _host_valido(self):
        return self.headers.get("Host", "") in (f"127.0.0.1:{Estado.puerto}", f"localhost:{Estado.puerto}")

    # ---------------------------------------------------------------- rutas
    def do_GET(self):
        if not self._host_valido():
            return self._json(403, {"ok": False, "error": "Host no permitido."})
        ruta = urlparse(self.path).path
        if ruta in ("/", "/index.html"):
            return self._responder(200, HTML.read_bytes(), "text/html; charset=utf-8")
        if ruta == "/api/modelos":
            return self._json(200, {"modelos": modelos_disponibles(), "defecto": Estado.modelo_defecto,
                                    "umbrales": {"alta": UMBRAL_ALTA, "media": UMBRAL_MEDIA}})
        self._json(404, {"ok": False, "error": "No encontrado."})

    def do_POST(self):
        if not self._host_valido():
            return self._json(403, {"ok": False, "error": "Host no permitido."})
        url = urlparse(self.path)
        if url.path != "/api/predecir":
            return self._json(404, {"ok": False, "error": "No encontrado."})
        if self.headers.get("X-Reco") != "1":
            return self._json(403, {"ok": False, "error": "Peticion no permitida."})
        modelo = parse_qs(url.query).get("modelo", [Estado.modelo_defecto])[0]
        if modelo not in modelos_disponibles():
            return self._json(400, {"ok": False, "error": f"Modelo desconocido: {modelo}."})
        try:
            largo = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._json(411, {"ok": False, "error": "Falta Content-Length."})
        if largo <= 0:
            return self._json(400, {"ok": False, "error": "Foto vacia."})
        if largo > MAX_BYTES:
            return self._json(413, {"ok": False, "error": "La foto pesa mas de 25 MB."})
        datos = self.rfile.read(largo)
        try:
            rgb = cargar_rgb(io.BytesIO(datos))     # respeta EXIF y .heic, igual que el resto del proyecto
        except Exception:
            return self._json(400, {"ok": False, "error": "No se pudo leer la imagen (formato no valido)."})
        try:
            r = Estado.predictor(modelo).predecir_rgb(rgb)
        except Exception:
            return self._json(500, {"ok": False, "error": "Error interno al analizar la foto."})
        if r is None:
            return self._json(200, {"ok": False, "error": "No se detecto ningun rostro. Prueba con otra foto."})
        self._json(200, {"ok": True, "persona": r["persona"], "confianza": round(r["confianza"], 4),
                         "nivel": r["nivel"], "n_rostros": r["n_rostros"], "modelo": r["modelo"],
                         "cara": png_base64(r["cara"])})


def main():
    ap = argparse.ArgumentParser(description="Front local para probar el modelo de reconocimiento facial.")
    ap.add_argument("--modelo", default="ridge", help="modelo inicial: ridge | svm | knn (default: ridge)")
    ap.add_argument("--puerto", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1",
                    help="direccion donde escuchar (default 127.0.0.1; 0.0.0.0 solo dentro de Docker)")
    ap.add_argument("--sin-navegador", action="store_true", help="no abrir el navegador automaticamente")
    args = ap.parse_args()

    disponibles = modelos_disponibles()
    if not disponibles:
        raise SystemExit("ERROR: no hay ningun models/*_antelope.joblib. Copia ahi el archivo del modelo (por ejemplo ridge_antelope.joblib) "
                         "o exportalo con src/exportar_modelos.py.")
    if args.modelo not in disponibles:
        raise SystemExit(f"ERROR: no existe el modelo '{args.modelo}'. Disponibles: {', '.join(disponibles)}")
    Estado.puerto, Estado.modelo_defecto = args.puerto, args.modelo

    print(f"Cargando el modelo '{args.modelo}' ...")
    Estado.predictor(args.modelo)                   # carga ahora para que la primera foto no tarde
    try:
        servidor = ThreadingHTTPServer((args.host, args.puerto), Manejador)
    except OSError as e:
        raise SystemExit(f"ERROR: no se pudo abrir el puerto {args.puerto} ({e}). Prueba con --puerto 8000.")
    servidor.daemon_threads = True
    url = f"http://127.0.0.1:{args.puerto}"
    print(f"Listo: {url}   (Ctrl+C para salir)")
    if not args.sin_navegador:
        webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando ...")
    finally:
        servidor.server_close()
        Estado.cerrar()


if __name__ == "__main__":
    main()

"""Demo en vivo: detecta rostros con la cámara y dice quién es (Modelo A).

Uso (desde la raíz del repo):
    python reporte/demo_webcam.py --modelo resultados/modelo_clasico_svm_rbf/mejor_modelo/modelo.joblib

Presionen 'q' para salir.
Nota: el modelo siempre elige a alguien del grupo; una persona desconocida será
clasificada como la más parecida.
"""
import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import caracteristicas_hog, crear_detector, recortar_con_margen  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", type=Path, required=True, help="archivo modelo.joblib")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--detector", choices=["auto", "haar", "mediapipe"], default="auto")
    ap.add_argument("--margen", type=float, default=0.2, help="mismo margen que en recortar_rostros.py")
    args = ap.parse_args()

    import cv2

    paquete = joblib.load(args.modelo)
    modelo, tam = paquete["modelo"], paquete["tam"]
    detectar = crear_detector(args.detector)

    captura = cv2.VideoCapture(args.camara)
    if not captura.isOpened():
        sys.exit("No se pudo abrir la cámara.")

    while True:
        ok, cuadro = captura.read()
        if not ok:
            break
        for caja in detectar(cuadro):
            recorte = recortar_con_margen(cuadro, caja, args.margen)
            if recorte.size == 0:
                continue
            nombre = modelo.predict([caracteristicas_hog(recorte, tam)])[0]
            x, y, w, h = caja
            cv2.rectangle(cuadro, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(cuadro, str(nombre), (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.imshow("Detector de rostros - q para salir", cuadro)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

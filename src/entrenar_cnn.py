"""Modelo C: red neuronal convolucional (CNN) para reconocer rostros del grupo.

Dos modos, pensados para compararlos en el reporte:

  --modo finetune   CNN PREENTRENADA en ImageNet (ResNet) + fine-tuning. Es la opción recomendada:
                    con pocas fotos por persona, partir de pesos ya entrenados rinde mucho más.
  --modo scratch    CNN pequeña entrenada DESDE CERO. Sirve de línea base para mostrar cuánto
                    ayuda el preentrenamiento (con pocos datos suele quedar bastante por debajo).

Usa las mismas particiones y la misma función guardar_resultados() que entrenar_vit.py, así que el
equipo de reporte puede compararlo directo con HOG+SVM y con ViT.

Uso (desde la raíz del repo; con GPU, por ejemplo en Colab):
    python entrenar_cnn.py --modo finetune --arq resnet50 --epocas 15
    python entrenar_cnn.py --modo finetune --arq resnet18 --epocas 15 --batch 32
    python entrenar_cnn.py --modo scratch --epocas 60 --tam 112
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Busca comun.py subiendo por las carpetas (funciona con el repo plano o con entrenamiento/cnn/).
for _p in Path(__file__).resolve().parents:
    if (_p / "comun.py").exists():
        sys.path.insert(0, str(_p))
        break
from comun import PARTICIONES, RESULTADOS, guardar_resultados  # noqa: E402

MEDIA_IMAGENET, STD_IMAGENET = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ modelos
class CNNPequena(nn.Module):
    """4 bloques (Conv-BN-ReLU x2 + MaxPool) -> pooling global -> clasificador. ~1.2 M parámetros."""

    def __init__(self, n_clases):
        super().__init__()

        def bloque(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.rasgos = nn.Sequential(bloque(3, 32), bloque(32, 64), bloque(64, 128), bloque(128, 256))
        self.clasificador = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4),
                                          nn.Linear(256, n_clases))

    def forward(self, x):
        return self.clasificador(self.rasgos(x))


def crear_modelo(modo, arq, n_clases):
    if modo == "scratch":
        return CNNPequena(n_clases)
    if arq == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif arq == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"arquitectura desconocida: {arq}")
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n_clases))  # cabeza nueva
    return m


def parametros_cabeza(modelo):
    return modelo.fc.parameters()


# ------------------------------------------------------------------ entrenamiento
def evaluar(modelo, cargador, dispositivo):
    modelo.eval()
    reales, predichos = [], []
    with torch.no_grad():
        for x, y in cargador:
            predichos += modelo(x.to(dispositivo)).argmax(dim=1).cpu().tolist()
            reales += y.tolist()
    return reales, predichos


def una_epoca(modelo, cargador, criterio, optimizador, dispositivo):
    modelo.train()
    total, perdida = 0, 0.0
    for x, y in cargador:
        x, y = x.to(dispositivo), y.to(dispositivo)
        optimizador.zero_grad()
        loss = criterio(modelo(x), y)
        loss.backward()
        optimizador.step()
        perdida += loss.item() * len(y)
        total += len(y)
    return perdida / total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", choices=["finetune", "scratch"], default="finetune")
    ap.add_argument("--arq", choices=["resnet18", "resnet50"], default="resnet50", help="solo en modo finetune")
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--congelar", type=int, default=3,
                    help="finetune: épocas iniciales entrenando solo la cabeza (el resto de la red congelada)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="por defecto: 1e-4 (finetune) o 1e-3 (scratch)")
    ap.add_argument("--tam", type=int, default=None, help="lado de la imagen; por defecto 224 (finetune) o 112 (scratch)")
    ap.add_argument("--workers", type=int, default=2, help="en Windows, si da error, usa 0")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    finetune = args.modo == "finetune"
    tam = args.tam or (224 if finetune else 112)
    lr = args.lr or (1e-4 if finetune else 1e-3)
    nombre = f"cnn_{args.arq}" if finetune else "cnn_scratch"

    torch.manual_seed(args.semilla)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {dispositivo}" + (" (sin GPU será lento; usen Colab)" if dispositivo == "cpu" else ""))

    # Con preentrenado hay que normalizar como ImageNet; desde cero basta con centrar en 0.5.
    norm = transforms.Normalize(MEDIA_IMAGENET, STD_IMAGENET) if finetune else transforms.Normalize([0.5] * 3, [0.5] * 3)
    fuerza = 0.2 if finetune else 0.3
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(tam, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=fuerza, contrast=fuerza, saturation=fuerza),
        transforms.ToTensor(), norm,
    ])
    tf_eval = transforms.Compose([transforms.Resize((tam, tam)), transforms.ToTensor(), norm])

    ds_train = datasets.ImageFolder(PARTICIONES / "train", tf_train)
    ds_val = datasets.ImageFolder(PARTICIONES / "val", tf_eval)
    ds_test = datasets.ImageFolder(PARTICIONES / "test", tf_eval)
    assert ds_train.classes == ds_val.classes == ds_test.classes, \
        "Las tres particiones deben tener las mismas personas (revisen particionar.py)."
    clases = ds_train.classes
    print(f"personas={len(clases)} train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")

    pin = dispositivo == "cuda"
    cargador_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=pin)
    cargador_val = DataLoader(ds_val, batch_size=args.batch, num_workers=args.workers, pin_memory=pin)
    cargador_test = DataLoader(ds_test, batch_size=args.batch, num_workers=args.workers, pin_memory=pin)  # sin shuffle

    # Pesos por clase: compensan que unas personas tengan más fotos que otras.
    conteo = Counter(ds_train.targets)
    pesos = torch.tensor([len(ds_train) / (len(clases) * conteo[i]) for i in range(len(clases))], dtype=torch.float)
    criterio = nn.CrossEntropyLoss(weight=pesos.to(dispositivo), label_smoothing=0.1)

    modelo = crear_modelo(args.modo, args.arq, len(clases)).to(dispositivo)

    # Fase 1 (solo finetune): congelar el cuerpo y entrenar la cabeza nueva. Fase 2: descongelar todo con lr bajo.
    fase_cabeza = args.congelar if finetune else 0
    if fase_cabeza:
        for p in modelo.parameters():
            p.requires_grad = False
        for p in parametros_cabeza(modelo):
            p.requires_grad = True
        optimizador = torch.optim.AdamW(parametros_cabeza(modelo), lr=lr * 10, weight_decay=0.01)
    else:
        optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=max(1, args.epocas - fase_cabeza))

    carpeta = RESULTADOS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_pesos = carpeta / "mejor_modelo.pt"
    historial, mejor_acc = [], -1.0

    for epoca in range(1, args.epocas + 1):
        if fase_cabeza and epoca == fase_cabeza + 1:
            print("-> Descongelando toda la red (fine-tuning completo)")
            for p in modelo.parameters():
                p.requires_grad = True
            optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=args.epocas - fase_cabeza)

        perdida = una_epoca(modelo, cargador_train, criterio, optimizador, dispositivo)
        if epoca > fase_cabeza:
            scheduler.step()
        reales, predichos = evaluar(modelo, cargador_val, dispositivo)
        acc_val = sum(r == p for r, p in zip(reales, predichos)) / len(reales)
        historial.append((epoca, perdida, acc_val))
        print(f"Época {epoca:>2}/{args.epocas} pérdida train={perdida:.4f} acc val={acc_val:.4f}")
        if acc_val > mejor_acc:
            mejor_acc = acc_val
            torch.save({"state_dict": modelo.state_dict(), "clases": clases, "modo": args.modo,
                        "arq": args.arq, "tam": tam}, ruta_pesos)

    with open(carpeta / "historial.csv", "w", newline="", encoding="utf-8") as f:  # para graficar en el reporte
        w = csv.writer(f)
        w.writerow(["epoca", "perdida_train", "acc_val"])
        w.writerows(historial)
    (carpeta / "clases.json").write_text(json.dumps(clases, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nMejor acc en validación: {mejor_acc:.4f}. Evaluando en prueba (una sola vez)...")
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=dispositivo)["state_dict"])
    reales, predichos = evaluar(modelo, cargador_test, dispositivo)
    rutas = [Path(ruta) for ruta, _ in ds_test.samples]
    guardar_resultados(nombre, [clases[i] for i in reales], [clases[i] for i in predichos], clases, rutas)


if __name__ == "__main__":
    main()

"""Modelo B: fine-tuning de un Vision Transformer (ViT) preentrenado.

Uso (desde la raíz del repo; con GPU, por ejemplo en Google Colab):
    python entrenamiento/transformer/entrenar_vit.py --epocas 10
    python entrenamiento/transformer/entrenar_vit.py --epocas 15 --lr 2e-5 --batch 8
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comun import PARTICIONES, RESULTADOS, guardar_resultados  # noqa: E402


def evaluar(modelo, cargador, dispositivo):
    modelo.eval()
    reales, predichos = [], []
    with torch.no_grad():
        for x, y in cargador:
            logits = modelo(pixel_values=x.to(dispositivo)).logits
            predichos += logits.argmax(dim=1).cpu().tolist()
            reales += y.tolist()
    return reales, predichos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.semilla)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {dispositivo}" + ("  (sin GPU será muy lento; usen Colab)" if dispositivo == "cpu" else ""))

    norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # preprocesamiento de ViT
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        norm,
    ])
    tf_eval = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

    ds_train = datasets.ImageFolder(PARTICIONES / "train", tf_train)
    ds_val = datasets.ImageFolder(PARTICIONES / "val", tf_eval)
    ds_test = datasets.ImageFolder(PARTICIONES / "test", tf_eval)
    assert ds_train.classes == ds_val.classes == ds_test.classes, \
        "Las tres particiones deben tener las mismas personas (revisen particionar.py)."
    clases = ds_train.classes
    print(f"personas={len(clases)}  train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}")

    cargador_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True)
    cargador_val = DataLoader(ds_val, batch_size=args.batch)
    cargador_test = DataLoader(ds_test, batch_size=args.batch)  # sin shuffle: el orden coincide con ds_test.samples

    modelo = ViTForImageClassification.from_pretrained(
        args.modelo,
        num_labels=len(clases),
        id2label=dict(enumerate(clases)),
        label2id={c: i for i, c in enumerate(clases)},
    ).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.01)

    carpeta_modelo = RESULTADOS / "vit" / "mejor_modelo"
    mejor_acc = -1.0
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        perdida_total = 0.0
        for x, y in cargador_train:
            x, y = x.to(dispositivo), y.to(dispositivo)
            optimizador.zero_grad()
            salida = modelo(pixel_values=x, labels=y)
            salida.loss.backward()
            optimizador.step()
            perdida_total += salida.loss.item() * len(y)

        reales, predichos = evaluar(modelo, cargador_val, dispositivo)
        acc_val = sum(r == p for r, p in zip(reales, predichos)) / len(reales)
        print(f"Época {epoca:>2}/{args.epocas}  pérdida train={perdida_total / len(ds_train):.4f}  acc val={acc_val:.4f}")
        if acc_val > mejor_acc:
            mejor_acc = acc_val
            modelo.save_pretrained(carpeta_modelo)

    print(f"\nMejor acc en validación: {mejor_acc:.4f}. Evaluando en prueba (una sola vez)...")
    modelo = ViTForImageClassification.from_pretrained(carpeta_modelo).to(dispositivo)
    reales, predichos = evaluar(modelo, cargador_test, dispositivo)
    rutas = [Path(ruta) for ruta, _ in ds_test.samples]
    guardar_resultados("vit", [clases[i] for i in reales], [clases[i] for i in predichos], clases, rutas)


if __name__ == "__main__":
    main()

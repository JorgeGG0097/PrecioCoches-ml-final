# -*- coding: utf-8 -*-
"""
modelo_imagenes.py  --  Clasificador de danos en coches (EfficientNet-B0, 4 clases).

Uso:
    python modelo_imagenes.py                          # entrena
    python modelo_imagenes.py --test <ruta_imagen>     # predice una imagen
    python modelo_imagenes.py --eval                   # muestra metricas guardadas

Clases:
    0 -> sin_danos
    1 -> leve
    2 -> moderado
    3 -> severo

Salida:
    modelo_cnn.pt              pesos del mejor modelo
    modelo_cnn_metricas.json   metricas de entrenamiento
"""

import sys
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image

RUTA_BASE  = Path(__file__).parent
DIR_DATOS  = RUTA_BASE / "dataset"
MODELO_PT  = RUTA_BASE / "modelo_cnn.pt"
METRICAS_J = RUTA_BASE / "modelo_cnn_metricas.json"

CLASES = ['sin_danos', 'leve', 'moderado', 'severo']

# ── Hiperparametros ────────────────────────────────────────────────────────────
EPOCHS     = 15
BATCH_SIZE = 32
LR         = 1e-4
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTENSIONES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ────────────────────────────────────────────────────────────────────
class DamageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform
        self.targets   = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ruta, label = self.samples[idx]
        img = Image.open(ruta).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def _cargar_split(split: str):
    muestras = []
    for label, cls in enumerate(CLASES):
        carpeta = DIR_DATOS / split / cls
        if not carpeta.exists():
            raise FileNotFoundError(f"No se encontro: {carpeta}")
        for ruta in carpeta.iterdir():
            if ruta.suffix.lower() in EXTENSIONES:
                muestras.append((str(ruta), label))
    return muestras


def construir_datasets():
    train_ds = DamageDataset(_cargar_split("train"), transform=TRAIN_TF)
    val_ds   = DamageDataset(_cargar_split("val"),   transform=VAL_TF)

    counts = [train_ds.targets.count(i) for i in range(len(CLASES))]
    print(f"Clases: {CLASES}")
    for cls, n in zip(CLASES, counts):
        print(f"  train/{cls}: {n}")
    print(f"  val total: {len(val_ds)}")
    return train_ds, val_ds


def construir_sampler(train_ds):
    counts  = [train_ds.targets.count(i) for i in range(len(CLASES))]
    weights = [1.0 / counts[l] for l in train_ds.targets]
    return WeightedRandomSampler(weights, len(weights))


# ── Modelo ─────────────────────────────────────────────────────────────────────
def construir_modelo(n_clases=4):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for name, param in model.features.named_parameters():
        layer_idx = int(name.split(".")[0]) if name.split(".")[0].isdigit() else -1
        param.requires_grad = layer_idx >= 6
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, n_clases),
    )
    return model.to(DEVICE)


# ── Metricas multiclase ────────────────────────────────────────────────────────
def calcular_f1_macro(all_preds, all_labels, n_clases):
    f1s = []
    for c in range(n_clases):
        tp = sum(p == c and l == c for p, l in zip(all_preds, all_labels))
        fp = sum(p == c and l != c for p, l in zip(all_preds, all_labels))
        fn = sum(p != c and l == c for p, l in zip(all_preds, all_labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return sum(f1s) / len(f1s), f1s


# ── Entrenamiento ──────────────────────────────────────────────────────────────
def entrenar():
    print(f"Dispositivo: {DEVICE}")
    train_ds, val_ds = construir_datasets()

    sampler      = construir_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    model     = construir_modelo(len(CLASES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mejor_val_acc = 0.0
    historial     = []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += imgs.size(0)
        train_acc  = correct / total
        train_loss = loss_sum / total

        # ── Val ──
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels  = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(1)
                val_correct   += (preds == labels).sum().item()
                val_total     += imgs.size(0)
                all_preds     += preds.cpu().tolist()
                all_labels    += labels.cpu().tolist()

        val_acc        = val_correct / val_total
        f1_macro, f1s  = calcular_f1_macro(all_preds, all_labels, len(CLASES))

        scheduler.step()
        historial.append({
            "epoch": epoch,
            "train_acc": round(train_acc, 4),
            "val_acc":   round(val_acc, 4),
            "f1_macro":  round(f1_macro, 4),
        })
        print(f"Epoch {epoch:2}/{EPOCHS}  train={train_acc:.3f}  val={val_acc:.3f}  F1_macro={f1_macro:.3f}  "
              + "  ".join(f"{c}={f:.2f}" for c, f in zip(CLASES, f1s)))

        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            torch.save(model.state_dict(), MODELO_PT)
            print(f"             -> Mejor modelo guardado (val_acc={val_acc:.4f})")

    elapsed = round(time.time() - t0, 1)
    mejor   = max(historial, key=lambda x: x["val_acc"])
    _, f1s_mejor = calcular_f1_macro(all_preds, all_labels, len(CLASES))

    metricas = {
        "clases":        CLASES,
        "n_train":       len(train_ds),
        "n_val":         len(val_ds),
        "epochs":        EPOCHS,
        "mejor_epoch":   mejor["epoch"],
        "val_acc":       mejor["val_acc"],
        "f1_macro":      mejor["f1_macro"],
        "f1_por_clase":  {c: round(f, 4) for c, f in zip(CLASES, f1s_mejor)},
        "tiempo_seg":    elapsed,
        "dispositivo":   str(DEVICE),
        "historial":     historial,
    }
    with open(METRICAS_J, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"\nEntrenamiento completado en {elapsed}s")
    print(f"Mejor val_acc: {mejor_val_acc:.4f}  |  guardado en {MODELO_PT}")

    # ── Exportar a ONNX (mas ligero para Streamlit Cloud) ──────────────────────
    try:
        model_export = construir_modelo(len(CLASES))
        model_export.load_state_dict(torch.load(MODELO_PT, map_location="cpu", weights_only=True))
        model_export.eval()
        dummy  = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        onnx_path = str(RUTA_BASE / "modelo_cnn.onnx")
        torch.onnx.export(
            model_export, dummy, onnx_path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        print(f"Exportado a ONNX: {onnx_path}")
    except Exception as e:
        print(f"Exportacion ONNX fallida (no critico): {e}")

    return metricas


# ── Prediccion ─────────────────────────────────────────────────────────────────
def predecir(ruta_imagen: str) -> dict:
    if not os.path.exists(MODELO_PT):
        raise FileNotFoundError(f"Modelo no encontrado: {MODELO_PT}. Entrena primero.")

    model = construir_modelo(len(CLASES))
    model.load_state_dict(torch.load(MODELO_PT, map_location=DEVICE, weights_only=True))
    model.eval()

    img    = Image.open(ruta_imagen).convert("RGB")
    tensor = VAL_TF(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

    clase_idx = probs.index(max(probs))
    return {
        "clase":       CLASES[clase_idx],
        "clase_idx":   clase_idx,
        "probabilidades": {c: round(p, 4) for c, p in zip(CLASES, probs)},
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        if idx + 1 < len(sys.argv):
            res = predecir(sys.argv[idx + 1])
            print(f"\nClase: {res['clase']}")
            for c, p in res["probabilidades"].items():
                print(f"  {c}: {p:.1%}")
        else:
            print("Uso: python modelo_imagenes.py --test <ruta_imagen>")
    elif "--eval" in sys.argv:
        if not os.path.exists(METRICAS_J):
            print("No hay metricas. Entrena primero.")
        else:
            with open(METRICAS_J, encoding="utf-8") as f:
                m = json.load(f)
            print(f"val_acc  : {m['val_acc']:.4f}")
            print(f"F1 macro : {m['f1_macro']:.4f}")
            print(f"Mejor epoch: {m['mejor_epoch']}/{m['epochs']}")
            for c, f in m["f1_por_clase"].items():
                print(f"  F1 {c}: {f:.4f}")
    else:
        entrenar()

# -*- coding: utf-8 -*-
"""
modelo_imagenes.py  --  Entrena un clasificador de danos en coches (EfficientNet-B0 fine-tuning).

Uso:
    python modelo_imagenes.py           # entrena y guarda modelo_cnn.pt
    python modelo_imagenes.py --test <ruta_imagen>  # predice una imagen
    python modelo_imagenes.py --eval    # evalua el modelo guardado en el conjunto de validacion

Clases:
    0 -> sin_danos   (coche en buen estado visual)
    1 -> danado      (coche con golpes, abolladuras o rayones)

Salida:
    modelo_cnn.pt            modelo entrenado (pesos completos)
    modelo_cnn_metricas.json metricas de entrenamiento para la memoria
"""

import sys
import os
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image

RUTA_BASE   = Path(__file__).parent
DIR_DATOS   = RUTA_BASE / "fotos"
MODELO_PT   = RUTA_BASE / "modelo_cnn.pt"
METRICAS_J  = RUTA_BASE / "modelo_cnn_metricas.json"

# ── Hiperparametros ────────────────────────────────────────────────────────────
EPOCHS      = 12
BATCH_SIZE  = 32
LR          = 1e-4
VAL_SPLIT   = 0.2
SEED        = 42
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


CLASES     = ['danado', 'sin_danos']   # 0 = danado, 1 = sin_danos
EXTENSIONES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


class DamageDataset(Dataset):
    """Lee solo las carpetas 'danado' y 'sin_danos', ignora el resto."""
    def __init__(self, samples, transform=None):
        self.samples   = samples   # lista de (ruta, label)
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


def _cargar_muestras():
    muestras = []
    for label, cls in enumerate(CLASES):
        carpeta = DIR_DATOS / cls
        if not carpeta.exists():
            raise FileNotFoundError(f"No se encontro: {carpeta}")
        for ruta in carpeta.iterdir():
            if ruta.suffix.lower() in EXTENSIONES:
                muestras.append((str(ruta), label))
    return muestras


# ── Construir datasets ─────────────────────────────────────────────────────────
def construir_datasets():
    todas = _cargar_muestras()
    random.seed(SEED)
    random.shuffle(todas)
    split     = int(len(todas) * VAL_SPLIT)
    train_muestras = todas[split:]
    val_muestras   = todas[:split]

    train_ds = DamageDataset(train_muestras, transform=TRAIN_TF)
    val_ds   = DamageDataset(val_muestras,   transform=VAL_TF)

    print(f"Clases: {CLASES}  (0=danado, 1=sin_danos)")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds, CLASES


def construir_sampler(train_ds):
    labels = train_ds.targets
    counts = [labels.count(c) for c in range(len(CLASES))]
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights))


# ── Modelo ─────────────────────────────────────────────────────────────────────
def construir_modelo(n_clases=2):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Congelar capas base, solo entrenar las ultimas + clasificador
    for name, param in model.features.named_parameters():
        layer_idx = int(name.split(".")[0]) if name.split(".")[0].isdigit() else -1
        param.requires_grad = layer_idx >= 6   # descongelar bloques 6 y 7
    # Reemplazar clasificador
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, n_clases),
    )
    return model.to(DEVICE)


# ── Entrenamiento ──────────────────────────────────────────────────────────────
def entrenar():
    print(f"Dispositivo: {DEVICE}")
    train_ds, val_ds, clases = construir_datasets()

    sampler      = construir_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    model     = construir_modelo(len(clases))
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
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                preds = out.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)
                # Para F1 — clase 1 = danado
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

        val_acc  = val_correct / val_total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        scheduler.step()
        historial.append({"epoch": epoch, "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4), "f1": round(f1, 4)})
        print(f"Epoch {epoch:2}/{EPOCHS}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  F1={f1:.3f}")

        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            torch.save(model.state_dict(), MODELO_PT)
            print(f"             -> Mejor modelo guardado (val_acc={val_acc:.4f})")

    elapsed = round(time.time() - t0, 1)
    mejor = max(historial, key=lambda x: x["val_acc"])
    metricas = {
        "clases":      clases,
        "n_train":     len(train_ds),
        "n_val":       len(val_ds),
        "epochs":      EPOCHS,
        "mejor_epoch": mejor["epoch"],
        "val_acc":     mejor["val_acc"],
        "f1_danado":   mejor["f1"],
        "tiempo_seg":  elapsed,
        "dispositivo": str(DEVICE),
        "historial":   historial,
    }
    with open(METRICAS_J, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"\nEntrenamiento completado en {elapsed}s")
    print(f"Mejor val_acc: {mejor_val_acc:.4f}  |  modelo guardado en {MODELO_PT}")
    print(f"Metricas guardadas en {METRICAS_J}")
    return metricas


# ── Prediccion de una imagen ───────────────────────────────────────────────────
def predecir(ruta_imagen: str, umbral: float = 0.5) -> dict:
    if not os.path.exists(MODELO_PT):
        raise FileNotFoundError(f"No se encontro el modelo: {MODELO_PT}. Ejecuta primero sin --test.")

    model = construir_modelo(2)
    model.load_state_dict(torch.load(MODELO_PT, map_location=DEVICE, weights_only=True))
    model.eval()

    img = Image.open(ruta_imagen).convert("RGB")
    tensor = VAL_TF(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    prob_danado = float(probs[1])
    clase = "danado" if prob_danado >= umbral else "sin_danos"

    return {
        "clase":       clase,
        "prob_danado": round(prob_danado, 4),
        "prob_sano":   round(float(probs[0]), 4),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        if idx + 1 < len(sys.argv):
            ruta = sys.argv[idx + 1]
            resultado = predecir(ruta)
            print(f"\nImagen: {ruta}")
            print(f"Clase predicha : {resultado['clase']}")
            print(f"Prob. danado   : {resultado['prob_danado']:.1%}")
            print(f"Prob. sin danos: {resultado['prob_sano']:.1%}")
        else:
            print("Indica la ruta de la imagen: python modelo_imagenes.py --test <ruta>")
    elif "--eval" in sys.argv:
        if not os.path.exists(METRICAS_J):
            print("No hay metricas guardadas. Entrena primero.")
        else:
            with open(METRICAS_J, encoding="utf-8") as f:
                m = json.load(f)
            print(f"val_acc  : {m['val_acc']:.4f}")
            print(f"F1 danado: {m['f1_danado']:.4f}")
            print(f"Mejor epoch: {m['mejor_epoch']}/{m['epochs']}")
    else:
        entrenar()

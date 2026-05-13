# -*- coding: utf-8 -*-
"""
organizar_kaggle.py  --  Mueve las imagenes del dataset de Kaggle a las carpetas de entrenamiento.

Uso:
    python organizar_kaggle.py

Espera encontrar la estructura del dataset "Car Damage Detection" de anujms:
    fotos/training/00-damage/   -> se copia a fotos/danado/
    fotos/training/01-whole/    -> se copia a fotos/sin_danos/

Las fotos de Coches.net ya descargadas en fotos/sin_etiquetar/ TAMBIEN se
mueven a fotos/sin_danos/ automaticamente (son de concesionario, por tanto sin danos).
"""

import os
import shutil
import glob

BASE     = os.path.dirname(os.path.abspath(__file__))
DIR_FTOS = os.path.join(BASE, "fotos")

DATA1A       = os.path.join(DIR_FTOS, "data1a")
SPLITS_DANO  = [
    os.path.join(DATA1A, "training",   "00-damage"),
    os.path.join(DATA1A, "validation", "00-damage"),
]
SPLITS_SANO  = [
    os.path.join(DATA1A, "training",   "01-whole"),
    os.path.join(DATA1A, "validation", "01-whole"),
]
DST_DANO     = os.path.join(DIR_FTOS, "danado")
DST_SANO     = os.path.join(DIR_FTOS, "sin_danos")
SRC_COCHES   = os.path.join(DIR_FTOS, "sin_etiquetar")

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}


def copiar_carpeta(origen, destino, prefijo=""):
    if not os.path.exists(origen):
        print(f"  [AVISO] No existe: {origen}")
        return 0
    os.makedirs(destino, exist_ok=True)
    n = 0
    for ruta in glob.glob(os.path.join(origen, "**", "*"), recursive=True):
        if os.path.isfile(ruta) and os.path.splitext(ruta)[1].lower() in EXTENSIONES:
            nombre = f"{prefijo}{os.path.basename(ruta)}"
            dst = os.path.join(destino, nombre)
            if not os.path.exists(dst):
                shutil.copy2(ruta, dst)
                n += 1
    return n


def main():
    os.makedirs(DST_DANO, exist_ok=True)
    os.makedirs(DST_SANO, exist_ok=True)

    print("Organizando imagenes Kaggle...")
    n_dano = sum(copiar_carpeta(src, DST_DANO, prefijo="kg_") for src in SPLITS_DANO)
    n_sano = sum(copiar_carpeta(src, DST_SANO, prefijo="kg_") for src in SPLITS_SANO)
    print(f"  Kaggle danado    -> {DST_DANO} : {n_dano} fotos copiadas")
    print(f"  Kaggle sin danos -> {DST_SANO} : {n_sano} fotos copiadas")

    print("\nMoviendo fotos de Coches.net (sin_etiquetar) a sin_danos...")
    n_coches = copiar_carpeta(SRC_COCHES, DST_SANO, prefijo="cn_")
    print(f"  Coches.net -> {DST_SANO} : {n_coches} fotos copiadas")

    total_dano = len([f for f in os.listdir(DST_DANO) if os.path.splitext(f)[1].lower() in EXTENSIONES])
    total_sano = len([f for f in os.listdir(DST_SANO) if os.path.splitext(f)[1].lower() in EXTENSIONES])

    print(f"\nDataset final:")
    print(f"  danado    : {total_dano} imagenes")
    print(f"  sin_danos : {total_sano} imagenes")

    if total_dano == 0 or total_sano == 0:
        print("\n[ERROR] Alguna clase tiene 0 imagenes. Revisa las rutas del dataset de Kaggle.")
    elif total_sano > total_dano * 3:
        print(f"\n[AVISO] Desbalance: {total_sano} sin danos vs {total_dano} danados.")
        print("  Considera usar class_weight='balanced' al entrenar el modelo.")
    else:
        print("\nDataset listo para entrenar. Ejecuta: python modelo_imagenes.py")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
descargar_fotos.py  --  Descarga las fotos de coches_scrapeados.csv para etiquetado manual.

Uso:
    python descargar_fotos.py              # descarga todas las fotos disponibles
    python descargar_fotos.py --max 300    # descarga maximo 300 fotos

Las fotos se guardan en:
    fotos/sin_etiquetar/   <-- donde van a parar al descargar

Despues de descargar, etiqueta manualmente moviendo cada foto a:
    fotos/danado/          <-- coche con golpes, abolladuras, rayones evidentes
    fotos/sin_danos/       <-- coche en buen estado visual

Nombre de archivo: <idx>_<marca>_<modelo>_<anio>.jpg
El indice permite relacionar la foto con la fila del CSV.
"""

import sys
import os
import time
import re
import urllib.request
import pandas as pd

CSV_ENTRADA  = "coches_scrapeados.csv"
DIR_FOTOS    = "fotos"
DIR_SIN_ETI  = os.path.join(DIR_FOTOS, "sin_etiquetar")
DIR_DANADO   = os.path.join(DIR_FOTOS, "danado")
DIR_SIN_DANO = os.path.join(DIR_FOTOS, "sin_danos")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 10
PAUSA   = 0.4   # segundos entre descargas


def _nombre_seguro(texto: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '_', str(texto).strip())[:20]


def descargar_fotos(max_fotos: int = 0):
    for d in (DIR_SIN_ETI, DIR_DANADO, DIR_SIN_DANO):
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(CSV_ENTRADA):
        print(f"No se encontro {CSV_ENTRADA}. Ejecuta primero scraper_coches.py")
        return

    df = pd.read_csv(CSV_ENTRADA, encoding="utf-8-sig")
    if "url_foto" not in df.columns:
        print("El CSV no tiene columna 'url_foto'. Actualiza el scraper y vuelve a ejecutarlo.")
        return

    df_con_foto = df[df["url_foto"].notna() & (df["url_foto"] != "")].copy()
    print(f"Anuncios totales: {len(df)} | Con foto: {len(df_con_foto)}")

    if max_fotos > 0:
        df_con_foto = df_con_foto.head(max_fotos)
        print(f"Descargando maximo {max_fotos} fotos...")
    else:
        print(f"Descargando {len(df_con_foto)} fotos...")

    descargadas = 0
    errores     = 0

    for idx, row in df_con_foto.iterrows():
        marca  = _nombre_seguro(row.get("marca",  ""))
        modelo = _nombre_seguro(row.get("modelo", ""))
        anio   = str(int(row.get("año", 0))) if row.get("año", 0) else "0000"
        nombre = f"{idx:05d}_{marca}_{modelo}_{anio}.jpg"
        ruta   = os.path.join(DIR_SIN_ETI, nombre)

        if os.path.exists(ruta):
            descargadas += 1
            continue

        try:
            req = urllib.request.Request(row["url_foto"], headers=HEADERS)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                datos = resp.read()
            with open(ruta, "wb") as f:
                f.write(datos)
            descargadas += 1
            if descargadas % 50 == 0:
                print(f"  {descargadas}/{len(df_con_foto)} descargadas...")
            time.sleep(PAUSA)
        except Exception as ex:
            errores += 1

    print(f"\nListo: {descargadas} fotos en '{DIR_SIN_ETI}', {errores} errores")
    print("\nSiguiente paso:")
    print(f"  Abre la carpeta '{DIR_SIN_ETI}' y mueve cada foto a:")
    print(f"    '{DIR_DANADO}'   -> si el coche tiene golpes, abolladuras o rayones visibles")
    print(f"    '{DIR_SIN_DANO}' -> si el coche parece estar en buen estado visual")
    print(f"  Con ~300-500 fotos etiquetadas es suficiente para entrenar el modelo.")


if __name__ == "__main__":
    max_arg = 0
    if "--max" in sys.argv:
        try:
            max_arg = int(sys.argv[sys.argv.index("--max") + 1])
        except (IndexError, ValueError):
            pass
    descargar_fotos(max_arg)

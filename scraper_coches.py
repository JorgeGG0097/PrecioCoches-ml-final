# -*- coding: utf-8 -*-
"""
scraper_coches.py  —  Descarga anuncios de Coches.net y genera coches_scrapeados.csv
Uso:
    python scraper_coches.py            # scraping completo (MAX_PAGINAS)
    python scraper_coches.py --test     # 1 pagina, imprime estructura y 3 anuncios
    python scraper_coches.py --test-foto  # muestra la URL de foto extraida en 5 anuncios
"""

import sys, time, re
import pandas as pd
from datetime import datetime
from playwright.sync_api import sync_playwright

# ── Configuración ──────────────────────────────────────────────────────────────
MAX_PAGINAS  = 50          # páginas a descargar (~30 anuncios/página ≈ 1.500 anuncios)
PAUSA_SEG    = 1.5         # pausa entre páginas
SALIDA_CSV   = "coches_scrapeados.csv"
URL_BASE     = "https://www.coches.net/segunda-mano/?pg={}"
HEADLESS     = True        # False = abre ventana visible (útil para depurar)

# ── Normalización de marcas ────────────────────────────────────────────────────
NORM_MARCAS = {
    "alfa":       "Alfa Romeo",
    "citro":      "Citroen",
    "citroën":    "Citroen",
    "mercedes":   "Mercedes-Benz",
    "land":       "Land Rover",
}

def normalizar_marca(marca: str) -> str:
    if not marca:
        return ""
    key = marca.strip().lower()
    for patron, valor in NORM_MARCAS.items():
        if key.startswith(patron):
            return valor
    return marca.strip().title()

# ── Etiqueta DGT ───────────────────────────────────────────────────────────────
def etiqueta_dgt(combustible: str, año: int) -> str:
    c = str(combustible).lower()
    if "electr" in c:                               return "0_EMISIONES"
    if "enchufable" in c or "plug" in c:            return "0_EMISIONES"
    if "híbrido" in c or "hybrid" in c:             return "ECO"
    if "gasolina" in c:
        return "C" if año >= 2006 else "B" if año >= 2001 else "Sin etiqueta"
    if "diésel" in c or "diesel" in c:
        return "C" if año >= 2015 else "B" if año >= 2006 else "Sin etiqueta"
    return "Sin etiqueta"

# ── Extraer número de un texto (formato español: . = miles, , = decimales) ──────
def extraer_numero(texto: str) -> float:
    nums = re.sub(r'[^\d,.]', '', texto)
    if ',' in nums and '.' in nums:          # "29.900,50" → 29900.50
        nums = nums.replace('.', '').replace(',', '.')
    elif ',' in nums:                         # "29,90" → 29.90
        nums = nums.replace(',', '.')
    elif '.' in nums:
        parts = nums.split('.')
        if len(parts[-1]) == 3:              # "29.900" → 29900 (miles)
            nums = nums.replace('.', '')
    try:    return float(nums)
    except: return 0.0

# ── Parsear una tarjeta de anuncio ─────────────────────────────────────────────
def parsear_tarjeta(card) -> dict | None:
    try:
        texto = card.inner_text()

        # Precio — buscar el elemento exacto con el importe
        precio_el = card.query_selector('[data-testid*="price"], [class*=cashAmount], [class*=Amount]')
        if not precio_el:
            precio_el = card.query_selector('[class*=Price] p, [class*=Price] span')
        precio = extraer_numero(precio_el.inner_text()) if precio_el else 0.0

        # Título (marca + modelo)
        titulo_el = card.query_selector('h2, h3, [class*=title], [class*=Title], [class*=name]')
        titulo = titulo_el.inner_text().strip() if titulo_el else ""

        # Link
        link_el = card.query_selector('a')
        url = link_el.get_attribute('href') if link_el else ""
        if url and not url.startswith('http'):
            url = "https://www.coches.net" + url

        # Foto principal — iteramos todos los imgs hasta encontrar la foto del coche
        url_foto = ""
        for img_el in card.query_selector_all('img'):
            for attr in ('src', 'data-src', 'data-original', 'data-lazy', 'data-lazy-src'):
                val = img_el.get_attribute(attr) or ""
                if (val and not val.startswith('data:')
                        and '.svg' not in val
                        and 'placeholder' not in val.lower()
                        and 'icon' not in val.lower()):
                    url_foto = val
                    break
            if url_foto:
                if not url_foto.startswith('http'):
                    url_foto = "https://www.coches.net" + url_foto
                break

        # Intentar extraer año, km, combustible del texto completo de la tarjeta
        año_m = re.search(r'\b(19[9]\d|20[0-2]\d)\b', texto)
        año   = int(año_m.group(1)) if año_m else 0

        # Fallback año desde URL: "...-2017-en-..."
        if año == 0 and url:
            url_año = re.search(r'-(\d{4})-en-', url)
            if url_año:
                año = int(url_año.group(1))

        km_m  = re.search(r'([\d\.]+)\s*km', texto, re.IGNORECASE)
        km    = int(km_m.group(1).replace('.','')) if km_m else 0

        cv_m  = re.search(r'(\d+)\s*[Cc][Vv]', texto)
        cv    = int(cv_m.group(1)) if cv_m else 0

        # Combustible — nombres exactos del dataset de entrenamiento
        comb = ""
        COMBUST = [
            ("Eléctrico",          ["electrico", "eléctrico", "electric"]),
            ("Híbrido Enchufable", ["enchufable", "plug-in", "plug in", "phev"]),
            ("Híbrido",            ["híbrido", "hibrido", "hybrid", "mhev", "hev"]),
            ("Gasolina",           ["gasolina", "tfsi", "tsi", "gdi", "benzin"]),
            ("Diésel",             ["diésel", "diesel", "tdi", "hdi", "cdi", "dci", "bluehdi"]),
            ("Glp",                ["glp"]),
            ("Gnc",                ["gnc"]),
        ]
        texto_l = texto.lower()
        for nombre, variantes in COMBUST:
            if any(v in texto_l for v in variantes):
                comb = nombre
                break

        # Transmisión — "Automática" exacto como en el dataset de entrenamiento
        PATRON_AUTO = [
            "automático", "automática", "automat",
            "s tronic", "dsg", "dct", "tiptronic", "steptronic",
            "cvt", "xtronic", "powershift", "e-drive", "edrive",
            " at ", "6at", "7at", "8at", "9at",
        ]
        trans = "Automática" if any(p in texto_l for p in PATRON_AUTO) else "Manual"
        # Eléctricos puros: siempre automáticos
        if comb == "Eléctrico":
            trans = "Automática"

        # Tipo venta
        tipo_venta = "Usado"
        for tv in ["Km 0", "km 0", "KM 0", "Casi nuevo", "Demo"]:
            if tv in texto:
                tipo_venta = tv.title()
                break

        # Provincia: selector específico → heurística → URL fallback
        prov_el = card.query_selector('[class*=location], [class*=Location], [class*=province], [class*=city], [class*=ubica]')
        if prov_el:
            provincia = prov_el.inner_text().strip()
        else:
            IGNORAR = {"reservable", "comparar", "profesional", "nuevo", "usado", "garantia",
                       "iva", "precio", "contado", "financiado", "km 0", "con", "sin",
                       "mes", "tae", "ver", "entrada", "meses", "incluido", "justo",
                       "envio", "envío", "disponible", "outlet", "demo", "certificado"}
            lineas = [l.strip() for l in texto.split('\n') if l.strip()]
            provincia = ""
            for l in reversed(lineas):
                low = l.lower()
                if (not re.search(r'\d', l) and 4 <= len(l) <= 40
                        and not any(ign in low for ign in IGNORAR)):
                    provincia = l
                    break
        # Fallback provincia desde URL: "-en-madrid-"
        if not provincia and url:
            url_prov = re.search(r'-en-([a-z][a-z-]+)-\d+', url)
            if url_prov:
                provincia = url_prov.group(1).replace('-', ' ').title()

        # Marca y modelo desde título
        partes = titulo.split(' ', 1)
        marca  = normalizar_marca(partes[0]) if partes else ""
        modelo = partes[1].strip() if len(partes) > 1 else titulo

        if precio <= 500 or año < 1990:
            return None

        antiguedad = datetime.now().year - año
        km_por_año = round(km / max(antiguedad, 1)) if km > 0 else 0

        return {
            "marca":              marca,
            "modelo":             modelo,
            "año":                año,
            "kilometraje_km":     km,
            "potencia_cv":        cv,
            "combustible":        comb,
            "transmision":        trans,
            "etiqueta_ambiental": etiqueta_dgt(comb, año),
            "tipo_venta":         tipo_venta,
            "precio_eur":         precio,
            "provincia":          provincia,
            "url":                url,
            "url_foto":           url_foto,
            "antiguedad":         antiguedad,
            "km_por_año":         km_por_año,
            "fecha_scraping":     datetime.now().strftime("%Y-%m-%d"),
        }
    except Exception as ex:
        return None

# ── Scraping de una página ─────────────────────────────────────────────────────
def scrapear_pagina(page, num_pag: int, modo_test: bool = False) -> list[dict]:
    url = URL_BASE.format(num_pag)
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(2000)

    # Scroll incremental para activar carga lazy
    for _ in range(6):
        page.evaluate("window.scrollBy(0, 900)")
        page.wait_for_timeout(400)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(800)

    # Descubrir selector de tarjetas
    selector = None
    cards = []
    for s in [
        '[class*=CardBasic]',
        '[class*=card-basic]',
        '[class*=VehicleCard]',
        'article',
        '[class*=listing-item]',
        '[class*=search-item]',
        '[class*=mt-Card]',
    ]:
        cards = page.query_selector_all(s)
        if len(cards) > 3:
            selector = s
            break

    if modo_test:
        print(f"\nURL: {url}")
        print(f"Selector encontrado: {selector!r}")
        print(f"Total tarjetas: {len(cards)}")
        if selector and cards:
            print("\n--- Texto tarjeta 1 ---")
            print(cards[0].inner_text()[:600])
            if len(cards) > 1:
                print("\n--- Texto tarjeta 2 ---")
                print(cards[1].inner_text()[:400])
        return []

    # modo test-foto: parsea y muestra url_foto de los primeros 5 anuncios
    if getattr(scrapear_pagina, '_modo_test_foto', False):
        resultados = []
        for card in cards[:10]:
            r = parsear_tarjeta(card)
            if r:
                resultados.append(r)
                if len(resultados) >= 5:
                    break
        for i, r in enumerate(resultados, 1):
            print(f"\n[{i}] {r['marca']} {r['modelo']} ({r['año']}) — {r['precio_eur']:,.0f} EUR")
            print(f"    url_foto: {r['url_foto'] or '(sin foto)'}")
        return resultados

    if not selector:
        return []

    registros = []
    for card in page.query_selector_all(selector):
        r = parsear_tarjeta(card)
        if r:
            registros.append(r)
    return registros

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    test      = "--test"      in sys.argv
    test_foto = "--test-foto" in sys.argv

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="es-ES",
            viewport={"width": 1366, "height": 768},
            extra_http_headers={"Accept-Language": "es-ES,es;q=0.9"},
        )
        # Ocultar rastros de automatizacion
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()

        if test:
            scrapear_pagina(page, 1, modo_test=True)
            browser.close()
            return

        if test_foto:
            scrapear_pagina._modo_test_foto = True
            print("--- Test extraccion de fotos (pag 1) ---")
            scrapear_pagina(page, 1)
            browser.close()
            return

        # Scraping completo
        registros = []
        print(f"Iniciando scraping ({MAX_PAGINAS} páginas)...")
        for pag in range(1, MAX_PAGINAS + 1):
            try:
                nuevos = scrapear_pagina(page, pag)
                registros.extend(nuevos)
                print(f"  Pág {pag:3}/{MAX_PAGINAS} — {len(nuevos):3} anuncios válidos (total: {len(registros)})")
                if not nuevos and pag > 3:
                    print("  Sin resultados, fin del scraping.")
                    break
                time.sleep(PAUSA_SEG)
            except KeyboardInterrupt:
                print("\nInterrumpido.")
                break
            except Exception as ex:
                print(f"  Error en pág {pag}: {ex}")
                continue

        browser.close()

    if not registros:
        print("No se obtuvieron registros.")
        return

    df = pd.DataFrame(registros)
    df.to_csv(SALIDA_CSV, index=False, encoding="utf-8-sig")
    con_foto = df["url_foto"].notna() & (df["url_foto"] != "")
    print(f"\nGuardado: {SALIDA_CSV}  ({len(df)} anuncios, {con_foto.sum()} con foto)")
    print(df[["marca", "modelo", "año", "kilometraje_km", "precio_eur", "url_foto"]].head(5).to_string())

if __name__ == "__main__":
    main()

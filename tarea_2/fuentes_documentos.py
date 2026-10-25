"""Corpus de documentos: de dónde sale cada una de las 25 imágenes.

- Impresos nítidos y fotos: capturas propias de periódicos (El Comercio y La Crónica, 1939 y 1945),
  leyes del Estatuto Electoral de 1939 y el folleto «Tres discursos» del Bloque Antifascista
  Peruano (1945). Están en PDF en una carpeta local; cada documento es una página del PDF.
- Manuscritos: cartas y postales del siglo XX que la Biblioteca Nacional del Perú y el Archivo
  José Carlos Mariátegui publicaron en Wikimedia Commons con licencias libres.

    from fuentes_documentos import preparar_documentos
    catalogo = preparar_documentos()
"""
from __future__ import annotations

import io
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import pymupdf
from PIL import Image

RAIZ = Path(__file__).resolve().parent
CARPETA = RAIZ / "datos" / "raw" / "documentos"
UA = {"User-Agent": "Tarea2-PythonIntermedio-PUCP/1.0 (proyecto academico; https://github.com/saffron29/tarea_2)"}

# Carpeta local con las capturas propias (no se versiona: son ~300 MB de PDF)
FUENTES_PROPIAS = Path(r"C:\Users\USER\Desktop\Documentos Frank\2024-0\Prácticas Preprofesionales - Asistencia a Fernando Tuesta")

COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath/"
API = "https://commons.wikimedia.org/w/api.php"


def _get(url: str) -> bytes:
    for intento in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(15 * (intento + 1))
    raise RuntimeError(url)


def bajar_commons(titulo: str, destino: Path) -> Path:
    """Baja un archivo de Commons una sola vez."""
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(_get(COMMONS + urllib.parse.quote(titulo.removeprefix("File:"))))
        time.sleep(random.uniform(1, 2))  # cortesía con el servidor
    return destino


def licencia_commons(titulo: str) -> dict:
    """Licencia y autor declarados en la página del archivo."""
    url = API + "?" + urllib.parse.urlencode({"action": "query", "titles": titulo, "prop": "imageinfo",
                                             "iiprop": "extmetadata|url", "format": "json"})
    pagina = next(iter(json.loads(_get(url))["query"]["pages"].values()))
    info = pagina["imageinfo"][0]
    meta = info["extmetadata"]
    time.sleep(random.uniform(1, 2))
    return {"licencia": meta.get("LicenseShortName", {}).get("value", ""), "fuente_url": info["descriptionurl"]}


def pagina_a_imagen(pdf: Path, numero: int, destino: Path, dpi: int = 200, max_lado: int = 2000) -> Path:
    """Una página de un PDF -> imagen. Si la página es una sola imagen incrustada, se usa tal cual
    (sin volver a rasterizarla); si no, se renderiza a `dpi`."""
    if destino.exists():
        return destino
    doc = pymupdf.open(pdf)
    pagina = doc[numero - 1]
    imagenes = pagina.get_images()
    if len(imagenes) == 1:
        datos = doc.extract_image(imagenes[0][0])
        imagen = Image.open(io.BytesIO(datos["image"]))
    else:
        pix = pagina.get_pixmap(dpi=dpi)
        imagen = Image.open(io.BytesIO(pix.tobytes("png")))
    imagen = imagen.convert("RGB")
    imagen.thumbnail((max_lado, max_lado))  # en CPU, el detector tarda según los píxeles
    destino.parent.mkdir(parents=True, exist_ok=True)
    imagen.save(destino, quality=92)
    return destino

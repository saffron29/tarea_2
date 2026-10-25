"""Copia liviana y publicable del corpus para el dashboard (datos/ejemplos/).

- Imágenes: solo los documentos con licencia que permite publicarlos (imagen_publicable), reducidos a
  1400 px y JPEG de calidad 80. Las fotos del folleto de 1945 no se copian. Los nombres de
  particulares (revisión de datos/privado/revision_nombres.csv) se tapan con un rectángulo negro,
  ubicado con las cajas de línea del OCR.
- Audio: los 32 clips (dominio público, CC0 o CC BY 4.0) en OGG Vorbis mono de 16 kHz.

    uv run python scripts/preparar_ejemplos.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from PIL import Image, ImageDraw
from rapidfuzz import fuzz

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from procesar_documentos import CATALOGO  # noqa: E402

DESTINO = RAIZ / "datos" / "ejemplos"
DOCS = RAIZ / "datos" / "raw" / "documentos"
REVISION = RAIZ / "datos" / "privado" / "revision_nombres.csv"


def nombres_particulares() -> list[str]:
    if not REVISION.exists():
        return []
    revision = pd.read_csv(REVISION)
    return revision.loc[revision["decision"] == "particular", "nombre"].tolist()


def tachar(imagen: Image.Image, doc_id: str, nombres: list[str]) -> int:
    """Tapa, en cada línea del OCR que contiene un nombre particular, el tramo horizontal del nombre."""
    lineas = json.loads((DOCS / "ocr" / f"{doc_id}.json").read_text(encoding="utf-8"))["lineas"]
    dibujo = ImageDraw.Draw(imagen)
    tachados = 0
    for linea in lineas:
        texto = linea["texto"]
        for nombre in nombres:
            apellido = nombre.split()[-1].lower()
            # el apellido debe estar como palabra (evita falsos positivos en el texto ilegible de los manuscritos)
            if not any(fuzz.ratio(apellido, palabra.lower().strip(".,;:")) >= 85 for palabra in texto.split()):
                continue
            alineado = fuzz.partial_ratio_alignment(nombre.lower(), texto.lower())
            if alineado.score < 85:
                continue
            ancho = linea["x1"] - linea["x0"]
            x0 = linea["x0"] + ancho * alineado.dest_start / max(len(texto), 1)
            x1 = linea["x0"] + ancho * alineado.dest_end / max(len(texto), 1)
            dibujo.rectangle([x0 - 6, linea["y0"] - 3, x1 + 6, linea["y1"] + 3], fill="black")
            tachados += 1
    return tachados


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    nombres = nombres_particulares()
    for doc_id, *_, publicable in CATALOGO:
        if not publicable:
            continue
        imagen = Image.open(DOCS / f"{doc_id}.jpg").convert("RGB")
        if nombres and (n := tachar(imagen, doc_id, nombres)):
            print(f"{doc_id}: {n} nombre(s) de particulares tapado(s)")
        imagen.thumbnail((1400, 1400))
        imagen.save(DESTINO / f"{doc_id}.jpg", quality=80, optimize=True)
    for wav in sorted((RAIZ / "datos" / "raw" / "audio" / "wav").glob("clip_*.wav")):
        destino = DESTINO / f"{wav.stem}.ogg"
        if destino.exists():
            continue
        audio, sr = sf.read(wav, dtype="float32")
        # libsndfile desborda la pila si se le pasa un OGG largo de una vez: se escribe por bloques de 10 s
        with sf.SoundFile(destino, "w", sr, 1, format="OGG", subtype="VORBIS") as salida:
            for inicio in range(0, len(audio), sr * 10):
                salida.write(audio[inicio:inicio + sr * 10])
    total = sum(p.stat().st_size for p in DESTINO.iterdir()) / 1e6
    print(f"{len(list(DESTINO.iterdir()))} archivos · {total:.1f} MB en {DESTINO}")


if __name__ == "__main__":
    main()

"""Copia liviana y publicable del corpus para el dashboard (datos/ejemplos/).

- Imágenes: solo los documentos con licencia que permite publicarlos (imagen_publicable), reducidos a
  1400 px y JPEG de calidad 80. Las fotos del folleto de 1945 no se copian.
- Audio: los 32 clips (dominio público, CC0 o CC BY 4.0) en OGG Vorbis mono de 16 kHz.

    uv run python scripts/preparar_ejemplos.py
"""
import sys
from pathlib import Path

import soundfile as sf
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from procesar_documentos import CATALOGO  # noqa: E402

DESTINO = RAIZ / "datos" / "ejemplos"


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    for doc_id, *_, publicable in CATALOGO:
        if publicable:
            imagen = Image.open(RAIZ / "datos" / "raw" / "documentos" / f"{doc_id}.jpg").convert("RGB")
            imagen.thumbnail((1400, 1400))
            imagen.save(DESTINO / f"{doc_id}.jpg", quality=80, optimize=True)
    for wav in sorted((RAIZ / "datos" / "raw" / "audio" / "wav").glob("clip_*.wav")):
        audio, sr = sf.read(wav, dtype="float32")
        # libsndfile desborda la pila si se le pasa un OGG largo de una vez: se escribe por bloques de 10 s
        with sf.SoundFile(DESTINO / f"{wav.stem}.ogg", "w", sr, 1, format="OGG", subtype="VORBIS") as salida:
            for inicio in range(0, len(audio), sr * 10):
                salida.write(audio[inicio:inicio + sr * 10])
    total = sum(p.stat().st_size for p in DESTINO.iterdir()) / 1e6
    print(f"{len(list(DESTINO.iterdir()))} archivos · {total:.1f} MB en {DESTINO}")


if __name__ == "__main__":
    main()

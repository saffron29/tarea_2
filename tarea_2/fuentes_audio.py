"""Descarga del corpus de audio: noticieros peruanos de la década de 1940 en Wikimedia Commons.

Los videos fueron digitalizados por la Biblioteca Nacional del Perú (BNP) a partir de cintas de
nitrato de 35 mm y publicados en Commons con licencia Public Domain Mark (del propietario), CC0
o CC BY 4.0. Se baja la versión liviana de 240p (el audio Opus es el mismo) y se extrae el audio
a WAV mono de 16 kHz, el formato que esperan todos los modelos de voz de la sesión 13.

    from fuentes_audio import bajar_noticieros
    atribucion = bajar_noticieros()

Buenas prácticas de descarga: User-Agent descriptivo, pausa de 1 a 2 s entre pedidos y nada se
vuelve a bajar si ya está en disco.
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

CARPETA = Path(__file__).resolve().parent / "datos" / "raw" / "audio"
SR = 16_000
UA = {"User-Agent": "Tarea2-PythonIntermedio-PUCP/1.0 (proyecto academico; https://github.com/saffron29/tarea_2)"}
API = "https://commons.wikimedia.org/w/api.php"

# id local -> título del archivo en Commons (orden alfabético del título, tal como aparece en la
# categoría "Videos of the 1940s from Peru"; el último es una película de 1942 con diálogos)
NOTICIEROS = {
    "clip_001": "File:Centenario del nacimiento de Don José Antonio Miro Quesada.webm",
    "clip_002": "File:Chosica celebra su cincuentenario.webm",
    "clip_003": "File:Del Viejo Lima (Los Descalzos).webm",
    "clip_004": "File:Desfile militar por fiestas patrias.webm",
    "clip_005": "File:El balcón de la libertad, una película documental.webm",
    "clip_006": "File:El Dr. Eduardo Santos en el Country Club.webm",
    "clip_007": "File:El regreso del cardenal Guevara.webm",
    "clip_008": "File:El Satipo.webm",
    "clip_009": "File:En el museo de Antropología y Arqueología.webm",
    "clip_010": "File:En las ruinas de Chan Chan.webm",
    "clip_011": "File:Inmigrantes.webm",
    "clip_012": "File:Instalación del Instituto Indigenista Peruano.webm",
    "clip_013": "File:Inti Raymi, fiesta milenaria.webm",
    "clip_014": "File:Mar Y Playa En Miraflores.webm",
    "clip_015": "File:Muere un sabio peruano (Fallecimiento de Julio C. Tello).webm",
    "clip_016": "File:Nueva Central de la Asistencia Pública.webm",
    "clip_017": 'File:Playas de "Agua Dulce".webm',
    "clip_018": "File:Preparativos de la expedición Kon Tiki.webm",
    "clip_019": "File:Puericultorio Pérez Araníbar, sección Víctor Larco Herrera.webm",
    "clip_020": "File:Regatas En Chorrillos.webm",
    "clip_021": "File:Reunión de boy-scouts en Lima.webm",
    "clip_022": "File:Sacsayhuamán revive las legendarias Fiestas del Sol.webm",
    "clip_023": "File:Scouts peruanos.webm",
    "clip_024": "File:Se inaugura un teatro en Ancón.webm",
    "clip_025": "File:Semana Santa en Ayacucho.webm",
    "clip_026": "File:Tarma, la ciudad florida de la sierra.webm",
    "clip_027": "File:Toma de mando del presidente Bustamante y Rivero.webm",
    "clip_028": "File:Una playa pintoresca.webm",
    "clip_029": "File:Verano En La Punta.webm",
    "clip_030": "File:Visiones de la provincia de Cajamarca.webm",
    "clip_031": "File:Visiones de la provincia de Huancayo.webm",
    "clip_032": "File:Alerta en la Frontera (Evd42DL0ySs).webm",
}

# De la película (54 min) solo se usa un tramo con diálogos para la diarización
TRAMOS = {"clip_032": (600.0, 900.0)}


def _api(parametros: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(dict(parametros, format="json"))
    for intento in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:  # demasiados pedidos: esperar y reintentar
                raise
            time.sleep(15 * (intento + 1))
    raise RuntimeError("Commons no respondió")


def _limpiar(html: str) -> str:
    texto = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", texto).strip()


def metadatos() -> pd.DataFrame:
    """Licencia, autor, fecha y URLs de cada video, leídos de la API de Commons."""
    filas = []
    titulos = list(NOTICIEROS.values())
    for i in range(0, len(titulos), 20):
        d = _api({"action": "query", "titles": "|".join(titulos[i:i + 20]), "prop": "videoinfo",
                  "viprop": "url|size|extmetadata|derivatives"})
        for p in d["query"]["pages"].values():
            vi = p["videoinfo"][0]
            em = vi.get("extmetadata", {})
            liviano = [x["src"] for x in vi.get("derivatives", []) if str(x.get("transcodekey", "")).startswith("240p")]
            filas.append({
                "titulo_commons": p["title"],
                "licencia": _limpiar(em.get("LicenseShortName", {}).get("value")),
                "autor": _limpiar(em.get("Artist", {}).get("value")),
                "fecha_original": _limpiar(em.get("DateTimeOriginal", {}).get("value"))[:40],
                "descripcion": _limpiar(em.get("ImageDescription", {}).get("value"))[:300],
                "fuente_url": vi["descriptionurl"],
                "url_descarga": liviano[0] if liviano else vi["url"],
            })
        time.sleep(random.uniform(1, 2))
    tabla = pd.DataFrame(filas)
    ids = {titulo: clave for clave, titulo in NOTICIEROS.items()}
    tabla.insert(0, "id", tabla["titulo_commons"].map(ids))
    return tabla.sort_values("id").reset_index(drop=True)


def a_wav16k(origen: Path, destino: Path, tramo: tuple[float, float] | None = None) -> float:
    """Extrae el audio del video a WAV mono de 16 kHz. Devuelve la duración en segundos."""
    from faster_whisper import decode_audio  # usa PyAV: no hace falta instalar ffmpeg
    audio = decode_audio(str(origen), sampling_rate=SR)
    if tramo:
        audio = audio[int(tramo[0] * SR):int(tramo[1] * SR)]
    sf.write(destino, audio.astype(np.float32), SR)
    return len(audio) / SR


def bajar_noticieros() -> pd.DataFrame:
    """Baja los videos (240p) y deja su audio en datos/raw/audio/wav/. Devuelve la atribución."""
    (CARPETA / "originales").mkdir(parents=True, exist_ok=True)
    (CARPETA / "wav").mkdir(parents=True, exist_ok=True)
    indice = CARPETA / "atribucion.csv"
    tabla = pd.read_csv(indice) if indice.exists() else metadatos()
    duraciones = []
    for fila in tabla.itertuples():
        video = CARPETA / "originales" / f"{fila.id}.webm"
        wav = CARPETA / "wav" / f"{fila.id}.wav"
        if not video.exists():
            with urllib.request.urlopen(urllib.request.Request(fila.url_descarga, headers=UA), timeout=300) as r:
                video.write_bytes(r.read())
            time.sleep(random.uniform(1, 2))  # cortesía con el servidor
        if not wav.exists():
            a_wav16k(video, wav, TRAMOS.get(fila.id))
        duraciones.append(round(sf.info(wav).duration, 1))
    tabla["duracion_seg"] = duraciones
    tabla["tramo_usado"] = [f"{TRAMOS[i][0]:.0f}-{TRAMOS[i][1]:.0f} s" if i in TRAMOS else "completo" for i in tabla["id"]]
    tabla.to_csv(indice, index=False)
    return tabla


if __name__ == "__main__":
    t = bajar_noticieros()
    print(t[["id", "licencia", "fecha_original", "duracion_seg"]].to_string(index=False))
    print(f"\nTotal: {t['duracion_seg'].sum() / 60:.1f} min en {len(t)} clips")

"""ASR del corpus con voz.py (sesión 13): VAD + faster-whisper + diarización + exportación.

Cada clip deja en datos/raw/audio/salidas/ su tabla de segmentos (CSV), subtítulos (SRT) y
transcripción para leer (TXT). Si un clip ya fue procesado, se lee del disco y no se repite.

    uv run python procesar_audio.py        # procesa todo lo que falte
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from voz import SR, Segmento, Transcriptor, a_srt, a_tabla, a_texto, cargar_audio, diarizar  # noqa: E402

AUDIO = RAIZ / "datos" / "raw" / "audio"
SALIDAS = AUDIO / "salidas"
MODELO_ASR = "faster-whisper"          # Whisper large-v3-turbo en CTranslate2, int8, CPU
CONFIG_ASR = "faster-whisper large-v3-turbo (int8, CPU, beam 5, VAD Silero on, idioma=es)"
UMBRAL_DIARIZACION = 0.7               # el mismo valor por defecto de voz.diarizar


def procesar_clip(clip_id: str, transcriptor: Transcriptor | None = None) -> dict:
    """Transcribe y diariza un clip. Devuelve un resumen y deja CSV/SRT/TXT en disco."""
    SALIDAS.mkdir(parents=True, exist_ok=True)
    resumen_json = SALIDAS / f"{clip_id}.json"
    if resumen_json.exists():
        return json.loads(resumen_json.read_text(encoding="utf-8"))
    audio = cargar_audio(AUDIO / "wav" / f"{clip_id}.wav")
    t0 = time.perf_counter()
    segmentos = transcriptor.segmentos(audio, idioma="es")
    segundos_asr = time.perf_counter() - t0
    if segmentos:
        segmentos = diarizar(audio, segmentos, umbral=UMBRAL_DIARIZACION)
    tabla = a_tabla(segmentos)
    tabla.to_csv(SALIDAS / f"{clip_id}.csv", index=False)
    (SALIDAS / f"{clip_id}.srt").write_text(a_srt(segmentos), encoding="utf-8")
    (SALIDAS / f"{clip_id}.txt").write_text(a_texto(segmentos), encoding="utf-8")
    resumen = {
        "id": clip_id,
        "duracion_seg": round(len(audio) / SR, 1),
        "segundos_voz": round(float((tabla["fin"] - tabla["inicio"]).sum()), 1) if len(tabla) else 0.0,
        "n_segmentos": len(tabla),
        "n_hablantes": int(tabla["hablante"].nunique()) if len(tabla) else 0,
        "segundos_procesamiento": round(segundos_asr, 1),
        "segmentos": tabla.to_dict(orient="records"),
    }
    resumen_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    return resumen


def diarizar_por_tramos(clip_id: str, transcriptor: Transcriptor | None = None, silencio_min: float = 0.3,
                        max_tramo: float = 12.0) -> dict:
    """Segunda pasada para clips con varias voces: el VAD corta tramos cortos (una sola voz cada uno),
    se transcribe cada tramo por separado y después se diariza. Resultado en caché."""
    archivo = SALIDAS / f"{clip_id}_tramos.json"
    if archivo.exists():
        return json.loads(archivo.read_text(encoding="utf-8"))
    from voz import tramos_de_voz
    transcriptor = transcriptor or Transcriptor(MODELO_ASR)
    audio = cargar_audio(AUDIO / "wav" / f"{clip_id}.wav")
    t0 = time.perf_counter()
    segmentos = []
    for a, b in tramos_de_voz(audio, silencio_min=silencio_min, max_tramo=max_tramo):
        texto = transcriptor.transcribir(audio[int(a * SR):int(b * SR)], "es").strip()
        if texto:
            segmentos.append(Segmento(round(a, 2), round(b, 2), texto))
    segmentos = diarizar(audio, segmentos, umbral=UMBRAL_DIARIZACION)
    resultado = {"id": clip_id, "segundos_procesamiento": round(time.perf_counter() - t0, 1),
                 "segmentos": a_tabla(segmentos).to_dict(orient="records")}
    archivo.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


if __name__ == "__main__":
    ids = sorted(p.stem for p in (AUDIO / "wav").glob("clip_*.wav"))
    pendientes = [i for i in ids if not (SALIDAS / f"{i}.json").exists()]
    pendientes.sort(key=lambda i: i != "clip_032")  # primero el clip con diálogos: prueba la diarización
    print(f"{len(pendientes)} clips por procesar de {len(ids)}", flush=True)
    if pendientes:
        t = Transcriptor(MODELO_ASR)
        for i in pendientes:
            r = procesar_clip(i, t)
            print(f"{i}: {r['duracion_seg']:6.1f} s · {r['n_segmentos']:3d} segmentos · "
                  f"{r['n_hablantes']} hablantes · {r['segundos_procesamiento']:6.1f} s de cómputo", flush=True)

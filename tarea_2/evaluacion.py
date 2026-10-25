"""Parte 2: los mismos 10 documentos y 10 clips leídos por varios modelos (salidas en caché).

Las configuraciones se fijaron antes de tener las transcripciones de referencia y no se ajustan
después de ver las métricas.

Documentos
- paddleocr_base: PaddleOCR 3.7 · PP-OCRv6 sin pasos previos (la salida de la Parte 1).
- paddleocr_preproc: el mismo modelo con los tres pasos previos de PaddleOCR activados
  (orientación de página, desdoblado UVDoc y orientación de cada línea).
- easyocr: EasyOCR 1.7 (detector CRAFT + reconocedor latino) con idioma español; otra familia de OCR.

Audio
- fw_small: faster-whisper small en CPU (int8, beam 5, VAD), la línea base pedida.
- fw_turbo: faster-whisper large-v3-turbo en CPU (int8, beam 5, VAD), el modelo del corpus.
- qwen3_asr: Qwen3-ASR 0.6B (Alibaba, Apache 2.0) en CPU, con los tramos del VAD de voz.py.

    uv run python evaluacion.py asr      # corre lo que falte
    uv run python evaluacion.py ocr
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
from kit_transcripcion import MUESTRA_CLIPS, MUESTRA_DOCS  # noqa: E402

CACHE = RAIZ / "datos" / "raw" / "evaluacion"
DOCS = RAIZ / "datos" / "raw" / "documentos"
WAV = RAIZ / "datos" / "raw" / "audio" / "wav"
HILOS = 4

MODELOS_OCR = {
    "paddleocr_base": "PaddleOCR 3.7 · PP-OCRv6 (configuración base de la clase)",
    "paddleocr_preproc": "PaddleOCR 3.7 · PP-OCRv6 + orientación, desdoblado UVDoc y orientación de líneas",
    "easyocr": "EasyOCR 1.7 · CRAFT + reconocedor latino (es)",
}
MODELOS_ASR = {
    "fw_small": "faster-whisper small (int8, CPU, beam 5, VAD) · línea base",
    "fw_turbo": "faster-whisper large-v3-turbo (int8, CPU, beam 5, VAD)",
    "qwen3_asr": "Qwen3-ASR 0.6B (CPU, float32, tramos VAD de voz.py)",
}


def _guardar(archivo: Path, datos: dict) -> dict:
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    return datos


def _leer(archivo: Path) -> dict | None:
    return json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else None


# ───────────────────────────────────────────────────────── OCR
def _crear_ocr(modelo: str):
    if modelo == "paddleocr_preproc":
        from paddleocr import PaddleOCR
        return PaddleOCR(device="cpu", enable_mkldnn=False, cpu_threads=HILOS, use_doc_orientation_classify=True,
                         use_doc_unwarping=True, use_textline_orientation=True)
    if modelo == "easyocr":
        import easyocr
        import torch
        torch.set_num_threads(HILOS)
        return easyocr.Reader(["es"], gpu=False, verbose=False)
    raise ValueError(modelo)


def _leer_imagen(lector, modelo: str, ruta: Path) -> list[str]:
    if modelo == "paddleocr_preproc":
        return list(lector.predict(str(ruta))[0]["rec_texts"])
    if modelo == "easyocr":
        # detail=1: (caja, texto, confianza); se ordena de arriba abajo y de izquierda a derecha
        filas = lector.readtext(str(ruta), detail=1, paragraph=False)
        filas.sort(key=lambda f: (round(min(p[1] for p in f[0]) / 20), min(p[0] for p in f[0])))
        return [f[1] for f in filas]
    raise ValueError(modelo)


def ocr(modelo: str, doc_id: str, lector=None) -> dict:
    archivo = CACHE / f"ocr_{modelo}" / f"{doc_id}.json"
    if (hecho := _leer(archivo)) is not None:
        return hecho
    if modelo == "paddleocr_base":  # la salida de la Parte 1
        base = json.loads((DOCS / "ocr" / f"{doc_id}.json").read_text(encoding="utf-8"))
        return {"id": doc_id, "modelo": modelo, "texto": "\n".join(l["texto"] for l in base["lineas"]), "segundos": base["segundos"]}
    t0 = time.perf_counter()
    lineas = _leer_imagen(lector, modelo, DOCS / f"{doc_id}.jpg")
    return _guardar(archivo, {"id": doc_id, "modelo": modelo, "texto": "\n".join(lineas),
                              "segundos": round(time.perf_counter() - t0, 2)})


# ───────────────────────────────────────────────────────── ASR
def _crear_asr(modelo: str):
    if modelo == "fw_small":
        from faster_whisper import WhisperModel
        return WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=HILOS)
    if modelo == "fw_turbo":
        from faster_whisper import WhisperModel
        return WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=HILOS)
    if modelo == "qwen3_asr":
        import torch
        from voz import Transcriptor
        torch.set_num_threads(HILOS)
        return Transcriptor("qwen3-asr-0.6b", device="cpu")
    raise ValueError(modelo)


def asr(modelo: str, clip_id: str, transcriptor=None) -> dict:
    archivo = CACHE / f"asr_{modelo}" / f"{clip_id}.json"
    if (hecho := _leer(archivo)) is not None:
        return hecho
    from voz import SR, cargar_audio
    audio = cargar_audio(WAV / f"{clip_id}.wav")
    t0 = time.perf_counter()
    if modelo.startswith("fw_"):
        segmentos, _ = transcriptor.transcribe(audio, language="es", beam_size=5, vad_filter=True)
        texto = " ".join(s.text.strip() for s in segmentos)
    else:
        texto = " ".join(s.texto for s in transcriptor.segmentos(audio, idioma="es"))
    segundos = time.perf_counter() - t0
    duracion = len(audio) / SR
    return _guardar(archivo, {"id": clip_id, "modelo": modelo, "texto": texto, "segundos": round(segundos, 2),
                              "duracion_seg": round(duracion, 1), "factor_tiempo_real": round(segundos / duracion, 3)})


def correr(modalidad: str):
    if modalidad == "asr":
        for modelo in MODELOS_ASR:
            pendientes = [c for c in MUESTRA_CLIPS if not (CACHE / f"asr_{modelo}" / f"{c}.json").exists()]
            if not pendientes:
                continue
            t0 = time.perf_counter()
            transcriptor = _crear_asr(modelo)
            print(f"{modelo}: modelo cargado en {time.perf_counter() - t0:.0f} s", flush=True)
            for c in pendientes:
                r = asr(modelo, c, transcriptor)
                print(f"  {c}: factor de tiempo real {r['factor_tiempo_real']:.2f} · {r['texto'][:70]}", flush=True)
            del transcriptor
    else:
        for modelo in [m for m in MODELOS_OCR if m != "paddleocr_base"]:
            pendientes = [d for d in MUESTRA_DOCS if not (CACHE / f"ocr_{modelo}" / f"{d}.json").exists()]
            if not pendientes:
                continue
            lector = _crear_ocr(modelo)
            for d in pendientes:
                r = ocr(modelo, d, lector)
                print(f"  {modelo} {d}: {r['segundos']:.1f} s · {r['texto'][:70]!r}", flush=True)
            del lector


if __name__ == "__main__":
    correr(sys.argv[1])

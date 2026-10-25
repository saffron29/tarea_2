"""Corre la extracción del LLM sobre los 57 registros publicados (la caché la reutiliza el notebook 02)."""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import extraccion_llm as ex  # noqa: E402

documentos = pd.read_csv(RAIZ / "datos" / "documentos.csv")
audios = pd.read_csv(RAIZ / "datos" / "transcripciones.csv")
registros = [(f.id, f.texto_ocr, "documento") for f in documentos.itertuples()] + [(f.id, f.texto_asr, "audio") for f in audios.itertuples()]
if "--reverso" in sys.argv:  # un segundo proceso recorre la lista desde el final: ninguno repite lo ya guardado
    registros = registros[::-1]
tok = modelo = None
for registro_id, texto, modalidad in registros:
    if (ex.CACHE / f"{registro_id}.json").exists():
        continue
    if not (ex.CACHE / f"{registro_id}.json").exists() and modelo is None:
        tok, modelo = ex.cargar_modelo()
    r = ex.extraer(registro_id, texto, modalidad, tok, modelo)
    print(f"{registro_id}: JSON válido={r['json_valido']} · {r['segundos']} s · {r['tokens_entrada']} tokens", flush=True)

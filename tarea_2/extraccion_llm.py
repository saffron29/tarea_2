"""Extracción estructurada con un LLM local (Qwen3-1.7B, Apache 2.0) sobre los textos OCR y ASR.

El modelo recibe un prompt cerrado con el esquema JSON y la regla de usar null cuando un dato no
aparece en el texto. Corre en CPU con decodificación voraz (sin muestreo): el mismo texto da
siempre el mismo JSON. Cada respuesta se guarda en datos/raw/llm/<id>.json (caché).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
CACHE = RAIZ / "datos" / "raw" / "llm"
MODELO_LLM = "Qwen/Qwen3-1.7B"
CONFIG_LLM = "Qwen/Qwen3-1.7B · transformers en CPU (float32) · decodificación voraz · sin modo de razonamiento · máx. 350 tokens"
MAX_CARACTERES = 2500  # los textos más largos se recortan: en CPU el tiempo crece con el prompt

TEMAS = ["elecciones y partidos", "gobierno y Estado", "relaciones internacionales y guerra",
         "cultura, patrimonio y fiestas", "sociedad y vida cotidiana", "religión",
         "educación y ciencia", "economía y obras públicas", "correspondencia personal"]

ESQUEMA_DOCUMENTO = """{
  "modalidad": "documento",
  "tema": uno de TEMAS,
  "resumen": "una o dos oraciones fieles al texto",
  "emisor": "quién firma o emite el documento, o null",
  "destinatario": "a quién va dirigido, o null",
  "fecha": "fecha tal como aparece en el texto, o null",
  "lugar": "lugar tal como aparece en el texto, o null",
  "monto": "cantidad de dinero tal como aparece, o null"
}"""

ESQUEMA_AUDIO = """{
  "modalidad": "audio",
  "tema": uno de TEMAS,
  "resumen": "una o dos oraciones fieles a la transcripción",
  "lugar": "lugar principal mencionado, o null",
  "fecha": "fecha mencionada en el audio, o null",
  "citas_clave": ["hasta 2 frases copiadas literalmente de la transcripción"],
  "numero_hablantes": número de hablantes según las marcas [H1], [H2]..., o null si no hay marcas
}"""

INSTRUCCIONES = """Eres un archivero que cataloga documentos y noticieros peruanos de 1910 a 1950.
Lee el texto y devuelve SOLO un objeto JSON válido con exactamente estas claves:
{esquema}

TEMAS = {temas}

Reglas:
- Usa null si el dato no aparece explícitamente en el texto. No inventes ni completes datos.
- Copia fechas, lugares, nombres y montos tal como están escritos.
- El texto viene de OCR o de reconocimiento de voz y puede traer errores: no los corrijas en las citas.
- No agregues claves nuevas ni texto fuera del JSON."""


def construir_prompt(texto: str, modalidad: str) -> list[dict]:
    esquema = ESQUEMA_DOCUMENTO if modalidad == "documento" else ESQUEMA_AUDIO
    sistema = INSTRUCCIONES.replace("{esquema}", esquema).replace("{temas}", json.dumps(TEMAS, ensure_ascii=False))
    recorte = texto[:MAX_CARACTERES] + (" [...]" if len(texto) > MAX_CARACTERES else "")
    return [{"role": "system", "content": sistema},
            {"role": "user", "content": f"Texto ({modalidad}):\n\"\"\"\n{recorte}\n\"\"\""}]


def cargar_modelo():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELO_LLM)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_LLM, dtype=torch.float32)
    modelo.eval()
    return tok, modelo


def leer_json(respuesta: str) -> dict | None:
    """Primer objeto JSON de la respuesta; None si no es válido."""
    m = re.search(r"\{.*\}", respuesta, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def extraer(registro_id: str, texto: str, modalidad: str, tok=None, modelo=None) -> dict:
    """JSON extraído para un registro (con caché en disco)."""
    archivo = CACHE / f"{registro_id}.json"
    if archivo.exists():
        return json.loads(archivo.read_text(encoding="utf-8"))
    import torch
    CACHE.mkdir(parents=True, exist_ok=True)
    mensajes = construir_prompt(texto, modalidad)
    entrada = tok.apply_chat_template(mensajes, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt", return_dict=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        salida = modelo.generate(**entrada, max_new_tokens=350, do_sample=False)
    respuesta = tok.decode(salida[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    campos = leer_json(respuesta)
    resultado = {"id": registro_id, "modalidad": modalidad, "respuesta": respuesta, "campos": campos,
                 "json_valido": campos is not None, "segundos": round(time.perf_counter() - t0, 1),
                 "tokens_entrada": int(entrada["input_ids"].shape[1])}
    archivo.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


# ───────────────────────────────────────────────────────── tabla publicable
CLAVES = {"documento": ["modalidad", "tema", "resumen", "emisor", "destinatario", "fecha", "lugar", "monto"],
          "audio": ["modalidad", "tema", "resumen", "lugar", "fecha", "citas_clave", "numero_hablantes"]}


def depurar(campos, modalidad: str):
    """Deja solo las claves del esquema, en orden. Devuelve (campos, número de claves inventadas descartadas)."""
    if not isinstance(campos, dict):
        return None, 0
    extra = [k for k in campos if k not in CLAVES[modalidad]]
    return {k: campos.get(k) for k in CLAVES[modalidad]}, len(extra)


def tabla_publicable(salidas, reemplazos: dict, fecha: str):
    """datos/extraccion_llm.csv: id, modalidad, campos_json (seudonimizado), modelo_llm, fecha y controles."""
    import pandas as pd
    from seudonimos import aplicar
    filas = []
    for s in salidas:
        campos, extra = depurar(s["campos"], s["modalidad"])
        texto_json = aplicar(json.dumps(campos, ensure_ascii=False), reemplazos)[0] if campos else s["respuesta"]
        filas.append({"id": s["id"], "modalidad": s["modalidad"], "campos_json": texto_json, "modelo_llm": CONFIG_LLM,
                      "fecha": fecha, "json_valido": s["json_valido"], "claves_extra_descartadas": extra})
    return pd.DataFrame(filas)

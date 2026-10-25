"""Seudonimización: candidatos con NER y reemplazo con voz.seudonimizar (sesión 13).

Flujo:
1. `candidatos(textos)`: un modelo NER en español marca las entidades de tipo persona (PER).
2. Una persona revisa la lista y decide, nombre por nombre, si es figura pública en su rol
   público (se conserva) o particular (se reemplaza). La decisión queda en
   datos/privado/revision_nombres.csv, que NO se versiona porque contiene los nombres reales.
3. `aplicar(texto, reemplazos)`: cambia cada nombre particular por su seudónimo, también cuando el
   OCR o el ASR lo escribieron mal (comparación difusa de voz.seudonimizar).
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from voz import seudonimizar

MODELO_NER = "mrm8488/bert-spanish-cased-finetuned-ner"


def candidatos(textos: dict[str, str], minimo_letras: int = 4) -> pd.DataFrame:
    """Personas detectadas por el NER, con cuántas veces y en qué registros aparecen."""
    from transformers import pipeline
    ner = pipeline("token-classification", model=MODELO_NER, aggregation_strategy="simple", device="cpu")
    apariciones = defaultdict(lambda: {"n": 0, "ids": set(), "contexto": ""})
    for clave, texto in textos.items():
        # el NER acepta hasta 512 tokens: se procesa por párrafos o líneas
        trozos = [t for t in re.split(r"\n+", texto) if t.strip()]
        for trozo in trozos:
            for ent in ner(trozo[:1500]):
                nombre = re.sub(r"\s+", " ", ent["word"].replace("##", "")).strip(" .,;:")
                if ent["entity_group"] != "PER" or len(nombre) < minimo_letras or ent["score"] < 0.80:
                    continue
                a = apariciones[nombre]
                a["n"] += 1
                a["ids"].add(clave)
                if not a["contexto"]:
                    i = trozo.find(nombre)
                    a["contexto"] = trozo[max(0, i - 60): i + len(nombre) + 60].replace("\n", " ")
    filas = [{"nombre": k, "apariciones": v["n"], "ids": ", ".join(sorted(v["ids"])), "contexto": v["contexto"]}
             for k, v in apariciones.items()]
    return pd.DataFrame(filas).sort_values(["apariciones", "nombre"], ascending=[False, True]).reset_index(drop=True)


def tabla_de_reemplazos(revision: pd.DataFrame) -> dict[str, str]:
    """De la revisión humana a {nombre real: seudónimo}. Solo los marcados como particulares."""
    particulares = revision[revision["decision"].str.strip().str.lower() == "particular"].copy()
    # un mismo particular puede venir con variantes («Burzagli», «Ernesto Burzagli»): mismo seudónimo
    particulares["persona"] = particulares["persona"].fillna(particulares["nombre"])
    codigos = {p: f"[PERSONA_{i:02d}]" for i, p in enumerate(dict.fromkeys(particulares["persona"]), start=1)}
    # primero los nombres más largos, para que «Ernesto Burzagli» se reemplace antes que «Burzagli»
    pares = sorted(zip(particulares["nombre"], particulares["persona"]), key=lambda x: -len(x[0]))
    return {nombre: codigos[persona] for nombre, persona in pares}


def aplicar(texto: str, reemplazos: dict[str, str]) -> tuple[str, int]:
    """Texto seudonimizado y cuántos reemplazos se hicieron."""
    if not isinstance(texto, str) or not texto:
        return texto, 0
    nuevo = seudonimizar(texto, reemplazos)
    cambios = sum(nuevo.count(s) - texto.count(s) for s in set(reemplazos.values()))
    return nuevo, cambios

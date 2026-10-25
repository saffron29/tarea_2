"""Escribe 02_evaluacion.ipynb. Los textos @@...@@ se reemplazan con la interpretación de los resultados."""
import json
import sys
from pathlib import Path

import nbformat
from nb_util import code, md, notebook

RAIZ = Path(__file__).resolve().parents[1]
TEXTOS = json.loads((RAIZ / "scripts" / "textos_nb02.json").read_text(encoding="utf-8")) if (RAIZ / "scripts" / "textos_nb02.json").exists() else {}


def t(clave):
    return TEXTOS.get(clave, f"*(pendiente: {clave})*")


celdas = [
md("""
# Tarea 2 · Parte 2 — De los modelos base a la extracción estructurada

Medimos qué tan bien leen y escuchan los modelos, con **los mismos 10 documentos y 10 clips** de la Parte 1
y transcripciones de referencia hechas a mano. Luego un LLM local convierte cada texto en un JSON con campos
estructurados, y una auditoría humana de 20 casos mide cuánto se puede confiar en él.

1. Muestra de evaluación y referencias manuales
2. Normalización y métricas (CER y WER)
3. Línea base sin ajuste: PaddleOCR y faster-whisper *small*
4. Modelos alternativos: PaddleOCR con preprocesamiento, EasyOCR, Whisper *turbo* y Qwen3-ASR
5. Casos difíciles: manuscritos y audio con música
6. Extracción estructurada con Qwen3-1.7B y auditoría humana
7. Comparativa y decisión

**Regla de juego:** las configuraciones de todos los modelos (módulo `evaluacion.py`) se fijaron y se
corrieron **antes** de tener las referencias manuales; ningún parámetro se ajustó después de ver las métricas.
"""),
code("""
import json
import os
import re
import sys
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import jiwer
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

sys.path.insert(0, "scripts")
import evaluacion as ev
import extraccion_llm as ex
import seudonimos
import voz
from kit_auditoria import muestra as muestra_auditoria
from kit_transcripcion import MUESTRA_CLIPS, MUESTRA_DOCS

import logging
import transformers
logging.getLogger().setLevel(logging.ERROR)
for nombre in list(logging.root.manager.loggerDict):
    logging.getLogger(nombre).setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

pio.renderers.default = "plotly_mimetype+png"
pio.renderers["png"].width, pio.renderers["png"].height = 950, 480
pd.set_option("display.max_colwidth", 110)

DATOS = Path("datos")
documentos = pd.read_csv(DATOS / "documentos.csv", dtype={"texto_referencia": str})
transcripciones = pd.read_csv(DATOS / "transcripciones.csv", dtype={"texto_referencia": str})
from importlib.metadata import version
print(f"jiwer {version('jiwer')} · documentos {documentos.shape} · transcripciones {transcripciones.shape}")
"""),
md("""
## 1. Muestra de evaluación (sin datos nuevos)

Se eligieron de los corpus de la Parte 1, antes de correr cualquier métrica:

- **10 documentos** de los tres tipos (3 impresos, 3 fotos y 4 manuscritos), todos cortos para que la
  transcripción manual sea completa y cuidadosa;
- **10 clips** de tres orígenes distintos (Productora Huascarán, Nacional Films y productoras no
  identificadas) y condiciones de grabación variadas: narración limpia, narración con música de fondo y
  ambiente de playa o de calle.
"""),
code("""
muestra_docs = documentos.set_index("id").loc[MUESTRA_DOCS, ["tipo", "descripcion", "n_caracteres"]]
muestra_clips = transcripciones.set_index("id").loc[MUESTRA_CLIPS, ["titulo", "productora", "fecha_original", "duracion_seg", "segundos_voz"]]
print(muestra_docs["tipo"].value_counts().to_dict(), "·", muestra_clips["productora"].value_counts().to_dict())
display(muestra_docs)
display(muestra_clips)
"""),
md("""
### 1.1 Referencias manuales

El estudiante transcribió a mano los 20 casos en una página local (`scripts/kit_transcripcion.py`): cada
imagen o audio con un cuadro de texto, **sin ver las salidas de los modelos**. Documentos: línea por línea,
con la ortografía original. Audios: todo lo dicho, sin marcas de hablante.

Las referencias se publican en la columna `texto_referencia` con la misma seudonimización que los textos
de los modelos.
"""),
code("""
referencias = json.loads((DATOS / "raw" / "transcripcion_manual" / "referencias_manuales.json").read_text(encoding="utf-8"))
faltan = [i for i in MUESTRA_DOCS + MUESTRA_CLIPS if not referencias.get(i, "").strip()]
print(f"Referencias manuales: {sum(1 for i in MUESTRA_DOCS + MUESTRA_CLIPS if referencias.get(i, '').strip())} de 20 · faltan: {faltan}")

revision = pd.read_csv(DATOS / "privado" / "revision_nombres.csv")
reemplazos = seudonimos.tabla_de_reemplazos(revision)
publicar = {i: seudonimos.aplicar(referencias[i], reemplazos)[0] for i in MUESTRA_DOCS + MUESTRA_CLIPS}

documentos["texto_referencia"] = documentos["id"].map(publicar).fillna("")
transcripciones["texto_referencia"] = transcripciones["id"].map(publicar).fillna("")
documentos.to_csv(DATOS / "documentos.csv", index=False)
transcripciones.to_csv(DATOS / "transcripciones.csv", index=False)
pd.DataFrame({"id": list(publicar), "caracteres_referencia": [len(referencias[i]) for i in publicar],
              "palabras_referencia": [len(referencias[i].split()) for i in publicar]}).set_index("id").T
"""),
md("""
## 2. Normalización y métricas

Referencia y salida automática pasan por **el mismo** criterio, `voz.normalizar` de la sesión 13
(minúsculas, sin puntuación, espacios simples; las tildes se conservan), después de quitar las marcas
que no son texto ni palabras habladas: `[HABLANTE_n]`, `[ilegible]`, `[inaudible]`, `[PERSONA_nn]`.

$$\\text{CER} = \\frac{S + D + I}{N_\\text{caracteres de la referencia}} \\qquad \\text{WER} = \\frac{S + D + I}{N_\\text{palabras de la referencia}}$$

donde $S$, $D$ e $I$ son sustituciones, borrados e inserciones (distancia de Levenshtein). Reportamos dos
cifras: el **CER/WER agregado** (todos los errores sobre todos los caracteres o palabras de la muestra, como
`voz.cer` y `voz.wer`) y el **promedio por caso**, que da el mismo peso a un documento corto que a uno largo.

Las métricas se calculan sobre los textos **sin seudonimizar** (en local): así un nombre mal leído cuenta
como error, aunque en la publicación se cambie por un seudónimo.
"""),
code("""
MARCAS = re.compile(r"\\[(?:HABLANTE_\\d+|ilegible|inaudible|PERSONA_\\d+)\\]", flags=re.IGNORECASE)


def limpiar(texto: str) -> str:
    return voz.normalizar(MARCAS.sub(" ", str(texto or "")))


def cer(referencia, hipotesis):
    return jiwer.cer(limpiar(referencia), limpiar(hipotesis) or "∅")


def wer(referencia, hipotesis):
    return jiwer.wer(limpiar(referencia), limpiar(hipotesis) or "∅")


def agregado(metrica, referencias_, hipotesis_):
    return metrica([limpiar(r) for r in referencias_], [limpiar(h) or "∅" for h in hipotesis_])


print(limpiar("[HABLANTE_1] El Jurado Provincial de Elecciones de Lima, en su local; ¡Gracias!"))
"""),
md("""
## 3. Línea base sin ajuste

- **Documentos:** PaddleOCR 3.7 con PP-OCRv6 y la configuración de la clase, tal como corrió en la Parte 1.
- **Audio:** faster-whisper *small* en CPU (int8, `beam_size=5`, VAD). El **factor de tiempo real** es el
  tiempo de procesamiento dividido entre la duración del audio: 0,5 significa que un minuto de audio se
  transcribe en 30 segundos.
"""),
code("""
tipo = documentos.set_index("id")["tipo"]
filas = []
for modelo, nombre in ev.MODELOS_OCR.items():
    for doc_id in MUESTRA_DOCS:
        r = ev.ocr(modelo, doc_id)
        filas.append({"id": doc_id, "tipo": tipo[doc_id], "modelo": modelo, "descripcion_modelo": nombre,
                      "texto": r["texto"], "segundos": r["segundos"], "cer": cer(referencias[doc_id], r["texto"])})
eval_docs = pd.DataFrame(filas)

filas = []
for modelo, nombre in ev.MODELOS_ASR.items():
    for clip_id in MUESTRA_CLIPS:
        r = ev.asr(modelo, clip_id)
        filas.append({"id": clip_id, "modelo": modelo, "descripcion_modelo": nombre, "texto": r["texto"],
                      "segundos": r["segundos"], "duracion_seg": r["duracion_seg"], "factor_tiempo_real": r["factor_tiempo_real"],
                      "wer": wer(referencias[clip_id], r["texto"])})
eval_audio = pd.DataFrame(filas)

base_docs = eval_docs[eval_docs["modelo"] == "paddleocr_base"]
base_audio = eval_audio[eval_audio["modelo"] == "fw_small"]
cer_base = agregado(jiwer.cer, [referencias[i] for i in base_docs["id"]], base_docs["texto"])
wer_base = agregado(jiwer.wer, [referencias[i] for i in base_audio["id"]], base_audio["texto"])
print(f"Línea base · PaddleOCR: CER agregado {cer_base:.3f} (promedio por documento {base_docs['cer'].mean():.3f})")
print(f"Línea base · faster-whisper small: WER agregado {wer_base:.3f} (promedio por clip {base_audio['wer'].mean():.3f}) · "
      f"factor de tiempo real {base_audio['segundos'].sum() / base_audio['duracion_seg'].sum():.2f}")
"""),
code("""
base_docs[["id", "tipo", "cer"]].round(3).set_index("id").T
"""),
code("""
base_audio[["id", "wer", "factor_tiempo_real"]].round(3).set_index("id").T
"""),
md(t("interpretacion_base")),
md("""
## 4. Modelos alternativos sobre las mismas muestras

| Modalidad | Modelo | Por qué |
|---|---|---|
| Documentos | **PaddleOCR + preprocesamiento** (orientación de página, desdoblado UVDoc, orientación de líneas) | mismo reconocedor; prueba si «enderezar» la imagen ayuda en las fotos de papel curvado |
| Documentos | **EasyOCR** (CRAFT + reconocedor latino, `es`) | otra familia de OCR abierta, instalable con pip y sin binarios del sistema (Tesseract necesitaba permisos de administrador en esta PC) |
| Audio | **faster-whisper large-v3-turbo** | el modelo usado para el corpus: mismo motor que la línea base, 3,5 veces más parámetros |
| Audio | **Qwen3-ASR 0.6B** (Alibaba, Apache 2.0) | otra familia de ASR vista en la sesión 13, del tamaño de *small*; corre sobre los tramos del VAD de `voz.py` |

Todos los tiempos se midieron en la misma laptop (CPU, 4 hilos), con el modelo ya cargado.
"""),
code("""
def tabla_modelos(evaluacion, metrica, refs, columna):
    filas = []
    for modelo, grupo in evaluacion.groupby("modelo", sort=False):
        filas.append({
            "modelo": grupo["descripcion_modelo"].iloc[0], "clave": modelo,
            f"{columna} agregado": agregado(metrica, [refs[i] for i in grupo["id"]], grupo["texto"]),
            f"{columna} promedio por caso": grupo[columna.lower()].mean(),
            "segundos por caso": grupo["segundos"].median(),
        })
    return pd.DataFrame(filas)


tabla_ocr = tabla_modelos(eval_docs, jiwer.cer, referencias, "CER")
tabla_asr = tabla_modelos(eval_audio, jiwer.wer, referencias, "WER")
tabla_asr["factor de tiempo real"] = [g["segundos"].sum() / g["duracion_seg"].sum() for _, g in eval_audio.groupby("modelo", sort=False)]
display(tabla_ocr.round(3))
display(tabla_asr.round(3))
"""),
code("""
cer_tipo = eval_docs.groupby(["tipo", "modelo"], as_index=False)["cer"].mean()
base_tipo = cer_tipo[cer_tipo["modelo"] == "paddleocr_base"].set_index("tipo")["cer"]
mejor_manuscrito = cer_tipo[cer_tipo["tipo"] == "manuscrito"]["cer"].min()
fig = px.bar(cer_tipo, x="tipo", y="cer", color="modelo", barmode="group", text_auto=".2f",
             category_orders={"tipo": ["impreso", "foto", "manuscrito"], "modelo": list(ev.MODELOS_OCR)},
             title=f"Con PaddleOCR el CER es {base_tipo['foto']:.2f} en fotos y {base_tipo['impreso']:.2f} en impresos, pero "
                   f"{base_tipo['manuscrito']:.2f} en manuscritos:<br>ningún modelo baja de {mejor_manuscrito:.2f} con la letra a mano",
             labels={"cer": "CER promedio", "tipo": ""})
fig.show()
"""),
code("""
agr = tabla_asr.set_index("clave")
fig = px.scatter(eval_audio, x="factor_tiempo_real", y="wer", color="modelo", hover_data=["id"],
                 category_orders={"modelo": list(ev.MODELOS_ASR)},
                 title=f"Whisper turbo comete {agr.loc['fw_small', 'WER agregado'] / agr.loc['fw_turbo', 'WER agregado']:.1f} veces menos errores que small "
                       f"(WER {agr.loc['fw_turbo', 'WER agregado']:.2f} vs. {agr.loc['fw_small', 'WER agregado']:.2f})<br>a cambio de "
                       f"{agr.loc['fw_turbo', 'factor de tiempo real'] / agr.loc['fw_small', 'factor de tiempo real']:.1f} veces más tiempo de CPU; "
                       f"Qwen3-ASR no lo supera en ninguna de las dos cosas",
                 labels={"factor_tiempo_real": "factor de tiempo real (menos es más rápido)", "wer": "WER del clip"})
fig.show()
"""),
md(t("interpretacion_alternativos")),
md("""
## 5. Casos difíciles

### 5.1 Manuscritos
"""),
code("""
for doc_id in [d for d in MUESTRA_DOCS if tipo[d] == "manuscrito"]:
    print(f"━━ {doc_id} · referencia: {referencias[doc_id][:160]!r}")
    for f in eval_docs[eval_docs["id"] == doc_id].itertuples():
        print(f"   {f.modelo:18s} CER {f.cer:.2f} · {f.texto[:160]!r}")
    print()
"""),
md(t("interpretacion_manuscritos")),
md("""
### 5.2 Audio con música y ambiente

Los noticieros mezclan narración con música orquestal y sonido ambiente. Como medida aproximada de
cuánta música o ruido hay en cada clip usamos la **fracción del clip sin voz** según el VAD.
"""),
code("""
sin_voz = 1 - transcripciones.set_index("id")["segundos_voz"] / transcripciones.set_index("id")["duracion_seg"]
eval_audio["fraccion_sin_voz"] = eval_audio["id"].map(sin_voz)
display(eval_audio.pivot_table(index=["id", "fraccion_sin_voz"], columns="modelo", values="wer").round(2).sort_index(level=1))
peor_clip = eval_audio[eval_audio["modelo"] == "fw_turbo"].sort_values("wer").iloc[-1]["id"]
alineacion = jiwer.process_words(limpiar(referencias[peor_clip]), limpiar(eval_audio.query("modelo == 'fw_turbo' and id == @peor_clip")["texto"].iloc[0]))
print(f"Clip con más errores para turbo: {peor_clip}")
print(jiwer.visualize_alignment(alineacion, show_measures=False)[:1500])
"""),
md(t("interpretacion_audio")),
md("""
## 6. Extracción estructurada con un LLM local

**Modelo:** Qwen3-1.7B (Alibaba, Apache 2.0), en la laptop con `transformers`: ningún texto sale a una API.
Decodificación voraz (sin muestreo) y sin modo de razonamiento, así que el mismo texto da siempre el mismo JSON.

**Prompt estructurado** (en `extraccion_llm.py`): el modelo recibe el esquema, una lista cerrada de temas y
cuatro reglas: `null` si el dato no aparece, copiar fechas y nombres tal como están, no corregir errores
de OCR en las citas y no agregar claves. Los textos se recortan a 2500 caracteres para acotar el tiempo
en CPU.
"""),
code("""
print(ex.construir_prompt("(texto)", "documento")[0]["content"])
"""),
code("""
registros = pd.concat([
    documentos.assign(modalidad="documento", texto=documentos["texto_ocr"])[["id", "modalidad", "texto"]],
    transcripciones.assign(modalidad="audio", texto=transcripciones["texto_asr"])[["id", "modalidad", "texto"]],
], ignore_index=True)

tok = modelo_llm = None
salidas = []
for r in registros.itertuples():
    if not (ex.CACHE / f"{r.id}.json").exists() and modelo_llm is None:
        tok, modelo_llm = ex.cargar_modelo()
    salidas.append(ex.extraer(r.id, r.texto, r.modalidad, tok, modelo_llm))
salidas = pd.DataFrame(salidas)
print(f"Registros: {len(salidas)} · JSON válidos: {salidas['json_valido'].sum()} · "
      f"tiempo mediano {salidas['segundos'].median():.0f} s por registro · total {salidas['segundos'].sum() / 60:.0f} min")
"""),
code("""
fecha_extraccion = pd.Timestamp.now(tz="America/Lima").strftime("%Y-%m-%d")
# solo las claves del esquema (si el modelo inventó otras, se descartan y se cuentan); el JSON se publica seudonimizado
extraccion = ex.tabla_publicable(salidas.to_dict(orient="records"), reemplazos, fecha_extraccion)
extraccion.to_csv(DATOS / "extraccion_llm.csv", index=False)
print(f"datos/extraccion_llm.csv: {extraccion.shape} · claves inventadas descartadas: {extraccion['claves_extra_descartadas'].sum()}")
extraccion.head(4)
"""),
code("""
temas = pd.Series([json.loads(j).get("tema") for j in extraccion.loc[extraccion["json_valido"], "campos_json"]]).value_counts()
nulos = {k: np.mean([json.loads(j).get(k) is None for j in extraccion.loc[(extraccion["json_valido"]) & (extraccion["modalidad"] == "documento"), "campos_json"]])
         for k in ["emisor", "destinatario", "fecha", "lugar", "monto"]}
print("Temas asignados:", temas.to_dict())
print("Fracción de documentos con null:", {k: round(float(v), 2) for k, v in nulos.items()})
"""),
md("""
### 6.1 Auditoría humana de 20 extracciones

**Muestra:** 10 documentos y 10 audios elegidos al azar (`random_state=2026`) con `scripts/kit_auditoria.py`,
que muestra el texto publicado al lado del JSON con un selector correcto/incorrecto por campo.

**Criterios de corrección** (fijados antes de auditar):

| Campo | Correcto si… |
|---|---|
| `tema` | es la categoría de la lista que mejor describe el contenido principal |
| `resumen` | todo lo que afirma está en el texto y recoge la idea principal |
| `emisor`, `destinatario`, `fecha`, `lugar`, `monto` | el valor está en el texto y cumple ese rol, o es `null` y el texto no trae el dato |
| `citas_clave` | cada cita está literalmente en la transcripción y es relevante |
| `numero_hablantes` | coincide con las marcas `[HABLANTE_n]` distintas |

Una extracción es **completamente correcta** solo si todos sus campos lo son.
"""),
code("""
muestra = muestra_auditoria(extraccion)
auditoria = json.loads((DATOS / "raw" / "auditoria" / "auditoria_llm.json").read_text(encoding="utf-8"))
filas = []
for r in muestra.itertuples():
    campos = json.loads(r.campos_json) if r.json_valido else {}
    juicios = {k: auditoria.get(f"{r.id}|{k}") for k in campos}
    filas.append({"id": r.id, "modalidad": r.modalidad, "campos": len(campos),
                  "campos_correctos": sum(v == "1" for v in juicios.values()),
                  "sin_revisar": sum(v is None for v in juicios.values()),
                  "completamente_correcta": bool(campos) and all(v == "1" for v in juicios.values()),
                  "campos_con_error": ", ".join(k for k, v in juicios.items() if v == "0"),
                  "nota": auditoria.get(f"{r.id}|nota", "")})
resultado_auditoria = pd.DataFrame(filas)
acuerdo = int(resultado_auditoria["completamente_correcta"].sum())
print(f"Extracciones completamente correctas: {acuerdo}/20 = {acuerdo / 20:.0%} · campos sin revisar: {resultado_auditoria['sin_revisar'].sum()}")
print(resultado_auditoria.groupby("modalidad")["completamente_correcta"].agg(["sum", "count"]).to_string())
"""),
code("""
por_campo = {}
for r in muestra.itertuples():
    for k in (json.loads(r.campos_json) if r.json_valido else {}):
        por_campo.setdefault(k, []).append(auditoria.get(f"{r.id}|{k}") == "1")
print("Exactitud por campo:", {k: f"{np.mean(v):.0%} (n={len(v)})" for k, v in por_campo.items()})
errores = resultado_auditoria[~resultado_auditoria["completamente_correcta"]][["id", "modalidad", "campos_con_error", "nota"]]
errores
"""),
md(t("interpretacion_llm")),
md("""
## 7. Comparativa y decisión
"""),
code("""
comparativa = pd.concat([
    tabla_ocr.assign(modalidad="documentos", metrica="CER").rename(columns={"CER agregado": "error agregado", "CER promedio por caso": "error promedio por caso"}),
    tabla_asr.assign(modalidad="audio", metrica="WER").rename(columns={"WER agregado": "error agregado", "WER promedio por caso": "error promedio por caso"}),
], ignore_index=True)[["modalidad", "modelo", "metrica", "error agregado", "error promedio por caso", "segundos por caso", "factor de tiempo real"]]
comparativa["rol"] = ["línea base" if c in ("paddleocr_base", "fw_small") else "alternativo" for c in pd.concat([tabla_ocr["clave"], tabla_asr["clave"]])]
comparativa.round(3)
"""),
md(t("decision")),
code("""
OCR_ELEGIDO, ASR_ELEGIDO = "paddleocr_base", "fw_turbo"
cer_elegido = float(tabla_ocr.set_index("clave").loc[OCR_ELEGIDO, "CER agregado"])
wer_elegido = float(tabla_asr.set_index("clave").loc[ASR_ELEGIDO, "WER agregado"])

eval_docs.drop(columns=["descripcion_modelo"]).to_csv(DATOS / "evaluacion_documentos.csv", index=False)
eval_audio.drop(columns=["descripcion_modelo"]).to_csv(DATOS / "evaluacion_audio.csv", index=False)
metricas = {
    "ocr_elegido": ev.MODELOS_OCR[OCR_ELEGIDO], "asr_elegido": ev.MODELOS_ASR[ASR_ELEGIDO],
    "cer_base": float(cer_base), "wer_base": float(wer_base), "cer_elegido": cer_elegido, "wer_elegido": wer_elegido,
    "acuerdo_llm": acuerdo,
    "tabla_comparativa": comparativa.round(3).to_dict(orient="records"),
    "auditoria_errores": errores.to_dict(orient="records"),
}
(DATOS / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Resultados Test: WER: {wer_elegido:.2f} | CER: {cer_elegido:.2f} | Acuerdo LLM-Humano: {acuerdo}/20")
"""),
md(t("reflexion")),
]

nb = notebook(celdas)
for celda in nb.cells:
    for clave, valor in TEXTOS.items():
        if clave.startswith("@@"):
            celda.source = celda.source.replace(clave, valor)
nbformat.write(nb, RAIZ / "02_evaluacion.ipynb")
print("ok:", len(nb.cells), "celdas")

"""Escribe 01_extraccion.ipynb. Después se ejecuta con:
    jupyter nbconvert --to notebook --execute --inplace 01_extraccion.ipynb
"""
from pathlib import Path

import nbformat
from nb_util import code, md, notebook

RAIZ = Path(__file__).resolve().parents[1]

celdas = [
md("""
# Tarea 2 · Parte 1 — Corpus y extracción (OCR + ASR)

> **Pregunta del proyecto:** ¿Pueden convertir documentos escaneados y entrevistas en audio en un archivo de texto estructurado y buscable?

Este notebook arma dos corpus del Perú del siglo XX y los convierte en texto: **25 documentos** (prensa
electoral de 1939 y 1945, leyes y manuscritos de 1912 a 1921) leídos con **PaddleOCR**, y **92 minutos de
noticieros cinematográficos de la década de 1940** transcritos con **faster-whisper**, VAD y diarización.

## Configuración inicial

| Dato | Valor |
|---|---|
| Fecha de extracción | 25 de octubre de 2026 (descargas de Wikimedia Commons y OCR/ASR locales) |
| OCR | PaddleOCR **3.7.0**, modelo PP-OCRv6 *medium* (detector + reconocedor multilingüe con español), motor PaddlePaddle 3.3.1 en CPU |
| ASR | faster-whisper **1.2.1** con **Whisper large-v3-turbo** (CTranslate2, int8, CPU), `beam_size=5`, VAD Silero, `language="es"` |
| Diarización | huellas de voz WeSpeaker ResNet34 (pyannote.audio 4.0.7) + clustering jerárquico (coseno, umbral 0,7), con `voz.py` de la sesión 13 |
| Entorno | Python 3.12, los grupos `docai` y `audio` del curso (`uv sync --group docai`, `uv sync --group audio`); versiones exactas impresas abajo |
| Equipo | laptop con Intel Core i5-8250U (4 núcleos), 24 GB de RAM, sin GPU utilizable (GeForce de 2 GB con driver CUDA 12.0) |

### Fuentes y licencias

| Corpus | Fuente | Licencia | Registros |
|---|---|---|---|
| Documentos impresos y fotos | Capturas propias de *El Comercio* y *La Crónica* (1939 y 1945), Leyes 8901 (1939) y folleto «Tres discursos» del Bloque Antifascista Peruano (1945) | Prensa: dominio público en el Perú (D. Leg. 822, art. 53: obra de persona jurídica con más de 70 años). Ley: texto oficial no protegido (art. 9). Folleto: uso académico, **sus imágenes no se publican** | 16 |
| Manuscritos | Biblioteca Nacional del Perú y archivo Burzagli en Wikimedia Commons (cartas de 1912 y 1917, tarjeta de 1912, documentos de 1921) | CC0, Public Domain Mark, dominio público y CC BY-SA 3.0 | 9 |
| Audio | 31 noticieros peruanos de 1940-1948 y un tramo de *Alerta en la frontera* (1942), digitalizados por la BNP y publicados en Wikimedia Commons | Public Domain Mark (del propietario), CC0, dominio público y CC BY 4.0 | 32 clips · 92 min |

El detalle por archivo (URL, licencia y cómo recuperarlo) se exporta al final a `datos/fuentes.csv`.
Los archivos originales y las salidas pesadas viven en `datos/raw/`, que no se versiona.
"""),
code("""
import hashlib
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"   # no probar la conexión a cada servidor de modelos
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import faster_whisper
import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddleocr
import pandas as pd
import plotly
import plotly.express as px
import plotly.io as pio
import pyannote.audio
import pymupdf
import torch
import transformers
from matplotlib.patches import Patch, Rectangle
from PIL import Image

import fuentes_audio
import procesar_audio as pa
import procesar_documentos as pd_ocr
import seudonimos
import voz
from fuentes_documentos import CARPETA as DOCS, FUENTES_PROPIAS

# Los gráficos de Plotly se guardan también como PNG para que se vean en GitHub
pio.renderers.default = "plotly_mimetype+png"
pio.renderers["png"].width, pio.renderers["png"].height = 950, 500
pd.set_option("display.max_colwidth", 90)
%config InlineBackend.figure_formats = ["jpeg"]

DATOS = Path("datos")
AUDIO = DATOS / "raw" / "audio"
COLORES = {"impreso": "#2a78d6", "foto": "#eb6834", "manuscrito": "#7a5195"}

print(f"PaddleOCR {paddleocr.__version__} · PaddlePaddle {paddle.__version__} · PyMuPDF {pymupdf.__version__}")
print(f"faster-whisper {faster_whisper.__version__} · modelo {voz.MODELOS['faster-whisper']['hf']}")
print(f"pyannote.audio {pyannote.audio.__version__} · torch {torch.__version__} · transformers {transformers.__version__}")
print(f"pandas {pd.__version__} · plotly {plotly.__version__} · Python {sys.version.split()[0]}")
"""),
md("""
---
# A · Documentos

## 1. El corpus: 25 documentos en tres niveles de dificultad

Cada documento es **una imagen**: una página de un PDF de capturas propias o un archivo de Commons. El
catálogo (`procesar_documentos.CATALOGO`) fija el tipo de cada uno con un criterio visual:

- **impreso**: texto tipográfico recto, buen contraste, fondo limpio;
- **foto**: imagen tomada con celular de un papel curvado o inclinado, o con sombras (el folleto de 1945
  fotografiado abierto sobre la mesa, columnas de periódico dobladas);
- **manuscrito**: el contenido principal está escrito a mano (cartas, tarjetas y formularios llenados a mano).
"""),
code("""
catalogo = pd_ocr.catalogo()
for fila in pd_ocr.CATALOGO:
    pd_ocr.preparar_imagen(fila)          # baja o recorta la imagen solo si no está en disco

catalogo["archivo"] = [f"datos/raw/documentos/{i}.jpg" for i in catalogo["id"]]
catalogo["ancho_px"] = [Image.open(DOCS / f"{i}.jpg").size[0] for i in catalogo["id"]]
catalogo["alto_px"] = [Image.open(DOCS / f"{i}.jpg").size[1] for i in catalogo["id"]]

print(catalogo["tipo"].value_counts().to_string())
catalogo[["id", "tipo", "descripcion", "licencia", "ancho_px", "alto_px"]]
"""),
md("""
### 1.1 Limpieza: duplicados por hash

Antes de elegir los 25 documentos revisamos **todas** las páginas de los PDF de capturas propias (no solo
las elegidas), más los archivos bajados de Commons. Dos archivos con el mismo hash SHA-256 son idénticos
byte a byte. Para las páginas de PDF se toma el hash de la imagen incrustada, que es la foto original.
"""),
code("""
def sha256(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


filas = []
for pdf in sorted(FUENTES_PROPIAS.glob("Elecciones 19*/Fuentes Primarias/*.pdf")):
    doc = pymupdf.open(pdf)
    for n, pagina in enumerate(doc, start=1):
        for xref in [img[0] for img in pagina.get_images()][:1]:     # la foto principal de la página
            filas.append({"archivo": f"{pdf.name} · p. {n}", "hash": sha256(doc.extract_image(xref)["image"])})
for original in sorted((DOCS / "originales").iterdir()):
    filas.append({"archivo": f"Commons · {original.name}", "hash": sha256(original.read_bytes())})

pool = pd.DataFrame(filas)
repetidos = pool[pool.duplicated("hash", keep=False)].sort_values("hash")
print(f"Archivos revisados: {len(pool)} · hashes distintos: {pool['hash'].nunique()} · "
      f"copias exactas sobrantes: {pool.duplicated('hash').sum()}")
repetidos.assign(hash=repetidos["hash"].str[:12])
"""),
code("""
# Los 25 documentos elegidos también deben ser distintos entre sí
hash_docs = {i: sha256((DOCS / f"{i}.jpg").read_bytes()) for i in catalogo["id"]}
print("Hashes distintos entre los 25 elegidos:", len(set(hash_docs.values())))
"""),
md("""
El hash solo detecta copias **exactas**. Dos escaneos del mismo papel tienen hashes distintos, así que
además comparamos una huella perceptual: la imagen reducida a 8×8 grises y binarizada contra su media
(*average hash*). Dos imágenes del mismo documento quedan a pocos bits de distancia.
"""),
code("""
def huella_visual(ruta) -> np.ndarray:
    gris = np.asarray(Image.open(ruta).convert("L").resize((8, 8), Image.Resampling.LANCZOS), dtype=float)
    return (gris > gris.mean()).flatten()


commons = {p.name: huella_visual(p) for p in sorted((DOCS / "originales").iterdir()) if p.suffix in (".jpg", ".png")}
pares = []
nombres = list(commons)
for i, a in enumerate(nombres):
    for b in nombres[i + 1:]:
        pares.append({"archivo_a": a, "archivo_b": b, "bits_distintos": int((commons[a] != commons[b]).sum())})
pd.DataFrame(pares).sort_values("bits_distintos").head(4)
"""),
md("""
`lima2_1921.jpg` y `socgeolima_1921.jpg` están a solo unos bits de distancia: son el mismo diploma de la
Sociedad Geográfica de Lima escaneado dos veces. Nos quedamos con la versión de mayor resolución
(`doc_024`). Este y los demás candidatos descartados durante la selección, con su motivo:
"""),
code("""
descartes_docs = pd.DataFrame(pd_ocr.DESCARTADOS, columns=["candidato", "motivo"])
descartes_docs
"""),
md("""
## 2. OCR con PaddleOCR

Parámetros (iguales para los 25 documentos):

| Parámetro | Valor | Por qué |
|---|---|---|
| Modelo | PP-OCRv6 *medium*: detector de líneas + reconocedor multilingüe (incluye tildes y eñes) | el de la clase, sin ajuste adicional |
| Idioma | sin `lang`: el reconocedor PP-OCRv6 es multilingüe y ya cubre el español | igual que en la clase |
| `use_doc_orientation_classify`, `use_doc_unwarping`, `use_textline_orientation` | `False` | configuración base de la clase |
| Motor | PaddlePaddle en CPU, `enable_mkldnn=False`, 4 hilos | el motor PyTorch de la clase tardó más de 10 min por imagen en esta CPU, y oneDNN falla en Paddle 3.3.1 para Windows |
| Preprocesamiento | página extraída del PDF sin recomprimir; imagen reducida a **2000 px** en su lado mayor; recorte de la parte manuscrita en la tarjeta de Palma | en CPU el detector tarda según el número de píxeles |

El resultado crudo de cada imagen (texto, confianza y caja de cada línea) queda como respaldo en
`datos/raw/documentos/ocr/`. Si ya existe, se lee de ahí y no se recalcula.
"""),
code("""
ocr = None
lecturas = {}
for doc_id in catalogo["id"]:
    if not (DOCS / "ocr" / f"{doc_id}.json").exists() and ocr is None:
        ocr = pd_ocr.crear_ocr()                       # solo se carga si falta algún documento
    resultado = pd_ocr.ocr_documento(ocr, doc_id)
    lecturas[doc_id] = pd.DataFrame(resultado["lineas"])
    catalogo.loc[catalogo["id"] == doc_id, "segundos_ocr"] = resultado["segundos"]

catalogo["texto_ocr_crudo"] = ["\\n".join(lecturas[i]["texto"]) if len(lecturas[i]) else "" for i in catalogo["id"]]
catalogo["n_lineas"] = [len(lecturas[i]) for i in catalogo["id"]]
catalogo["confianza_mediana"] = [lecturas[i]["confianza"].median() if len(lecturas[i]) else 0 for i in catalogo["id"]]
catalogo["pct_lineas_dudosas"] = [100 * (lecturas[i]["confianza"] < 0.90).mean() if len(lecturas[i]) else 100 for i in catalogo["id"]]
catalogo["n_caracteres"] = catalogo["texto_ocr_crudo"].str.len()

print(f"OCR de {len(catalogo)} documentos · {catalogo['segundos_ocr'].sum() / 60:.0f} min de cómputo en CPU "
      f"({catalogo['segundos_ocr'].median():.0f} s por página, mediana)")
(catalogo.groupby("tipo")
 .agg(documentos=("id", "count"), lineas=("n_lineas", "sum"), confianza_mediana=("confianza_mediana", "median"),
      pct_lineas_dudosas=("pct_lineas_dudosas", "mean"), segundos_por_pagina=("segundos_ocr", "median"))
 .round(2).loc[["impreso", "foto", "manuscrito"]])
"""),
md("""
### 2.1 Ver lo que ve el modelo

Como en la clase, dibujamos la caja de cada línea detectada y la pintamos según la confianza del
reconocedor: verde 0,90 o más, amarillo 0,70 a 0,90 y rojo menos de 0,70.
"""),
code("""
BIEN, DUDA, MAL = "#0ca30c", "#fab219", "#d03b3b"
LEYENDA = [Patch(facecolor="none", edgecolor=c, linewidth=2, label=t)
           for c, t in [(BIEN, "alta (≥ 0.90)"), (DUDA, "media (0.70-0.90)"), (MAL, "baja (< 0.70)")]]


def dibujar(doc_id, eje):
    eje.imshow(Image.open(DOCS / f"{doc_id}.jpg").convert("RGB"))
    for f in lecturas[doc_id].itertuples():
        color = BIEN if f.confianza >= 0.90 else DUDA if f.confianza >= 0.70 else MAL
        eje.add_patch(Rectangle((f.x0, f.y0), f.x1 - f.x0, f.y1 - f.y0, fill=False, linewidth=1.4, edgecolor=color))
    fila = catalogo.set_index("id").loc[doc_id]
    eje.set_title(f"{doc_id} · {fila.tipo} · mediana {fila.confianza_mediana:.2f}", fontsize=10)
    eje.axis("off")


fig, ejes = plt.subplots(1, 3, figsize=(15, 6.5))
for eje, doc_id in zip(ejes, ["doc_008", "doc_010", "doc_021"]):
    dibujar(doc_id, eje)
fig.legend(handles=LEYENDA, title="Confianza del reconocedor", loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()
"""),
code("""
for doc_id in ["doc_008", "doc_010", "doc_021"]:
    print(f"── {doc_id}:", " · ".join(lecturas[doc_id]["texto"].head(8)), "\\n")
"""),
md("@@INTERPRETACION_OCR@@"),
md("""
### 2.2 ¿Hay documentos ilegibles?

El criterio de descarte es humano: un documento es **ilegible** si una persona tampoco puede leerlo. Los 25
se revisaron a ojo al armar el catálogo y todos se pueden leer. Como alarma automática marcamos los
documentos en los que el OCR devolvió menos de 3 líneas o una confianza mediana menor a 0,80: esos van
a revisión, pero no se descartan si el papel es legible.
"""),
code("""
alerta = catalogo[(catalogo["n_lineas"] < 3) | (catalogo["confianza_mediana"] < 0.80)]
print(f"Documentos en alerta: {len(alerta)} · descartados por ilegibles: 0")
alerta[["id", "tipo", "descripcion", "n_lineas", "confianza_mediana", "pct_lineas_dudosas"]].round(2)
"""),
md("""
---
# B · Audio

## 3. El corpus: noticieros peruanos de la década de 1940

`fuentes_audio.py` reutiliza la idea de `datos.py` de la sesión 13. Consulta la API de Commons para
obtener la licencia y el autor de cada archivo, baja la versión liviana (240p; el audio Opus es el mismo
del original) con un User-Agent descriptivo y pausas de 1 a 2 s, y extrae el audio a WAV mono de 16 kHz.
De la película *Alerta en la frontera* (54 min) se usa solo un tramo de 5 min.
"""),
code("""
atribucion = fuentes_audio.bajar_noticieros()          # no vuelve a bajar lo que ya está en disco
atribucion["fecha_original"] = atribucion["fecha_original"].str.replace(r"date QS.*$", "", regex=True).str.strip()
atribucion["productora"] = (atribucion["autor"].str.extract(r"(Productora Huascarán|Nacional Films|National Films|Perú Sono Film|Peru Sono Film|Ministerio de Gobierno|Herrmann)", expand=False)
                            .replace({"National Films": "Nacional Films", "Peru Sono Film": "Perú Sono Film", "Herrmann": "Nacional Films (película)"})
                            .fillna("productora no identificada"))
atribucion["titulo"] = atribucion["titulo_commons"].str.removeprefix("File:").str.removesuffix(".webm")

print(f"{len(atribucion)} clips · {atribucion['duracion_seg'].sum() / 60:.1f} min · licencias: {atribucion['licencia'].value_counts().to_dict()}")
atribucion[["id", "titulo", "productora", "fecha_original", "licencia", "duracion_seg", "tramo_usado"]]
"""),
code("""
# Duplicados por hash entre los videos descargados
hash_audio = {p.stem: sha256(p.read_bytes()) for p in sorted((AUDIO / "originales").glob("clip_*.webm"))}
print(f"Videos: {len(hash_audio)} · hashes distintos: {len(set(hash_audio.values()))}")
"""),
md("""
## 4. ASR: VAD + faster-whisper + diarización

`procesar_audio.py` aplica a cada clip el flujo de `voz.py` de la sesión 13:

1. `cargar_audio`: cualquier formato → arreglo mono de 16 kHz;
2. `Transcriptor("faster-whisper").segmentos(audio, idioma="es")`: **VAD Silero activado** (no se le
   manda silencio ni música sola al modelo) y transcripción con Whisper large-v3-turbo en int8;
3. `diarizar`: una huella de voz por segmento y clustering jerárquico con umbral 0,7;
4. exportación de cada clip a **CSV** (segmentos), **SRT** (subtítulos) y **TXT** (lectura por turnos) en
   `datos/raw/audio/salidas/`.

Los resultados se guardan en caché: la transcripción completa tardó alrededor de una hora en esta CPU.
"""),
code("""
transcriptor = None
resultados = {}
for clip_id in atribucion["id"]:
    if not (pa.SALIDAS / f"{clip_id}.json").exists() and transcriptor is None:
        transcriptor = voz.Transcriptor(pa.MODELO_ASR)
    resultados[clip_id] = pa.procesar_clip(clip_id, transcriptor)

resumen_asr = pd.DataFrame([{k: v for k, v in r.items() if k != "segmentos"} for r in resultados.values()])
resumen_asr["factor_tiempo_real"] = (resumen_asr["segundos_procesamiento"] / resumen_asr["duracion_seg"]).round(2)
print(f"Audio: {resumen_asr['duracion_seg'].sum() / 60:.1f} min · voz detectada por el VAD: "
      f"{resumen_asr['segundos_voz'].sum() / 60:.1f} min · cómputo: {resumen_asr['segundos_procesamiento'].sum() / 60:.0f} min "
      f"(factor de tiempo real {resumen_asr['segundos_procesamiento'].sum() / resumen_asr['duracion_seg'].sum():.2f})")
resumen_asr.head(8)
"""),
md("""
### 4.1 Alucinaciones: el «¡Gracias!» de Whisper

Aun con el VAD, en los cierres musicales de algunos noticieros Whisper escribió **«¡Gracias!»**. Es la
misma alucinación que vimos en la sesión 13: el modelo aprendió de subtítulos de YouTube y la escribe
cuando oye música o aplausos. La revisión manual de las transcripciones encontró además una frase
repetida dos veces seguidas.

Regla de limpieza, definida antes de calcular métricas: se elimina un segmento si su texto normalizado
está en la lista de alucinaciones conocidas de Whisper, o si repite literalmente al segmento anterior.
"""),
code("""
ALUCINACIONES = {"gracias", "muchas gracias", "gracias por ver el video", "suscríbete",
                 "subtítulos realizados por la comunidad de amara org"}

eliminados = []
for clip_id, r in resultados.items():
    limpios, anterior = [], None
    for s in r["segmentos"]:
        normal = voz.normalizar(s["texto"])
        if normal in ALUCINACIONES:
            eliminados.append({"id": clip_id, "inicio": s["inicio"], "texto": s["texto"], "motivo": "alucinación conocida de Whisper"})
        elif normal == anterior:
            eliminados.append({"id": clip_id, "inicio": s["inicio"], "texto": s["texto"], "motivo": "repetición literal del segmento anterior"})
        else:
            limpios.append(s)
        anterior = normal
    r["segmentos_limpios"] = limpios

eliminados = pd.DataFrame(eliminados)
print(f"Segmentos eliminados: {len(eliminados)} en {eliminados['id'].nunique()} clips")
eliminados
"""),
md("""
### 4.2 Diarización del clip con varios hablantes

Con los segmentos de faster-whisper, la diarización no separó voces en ningún noticiero, ni siquiera en
la **toma de mando de Bustamante y Rivero (clip_027, 1945)**, donde se oye al narrador y, en dos
momentos, a los oradores del Congreso. La causa: faster-whisper arma segmentos largos (frases completas)
que pueden mezclar dos voces, y la huella de voz de una mezcla no se parece a ninguna de las dos.

`voz.py` anticipa este problema: *«tramos cortos (pausas de 0,3 s, máximo 12 s) ayudan a separar
hablantes: cada tramo debería tener una sola voz»*. Para este clip hacemos una segunda pasada: primero
el VAD corta en tramos cortos, se transcribe cada tramo por separado y después se diariza.
"""),
code("""
diarizado_027 = pa.diarizar_por_tramos("clip_027")
tabla_027 = pd.DataFrame(diarizado_027["segmentos"])
voz_por_hablante = (tabla_027.assign(segundos=tabla_027["fin"] - tabla_027["inicio"])
                    .groupby("hablante").agg(tramos=("texto", "count"), segundos=("segundos", "sum")).round(1))
print(f"clip_027: {len(tabla_027)} tramos · hablantes detectados: {tabla_027['hablante'].nunique()}")
voz_por_hablante
"""),
code("""
print(voz.a_texto([voz.Segmento(**s) for s in diarizado_027["segmentos"]])[:2500])
"""),
md("@@INTERPRETACION_DIARIZACION@@"),
code("""
# El clip_027 usa la segunda pasada; los demás, los segmentos de faster-whisper ya limpios
for clip_id, r in resultados.items():
    if clip_id == "clip_027":
        r["segmentos_limpios"] = [s for s in diarizado_027["segmentos"] if voz.normalizar(s["texto"]) not in ALUCINACIONES]
    segmentos = [voz.Segmento(**s) for s in r["segmentos_limpios"]]
    base = pa.SALIDAS / f"{clip_id}_limpio"
    voz.a_tabla(segmentos).to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".srt").write_text(voz.a_srt(segmentos), encoding="utf-8")
    base.with_suffix(".txt").write_text(voz.a_texto(segmentos), encoding="utf-8")

print("Exportados por clip (CSV, SRT y TXT):", len(list(pa.SALIDAS.glob("*_limpio.srt"))))
print((pa.SALIDAS / "clip_027_limpio.srt").read_text(encoding="utf-8")[:600])
"""),
md("""
### 4.3 ¿Hay audios sin voz?

Regla: un clip se descarta si el VAD encuentra menos de 5 segundos de voz o si, después de la limpieza,
no queda ningún segmento con texto.
"""),
code("""
def texto_con_hablantes(segmentos):
    \"\"\"Une los segmentos en turnos con marcas [HABLANTE_n], como pide el enunciado.\"\"\"
    turnos = []
    for s in segmentos:
        marca = "HABLANTE_" + s["hablante"].lstrip("H")
        if turnos and turnos[-1][0] == marca:
            turnos[-1][1].append(s["texto"])
        else:
            turnos.append((marca, [s["texto"]]))
    return " ".join(f"[{m}] " + " ".join(t) for m, t in turnos)


clips = atribucion[["id", "titulo", "productora", "fecha_original", "licencia", "fuente_url", "duracion_seg"]].copy()
clips["segundos_voz"] = [resultados[i]["segundos_voz"] for i in clips["id"]]
clips["n_segmentos"] = [len(resultados[i]["segmentos_limpios"]) for i in clips["id"]]
clips["n_hablantes"] = [len({s["hablante"] for s in resultados[i]["segmentos_limpios"]}) for i in clips["id"]]
clips["texto_asr_crudo"] = [texto_con_hablantes(resultados[i]["segmentos_limpios"]) for i in clips["id"]]
sin_voz = clips[(clips["segundos_voz"] < 5) | (clips["n_segmentos"] == 0)]
print(f"Clips sin voz descartados: {len(sin_voz)} · clips con 2 o más hablantes: {(clips['n_hablantes'] >= 2).sum()} "
      f"({', '.join(clips.loc[clips['n_hablantes'] >= 2, 'id'])})")
"""),
md("""
---
## 5. Seudonimización

Criterio de publicación acordado para este proyecto: se **conservan** las figuras públicas mencionadas en su
rol público (presidentes, ministros, candidatos, autoridades, escritores e intelectuales conocidos) y se
**reemplazan por un seudónimo** los particulares (personas que aparecen a título privado). Ningún texto
contiene DNI, direcciones personales ni teléfonos.

1. Un modelo NER en español (`mrm8488/bert-spanish-cased-finetuned-ner`) propone las personas mencionadas.
2. **Revisión humana**: el estudiante clasificó cada nombre propuesto. La tabla con los nombres reales queda en
   `datos/privado/revision_nombres.csv`, fuera de Git; aquí solo se muestran los conteos.
3. `voz.seudonimizar` reemplaza los particulares, también cuando el OCR o el ASR los escribieron con errores
   (comparación difusa con `rapidfuzz`).
"""),
code("""
textos = {**dict(zip(catalogo["id"], catalogo["texto_ocr_crudo"])), **dict(zip(clips["id"], clips["texto_asr_crudo"]))}
candidatos = seudonimos.candidatos(textos)
print(f"Personas propuestas por el NER: {len(candidatos)} nombres distintos en {candidatos['apariciones'].sum()} menciones")
"""),
code("""
revision = pd.read_csv(DATOS / "privado" / "revision_nombres.csv")
print(revision["decision"].value_counts().to_string())
reemplazos = seudonimos.tabla_de_reemplazos(revision)
print(f"\\nParticulares distintos: {len(set(reemplazos.values()))} · variantes de nombre buscadas: {len(reemplazos)}")
"""),
code("""
conteo = {}
catalogo["texto_ocr"], catalogo["reemplazos"] = zip(*[seudonimos.aplicar(t, reemplazos) for t in catalogo["texto_ocr_crudo"]])
clips["texto_asr"], clips["reemplazos"] = zip(*[seudonimos.aplicar(t, reemplazos) for t in clips["texto_asr_crudo"]])
print(f"Entidades reemplazadas: {catalogo['reemplazos'].sum()} en documentos y {clips['reemplazos'].sum()} en audios "
      f"(total {catalogo['reemplazos'].sum() + clips['reemplazos'].sum()})")
pd.concat([catalogo.loc[catalogo["reemplazos"] > 0, ["id", "reemplazos"]], clips.loc[clips["reemplazos"] > 0, ["id", "reemplazos"]]])
"""),
md("""
---
## 6. Análisis exploratorio

Tres preguntas con un gráfico cada una. El título de cada gráfico es la respuesta.
"""),
code("@@EDA_1@@"),
code("@@EDA_2@@"),
code("@@EDA_3@@"),
md("@@INTERPRETACION_EDA@@"),
md("""
## 7. Validación: los extremos tienen que ser coherentes con los histogramas
"""),
code("""
cols = ["id", "tipo", "descripcion", "n_lineas", "n_caracteres", "confianza_mediana"]
print("Los 3 documentos con el OCR más largo:")
display(catalogo.nlargest(3, "n_caracteres")[cols].round(2))
print("Los 3 documentos con el OCR más corto:")
display(catalogo.nsmallest(3, "n_caracteres")[cols].round(2))
"""),
code("""
clips["n_palabras"] = clips["texto_asr"].map(lambda t: len(re.sub(r"\\[HABLANTE_\\d+\\]", " ", t).split()))
cols = ["id", "titulo", "duracion_seg", "segundos_voz", "n_palabras", "n_hablantes"]
print("Los 3 clips más largos:")
display(clips.nlargest(3, "duracion_seg")[cols])
print("Los 3 clips más cortos:")
display(clips.nsmallest(3, "duracion_seg")[cols])
"""),
md("@@INTERPRETACION_VALIDACION@@"),
md("""
## 8. Exportación
"""),
code("""
licencia_corta = {"CC0": "CC0", "Public Domain Mark": "Public Domain Mark", "Dominio público": "dominio público",
                  "CC BY-SA 3.0": "CC BY-SA 3.0"}
documentos = pd.DataFrame({
    "id": catalogo["id"],
    "archivo": catalogo["archivo"],
    "tipo": catalogo["tipo"],
    "fuente": [f"{f} ({licencia_corta.get(l, 'dominio público en el Perú' if 'D. Leg. 822, art. 53' in l else 'texto oficial no protegido' if 'art. 9' in l else 'uso académico, imagen no publicable')})"
               for f, l in zip(catalogo["fuente"], catalogo["licencia"])],
    "texto_ocr": catalogo["texto_ocr"],
    "modelo_ocr": pd_ocr.CONFIG_OCR,
    "texto_referencia": "",
    "descripcion": catalogo["descripcion"],
    "imagen_publicable": catalogo["imagen_publicable"],
    "n_lineas": catalogo["n_lineas"],
    "n_caracteres": catalogo["texto_ocr"].str.len(),
    "confianza_mediana": catalogo["confianza_mediana"].round(3),
})

transcripciones = pd.DataFrame({
    "id": clips["id"],
    "archivo_audio": [f"datos/raw/audio/wav/{i}.wav" for i in clips["id"]],
    "duracion_seg": clips["duracion_seg"],
    "idioma": "es (configurado; español peruano: noticieros de productoras limeñas, según la BNP)",
    "texto_asr": clips["texto_asr"],
    "modelo_asr": [pa.CONFIG_ASR + (" + segunda pasada por tramos VAD para diarizar" if i == "clip_027" else "") for i in clips["id"]],
    "n_hablantes": clips["n_hablantes"],
    "texto_referencia": "",
    "titulo": clips["titulo"],
    "productora": clips["productora"],
    "fecha_original": clips["fecha_original"],
    "segundos_voz": clips["segundos_voz"],
    "n_palabras": clips["n_palabras"],
})

fuentes = pd.concat([
    pd.DataFrame({
        "id": catalogo["id"], "modalidad": "documento", "fuente_url": catalogo["fuente_url"], "licencia": catalogo["licencia"],
        "acceso": [("Descarga desde Wikimedia Commons (fuentes_documentos.bajar_commons)" if o.startswith("File:")
                    else f"PDF de capturas propias del estudiante, «{Path(o).name}», página {p}; no se publica el PDF completo")
                   for o, p in zip(catalogo["origen"], catalogo["pagina"])],
    }),
    pd.DataFrame({
        "id": clips["id"], "modalidad": "audio", "fuente_url": clips["fuente_url"], "licencia": clips["licencia"],
        "acceso": "python fuentes_audio.py: baja la versión 240p desde Commons y extrae el audio a WAV mono de 16 kHz"
                  + pd.Series(["" if i != "clip_032" else " (tramo de 600 a 900 s)" for i in clips["id"]]),
    }),
], ignore_index=True)

documentos.to_csv(DATOS / "documentos.csv", index=False)
transcripciones.to_csv(DATOS / "transcripciones.csv", index=False)
fuentes.to_csv(DATOS / "fuentes.csv", index=False)
for nombre, tabla in [("documentos", documentos), ("transcripciones", transcripciones), ("fuentes", fuentes)]:
    print(f"datos/{nombre}.csv: {tabla.shape}")
"""),
code("""
documentos.head(5)
"""),
code("""
transcripciones.head(5)
"""),
code("""
fuentes.sample(5, random_state=1)
"""),
md("""
## Qué hicimos

| Paso | Herramienta | Resultado |
|---|---|---|
| Corpus de documentos | PDF propios + Commons, `pymupdf` | 25 imágenes (8 impresos, 8 fotos, 9 manuscritos), 4 candidatos descartados con motivo |
| Duplicados | SHA-256 + *average hash* | 0 copias exactas entre los elegidos; un diploma duplicado descartado |
| OCR | PaddleOCR 3.7 (PP-OCRv6) | texto, confianza y caja por línea; respaldo crudo en `datos/raw/documentos/ocr/` |
| Corpus de audio | API de Commons, `fuentes_audio.py` | 32 clips, 92 min, todos con licencia libre |
| ASR | `voz.py`: VAD + faster-whisper turbo | segmentos con tiempos; CSV, SRT y TXT por clip |
| Limpieza de audio | lista de alucinaciones + repeticiones | segmentos «¡Gracias!» eliminados; 0 clips sin voz |
| Diarización | WeSpeaker + clustering; segunda pasada por tramos VAD | clip multihablante con marcas `[HABLANTE_n]` |
| Seudonimización | NER + revisión humana + `voz.seudonimizar` | particulares reemplazados; figuras públicas conservadas |
| EDA y validación | Plotly Express | 3 gráficos con títulos analíticos; extremos revisados |
| Exportación | pandas | `documentos.csv`, `transcripciones.csv`, `fuentes.csv` |
"""),
]

nb = notebook(celdas)
nbformat.write(nb, RAIZ / "01_extraccion.ipynb")
print("ok:", RAIZ / "01_extraccion.ipynb", len(nb.cells), "celdas")

# Tarea 2 — Voces y documentos peruanos: del archivo escaneado al texto buscable

> **Pregunta del proyecto:** ¿Pueden convertir documentos escaneados y entrevistas en audio en un archivo de texto estructurado y buscable?

**Dashboard público:** @@URL_APP@@

**Respuesta corta:** @@RESPUESTA_CORTA@@

@@TABLA_RESULTADOS@@

---

## Corpus, fuentes y licencias

Todo el material es **peruano y del siglo XX** (1912-1948). Fecha de extracción: **25 de octubre de 2026**.

### A. Documentos (25)

| Tipo | N | Documentos | Fuente y licencia |
|---|---|---|---|
| Impreso nítido | 8 | Avisos y comunicados de *El Comercio* (1939, 1945) y *La Crónica* (1945); primera página de la Ley 8901, Estatuto Electoral (1939) | Capturas propias. Prensa: dominio público en el Perú (D. Leg. 822, art. 53: obra de persona jurídica publicada hace más de 70 años). Ley: texto oficial no protegido (art. 9) |
| Foto con sombra o inclinación | 8 | Folleto «Tres discursos» del Bloque Antifascista Peruano (1945), fotografiado abierto; columnas curvadas o inclinadas de *El Comercio* y *La Crónica* | Capturas propias. Las 4 imágenes del folleto **no se publican** (autores individuales, sin licencia de redistribución); su texto sí |
| Manuscrito | 9 | Carta de Carlos J. Bachmann a Ricardo Palma (1912, 4 páginas); carta de César Vallejo a José María Eguren (1917); tarjeta de Ricardo Palma (1912); tarjeta de circulación de la Policía de Lima y dos diplomas (1921) | Biblioteca Nacional del Perú y archivo Burzagli en Wikimedia Commons: CC0, Public Domain Mark, dominio público y CC BY-SA 3.0 |

Se revisaron 294 archivos candidatos (todas las páginas de los PDF propios y los archivos de Commons):
0 copias exactas por SHA-256. Se descartaron 4 candidatos, con su motivo en el notebook 01: un diploma
duplicado de menor resolución, dos cartas mecanografiadas y una postal de la que solo está la fotografía.

### B. Audio (32 clips, 92,1 minutos)

31 **noticieros cinematográficos peruanos de 1940 a 1948** (Productora Huascarán, Nacional Films, Perú Sono
Film y la Dirección de Prensa del Ministerio de Gobierno) y un tramo de 5 minutos de la película
*Alerta en la frontera* (1942). Los digitalizó la **Biblioteca Nacional del Perú** a partir de cintas de
nitrato de 35 mm y están publicados en Wikimedia Commons con licencias **Public Domain Mark** (26),
**CC0** (4), **dominio público** (1) y **CC BY 4.0** (1). Incluyen narración en español con música y
sonido ambiente; el clip_027 (toma de mando de José Luis Bustamante y Rivero, 1945) tiene además a un
orador grabado en el Congreso y es el audio multihablante del corpus.

La fuente, la licencia y la forma de recuperar cada archivo están en [`datos/fuentes.csv`](datos/fuentes.csv).
Los videos, el audio crudo y las salidas pesadas (`datos/raw/`) no se versionan; `fuentes_audio.py`
los vuelve a bajar (User-Agent descriptivo, pausas de 1 a 2 s).

### Seudonimización

Criterio: se **conservan** las figuras públicas en su rol público (presidentes, ministros, candidatos,
autoridades, escritores y personajes históricos) y se **reemplazan** los particulares por `[PERSONA_nn]`.
Un modelo NER en español propuso las personas, el estudiante revisó cada nombre y `voz.seudonimizar` hizo
el reemplazo, también sobre nombres mal escritos por el OCR. Resultado: **@@N_REEMPLAZOS@@ entidades
reemplazadas** (6 particulares: cuatro ciudadanos sorteados para el Jurado Provincial de 1945, un
particular de una delegación y el titular de los documentos personales de 1921). La tabla con los nombres
reales queda fuera de Git (`datos/privado/`). Ningún texto contiene DNI, direcciones ni teléfonos.

---

## Metodología

1. **`01_extraccion.ipynb`**: corpus, limpieza (hash y huella visual), OCR con PaddleOCR 3.7 (PP-OCRv6),
   ASR con `voz.py` de la sesión 13 (VAD + faster-whisper large-v3-turbo + diarización), limpieza de
   alucinaciones, seudonimización, EDA con 3 gráficos de Plotly y exportación a `datos/*.csv`.
2. **`02_evaluacion.ipynb`**: muestra fija de 10 documentos y 10 clips transcritos a mano, CER y WER con
   una misma normalización, línea base (PaddleOCR y faster-whisper *small*), modelos alternativos
   (PaddleOCR con preprocesamiento, EasyOCR, Whisper *turbo* y Qwen3-ASR), casos difíciles, extracción de
   campos JSON con **Qwen3-1.7B local** y auditoría humana de 20 extracciones.
3. **`app.py`**: dashboard de Streamlit que solo lee los resultados ya calculados.

## Hallazgos analíticos

@@HALLAZGOS@@

## Limitaciones técnicas

@@LIMITACIONES@@

---

## Cómo ejecutarlo en local

**Solo la app** (lee los CSV y JSON de `datos/`; no necesita modelos):

```bash
cd tarea_2
python -m venv .venv
.venv\Scripts\activate            # en macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Los notebooks** (OCR, ASR, LLM; entorno equivalente a los grupos `docai` y `audio` del curso):

```bash
cd tarea_2
uv venv --python 3.12
uv pip install -r requirements-notebooks.txt -c constraints.txt
uv run python fuentes_audio.py                 # baja los 32 clips (unos 450 MB) y extrae WAV de 16 kHz
uv run python procesar_documentos.py           # prepara las 25 imágenes y corre PaddleOCR (~1 h en CPU)
uv run python procesar_audio.py                # VAD + faster-whisper + diarización (~1 h en CPU)
uv run jupyter nbconvert --to notebook --execute --inplace 01_extraccion.ipynb
uv run python evaluacion.py asr && uv run python evaluacion.py ocr
uv run python scripts/correr_llm.py            # Qwen3-1.7B en CPU (~1 h)
uv run jupyter nbconvert --to notebook --execute --inplace 02_evaluacion.ipynb
```

Las capturas propias de prensa (PDF) no se publican: `procesar_documentos.py` las busca en la ruta local
indicada en `fuentes_documentos.FUENTES_PROPIAS`. Las transcripciones manuales de referencia y la
auditoría del LLM se hicieron con las páginas que generan `scripts/kit_transcripcion.py` y
`scripts/kit_auditoria.py`.

## Estructura

```
tarea_2/
├── 01_extraccion.ipynb        # OCR, ASR, limpieza, seudonimización y EDA (ejecutado)
├── 02_evaluacion.ipynb        # CER/WER, comparación de modelos, LLM y auditoría (ejecutado)
├── app.py                     # dashboard de Streamlit
├── requirements.txt           # dependencias de la app
├── requirements-notebooks.txt # entorno de los notebooks (+ constraints.txt)
├── voz.py                     # módulo de la sesión 13 (VAD, ASR, diarización, seudonimización, métricas)
├── fuentes_audio.py · fuentes_documentos.py · procesar_audio.py · procesar_documentos.py
├── seudonimos.py · extraccion_llm.py · evaluacion.py
├── scripts/                   # constructores de notebooks, kits de transcripción y auditoría, ejemplos
├── evidencias_ia/             # capturas de la conversación con Claude Code
├── datos/
│   ├── documentos.csv · transcripciones.csv · extraccion_llm.csv · fuentes.csv
│   ├── evaluacion_documentos.csv · evaluacion_audio.csv · metricas.json
│   └── ejemplos/              # imágenes y audios publicables que usa la app
└── modelos/                   # configuración de los modelos usados (sin pesos)
```

## Uso de IA

El proyecto se desarrolló con **Claude Code**. Las capturas de toda la conversación están en
[`evidencias_ia/`](evidencias_ia/), en orden. Las decisiones que requerían juicio humano las tomó el
estudiante: la selección del corpus, el criterio de seudonimización y la revisión de cada nombre, las 20
transcripciones de referencia y la auditoría de las 20 extracciones del LLM.

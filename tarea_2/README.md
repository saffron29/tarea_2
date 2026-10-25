# Tarea 2 — Voces y documentos peruanos: del archivo escaneado al texto buscable

> **Pregunta del proyecto:** ¿Pueden convertir documentos escaneados y entrevistas en audio en un archivo de texto estructurado y buscable?

**Dashboard público:** https://voces-documentos-peru.streamlit.app/

**Respuesta corta:** sí para la prensa impresa y los noticieros narrados, todavía no para la letra manuscrita.
Todo corrió en una laptop sin GPU:

- **Prensa impresa y fotografiada:** PaddleOCR deja el texto casi limpio (CER promedio de 0,01 en fotos y
  0,08 en impresos).
- **Noticieros:** faster-whisper *turbo* transcribe la narración con 4 errores cada 100 palabras (WER 0,04).
  Ese texto ya es buscable.
- **Manuscritos:** conservan un 38 % de error por carácter, con cualquiera de los tres modelos de OCR.
- **Extracción con un LLM local de 1,7B:** acierta todos los campos en 8 de 20 casos auditados. Sirve como
  borrador de catalogación, no como catálogo final.

| Etapa (muestra con referencia manual) | Línea base | Modelo elegido | Resultado |
|---|---|---|---|
| OCR · 10 documentos | PaddleOCR PP-OCRv6: CER 0,191 | el mismo: las dos alternativas tienen más error | **CER 0,19** (promedio por tipo: foto 0,01 · impreso 0,08 · manuscrito 0,38) |
| ASR · 10 clips | faster-whisper *small*: WER 0,111 | faster-whisper *large-v3-turbo* | **WER 0,04**, factor de tiempo real de 0,66 en CPU |
| Extracción JSON · 20 casos auditados | — | Qwen3-1.7B local | **8/20** extracciones completamente correctas; 83 % de los campos correctos |

`Resultados Test: WER: 0.04 | CER: 0.19 | Acuerdo LLM-Humano: 8/20`

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
el reemplazo, también sobre nombres mal escritos por el OCR.

**Resultado: 1 entidad reemplazada.** El NER propuso 106 nombres:

- **79 figuras públicas**, que se conservan;
- **26 descartados**, porque no eran nombres de persona (cargos, lugares o ruido del OCR);
- **1 particular**: quien preside una delegación en un informe de *La Crónica* (doc_015). En la imagen
  publicada de ese documento, su nombre está tapado con un rectángulo negro.

Por decisión del estudiante también se conservan dos casos dudosos:

- los cuatro ciudadanos que integran el Jurado Provincial de Elecciones de Lima en 1945, por su función
  electoral en un comunicado oficial;
- el comandante Ernesto Burzagli, por su rol diplomático y militar oficial en el Centenario de 1921.

La tabla con los nombres reales queda fuera de Git (`datos/privado/`). Ningún texto contiene DNI, teléfonos
ni direcciones personales. La única dirección del corpus es la de un local de propaganda electoral de 1945
(doc_007).

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

1. **El soporte pesa más que el modelo, y el error se arrastra hasta el LLM.**
   - Con el mismo PaddleOCR, el CER promedio es 0,01 en las fotos de 1945, 0,08 en los impresos y 0,38 en
     los manuscritos de 1912 a 1921. Ninguna de las alternativas baja de 0,38 con la letra a mano: ni el
     preprocesamiento de la página ni EasyOCR.
   - En los impresos el error no viene de leer mal sino de no detectar. En doc_001 el detector omitió la
     primera línea, y eso explica la mayor parte de su CER de 0,23: cuando el preprocesamiento la recupera,
     baja a 0,09.
   - El LLM no corrige estos errores, los disimula. Ninguno de los 4 manuscritos auditados quedó bien
     extraído: ante un texto ilegible el modelo escribe un resumen coherente y llena los campos con
     fragmentos sin sentido. En las páginas a varias columnas, el desorden de líneas del OCR se convirtió en
     fechas y lugares equivocados.
2. **En audio, el tamaño del modelo pesa más que la familia, y la música es el ruido que más complica.**
   - Whisper *large-v3-turbo* comete 2,5 veces menos errores que *small* (WER 0,04 frente a 0,11). Sigue
     siendo más rápido que el tiempo real en CPU.
   - Qwen3-ASR, del tamaño de *small*, apenas lo mejora (0,10).
   - El peor clip para turbo es el que tiene más tiempo sin voz, casi todo música (clip_024: 62 % del clip).
   - Whisper alucinó frases que nadie dice: «¡Gracias!», «¡Suscríbete!» y una oración repetida. Pasó sobre
     todo en tramos de música o aplausos. El filtro eliminó 6 segmentos en 5 clips y otros 4 en el discurso
     de 1945 (clip_027).
   - La diarización contó como segundo hablante un segmento que era casi todo marcha militar (clip_004:
     0,9 s de voz real en 16 s).
   - Para un archivo buscable, por tanto, no basta con el ASR: hacen falta el VAD y los filtros de
     alucinaciones y de falsos hablantes.

## Limitaciones técnicas

1. **Solo CPU.** Todo corrió en una laptop sin GPU, y eso condicionó las elecciones:
   - PaddleOCR tardó unos 2 minutos por página (mediana de 130 s en el corpus).
   - El LLM tardó 2,6 minutos por registro (148 min en total). Por eso se usó un modelo pequeño (Qwen3-1.7B)
     y se recortó la entrada a 2500 caracteres, lo que dejó fuera parte del texto de 2 documentos y 6 clips.
   - No se probaron modelos de visión-lenguaje para los manuscritos ni LLM más grandes, que con GPU podrían
     reducir los errores de rol y de resumen.
2. **Muestra pequeña y una sola persona.**
   - Las métricas salen de 10 documentos y 10 clips, y la auditoría del LLM, de 20 casos. Una sola persona
     escribió las referencias y auditó: no hay acuerdo entre anotadores.
   - Hay indicios de errores en la propia referencia. En clip_024, los tres modelos coinciden entre sí y no
     con ella. No se corrigió después de ver los resultados.
   - El CER también castiga el orden de lectura. En las páginas a varias columnas o con texto superpuesto,
     una línea bien leída pero fuera de lugar cuenta como error.
   - El criterio para valores copiados con errores de reconocimiento («Loguia» por Leguía) no se aplicó
     igual en todos los casos. Sin embargo, el 8/20 no cambia, porque esos casos tienen además otros campos
     incorrectos.

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
uv run python scripts/correr_llm.py            # Qwen3-1.7B en CPU (~2,5 h)
uv run jupyter nbconvert --to notebook --execute --inplace 02_evaluacion.ipynb
uv run python scripts/preparar_ejemplos.py     # imágenes (con particulares tapados) y OGG para la app
```

Las capturas propias de prensa (PDF) no se publican: `procesar_documentos.py` las busca en la ruta local
indicada en `fuentes_documentos.FUENTES_PROPIAS`. Los notebooks también leen la revisión de nombres
`datos/privado/revision_nombres.csv`, que no se versiona porque contiene los nombres reales. Las transcripciones manuales de referencia y la
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

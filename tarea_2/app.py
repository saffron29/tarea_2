"""Dashboard de la Tarea 2: documentos y noticieros peruanos del siglo XX convertidos en texto buscable.

Solo lee resultados ya calculados en los notebooks (CSV y JSON en datos/): no carga modelos de OCR,
ASR ni LLM, así que corre en el plan gratuito de Streamlit Community Cloud.

    streamlit run app.py
"""
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit Cloud ejecuta la app desde la raíz del repositorio: las rutas se arman desde este archivo
CARPETA = Path(__file__).resolve().parent
DATOS = CARPETA / "datos"
EJEMPLOS = DATOS / "ejemplos"

COLORES_TIPO = {"impreso": "#2a78d6", "foto": "#eb6834", "manuscrito": "#7a5195"}

st.set_page_config(page_title="Voces y documentos peruanos", page_icon="🎙️", layout="wide")


# ───────────────────────────────────────────────────────── datos
@st.cache_data
def cargar():
    documentos = pd.read_csv(DATOS / "documentos.csv", dtype={"texto_referencia": str}).fillna({"texto_referencia": ""})
    audios = pd.read_csv(DATOS / "transcripciones.csv", dtype={"texto_referencia": str}).fillna({"texto_referencia": ""})
    llm = pd.read_csv(DATOS / "extraccion_llm.csv") if (DATOS / "extraccion_llm.csv").exists() else pd.DataFrame(columns=["id", "campos_json"])
    eval_docs = pd.read_csv(DATOS / "evaluacion_documentos.csv") if (DATOS / "evaluacion_documentos.csv").exists() else pd.DataFrame()
    eval_audio = pd.read_csv(DATOS / "evaluacion_audio.csv") if (DATOS / "evaluacion_audio.csv").exists() else pd.DataFrame()
    metricas = json.loads((DATOS / "metricas.json").read_text(encoding="utf-8")) if (DATOS / "metricas.json").exists() else {}
    return documentos, audios, llm, eval_docs, eval_audio, metricas


documentos, audios, llm, eval_docs, eval_audio, metricas = cargar()
campos_llm = {fila.id: fila.campos_json for fila in llm.itertuples()}


def resaltar(texto: str, termino: str) -> str:
    """Marca en amarillo las apariciones del término buscado (sin distinguir mayúsculas)."""
    texto = str(texto).replace("\n", "  \n")
    if not termino:
        return texto
    return re.sub(f"({re.escape(termino)})", r":orange-background[\1]", texto, flags=re.IGNORECASE)


def mostrar_campos(registro_id: str):
    crudo = campos_llm.get(registro_id)
    if not isinstance(crudo, str):
        st.caption("Sin extracción del LLM para este registro.")
        return
    try:
        st.json(json.loads(crudo), expanded=True)
    except json.JSONDecodeError:
        st.code(crudo)


# ───────────────────────────────────────────────────────── encabezado
st.title("¿Se pueden convertir documentos escaneados y noticieros en texto estructurado y buscable?")
st.markdown(
    "Prensa electoral peruana de **1939 y 1945**, leyes, cartas manuscritas de **1912 a 1921** y **noticieros "
    "cinematográficos de la década de 1940** pasaron por un mismo pipeline: OCR con **PaddleOCR**, transcripción con "
    "**faster-whisper** (VAD y diarización), seudonimización de particulares y extracción de campos con un **LLM local "
    "(Qwen3-1.7B)**. La calidad se midió contra transcripciones hechas a mano de 10 documentos y 10 clips."
)

# ───────────────────────────────────────────────────────── filtros
with st.sidebar:
    st.header("Filtros")
    termino = st.text_input("Buscar una palabra en los textos", placeholder="p. ej. Jurado, Prado, Cusco")
    tipos = st.multiselect("Tipo de documento", list(COLORES_TIPO), default=list(COLORES_TIPO))
    productoras = sorted(audios["productora"].dropna().unique())
    elegidas = st.multiselect("Productora del noticiero", productoras, default=productoras)
    dur_max = int(audios["duracion_seg"].max()) + 1
    rango = st.slider("Duración del clip (segundos)", 0, dur_max, (0, dur_max))
    modelos_ocr = sorted(eval_docs["modelo"].unique()) if len(eval_docs) else []
    modelos_asr = sorted(eval_audio["modelo"].unique()) if len(eval_audio) else []
    modelo_ocr = st.selectbox("Modelo de OCR a mostrar en la evaluación", modelos_ocr) if modelos_ocr else None
    modelo_asr = st.selectbox("Modelo de ASR a mostrar en la evaluación", modelos_asr) if modelos_asr else None
    st.caption("Los textos publicados están revisados y seudonimizados: los particulares aparecen como [PERSONA_nn].")

docs_f = documentos[documentos["tipo"].isin(tipos)]
audios_f = audios[audios["productora"].isin(elegidas) & audios["duracion_seg"].between(*rango)]
if termino:
    docs_f = docs_f[docs_f["texto_ocr"].str.contains(termino, case=False, na=False, regex=False)]
    audios_f = audios_f[audios_f["texto_asr"].str.contains(termino, case=False, na=False, regex=False)]

# ───────────────────────────────────────────────────────── indicadores
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Documentos procesados", f"{len(documentos)}", help="8 impresos, 8 fotos y 9 manuscritos")
c2.metric("Minutos de audio transcritos", f"{audios['duracion_seg'].sum() / 60:.1f}", help=f"{len(audios)} clips")
if metricas:
    c3.metric("CER del OCR elegido", f"{metricas['cer_elegido']:.2f}",
              delta=f"{metricas['cer_elegido'] - metricas['cer_base']:+.2f} vs. base", delta_color="inverse",
              help=f"{metricas['ocr_elegido']} · 10 documentos con referencia manual")
    c4.metric("WER del ASR elegido", f"{metricas['wer_elegido']:.2f}",
              delta=f"{metricas['wer_elegido'] - metricas['wer_base']:+.2f} vs. base", delta_color="inverse",
              help=f"{metricas['asr_elegido']} · 10 clips con referencia manual")
    c5.metric("Concordancia LLM vs. auditoría", f"{metricas['acuerdo_llm']}/20",
              help="Extracciones completamente correctas en una auditoría humana de 20 casos al azar")

if termino:
    st.info(f"«{termino}» aparece en {len(docs_f)} documentos y {len(audios_f)} clips.")

tab_resumen, tab_docs, tab_audios, tab_eval = st.tabs(["📊 Resumen", "📄 Documentos", "🎙️ Audios", "🧪 Evaluación de modelos"])

# ───────────────────────────────────────────────────────── resumen
with tab_resumen:
    izq, der = st.columns(2)
    with izq:
        conteo = docs_f.groupby("tipo", as_index=False).agg(documentos=("id", "count"), caracteres=("n_caracteres", "median"))
        fig = px.bar(conteo, x="tipo", y="documentos", color="tipo", color_discrete_map=COLORES_TIPO, text="documentos",
                     title="Documentos por tipo (según los filtros)")
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="documentos")
        st.plotly_chart(fig, use_container_width=True)
    with der:
        fig = px.bar(audios_f.sort_values("duracion_seg"), x="duracion_seg", y="id", color="productora", orientation="h",
                     hover_data=["titulo", "fecha_original"], title="Duración de cada clip, por productora",
                     labels={"duracion_seg": "segundos", "id": ""})
        fig.update_layout(height=520, legend_title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabla de registros filtrados")
    tabla = pd.concat([
        docs_f.assign(modalidad="documento", texto=docs_f["texto_ocr"], detalle=docs_f["tipo"])[["id", "modalidad", "detalle", "descripcion", "texto"]],
        audios_f.assign(modalidad="audio", texto=audios_f["texto_asr"], detalle=audios_f["productora"], descripcion=audios_f["titulo"])[["id", "modalidad", "detalle", "descripcion", "texto"]],
    ], ignore_index=True)
    st.dataframe(tabla, use_container_width=True, hide_index=True, height=360)
    st.download_button("Descargar CSV (textos seudonimizados)", tabla.to_csv(index=False).encode("utf-8"),
                       file_name="registros_filtrados.csv", mime="text/csv")

# ───────────────────────────────────────────────────────── explorador de documentos
with tab_docs:
    if docs_f.empty:
        st.warning("Ningún documento cumple los filtros.")
    else:
        etiquetas = {f"{f.id} · {f.tipo} · {f.descripcion}": f.id for f in docs_f.itertuples()}
        doc_id = etiquetas[st.selectbox("Elige un documento", list(etiquetas))]
        fila = documentos.set_index("id").loc[doc_id]
        izq, der = st.columns([1, 1.2])
        with izq:
            imagen = EJEMPLOS / f"{doc_id}.jpg"
            if fila["imagen_publicable"] and imagen.exists():
                st.image(str(imagen), caption=fila["fuente"], use_container_width=True)
            else:
                st.info("La imagen de este documento no se publica (folleto sin licencia de redistribución). "
                        "El texto y los metadatos sí están disponibles.")
            st.markdown(f"**Tipo:** {fila['tipo']} · **Fuente:** {fila['fuente']}")
            st.markdown(f"**Modelo OCR:** {fila['modelo_ocr']}")
            st.markdown("**Campos extraídos por el LLM**")
            mostrar_campos(doc_id)
        with der:
            st.markdown(f"**Texto OCR** · {fila['n_lineas']} líneas · confianza mediana {fila['confianza_mediana']:.2f}")
            st.markdown(resaltar(fila["texto_ocr"], termino))
            if fila["texto_referencia"].strip():
                with st.expander("Texto de referencia (transcripción manual) y CER", expanded=True):
                    st.markdown(resaltar(fila["texto_referencia"], termino))
                    if len(eval_docs):
                        cer = eval_docs[eval_docs["id"] == doc_id][["modelo", "cer"]].sort_values("cer")
                        st.dataframe(cer.round(3), hide_index=True, use_container_width=True)

# ───────────────────────────────────────────────────────── explorador de audios
with tab_audios:
    if audios_f.empty:
        st.warning("Ningún clip cumple los filtros.")
    else:
        etiquetas = {f"{f.id} · {f.titulo} ({f.fecha_original})": f.id for f in audios_f.itertuples()}
        clip_id = etiquetas[st.selectbox("Elige un clip", list(etiquetas))]
        fila = audios.set_index("id").loc[clip_id]
        izq, der = st.columns([1, 1.2])
        with izq:
            sonido = EJEMPLOS / f"{clip_id}.ogg"
            if sonido.exists():
                st.audio(str(sonido), format="audio/ogg")
            st.markdown(f"**{fila['titulo']}** · {fila['productora']} · {fila['fecha_original']}")
            st.markdown(f"**Duración:** {fila['duracion_seg']:.0f} s · **voz detectada:** {fila['segundos_voz']:.0f} s · "
                        f"**hablantes:** {fila['n_hablantes']}")
            st.markdown(f"**Modelo ASR:** {fila['modelo_asr']}")
            st.markdown("**Campos extraídos por el LLM**")
            mostrar_campos(clip_id)
        with der:
            st.markdown("**Transcripción** (las marcas indican quién habla)")
            turnos = re.split(r"(\[HABLANTE_\d+\])", fila["texto_asr"])
            texto = ""
            for parte in turnos:
                if re.fullmatch(r"\[HABLANTE_\d+\]", parte):
                    texto += f"\n\n**{parte}** "
                else:
                    texto += parte
            st.markdown(resaltar(texto.strip(), termino))
            if len(eval_audio) and clip_id in set(eval_audio["id"]):
                with st.expander("Comparación de modelos sobre este clip", expanded=True):
                    st.markdown("**Referencia manual:** " + resaltar(fila["texto_referencia"], termino))
                    for f in eval_audio[eval_audio["id"] == clip_id].sort_values("wer").itertuples():
                        st.markdown(f"**{f.modelo}** · WER {f.wer:.2f} · factor de tiempo real {f.factor_tiempo_real:.2f}")
                        st.caption(f.texto)

# ───────────────────────────────────────────────────────── evaluación
with tab_eval:
    if not metricas:
        st.info("Las métricas aparecen cuando se ejecuta el notebook 02_evaluacion.ipynb.")
    else:
        st.markdown("Todos los modelos se evaluaron sobre **los mismos 10 documentos y 10 clips**, con transcripciones "
                    "de referencia hechas a mano y el mismo criterio de normalización (minúsculas, sin puntuación ni marcas "
                    "de hablante).")
        st.dataframe(pd.DataFrame(metricas["tabla_comparativa"]), hide_index=True, use_container_width=True)
        izq, der = st.columns(2)
        with izq:
            if len(eval_docs):
                datos = eval_docs.groupby(["tipo", "modelo"], as_index=False)["cer"].mean()
                fig = px.bar(datos, x="tipo", y="cer", color="modelo", barmode="group",
                             title="CER promedio por tipo de documento", labels={"cer": "CER", "tipo": ""})
                st.plotly_chart(fig, use_container_width=True)
                if modelo_ocr:
                    st.dataframe(eval_docs[eval_docs["modelo"] == modelo_ocr][["id", "tipo", "cer", "segundos"]].round(3),
                                 hide_index=True, use_container_width=True)
        with der:
            if len(eval_audio):
                fig = px.scatter(eval_audio, x="factor_tiempo_real", y="wer", color="modelo", hover_data=["id"],
                                 title="WER contra velocidad en cada clip", labels={"wer": "WER", "factor_tiempo_real": "factor de tiempo real (menos es más rápido)"})
                st.plotly_chart(fig, use_container_width=True)
                if modelo_asr:
                    st.dataframe(eval_audio[eval_audio["modelo"] == modelo_asr][["id", "wer", "factor_tiempo_real"]].round(3),
                                 hide_index=True, use_container_width=True)
        if "auditoria_errores" in metricas:
            st.subheader("Auditoría humana del LLM: ejemplos de errores")
            st.dataframe(pd.DataFrame(metricas["auditoria_errores"]), hide_index=True, use_container_width=True)

st.caption("Fuentes: capturas propias de El Comercio y La Crónica (1939, 1945), Biblioteca Nacional del Perú y Wikimedia "
           "Commons (CC0, Public Domain Mark, CC BY-SA 3.0, CC BY 4.0). Detalle por archivo en datos/fuentes.csv.")

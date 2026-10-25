"""OCR del corpus de documentos con PaddleOCR 3.7 (sesión 10).

Configuración base, la misma de la clase: modelo PP-OCRv6 (detector + reconocedor multilingüe que
cubre el español), motor PyTorch en CPU y sin pasos previos (no gira la página, no la «plancha»
ni gira las líneas). El resultado crudo de cada imagen (texto, confianza y caja de cada línea) se
guarda en datos/raw/documentos/ocr/ como respaldo.

    uv run python procesar_documentos.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from fuentes_documentos import CARPETA, FUENTES_PROPIAS, bajar_commons, pagina_a_imagen  # noqa: E402

E39 = "Elecciones 1939/Fuentes Primarias/"
E45 = "Elecciones 1945/Fuentes Primarias/"
LIC_PROPIA = "Captura propia; dominio público en el Perú (D. Leg. 822, art. 53: obra de persona jurídica publicada hace más de 70 años)"
LIC_LEY = "Captura propia; texto oficial no protegido (D. Leg. 822, art. 9)"
LIC_FOLLETO = "Captura propia de un folleto de 1945 (uso académico; la imagen no se publica)"

# id, tipo, origen, página o recorte, descripción, fuente, licencia, ¿imagen publicable?
CATALOGO = [
    # ── impresos nítidos: capturas propias, texto recto y contrastado
    ("doc_001", "impreso", E39 + "El Comercio - 1939 - Campaña Electoral y Elecciones de 1939.pdf", 4, "El Comercio (1939): aviso de manifestación cívica en la Plaza de Acho", "El Comercio, Lima, 1939 (captura propia)", LIC_PROPIA, True),
    ("doc_002", "impreso", E39 + "El Comercio - 1939 - Convocatoria a Plebiscito y Elecciones.pdf", 14, "El Comercio (1939): «Mensaje del Presidente de la República»", "El Comercio, Lima, 1939 (captura propia)", LIC_PROPIA, True),
    ("doc_003", "impreso", E39 + "El Comercio - 1939 - Convocatoria a Plebiscito y Elecciones.pdf", 6, "El Comercio (1939): «Víveres para el Auxilio Social de España»", "El Comercio, Lima, 1939 (captura propia)", LIC_PROPIA, True),
    ("doc_004", "impreso", E39 + "Ley 8901 - Primera Parte de Estatuto Electoral de 1939.pdf", 1, "Ley 8901, Estatuto Electoral (1939), primera página", "Ley N.° 8901, 1939 (captura propia)", LIC_LEY, True),
    ("doc_005", "impreso", E45 + "El Comercio - Junio 1945.pdf", 11, "El Comercio (1945): aviso del Partido Nacional Laborista", "El Comercio, Lima, 1945 (captura propia)", LIC_PROPIA, True),
    ("doc_006", "impreso", E45 + "El Comercio - Marzo 1945.pdf", 6, "El Comercio (1945): «Se instaló el Jurado Provincial de Elecciones de Lima»", "El Comercio, Lima, 1945 (captura propia)", LIC_PROPIA, True),
    ("doc_007", "impreso", E45 + "El Comercio - Marzo 1945.pdf", 16, "El Comercio (1945): comunicado del Comité Central de la candidatura del Gral. Ureta", "El Comercio, Lima, 1945 (captura propia)", LIC_PROPIA, True),
    ("doc_008", "impreso", E45 + "La Crónica - Marzo 1945.pdf", 11, "La Crónica (1945): «Instalóse en el Senado el Jurado Provincial de Elecciones de Lima»", "La Crónica, Lima, 1945 (captura propia)", LIC_PROPIA, True),
    # ── fotos con sombras, curvatura o inclinación
    ("doc_009", "foto", E45 + "Tres Discursos - Bloque Antifascista Peruano (1945).pdf", 1, "«Tres discursos» (1945): portada y manifiesto, libro abierto fotografiado", "Bloque Antifascista Peruano, 1945 (foto propia)", LIC_FOLLETO, False),
    ("doc_010", "foto", E45 + "Tres Discursos - Bloque Antifascista Peruano (1945).pdf", 4, "«Tres discursos» (1945): página con sombra y curvatura", "Bloque Antifascista Peruano, 1945 (foto propia)", LIC_FOLLETO, False),
    ("doc_011", "foto", E45 + "Tres Discursos - Bloque Antifascista Peruano (1945).pdf", 9, "«Tres discursos» (1945): portadilla del discurso del jefe del Partido Aprista", "Bloque Antifascista Peruano, 1945 (foto propia)", LIC_FOLLETO, False),
    ("doc_012", "foto", E45 + "Tres Discursos - Bloque Antifascista Peruano (1945).pdf", 18, "«Tres discursos» (1945): portadilla del discurso del Partido Vanguardia Socialista", "Bloque Antifascista Peruano, 1945 (foto propia)", LIC_FOLLETO, False),
    ("doc_013", "foto", E45 + "El Comercio - Febrero 1945.pdf", 2, "El Comercio (1945): «Se declara el estado de beligerancia entre el Perú y Alemania y Japón»", "El Comercio, Lima, 1945 (foto propia)", LIC_PROPIA, True),
    ("doc_014", "foto", E45 + "El Comercio - Febrero 1945.pdf", 8, "El Comercio (1945): aviso del Partido Democrático Reformista", "El Comercio, Lima, 1945 (foto propia)", LIC_PROPIA, True),
    ("doc_015", "foto", E45 + "La Crónica - Marzo 1945.pdf", 12, "La Crónica (1945): informes de los ministros y avisos, columna curvada", "La Crónica, Lima, 1945 (foto propia)", LIC_PROPIA, True),
    ("doc_016", "foto", E39 + "El Comercio - 1939 - Campaña Electoral y Elecciones de 1939.pdf", 17, "El Comercio (1939): «La manifestación de ayer en la Plaza de Acho», columna inclinada", "El Comercio, Lima, 1939 (foto propia)", LIC_PROPIA, True),
    # ── manuscritos del siglo XX en Wikimedia Commons
    ("doc_017", "manuscrito", "File:4000004161-0003 - Carta de Carlos J. Bachmann. 1912 junio 10, Riberalta; para Ricardo Palma, Lima.pdf", 1, "Carta de Carlos J. Bachmann a Ricardo Palma (1912), p. 1", "Biblioteca Nacional del Perú, vía Wikimedia Commons", "CC0", True),
    ("doc_018", "manuscrito", "File:4000004161-0003 - Carta de Carlos J. Bachmann. 1912 junio 10, Riberalta; para Ricardo Palma, Lima.pdf", 2, "Carta de Carlos J. Bachmann a Ricardo Palma (1912), p. 2", "Biblioteca Nacional del Perú, vía Wikimedia Commons", "CC0", True),
    ("doc_019", "manuscrito", "File:4000004161-0003 - Carta de Carlos J. Bachmann. 1912 junio 10, Riberalta; para Ricardo Palma, Lima.pdf", 3, "Carta de Carlos J. Bachmann a Ricardo Palma (1912), p. 3", "Biblioteca Nacional del Perú, vía Wikimedia Commons", "CC0", True),
    ("doc_020", "manuscrito", "File:4000004161-0003 - Carta de Carlos J. Bachmann. 1912 junio 10, Riberalta; para Ricardo Palma, Lima.pdf", 4, "Carta de Carlos J. Bachmann a Ricardo Palma (1912), p. 4", "Biblioteca Nacional del Perú, vía Wikimedia Commons", "CC0", True),
    ("doc_021", "manuscrito", "File:Carta de César Vallejo a José María Eguren.pdf", 1, "Carta de César Vallejo a José María Eguren (Trujillo, 1917)", "Biblioteca Nacional del Perú, vía Wikimedia Commons", "Public Domain Mark", True),
    ("doc_022", "manuscrito", "File:Tarjeta de Ricardo Palma a Luis Antonio Eguiguren Escudero 24 octubre 1912.png", "recorte", "Tarjeta de Ricardo Palma (1912), recorte de la parte manuscrita", "Wikimedia Commons (Leguicall)", "Dominio público", True),
    ("doc_023", "manuscrito", "File:Targheta.jpg", None, "Tarjeta de circulación de la Policía de Lima (1921), llenada a mano", "Wikimedia Commons (archivo Burzagli)", "Public Domain Mark", True),
    ("doc_024", "manuscrito", "File:Socgeolima.jpg", None, "Diploma de la Sociedad Geográfica de Lima (1921), caligrafía y llenado a mano", "Wikimedia Commons (archivo Burzagli)", "CC BY-SA 3.0", True),
    ("doc_025", "manuscrito", "File:Perucent.jpg", None, "Diploma de la Medalla del Centenario (1921), caligrafía y llenado a mano", "Wikimedia Commons (archivo Burzagli)", "CC BY-SA 3.0", True),
]

# Candidatos revisados que NO entran al corpus (se reportan en el notebook)
DESCARTADOS = [
    ("Lima2.jpg (Commons)", "duplicado de contenido: es el mismo diploma de doc_024, escaneado a 806×1136 px; se conserva la versión de mayor resolución"),
    ("Carta de Carlos J. Bachmann, 1914 (Commons)", "no es manuscrito: la carta está mecanografiada"),
    ("Carta de César Vallejo a J. C. Mariátegui (Commons)", "no es manuscrito: la carta está mecanografiada"),
    ("Postal de Blanca Luz Brum y César Miró (Commons)", "solo es la fotografía: el lado escrito no está digitalizado"),
]

COMMONS_ARCHIVO = {
    "File:4000004161-0003 - Carta de Carlos J. Bachmann. 1912 junio 10, Riberalta; para Ricardo Palma, Lima.pdf": "bachmann1912.pdf",
    "File:Carta de César Vallejo a José María Eguren.pdf": "vallejo_eguren.pdf",
    "File:Tarjeta de Ricardo Palma a Luis Antonio Eguiguren Escudero 24 octubre 1912.png": "palma_tarjeta_1912.png",
    "File:Targheta.jpg": "targheta_1921.jpg",
    "File:Socgeolima.jpg": "socgeolima_1921.jpg",
    "File:Perucent.jpg": "perucent_1921.jpg",
}


def preparar_imagen(fila) -> Path:
    """Deja la imagen de cada documento en datos/raw/documentos/<id>.jpg (y el original en originales/)."""
    doc_id, _, origen, pagina = fila[:4]
    destino = CARPETA / f"{doc_id}.jpg"
    if destino.exists():
        return destino
    if origen.startswith("File:"):
        original = bajar_commons(origen, CARPETA / "originales" / COMMONS_ARCHIVO[origen])
        if original.suffix == ".pdf":
            return pagina_a_imagen(original, pagina, destino)
        imagen = Image.open(original).convert("RGB")
        if pagina == "recorte":  # la tarjeta de Palma trae abajo una transcripción impresa: solo el manuscrito
            ancho, alto = imagen.size
            imagen = imagen.crop((0, int(0.03 * alto), ancho, int(0.62 * alto)))
            imagen.save(CARPETA / "recortes" / f"{doc_id}_manuscrito.jpg", quality=92)
        imagen.thumbnail((3000, 3000))
        imagen.save(destino, quality=92)
        return destino
    return pagina_a_imagen(FUENTES_PROPIAS / origen, pagina, destino)


def catalogo() -> pd.DataFrame:
    tabla = pd.DataFrame(CATALOGO, columns=["id", "tipo", "origen", "pagina", "descripcion", "fuente", "licencia", "imagen_publicable"])
    tabla["fuente_url"] = ["https://commons.wikimedia.org/wiki/" + o.replace(" ", "_") if o.startswith("File:") else "local: " + o for o in tabla["origen"]]
    return tabla


# ───────────────────────────────────────────────────────── OCR
OPCIONES = dict(
    use_doc_orientation_classify=False,   # no girar la página completa
    use_doc_unwarping=False,              # no "planchar" hojas dobladas
    use_textline_orientation=False,       # no girar cada línea
)
CONFIG_OCR = "PaddleOCR 3.7.0 · PP-OCRv6 (det + rec multilingüe) · engine=transformers, CPU · sin orientación ni desdoblado"


def crear_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(engine="transformers", device="cpu", **OPCIONES)


def leer(ocr, ruta: Path) -> pd.DataFrame:
    """OCR de una imagen. Devuelve una tabla con una fila por línea de texto (como en la clase)."""
    r = ocr.predict(str(ruta))[0]
    cajas = np.asarray(r["rec_boxes"]).reshape(-1, 4)
    return pd.DataFrame({
        "texto": r["rec_texts"],
        "confianza": np.round(r["rec_scores"], 3),
        "x0": cajas[:, 0], "y0": cajas[:, 1], "x1": cajas[:, 2], "y1": cajas[:, 3],
    })


def ocr_documento(ocr, doc_id: str, carpeta_salida: str = "ocr") -> dict:
    """OCR con caché: el respaldo crudo queda en datos/raw/documentos/<carpeta_salida>/<id>.json."""
    salida = CARPETA / carpeta_salida / f"{doc_id}.json"
    if salida.exists():
        return json.loads(salida.read_text(encoding="utf-8"))
    salida.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    lineas = leer(ocr, CARPETA / f"{doc_id}.jpg")
    resultado = {"id": doc_id, "segundos": round(time.perf_counter() - t0, 2),
                 "lineas": lineas.astype({"x0": int, "y0": int, "x1": int, "y1": int}).to_dict(orient="records")}
    salida.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


if __name__ == "__main__":
    (CARPETA / "recortes").mkdir(parents=True, exist_ok=True)
    tabla = catalogo()
    for fila in CATALOGO:
        preparar_imagen(fila)
    print("Imágenes listas:", len(tabla), flush=True)
    ocr = None
    for doc_id in tabla["id"]:
        if not (CARPETA / "ocr" / f"{doc_id}.json").exists() and ocr is None:
            ocr = crear_ocr()
        r = ocr_documento(ocr, doc_id)
        conf = [l["confianza"] for l in r["lineas"]]
        print(f"{doc_id}: {len(r['lineas']):3d} líneas · confianza mediana {np.median(conf) if conf else 0:.2f} · {r['segundos']:.1f} s", flush=True)

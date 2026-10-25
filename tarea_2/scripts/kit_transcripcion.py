"""Genera una página HTML local para transcribir a mano la muestra de evaluación.

La página muestra cada documento (imagen ampliable) y cada clip (reproductor), con un cuadro de
texto por caso. Lo escrito se guarda en el navegador mientras se trabaja y, al final, el botón
«Descargar» baja un archivo referencias_manuales.json que el notebook 02 lee.

    uv run python scripts/kit_transcripcion.py
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

MUESTRA_DOCS = ["doc_001", "doc_002", "doc_008", "doc_011", "doc_012", "doc_014", "doc_021", "doc_022", "doc_023", "doc_025"]
MUESTRA_CLIPS = ["clip_001", "clip_006", "clip_007", "clip_012", "clip_015", "clip_017", "clip_018", "clip_021", "clip_024", "clip_029"]

INSTRUCCIONES = """
<h2>Cómo transcribir</h2>
<ul>
<li><b>Documentos:</b> escribe exactamente lo que dice la imagen, línea por línea, con la ortografía de la época
(tildes, mayúsculas y abreviaturas tal cual). Si una palabra se corta con guion al final de la línea, escríbela
como está («conven-» en una línea y «io» en la siguiente). Omite sellos, números de página y textos de anuncios
ajenos al documento. Si algo es ilegible, escribe <code>[ilegible]</code>.</li>
<li><b>Audios:</b> escribe todo lo que se dice, en orden, sin marcas de hablante ni de tiempo. Ignora la música.
Si no entiendes una palabra, escribe <code>[inaudible]</code>.</li>
<li>No mires las salidas de los modelos antes ni durante la transcripción: la referencia debe ser independiente.</li>
<li>Lo que escribes se guarda solo en este navegador. Al terminar, pulsa <b>Descargar referencias</b> y guarda
el archivo en <code>datos/raw/transcripcion_manual/</code>.</li>
</ul>
"""


def main():
    salida = RAIZ / "datos" / "raw" / "transcripcion_manual"
    salida.mkdir(parents=True, exist_ok=True)
    casos = [{"id": d, "tipo": "documento", "src": f"../documentos/{d}.jpg"} for d in MUESTRA_DOCS]
    casos += [{"id": c, "tipo": "audio", "src": f"../audio/wav/{c}.wav"} for c in MUESTRA_CLIPS]
    bloques = []
    for c in casos:
        if c["tipo"] == "documento":
            medio = f'<a href="{c["src"]}" target="_blank"><img src="{c["src"]}" loading="lazy"></a><p class="nota">Clic en la imagen para abrirla en grande.</p>'
        else:
            medio = f'<audio controls preload="none" src="{c["src"]}"></audio>'
        bloques.append(f'<section><h3>{c["id"]}</h3><div class="caso"><div class="medio">{medio}</div>'
                       f'<textarea id="{c["id"]}" placeholder="Transcripción manual de {c["id"]}"></textarea></div></section>')
    ids = json.dumps([c["id"] for c in casos])
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Transcripción manual · Tarea 2</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:24px auto;max-width:1300px;padding:0 16px;background:#fbfaf7;color:#1d1d1b}}
section{{border-top:1px solid #ddd;padding:12px 0}} .caso{{display:flex;gap:16px;align-items:flex-start}}
.medio{{flex:1}} img{{max-width:100%;max-height:640px;border:1px solid #ccc}} audio{{width:100%}}
textarea{{flex:1;min-height:320px;font-size:15px;line-height:1.45;padding:8px}}
.barra{{position:sticky;top:0;background:#fbfaf7;padding:8px 0;border-bottom:2px solid #2a78d6;z-index:2}}
button{{font-size:15px;padding:6px 14px;margin-right:8px}} .nota{{color:#777;font-size:12px}}
</style></head><body>
<h1>Transcripción manual de la muestra de evaluación</h1>{INSTRUCCIONES}
<div class="barra"><button onclick="descargar()">Descargar referencias</button><span id="estado"></span></div>
{''.join(bloques)}
<script>
const IDS = {ids};
const CLAVE = "tarea2_referencias";
let datos = {{}};
try {{ datos = JSON.parse(localStorage.getItem(CLAVE) || "{{}}"); }} catch (e) {{}}
function contar() {{
  const hechos = IDS.filter(i => (datos[i] || "").trim().length > 0).length;
  document.getElementById("estado").textContent = hechos + " de " + IDS.length + " casos con texto";
}}
for (const i of IDS) {{
  const t = document.getElementById(i);
  t.value = datos[i] || "";
  t.addEventListener("input", () => {{ datos[i] = t.value; try {{ localStorage.setItem(CLAVE, JSON.stringify(datos)); }} catch (e) {{}} contar(); }});
}}
contar();
function descargar() {{
  const blob = new Blob([JSON.stringify(datos, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "referencias_manuales.json"; a.click();
}}
</script></body></html>"""
    (salida / "index.html").write_text(html, encoding="utf-8")
    print(salida / "index.html")


if __name__ == "__main__":
    main()

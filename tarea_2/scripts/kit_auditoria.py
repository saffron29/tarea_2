"""Página HTML local para auditar a mano 20 extracciones del LLM (10 documentos y 10 audios al azar).

Para cada caso muestra el texto publicado (seudonimizado) y el JSON del LLM, con un selector
correcto/incorrecto por campo. El botón «Descargar» baja auditoria_llm.json, que lee el notebook 02.

    uv run python scripts/kit_auditoria.py
"""
import html
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SEMILLA = 2026

CRITERIOS = """
<h2>Criterios de corrección (definidos antes de auditar)</h2>
<ul>
<li><b>tema</b>: correcto si es la categoría de la lista que mejor describe el contenido principal.</li>
<li><b>resumen</b>: correcto si todo lo que afirma está en el texto (nada inventado) y recoge la idea principal.</li>
<li><b>emisor, destinatario, fecha, lugar, monto</b>: correcto si el valor aparece en el texto y cumple ese rol,
o si es <code>null</code> y el texto no trae ese dato. Incorrecto si es inventado, si tiene el rol equivocado o si
es <code>null</code> cuando el dato está claramente en el texto.</li>
<li><b>citas_clave</b>: correcto si cada cita está literalmente en la transcripción (se toleran diferencias de
puntuación) y es relevante.</li>
<li><b>numero_hablantes</b>: correcto si coincide con el número de marcas [HABLANTE_n] distintas del texto.</li>
<li><b>modalidad</b>: correcto si dice «documento» o «audio» según corresponda.</li>
<li>Un JSON inválido cuenta como extracción incorrecta. Una extracción es <b>completamente correcta</b> solo si
todos sus campos lo son.</li>
</ul>
"""


def muestra(llm: pd.DataFrame) -> pd.DataFrame:
    docs = llm[llm["modalidad"] == "documento"].sample(10, random_state=SEMILLA)
    audios = llm[llm["modalidad"] == "audio"].sample(10, random_state=SEMILLA)
    return pd.concat([docs, audios]).reset_index(drop=True)


def main():
    # la misma tabla que publica el notebook 02, armada desde la caché del LLM
    import extraccion_llm as ex
    import seudonimos
    salidas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ex.CACHE.glob("*.json"))]
    reemplazos = seudonimos.tabla_de_reemplazos(pd.read_csv(RAIZ / "datos" / "privado" / "revision_nombres.csv"))
    llm = ex.tabla_publicable(salidas, reemplazos, pd.Timestamp.now(tz="America/Lima").strftime("%Y-%m-%d"))
    llm.to_csv(RAIZ / "datos" / "extraccion_llm.csv", index=False)
    textos = {**dict(pd.read_csv(RAIZ / "datos" / "documentos.csv")[["id", "texto_ocr"]].values),
              **dict(pd.read_csv(RAIZ / "datos" / "transcripciones.csv")[["id", "texto_asr"]].values)}
    casos = muestra(llm)
    bloques = []
    for c in casos.itertuples():
        try:
            campos = json.loads(c.campos_json)
        except (json.JSONDecodeError, TypeError):
            campos = None
        if not isinstance(campos, dict):
            filas_campos = '<p><b>JSON inválido</b>: cuenta como incorrecta.</p>'
            claves = []
        else:
            claves = list(campos)
            filas_campos = "".join(
                f'<tr><td><b>{html.escape(k)}</b></td><td><pre>{html.escape(json.dumps(v, ensure_ascii=False, indent=1))}</pre></td>'
                f'<td><label><input type="radio" name="{c.id}|{k}" value="1">correcto</label> '
                f'<label><input type="radio" name="{c.id}|{k}" value="0">incorrecto</label></td></tr>' for k, v in campos.items())
        bloques.append(
            f'<section data-id="{c.id}" data-claves="{"|".join(claves)}"><h3>{c.id} · {c.modalidad}</h3><div class="caso">'
            f'<div class="texto"><pre>{html.escape(str(textos.get(c.id, ""))[:2500])}</pre></div>'
            f'<div class="campos"><table>{filas_campos}</table>'
            f'<textarea name="{c.id}|nota" placeholder="Nota: describe el error, si lo hay"></textarea></div></div></section>')
    page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Auditoría del LLM · Tarea 2</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:24px auto;max-width:1400px;padding:0 16px;background:#fbfaf7;color:#1d1d1b}}
section{{border-top:1px solid #ddd;padding:10px 0}} .caso{{display:flex;gap:16px}} .texto{{flex:1;max-height:520px;overflow:auto;background:#fff;border:1px solid #ddd;padding:6px}}
.campos{{flex:1}} pre{{white-space:pre-wrap;margin:0;font-size:13px}} td{{vertical-align:top;padding:4px;border-bottom:1px solid #eee}}
textarea{{width:100%;min-height:50px}} .barra{{position:sticky;top:0;background:#fbfaf7;padding:8px 0;border-bottom:2px solid #2a78d6;z-index:2}}
button{{font-size:15px;padding:6px 14px}}
</style></head><body><h1>Auditoría humana de 20 extracciones del LLM</h1>{CRITERIOS}
<div class="barra"><button onclick="descargar()">Descargar auditoría</button> <span id="estado"></span></div>
{''.join(bloques)}
<script>
const CLAVE = "tarea2_auditoria";
let datos = {{}};
try {{ datos = JSON.parse(localStorage.getItem(CLAVE) || "{{}}"); }} catch (e) {{}}
document.querySelectorAll("input[type=radio]").forEach(r => {{
  if (datos[r.name] === r.value) r.checked = true;
  r.addEventListener("change", () => {{ datos[r.name] = r.value; guardar(); }});
}});
document.querySelectorAll("textarea").forEach(t => {{
  t.value = datos[t.name] || "";
  t.addEventListener("input", () => {{ datos[t.name] = t.value; guardar(); }});
}});
function guardar() {{ try {{ localStorage.setItem(CLAVE, JSON.stringify(datos)); }} catch (e) {{}} contar(); }}
function contar() {{
  let total = 0, hechos = 0;
  document.querySelectorAll("section").forEach(s => {{
    const claves = s.dataset.claves ? s.dataset.claves.split("|") : [];
    total += claves.length; hechos += claves.filter(k => datos[s.dataset.id + "|" + k] !== undefined).length;
  }});
  document.getElementById("estado").textContent = hechos + " de " + total + " campos revisados";
}}
contar();
function descargar() {{
  const blob = new Blob([JSON.stringify(datos, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "auditoria_llm.json"; a.click();
}}
</script></body></html>"""
    salida = RAIZ / "datos" / "raw" / "auditoria"
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "index.html").write_text(page, encoding="utf-8")
    casos[["id", "modalidad"]].to_csv(salida / "muestra_auditoria.csv", index=False)
    print(salida / "index.html")


if __name__ == "__main__":
    sys.path.insert(0, str(RAIZ))
    main()

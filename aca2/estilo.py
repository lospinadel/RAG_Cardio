# -*- coding: utf-8 -*-
"""Sistema de diseño de CardioGuía RAG.

Es el mismo de CardioRiesgo (ACA 1 de Aprendizaje Automático): papel de ECG en el encabezado,
canales de monitor para las cifras, Saira ancha para títulos y números, Atkinson Hyperlegible Next
para el texto y el cian clínico como acento. Hay tema claro (trazo impreso) y oscuro (pantalla de monitor).
"""
import base64

import streamlit as st

TOKENS = {  # nombre: (tema claro, tema oscuro)
    "pantalla": ("#f6f7f9", "#060b16"),
    "panel": ("#ffffff", "#0c1424"),
    "panel-2": ("#f1f4f8", "#111b30"),
    "borde": ("#e2e6ee", "#1e2b47"),
    "borde-2": ("#cdd4e0", "#2a3a5c"),
    "tinta": ("#0f172a", "#eaf0fa"),
    "tinta-2": ("#475066", "#a9b6cc"),
    "tinta-3": ("#5d6880", "#8391a8"),
    "cian": ("#0b7493", "#2fd3f0"),    # acento de la interfaz y canal del modelo
    "verde": ("#0b7f47", "#2fe08f"),   # resultado bueno (siempre con texto)
    "rojo": ("#c4223a", "#ff5a6e"),    # alerta (siempre con texto)
    "ambar": ("#9a5a00", "#ffb648"),   # umbral de decisión
    "traza": ("#111827", "#2fe08f"),   # trazo del ECG: tinta en papel / fósforo en pantalla
    "hero": ("#fffbfb", "#0c1424"),    # papel milimetrado / vidrio del monitor
    "cuadricula": ("rgba(214, 69, 69, .2)", "rgba(122, 146, 190, .07)"),
    "cuadricula-fina": ("rgba(214, 69, 69, .09)", "rgba(0, 0, 0, 0)"),
    "brillo-traza": ("rgba(0, 0, 0, 0)", "rgba(47, 224, 143, .55)"),
}
FUENTE_CIFRAS = "Saira, 'Atkinson Hyperlegible Next', system-ui, sans-serif"
# Un color por guía, tomados de la paleta categórica de CardioRiesgo
COLOR_DOC = {"SCA": "#e7672b", "HTA": "#4292f4", "DIS": "#0b7493"}

RAIZ = (":root {\n" + "\n".join(f"  --cr-{n}: light-dark({c}, {o});" for n, (c, o) in TOKENS.items())
        + f"\n  --cr-cifras: {FUENTE_CIFRAS};\n  --cr-ease-out: cubic-bezier(0.23, 1, 0.32, 1);\n}}\n")

CSS = """
.block-container, [data-testid="stMainBlockContainer"] {max-width: min(100%, 84rem); padding-top: 3.6rem;}
[data-testid="stHeading"] h3, [data-testid="stHeading"] h4 {font-weight: 600;}

/* Encabezado: papel de ECG, marca y trazo */
section.cr-hero {position: relative; padding: 1.25rem 1.75rem 1.125rem; border: 1px solid var(--cr-borde);
  border-radius: 18px; background: var(--cr-hero); overflow: hidden; margin: 0;}
section.cr-hero::before {content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: linear-gradient(var(--cr-cuadricula) 1px, transparent 1px),
                    linear-gradient(90deg, var(--cr-cuadricula) 1px, transparent 1px),
                    linear-gradient(var(--cr-cuadricula-fina) 1px, transparent 1px),
                    linear-gradient(90deg, var(--cr-cuadricula-fina) 1px, transparent 1px);
  background-size: 30px 30px, 30px 30px, 6px 6px, 6px 6px;}
section.cr-hero > * {position: relative;}
section.cr-hero div.cr-hero-top {display: flex; align-items: flex-start; justify-content: space-between; gap: 0.75rem 1rem;
  flex-wrap: wrap; margin: 0 0 0.625rem;}
section.cr-hero ul.cr-chips {display: flex; flex-wrap: wrap; gap: 0.5rem; list-style: none; margin: 0; padding: 0;}
section.cr-hero ul.cr-chips li {margin: 0; padding: 0.3125rem 0.6875rem; font-size: .84rem; line-height: 1.2; color: var(--cr-tinta-2);
  border: 1px solid var(--cr-borde-2); border-radius: 999px; background: var(--cr-panel-2);}
section.cr-hero ul.cr-chips li::marker {content: none;}
section.cr-hero img.cr-logo-cun {height: 2.75rem; width: auto; padding: 0.375rem 0.625rem; background: #ffffff;
  border-radius: 10px; border: 1px solid var(--cr-borde); flex: none;}
section.cr-hero div.cr-marca {font-family: var(--cr-cifras); font-weight: 700; font-stretch: 125%;
  font-variation-settings: "wdth" 125; font-size: clamp(2.3rem, 2.6vw + 1.1rem, 3.5rem); line-height: 1;
  letter-spacing: -0.01em; color: var(--cr-tinta); margin: 0; padding: 0;}
section.cr-hero div.cr-marca span {color: var(--cr-cian);}
section.cr-hero svg.cr-ecg {display: block; width: 100%; height: auto; margin: 0.25rem 0 0.125rem; overflow: visible;}
section.cr-hero .cr-traza {fill: none; stroke: var(--cr-traza); stroke-width: 2.6; stroke-linecap: round;
  stroke-linejoin: round; filter: drop-shadow(0 0 5px var(--cr-brillo-traza));}
section.cr-hero .cr-punto {fill: var(--cr-traza);}
section.cr-hero .cr-onda {fill: none; stroke: var(--cr-traza); stroke-width: 1.6; transform-box: fill-box;
  transform-origin: center; animation: cr-latido 1.1s var(--cr-ease-out) infinite;}
section.cr-hero p.cr-lema {color: var(--cr-tinta-2); font-size: 1.02rem; line-height: 1.55; max-width: 72ch; margin: 0;}

/* Canales de monitor: nombre, cifra grande y nota */
div.cr-fila {container-type: inline-size;}
div.cr-canales {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.75rem; margin: 0.875rem 0 0.625rem;}
div.cr-canales.cr-n5 > .cr-canal:last-child {grid-column: 1 / -1;}
@container (min-width: 38.75rem) {
  div.cr-canales.cr-n4 {grid-template-columns: repeat(4, minmax(0, 1fr));}
  div.cr-canales.cr-n3 {grid-template-columns: repeat(3, minmax(0, 1fr));}
}
@container (min-width: 47.5rem) {
  div.cr-canales.cr-n5 {grid-template-columns: repeat(5, minmax(0, 1fr));}
  div.cr-canales.cr-n5 > .cr-canal:last-child {grid-column: auto;}
}
div.cr-canales .cr-canal {position: relative; display: grid; grid-row: span 3; grid-template-rows: subgrid; row-gap: 0.25rem;
  align-content: start; padding: 0.8125rem 0.9375rem 0.75rem; background: var(--cr-panel); border: 1px solid var(--cr-borde);
  border-radius: 12px; overflow: hidden; min-width: 0;}
div.cr-canales .cr-canal::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: var(--canal);}
div.cr-canales .cr-canal-nombre {font-size: .88rem; line-height: 1.3; color: var(--cr-tinta-2); align-self: start;}
div.cr-canales .cr-canal-valor {font-family: var(--cr-cifras); font-weight: 600; font-size: 2rem; line-height: 1.1;
  color: var(--canal); font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  align-self: end; text-shadow: 0 0 18px light-dark(rgba(0, 0, 0, 0), color-mix(in srgb, var(--canal) 35%, transparent));}
div.cr-canales .cr-chico .cr-canal-valor {font-size: 1.45rem; white-space: normal; overflow-wrap: anywhere; line-height: 1.15;}
div.cr-canales .cr-canal-nota {font-size: .8rem; line-height: 1.35; color: var(--cr-tinta-3);}
div.cr-canales .cr-c-datos {--canal: var(--cr-tinta);}
div.cr-canales .cr-c-modelo {--canal: var(--cr-cian);}
div.cr-canales .cr-c-umbral {--canal: var(--cr-ambar);}
div.cr-canales .cr-c-decision {--canal: var(--cr-verde);}

/* Entrada del encabezado: solo la primera vez */
div.cr-anim .cr-traza {stroke-dasharray: 1; stroke-dashoffset: 1; animation: cr-barrido 1.7s linear .15s forwards;}
div.cr-anim .cr-cabeza {opacity: 0; animation: cr-aparece 180ms ease-out 1.8s forwards;}
div.cr-anim .cr-canal {opacity: 0; transform: translateY(6px); animation: cr-entra 360ms var(--cr-ease-out) forwards;}
div.cr-anim .cr-canal:nth-child(1) {animation-delay: .55s;}
div.cr-anim .cr-canal:nth-child(2) {animation-delay: .61s;}
div.cr-anim .cr-canal:nth-child(3) {animation-delay: .67s;}
div.cr-anim .cr-canal:nth-child(4) {animation-delay: .73s;}
div.cr-anim .cr-canal:nth-child(5) {animation-delay: .79s;}
@keyframes cr-barrido {to {stroke-dashoffset: 0;}}
@keyframes cr-aparece {to {opacity: 1;}}
@keyframes cr-entra {to {opacity: 1; transform: none;}}
@keyframes cr-latido {from {transform: scale(1); opacity: .75;} to {transform: scale(3.2); opacity: 0;}}

/* Fuentes recuperadas: ficha con la barra del color de la guía */
div.cg-fuente {position: relative; border: 1px solid var(--cr-borde); border-radius: 12px; padding: 0.625rem 0.875rem 0.625rem 1.125rem;
  margin: 0.4rem 0; background: var(--cr-panel); font-size: .9rem; line-height: 1.45; overflow: hidden;}
div.cg-fuente::before {content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--c);}
div.cg-fuente .cab {display: flex; justify-content: space-between; gap: 0.625rem; font-weight: 700; color: var(--cr-tinta);}
div.cg-fuente .cab .cos {font-family: var(--cr-cifras); font-weight: 600; color: var(--cr-tinta-2); white-space: nowrap;
  font-variant-numeric: tabular-nums;}
div.cg-fuente .txt {color: var(--cr-tinta-2); margin-top: 0.25rem;}
span.cg-tag {display: inline-block; font-size: .72rem; font-weight: 700; padding: 0.0625rem 0.4375rem; border-radius: 999px;
  margin-left: 0.375rem; vertical-align: 0.0625rem; color: var(--cr-cian); background: color-mix(in srgb, var(--cr-cian) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--cr-cian) 35%, transparent);}
span.cg-tag.no {color: var(--cr-tinta-3); background: var(--cr-panel-2); border-color: var(--cr-borde);}
div.cg-kpi {display: flex; flex-wrap: wrap; gap: 0.25rem 1rem; font-size: .8rem; color: var(--cr-tinta-3); margin: 0.25rem 0 0.375rem;}
div.cg-kpi b {font-family: var(--cr-cifras); font-weight: 600; color: var(--cr-tinta-2); font-variant-numeric: tabular-nums;}

/* Pasos numerados (cómo funciona) */
ol.cr-pasos {list-style: none; counter-reset: paso; display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
  gap: 0.625rem; padding: 0; margin: 0.375rem 0 1.125rem;}
ol.cr-pasos li {counter-increment: paso; margin: 0; padding: 0.875rem 0.875rem 0.8125rem; border: 1px solid var(--cr-borde);
  border-radius: 12px; background: var(--cr-panel);}
ol.cr-pasos li::before {content: counter(paso); display: block; font-family: var(--cr-cifras); font-weight: 700;
  font-size: 1.5rem; line-height: 1; color: var(--cr-cian); margin-bottom: 0.5rem;}
ol.cr-pasos strong {display: block; color: var(--cr-tinta); font-size: 1rem; margin-bottom: 0.25rem;}
ol.cr-pasos span {display: block; color: var(--cr-tinta-2); font-size: .88rem; line-height: 1.45;}

/* Barra lateral */
div.cr-side .cr-side-marca {display: flex; align-items: center; gap: 0.625rem; margin: 0 0 0.375rem; color: var(--cr-tinta);
  font-family: var(--cr-cifras); font-weight: 700; font-size: 1.4rem; font-variation-settings: "wdth" 115;}
div.cr-side .cr-side-marca svg {color: var(--cr-cian); flex: none;}
div.cr-side p {font-size: .9rem; line-height: 1.5; color: var(--cr-tinta-2); margin: 0 0 0.25rem;}
div.cr-side .cr-side-titulo {font-family: var(--cr-cifras); font-weight: 600; font-size: 1rem; color: var(--cr-tinta);
  margin: 1rem 0 0.25rem; padding: 0;}
div.cr-side dl {margin: 0; padding: 0;}
div.cr-side dl > div {display: block; padding: 0.4375rem 0; margin: 0; border-bottom: 1px solid var(--cr-borde);}
div.cr-side dt {font-size: .86rem; color: var(--cr-tinta-2); margin: 0; font-weight: 400;}
div.cr-side dd {margin: 0.125rem 0 0; padding: 0; font-family: var(--cr-cifras); font-size: .92rem; color: var(--cr-tinta);
  font-variant-numeric: tabular-nums; overflow-wrap: anywhere;}
div.cr-side p.cr-side-aviso {margin-top: 1rem; padding: 0.625rem 0.75rem; border-radius: 10px; background: var(--cr-panel);
  border: 1px solid var(--cr-borde); font-size: .84rem;}

/* Pestañas, gráficos y botones */
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 0.25rem;}
[data-testid="stTabs"] button[data-baseweb="tab"] p {font-size: 1rem; font-weight: 600;}
[data-testid="stPlotlyChart"] {background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px; overflow: hidden;}
[data-testid^="stBaseButton"] {transition: transform 160ms var(--cr-ease-out), background-color 160ms ease;}
[data-testid^="stBaseButton"]:active {transform: scale(0.97);}

@media (max-width: 640px) {
  section.cr-hero {padding: 1.125rem 1.125rem 1rem;}
  section.cr-hero .cr-traza {stroke-width: 6;}
  section.cr-hero .cr-onda {stroke-width: 3;}
  div.cr-canales .cr-canal-valor {font-size: 1.75rem;}
}
@media (prefers-reduced-motion: reduce) {
  div.cr-anim .cr-traza {animation: none; stroke-dashoffset: 0;}
  div.cr-anim .cr-cabeza {animation: none; opacity: 1;}
  div.cr-anim .cr-canal {transform: none; animation: cr-aparece 200ms ease forwards;}
  section.cr-hero .cr-onda {animation: none; opacity: 0;}
  [data-testid^="stBaseButton"]:active {transform: none;}
}
"""

ICONO_MONITOR = ('<svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="15" rx="2"/>'
                 '<path d="M5 11h3l2-4 3 8 2-4h4"/><path d="M8 21h8"/></svg>')


def aplicar():
    """Inyecta tokens y estilos. Va una vez al comienzo de cada ejecución."""
    st.html(f"<style>{RAIZ}{CSS}</style>")


def bloque(html):
    """Inserta HTML propio. Se usa st.markdown porque conserva los SVG en línea (st.html los elimina);
    se quitan sangrías y líneas vacías para que Markdown no lo convierta en bloques de código."""
    st.markdown("\n".join(linea.strip() for linea in html.strip().splitlines() if linea.strip()),
                unsafe_allow_html=True)


def num(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def trazo_ecg():
    """Trazo de ECG estilizado (complejos P-QRS-T) para el encabezado."""
    y = 56
    d = [f"M0 {y}"]
    for x in (40, 280, 520, 760, 1000):
        d += [f"L{x} {y}", f"Q{x + 12} {y - 14} {x + 24} {y}", f"L{x + 36} {y}", f"L{x + 40} {y + 5}",
              f"L{x + 47} {y - 48}", f"L{x + 54} {y + 15}", f"L{x + 59} {y}", f"L{x + 74} {y}",
              f"Q{x + 90} {y - 22} {x + 106} {y}"]
    d.append(f"L1172 {y}")
    return " ".join(d)


def canal(nombre, valor, nota="", tipo="datos"):
    """Recuadro estilo monitor: nombre del canal, cifra grande y una nota (textos largos van en letra menor)."""
    nota_html = f'<span class="cr-canal-nota">{nota}</span>' if nota else ""
    chico = " cr-chico" if len(str(valor)) > 8 else ""
    return (f'<div class="cr-canal cr-c-{tipo}{chico}"><span class="cr-canal-nombre">{nombre}</span>'
            f'<span class="cr-canal-valor">{valor}</span>{nota_html}</div>')


def fila_canales(canales):
    bloque(f'<div class="cr-fila"><div class="cr-canales cr-n{len(canales)}">{"".join(canales)}</div></div>')


def encabezado(chips, lema, canales, ruta_logo=None, animar=False):
    """Papel de ECG con chips, logo, marca CardioGuía, trazo y lema; debajo, los canales del monitor."""
    try:
        logo = ('<img class="cr-logo-cun" alt="Corporación Unificada Nacional de Educación Superior, CUN" '
                f'src="data:image/png;base64,{base64.b64encode(ruta_logo.read_bytes()).decode()}">') if ruta_logo else ""
    except OSError:
        logo = ""
    bloque(f"""
<div class="cr-intro cr-fila{' cr-anim' if animar else ''}">
<section class="cr-hero" aria-label="Presentación">
  <div class="cr-hero-top">
  <ul class="cr-chips" aria-label="Contexto del proyecto">{''.join(f'<li>{c}</li>' for c in chips)}</ul>{logo}
  </div>
  <div class="cr-marca" role="heading" aria-level="1">CardioGuía <span>RAG</span></div>
  <svg class="cr-ecg" viewBox="0 0 1200 80" aria-hidden="true" focusable="false">
    <path class="cr-traza" pathLength="1" d="{trazo_ecg()}"/>
    <g class="cr-cabeza"><circle class="cr-onda" cx="1172" cy="56" r="4.5"/><circle class="cr-punto" cx="1172" cy="56" r="4.5"/></g>
  </svg>
  <p class="cr-lema">{lema}</p>
</section>
<div class="cr-canales cr-n{len(canales)}" aria-label="Resumen">{''.join(canales)}</div>
</div>""")

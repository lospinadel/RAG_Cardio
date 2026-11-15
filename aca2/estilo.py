# -*- coding: utf-8 -*-
"""Sistema de diseño de CardioGuía RAG.

Es el mismo de CardioRiesgo (ACA 1 de Aprendizaje Automático, Grupo 13): los tokens, el CSS
responsive, la escala de pantalla, el puente de tema claro/oscuro y el estilo de los gráficos
se copian tal cual de esa app. Al final van solo las piezas propias del chat (fuentes y citas).

Diseño: tema claro por defecto (como un ECG impreso) y oscuro (como un monitor de cabecera).
Cian para el modelo, verde para lo que sale bien, rojo para alertas y ámbar para el umbral.
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
    "cian": ("#0b7493", "#2fd3f0"),    # canal del modelo y acento de la interfaz
    "verde": ("#0b7f47", "#2fe08f"),   # riesgo bajo (siempre con ícono y texto)
    "rojo": ("#c4223a", "#ff5a6e"),    # riesgo alto (siempre con ícono y texto)
    "ambar": ("#9a5a00", "#ffb648"),   # umbral de decisión
    "traza": ("#111827", "#2fe08f"),   # trazo del ECG: tinta en papel / fósforo en pantalla
    "hero": ("#fffbfb", "#0c1424"),    # papel milimetrado / vidrio del monitor
    "cuadricula": ("rgba(214, 69, 69, .2)", "rgba(122, 146, 190, .07)"),
    "cuadricula-fina": ("rgba(214, 69, 69, .09)", "rgba(0, 0, 0, 0)"),
    "brillo-traza": ("rgba(0, 0, 0, 0)", "rgba(47, 224, 143, .55)"),
    "brillo-umbral": ("rgba(0, 0, 0, 0)", "rgba(255, 182, 72, .6)"),
}
OSCURO = {nombre: valores[1] for nombre, valores in TOKENS.items()}

# Colores con que se dibujan los gráficos de Plotly. Los de datos sirven en los dos
# temas (contraste >= 3:1 sobre blanco y sobre el panel oscuro); los de texto, rejilla
# y ejes se escriben con el valor oscuro y el CSS los cambia en el tema claro.
PANEL, PANEL_2, BORDE, BORDE_2 = OSCURO["panel"], OSCURO["panel-2"], OSCURO["borde"], OSCURO["borde-2"]
TINTA, TINTA_2, TINTA_3 = OSCURO["tinta"], OSCURO["tinta-2"], OSCURO["tinta-3"]
AMBAR = OSCURO["ambar"]
AZUL = "#4292f4"       # sin enfermedad / baja el riesgo
NARANJA = "#e7672b"    # con enfermedad / sube el riesgo
REJILLA = "#18233b"
EJE = BORDE_2
GRIS_BARRA = "#33415e"
TEXTO_CELDA = "#f8fafc"  # texto sobre celdas oscuras de los mapas de calor (igual en los dos temas)
COLOR_DX = {"Sin enfermedad": AZUL, "Con enfermedad": NARANJA}
GRAFICO_CLARO = {  # color oscuro con que se dibuja -> color en el tema claro
    TINTA: "#0f172a", TINTA_2: "#475066", TINTA_3: "#5d6880", REJILLA: "#e6e9ef", EJE: "#c9d0dc",
    GRIS_BARRA: "#b9c1ce", AMBAR: "#9a5a00", PANEL: "#ffffff", PANEL_2: "#ffffff", BORDE: "#e2e6ee",
}
FUENTE_CIFRAS = "Saira, 'Atkinson Hyperlegible Next', system-ui, sans-serif"

# Un color por guía, tomados de la paleta categórica de CardioRiesgo
COLOR_DOC = {"SCA": NARANJA, "HTA": AZUL, "DIS": "#7c66e0"}


def _rgb(color_hex):
    return "rgb({}, {}, {})".format(*(int(color_hex[i:i + 2], 16) for i in (1, 3, 5)))


RAIZ = (":root {\n" + "\n".join(f"  --cr-{n}: light-dark({c}, {o});" for n, (c, o) in TOKENS.items())
        + f"\n  --cr-cifras: {FUENTE_CIFRAS};\n  --cr-ease-out: cubic-bezier(0.23, 1, 0.32, 1);\n}}\n")
# Plotly escribe los colores en línea (style="fill: rgb(...)"); estas reglas los cambian según el tema
CSS_GRAFICOS = "\n".join(
    f'.js-plotly-plot [style*="{prop}: {_rgb(oscuro)}"] {{{prop}: light-dark({claro}, {oscuro}) !important;}}'
    for oscuro, claro in GRAFICO_CLARO.items() for prop in ("fill", "stroke"))

CSS = """
/* Escala fluida (mobile-first): hasta 1600 px de ancho la base es 16 px; en monitores grandes
   y televisores crece hasta 30 px, y el ancho del contenido (en rem) crece con ella */
html {font-size: clamp(16px, calc(16px + (100vw - 1600px) * 0.0075), 30px) !important;}
.block-container, [data-testid="stMainBlockContainer"] {max-width: min(100%, 100rem);
  padding: 4.2rem clamp(1rem, 3.5vw, 4rem) 3rem;}
[data-testid="stHeading"] h3 {font-weight: 600; font-size: 1.45rem;}

/* Encabezado: marca, trazo de ECG y canales del monitor */
section.cr-hero {position: relative; padding: 1.375rem 1.75rem 1.25rem; border: 1px solid var(--cr-borde);
  border-radius: 18px; background: var(--cr-hero); overflow: hidden; margin: 0;}
section.cr-hero::before {content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: linear-gradient(var(--cr-cuadricula) 1px, transparent 1px),
                    linear-gradient(90deg, var(--cr-cuadricula) 1px, transparent 1px),
                    linear-gradient(var(--cr-cuadricula-fina) 1px, transparent 1px),
                    linear-gradient(90deg, var(--cr-cuadricula-fina) 1px, transparent 1px);
  background-size: 30px 30px, 30px 30px, 6px 6px, 6px 6px;}
section.cr-hero > * {position: relative;}
section.cr-hero div.cr-hero-top {display: flex; align-items: flex-start; justify-content: space-between; gap: 0.75rem 1rem;
  flex-wrap: wrap; margin: 0 0 0.75rem;}
section.cr-hero ul.cr-chips {display: flex; flex-wrap: wrap; gap: 0.5rem; list-style: none; margin: 0; padding: 0;}
/* El logo va sobre una placa blanca para que sus verdes se vean igual en los dos temas */
section.cr-hero img.cr-logo-cun {height: 2.75rem; width: auto; padding: 0.375rem 0.625rem; background: #ffffff;
  border-radius: 10px; border: 1px solid var(--cr-borde); flex: none;}
section.cr-hero ul.cr-chips li {margin: 0; padding: 0.3125rem 0.6875rem; font-size: .84rem; line-height: 1.2; color: var(--cr-tinta-2);
  border: 1px solid var(--cr-borde-2); border-radius: 999px; background: var(--cr-panel-2);}
section.cr-hero ul.cr-chips li::marker {content: none;}
section.cr-hero div.cr-marca {font-family: var(--cr-cifras); font-weight: 700; font-stretch: 125%;
  font-variation-settings: "wdth" 125; font-size: clamp(2.5rem, 3.2vw + 1.2rem, 4rem); line-height: 1;
  letter-spacing: -0.01em; color: var(--cr-tinta); margin: 0; padding: 0;}
section.cr-hero svg.cr-ecg {display: block; width: 100%; height: auto; margin: 0.375rem 0 0.25rem; overflow: visible;}
section.cr-hero .cr-traza {fill: none; stroke: var(--cr-traza); stroke-width: 2.6; stroke-linecap: round;
  stroke-linejoin: round; filter: drop-shadow(0 0 5px var(--cr-brillo-traza));}
section.cr-hero .cr-punto {fill: var(--cr-traza);}
section.cr-hero .cr-onda {fill: none; stroke: var(--cr-traza); stroke-width: 1.6; transform-box: fill-box;
  transform-origin: center; animation: cr-latido 1.1s var(--cr-ease-out) infinite;}
section.cr-hero p.cr-lema {color: var(--cr-tinta-2); font-size: 1.04rem; line-height: 1.55; max-width: 68ch; margin: 0;}

div.cr-fila {container-type: inline-size;}
div.cr-canales {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.75rem; margin: 0.875rem 0 0.625rem;}
div.cr-canales.cr-n5 > .cr-canal:last-child {grid-column: 1 / -1;}
@container (min-width: 38.75rem) {
  div.cr-canales.cr-n4 {grid-template-columns: repeat(4, minmax(0, 1fr));}
  div.cr-canales.cr-n6 {grid-template-columns: repeat(3, minmax(0, 1fr));}
}
@container (min-width: 47.5rem) {
  div.cr-canales.cr-n5 {grid-template-columns: repeat(5, minmax(0, 1fr));}
  div.cr-canales.cr-n5 > .cr-canal:last-child {grid-column: auto;}
}
@container (min-width: 60rem) {
  div.cr-canales.cr-n6 {grid-template-columns: repeat(6, minmax(0, 1fr));}
  div.cr-canales.cr-n6 .cr-canal-valor {font-size: 1.8rem;}
}
div.cr-canales .cr-canal {position: relative; display: grid; grid-row: span 3; grid-template-rows: subgrid; row-gap: 0.25rem;
  align-content: start; padding: 0.8125rem 0.9375rem 0.75rem; background: var(--cr-panel); border: 1px solid var(--cr-borde);
  border-radius: 12px; overflow: hidden; min-width: 0;}
div.cr-canales .cr-canal::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: var(--canal);}
div.cr-canales .cr-canal-nombre {font-size: .88rem; line-height: 1.3; color: var(--cr-tinta-2); align-self: start;}
div.cr-canales .cr-canal-valor {font-family: var(--cr-cifras); font-weight: 600; font-size: 2.15rem; line-height: 1.1;
  color: var(--canal); font-variant-numeric: tabular-nums; white-space: nowrap; align-self: end;
  text-shadow: 0 0 18px light-dark(rgba(0, 0, 0, 0), color-mix(in srgb, var(--canal) 35%, transparent));}
div.cr-canales .cr-canal-nota {font-size: .8rem; line-height: 1.35; color: var(--cr-tinta-3);}
div.cr-canales .cr-c-datos {--canal: var(--cr-tinta);}
div.cr-canales .cr-c-modelo {--canal: var(--cr-cian);}
div.cr-canales .cr-c-decision {--canal: var(--cr-verde);}

/* Animación de entrada: solo la primera vez que se abre la app */
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

/* Lectura del riesgo */
div.cr-lectura {--estado: var(--cr-verde); position: relative; padding: 1rem 1.25rem 0.875rem; background: var(--cr-panel);
  border: 1px solid var(--cr-borde); border-radius: 16px; overflow: hidden;}
div.cr-lectura.cr-alto {--estado: var(--cr-rojo);}
div.cr-lectura::before {content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: var(--estado);}
div.cr-lectura .cr-lectura-cab {display: flex; justify-content: space-between; align-items: center; gap: 0.5rem 0.875rem;
  flex-wrap: wrap; font-size: .92rem; color: var(--cr-tinta-2);}
div.cr-lectura .cr-criterio {font-size: .82rem; color: var(--cr-tinta); padding: 0.25rem 0.625rem; border-radius: 999px;
  border: 1px solid var(--cr-borde-2); background: var(--cr-panel-2); white-space: nowrap;}
div.cr-lectura .cr-lectura-valor {font-family: var(--cr-cifras); font-weight: 600; font-size: clamp(4rem, 5vw + 1.4rem, 5.6rem);
  line-height: 1; margin: 0.875rem 0 0.75rem; color: var(--estado); font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
  text-shadow: 0 0 28px light-dark(rgba(0, 0, 0, 0), color-mix(in srgb, var(--estado) 40%, transparent));}
div.cr-lectura .cr-unidad {font-size: .4em; font-weight: 500; margin-left: .14em; color: var(--cr-tinta-2); text-shadow: none;}
div.cr-lectura .cr-estado {display: flex; align-items: center; gap: 0.625rem; padding: 0.625rem 0.75rem; border-radius: 10px;
  font-size: .98rem; line-height: 1.4; color: var(--cr-tinta); background: color-mix(in srgb, var(--estado) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--estado) 38%, transparent);}
div.cr-lectura .cr-estado svg {flex: none; color: var(--estado);}
div.cr-lectura .cr-barra {position: relative; display: flex; gap: 0.1875rem; margin: 1.25rem 0 0.375rem;}
div.cr-lectura .cr-seg {flex: 1; height: 0.875rem; border-radius: 3px; background: var(--cr-panel-2);
  box-shadow: inset 0 0 0 1px var(--cr-borde);}
div.cr-lectura .cr-seg.on {background: var(--estado);
  box-shadow: 0 0 8px light-dark(rgba(0, 0, 0, 0), color-mix(in srgb, var(--estado) 45%, transparent));}
div.cr-lectura .cr-umbral {position: absolute; top: -6px; bottom: -6px; width: 2px; margin-left: -0.0625rem; border-radius: 2px;
  background: var(--cr-ambar); box-shadow: 0 0 6px var(--cr-brillo-umbral);}
div.cr-lectura .cr-escala {position: relative; height: 20px; font-size: .8rem; color: var(--cr-tinta-3);}
div.cr-lectura .cr-escala span {position: absolute; top: 0; white-space: nowrap;}
div.cr-lectura .cr-escala .cr-e-umbral {color: var(--cr-ambar); transform: translateX(-50%);}
div.cr-lectura .cr-escala .cr-e-fin {right: 0;}
p.cr-nota {color: var(--cr-tinta-2); font-size: .94rem; line-height: 1.55; margin: 0.75rem 0.125rem 0;}
p.cr-nota b {color: var(--cr-tinta);}
@media (min-width: 992px) {
  [data-testid="stColumn"]:has(.cr-lectura) {position: sticky; top: 4.25rem; align-self: flex-start;}
}

/* Rótulos de grupo, avisos, fichas y pasos */
p.cr-rotulo {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.02rem; color: var(--cr-tinta); margin: 0 0 0.125rem;}
div.cr-aviso {display: flex; gap: 0.625rem; align-items: flex-start; padding: 0.75rem 0.875rem; border-radius: 12px;
  background: var(--cr-panel); border: 1px solid var(--cr-borde); color: var(--cr-tinta-2); font-size: .9rem; line-height: 1.5;}
div.cr-aviso svg {flex: none; margin-top: 0.0625rem; color: var(--cr-ambar);}
div.cr-fichas {display: grid; grid-template-columns: repeat(auto-fit, minmax(15.5rem, 1fr)); gap: 0.875rem; margin: 0.25rem 0 0.5rem;}
div.cr-fichas .cr-ficha {padding: 1.125rem 1.25rem; background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px;}
div.cr-fichas .cr-ficha-titulo {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.15rem; color: var(--cr-tinta);
  margin: 0 0 0.5rem; padding: 0;}
div.cr-fichas p {color: var(--cr-tinta-2); font-size: .96rem; line-height: 1.55; margin: 0;}
div.cr-fichas p b {color: var(--cr-tinta);}
dl.cr-dic {margin: 0; padding: 0;}
dl.cr-dic > div {padding: 0.5rem 0; border-bottom: 1px solid var(--cr-borde); margin: 0;}
dl.cr-dic > div:last-child {border-bottom: 0; padding-bottom: 0;}
dl.cr-dic dt {font-weight: 700; color: var(--cr-tinta); font-size: .95rem; margin: 0;}
dl.cr-dic dd {margin: 0.125rem 0 0; color: var(--cr-tinta-2); font-size: .9rem; line-height: 1.45;}
dl.cr-dic .cr-dic-val {display: block; margin-top: 0.125rem; font-family: var(--cr-cifras); font-size: .84rem; color: var(--cr-cian);}
ol.cr-pasos {list-style: none; counter-reset: paso; display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
  gap: 0.625rem; padding: 0; margin: 0.375rem 0 1.125rem;}
ol.cr-pasos li {counter-increment: paso; margin: 0; padding: 0.875rem 0.875rem 0.8125rem; border: 1px solid var(--cr-borde);
  border-radius: 12px; background: var(--cr-panel);}
ol.cr-pasos li::before {content: counter(paso); display: block; font-family: var(--cr-cifras); font-weight: 700;
  font-size: 1.5rem; line-height: 1; color: var(--cr-cian); margin-bottom: 0.5rem;}
ol.cr-pasos.cr-seis {grid-template-columns: repeat(2, 1fr);}
@media (min-width: 900px) {ol.cr-pasos.cr-seis {grid-template-columns: repeat(3, 1fr);}}
@media (min-width: 1300px) {ol.cr-pasos.cr-seis {grid-template-columns: repeat(6, 1fr);}}
ol.cr-pasos strong {display: block; color: var(--cr-tinta); font-size: 1rem; margin-bottom: 0.25rem;}
ol.cr-pasos span {display: block; color: var(--cr-tinta-2); font-size: .88rem; line-height: 1.45;}

/* Barra lateral */
div.cr-side .cr-side-marca {display: flex; align-items: center; gap: 0.625rem; margin: 0 0 0.375rem; color: var(--cr-tinta);
  font-family: var(--cr-cifras); font-weight: 700; font-size: 1.4rem; font-variation-settings: "wdth" 115;}
div.cr-side .cr-side-marca svg {color: var(--cr-verde); flex: none;}
div.cr-side p {font-size: .9rem; line-height: 1.5; color: var(--cr-tinta-2); margin: 0 0 0.25rem;}
div.cr-side .cr-side-titulo {font-family: var(--cr-cifras); font-weight: 600; font-size: 1rem; color: var(--cr-tinta);
  margin: 1.375rem 0 0.375rem; padding: 0;}
div.cr-side dl {margin: 0; padding: 0;}
div.cr-side dl > div {display: flex; justify-content: space-between; align-items: baseline; gap: 0.75rem; padding: 0.4375rem 0;
  margin: 0; border-bottom: 1px solid var(--cr-borde);}
div.cr-side dt {font-size: .86rem; color: var(--cr-tinta-2); margin: 0; font-weight: 400;}
div.cr-side dd {margin: 0; padding: 0; font-family: var(--cr-cifras); font-size: .95rem; color: var(--cr-tinta); text-align: right; font-variant-numeric: tabular-nums;}
div.cr-side ul {margin: 0; padding: 0 0 0 1.05rem;}
div.cr-side li {font-size: .9rem; color: var(--cr-tinta); margin: 0.1875rem 0; padding: 0;}
div.cr-side p.cr-side-aviso {margin-top: 1.25rem; padding: 0.625rem 0.75rem; border-radius: 10px; background: var(--cr-panel);
  border: 1px solid var(--cr-borde); font-size: .84rem;}

/* En pantallas grandes la barra lateral crece con la escala para que su texto no se parta */
@media (min-width: 1700px) {
  [data-testid="stSidebar"][data-collapsed="false"] {width: 19rem !important; min-width: 19rem !important;
    max-width: 19rem !important;}
}

.st-key-escala_pantalla {display: none !important;}

/* Gráficos dentro de una tarjeta */
[data-testid="stPlotlyChart"] {background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px; overflow: hidden;}

/* El selector de tema del menú se oculta para que el interruptor sea el único control
   (la app lo usa por dentro para cambiar de tema) */
[data-testid="stThemeSwitcher"],
[data-testid="stThemeSwitcher"] + [data-testid="stMainMenuDivider"] {display: none !important;}

/* Pestañas */
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 0.25rem;}
[data-testid="stTabs"] button[data-baseweb="tab"] p {font-size: 1rem; font-weight: 600;}

/* Respuesta al presionar (botones) */
[data-testid^="stBaseButton"] {transition: transform 160ms var(--cr-ease-out), background-color 160ms ease,
  border-color 160ms ease, color 160ms ease;}
[data-testid^="stBaseButton"]:active {transform: scale(0.97);}

/* Entrada de cada pestaña: contexto en lenguaje sencillo y una pregunta guía */
div.cr-entrada {max-width: 62rem; margin: 0.25rem 0 1rem;}
div.cr-entrada p {font-size: 1.04rem; line-height: 1.6; color: var(--cr-tinta-2); margin: 0 0 0.5rem;}
p.cr-guia {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.12rem; color: var(--cr-tinta);
  border-left: 3px solid var(--cr-cian); padding: 0.125rem 0 0.125rem 0.75rem; margin: 0.625rem 0 0;}

/* El problema contado como una escena */
div.cr-relato {display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); gap: 0.875rem; margin: 0.25rem 0 1rem;}
div.cr-relato .cr-escena {position: relative; padding: 1.125rem 1.25rem 1.125rem 1.375rem; background: var(--cr-panel);
  border: 1px solid var(--cr-borde); border-radius: 14px; overflow: hidden;}
div.cr-relato .cr-escena::before {content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--cr-cian);}
div.cr-relato .cr-e2::before {background: var(--cr-ambar);}
div.cr-relato .cr-e3::before {background: var(--cr-rojo);}
div.cr-relato p {font-size: 1rem; line-height: 1.6; color: var(--cr-tinta-2); margin: 0;}
div.cr-relato p b {color: var(--cr-tinta);}
div.cr-relato p.cr-guia {margin-top: 0.75rem;}

/* Flujo de la consulta a la decisión */
ol.cr-flujo {list-style: none; display: grid; grid-template-columns: repeat(auto-fit, minmax(10.5rem, 1fr)); gap: 1.25rem;
  padding: 0; margin: 0.375rem 0 1.25rem;}
ol.cr-flujo li {position: relative; margin: 0; padding: 0.9375rem 0.9375rem 0.875rem; border-radius: 12px;
  background: var(--cr-panel); border: 1px solid var(--cr-borde); border-top: 3px solid var(--cr-cian);}
ol.cr-flujo li:nth-child(4) {border-top-color: var(--cr-ambar);}
ol.cr-flujo li:nth-child(5) {border-top-color: var(--cr-verde);}
ol.cr-flujo li:not(:last-child)::after {content: "→"; position: absolute; right: -1.05rem; top: 50%; transform: translateY(-50%);
  font-family: var(--cr-cifras); font-weight: 700; color: var(--cr-tinta-3); font-size: 1.1rem;}
ol.cr-flujo strong {display: block; color: var(--cr-tinta); font-family: var(--cr-cifras); font-size: 1.05rem; margin-bottom: 0.25rem;}
ol.cr-flujo span {display: block; color: var(--cr-tinta-2); font-size: .88rem; line-height: 1.45;}

/* Comparaciones con barras HTML (pistas y votos de los modelos) */
div.cr-pistas {display: grid; align-items: start; grid-template-columns: repeat(auto-fit, minmax(min(24rem, 100%), 1fr)); gap: 0.875rem; margin: 0.5rem 0 1rem;}
div.cr-pista {margin-top: 0.75rem; padding: 1rem 1.125rem; background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px;}
div.cr-pista .cr-pista-tit {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.05rem; color: var(--cr-tinta); margin: 0 0 0.625rem;}
div.cr-pista .cr-pista-nota {font-size: .86rem; line-height: 1.5; color: var(--cr-tinta-2); margin: 0.625rem 0 0;}
div.cr-comp {display: grid; grid-template-columns: minmax(6rem, 38%) 1fr 3.25rem; gap: 0.625rem; align-items: center; margin: 0.375rem 0;}
div.cr-comp .cr-comp-et {font-size: .9rem; color: var(--cr-tinta-2); line-height: 1.3;}
div.cr-comp .cr-comp-pista {height: 0.875rem; border-radius: 999px; background: var(--cr-panel-2); box-shadow: inset 0 0 0 1px var(--cr-borde); overflow: hidden;}
div.cr-comp .cr-comp-barra {display: block; height: 100%; border-radius: 999px; background: #4292f4;
  transition: width 400ms var(--cr-ease-out);}
div.cr-comp .cr-comp-barra.cr-alta {background: #e7672b;}
div.cr-comp .cr-comp-val {font-family: var(--cr-cifras); font-weight: 600; font-size: 1rem; color: var(--cr-tinta);
  text-align: right; font-variant-numeric: tabular-nums;}

/* Los 4 resultados posibles */
div.cr-resultados {display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 0.75rem; margin: 0.5rem 0 0.875rem;}
div.cr-res {display: grid; align-content: start; gap: 0.25rem; padding: 1rem 1.125rem; background: var(--cr-panel); border: 1px solid var(--cr-borde);
  border-radius: 14px; border-top: 4px solid var(--cr-c);}
div.cr-res-vp {--cr-c: var(--cr-verde);} div.cr-res-vn {--cr-c: var(--cr-cian);}
div.cr-res-fp {--cr-c: var(--cr-ambar);} div.cr-res-fn {--cr-c: var(--cr-rojo);}
div.cr-res .cr-res-num {font-family: var(--cr-cifras); font-weight: 700; font-size: 2.6rem; line-height: 1; color: var(--cr-c);
  font-variant-numeric: tabular-nums;}
div.cr-res .cr-res-tit {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.08rem; color: var(--cr-tinta);}
div.cr-res .cr-res-txt {font-size: .88rem; line-height: 1.45; color: var(--cr-tinta-2);}
div.cr-metricas {display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 0.75rem; margin: 0 0 0.875rem;}
div.cr-metrica {padding: 0.875rem 1rem; border-radius: 12px; background: var(--cr-panel-2); border: 1px solid var(--cr-borde);}
div.cr-metrica p {font-size: .9rem; line-height: 1.5; color: var(--cr-tinta-2); margin: 0;}
div.cr-metrica .cr-metrica-tit {font-family: var(--cr-cifras); font-weight: 600; font-size: 1.05rem; color: var(--cr-tinta); margin-bottom: 0.25rem;}
div.cr-metrica .cr-metrica-nota {font-size: .84rem; margin-top: 0.375rem;}
span.cr-formula {display: block; margin-top: 0.25rem; font-family: var(--cr-cifras); font-weight: 600; color: var(--cr-cian);
  font-variant-numeric: tabular-nums;}

/* Bitácora del entrenamiento en vivo */
div.cr-bitacora {padding: 0.875rem 1rem; background: var(--cr-panel); border: 1px solid var(--cr-borde); border-radius: 14px;}
div.cr-bitacora ol {list-style: none; margin: 0.375rem 0 0; padding: 0;}
div.cr-bitacora li {display: flex; gap: 0.625rem; align-items: baseline; font-size: .9rem; line-height: 1.45; color: var(--cr-tinta-2);
  padding: 0.375rem 0; margin: 0; border-bottom: 1px solid var(--cr-borde);}
div.cr-bitacora li:first-child {color: var(--cr-tinta);}
div.cr-bitacora li:last-child {border-bottom: 0;}
span.cr-bit-paso {flex: none; min-width: 1.9rem; font-family: var(--cr-cifras); font-weight: 700; color: var(--cr-cian);
  font-variant-numeric: tabular-nums;}

/* Fichas con borde de color según lo que está en juego */
div.cr-fichas .cr-borde-rojo {border-top: 4px solid var(--cr-rojo);}
div.cr-fichas .cr-borde-ambar {border-top: 4px solid var(--cr-ambar);}
div.cr-fichas .cr-borde-verde {border-top: 4px solid var(--cr-verde);}
div.cr-side p.cr-side-como {margin: 0.25rem 0 0.75rem; font-size: .88rem;}

@media (max-width: 640px) {
  ol.cr-flujo {gap: 0.625rem;}
  ol.cr-flujo li:not(:last-child)::after {content: none;}
  div.cr-comp {grid-template-columns: 6.5rem 1fr 3rem;}
}
@media (prefers-reduced-motion: reduce) {
  div.cr-comp .cr-comp-barra {transition: none;}
}

@media (max-width: 640px) {
  section.cr-hero {padding: 1.125rem 1.125rem 1rem;}
  section.cr-hero .cr-traza {stroke-width: 6;}
  section.cr-hero .cr-onda {stroke-width: 3;}
  div.cr-canales .cr-canal-valor {font-size: 1.9rem;}
}
@media (prefers-reduced-motion: reduce) {
  div.cr-anim .cr-traza {animation: none; stroke-dashoffset: 0;}
  div.cr-anim .cr-cabeza {animation: none; opacity: 1;}
  div.cr-anim .cr-canal {transform: none; animation: cr-aparece 200ms ease forwards;}
  section.cr-hero .cr-onda {animation: none; opacity: 0;}
  [data-testid^="stBaseButton"]:active {transform: none;}
}

/* ---- Piezas propias de CardioGuía RAG (mismos tokens, radios y tipografía de CardioRiesgo) ---- */
section.cr-hero div.cr-marca span {color: var(--cr-cian);}
div.cr-canales .cr-c-umbral {--canal: var(--cr-ambar);}
div.cr-canales .cr-chico .cr-canal-valor {font-size: 1.6rem; white-space: normal; overflow-wrap: anywhere; line-height: 1.2;}
@container (min-width: 38.75rem) {div.cr-canales.cr-n3 {grid-template-columns: repeat(3, minmax(0, 1fr));}}

/* Fuentes recuperadas: ficha con la barra del color de la guía */
div.cg-fuente {position: relative; border: 1px solid var(--cr-borde); border-radius: 12px; padding: 0.625rem 0.875rem 0.625rem 1.125rem;
  margin: 0.4rem 0; background: var(--cr-panel); font-size: .9rem; line-height: 1.5; overflow: hidden;}
div.cg-fuente::before {content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--c);}
div.cg-fuente .cab {display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.25rem 0.625rem; font-weight: 700; color: var(--cr-tinta);}
div.cg-fuente .cab .cos {font-family: var(--cr-cifras); font-weight: 600; color: var(--cr-tinta-2); white-space: nowrap;
  font-variant-numeric: tabular-nums;}
div.cg-fuente .txt {color: var(--cr-tinta-2); margin-top: 0.25rem;}
span.cg-tag {display: inline-block; font-size: .72rem; font-weight: 700; padding: 0.0625rem 0.4375rem; border-radius: 999px;
  margin-left: 0.375rem; vertical-align: 0.0625rem; color: var(--cr-cian); background: color-mix(in srgb, var(--cr-cian) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--cr-cian) 35%, transparent);}
span.cg-tag.no {color: var(--cr-tinta-3); background: var(--cr-panel-2); border-color: var(--cr-borde);}
div.cg-kpi {display: flex; flex-wrap: wrap; gap: 0.25rem 1rem; font-size: .8rem; color: var(--cr-tinta-3); margin: 0.25rem 0 0.375rem;}
div.cg-kpi b {font-family: var(--cr-cifras); font-weight: 600; color: var(--cr-tinta-2); font-variant-numeric: tabular-nums;}
"""

ICONO_MARCA = ('<svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.8" '
               'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
               '<path d="M12 20.4s-7.6-4.5-9.3-9.3C1.5 7.7 3.5 4.6 6.9 4.6c2.1 0 3.7 1.2 5.1 3 1.4-1.8 3-3 5.1-3 '
               '3.4 0 5.4 3.1 4.2 6.5-1.7 4.8-9.3 9.3-9.3 9.3z"/><path d="M3.2 12.2h4.6l1.6-3 2.4 6.2 1.9-4.4 1 1.2h6.1"/></svg>')
ICONO_INFO = ('<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/>'
              '<path d="M12 11v5.5"/><path d="M12 7.6h.01"/></svg>')


# ---------------------------------------------------------------------------
# Utilidades de formato y piezas de interfaz
# ---------------------------------------------------------------------------
def aplicar():
    """Inyecta tokens y estilos. Va una vez al comienzo de cada ejecución."""
    st.html(f"<style>{RAIZ}{CSS}{CSS_GRAFICOS}</style>")


def num(x, dec=1):
    """Número con coma decimal (formato usado en Colombia)."""
    return f"{x:,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x, dec=1):
    return f"{num(100 * x, dec)}\u00a0%"


def canal(nombre, valor, nota="", tipo="datos"):
    """Recuadro estilo monitor: nombre del canal, cifra grande y una nota."""
    nota_html = f'<span class="cr-canal-nota">{nota}</span>' if nota else ""
    html = (f'<div class="cr-canal cr-c-{tipo}"><span class="cr-canal-nombre">{nombre}</span>'
            f'<span class="cr-canal-valor">{valor}</span>{nota_html}</div>')
    chico = " cr-chico" if len(str(valor)) > 8 else ""
    return html.replace('class="cr-canal cr-c-', f'class="cr-canal{chico} cr-c-', 1)


def bloque(html):
    """Inserta HTML propio. Se usa st.markdown porque conserva los SVG en línea (st.html los elimina);
    se quitan sangrías y líneas vacías para que Markdown no lo convierta en bloques de código."""
    st.markdown("\n".join(linea.strip() for linea in html.strip().splitlines() if linea.strip()),
                unsafe_allow_html=True)


def fila_canales(canales):
    bloque(f'<div class="cr-fila"><div class="cr-canales cr-n{len(canales)}">{"".join(canales)}</div></div>')


def aviso(texto):
    bloque(f'<div class="cr-aviso">{ICONO_INFO}<span>{texto}</span></div>')


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


def estilo(fig, alto=340, leyenda=True, titulo=None):
    fig.update_layout(
        height=alto,
        margin=dict(l=14, r=18, t=(64 if leyenda else 54) if titulo else 30, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        separators=",.",
        font=dict(family=FUENTE_CIFRAS, color=TINTA_2, size=14),
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None,
                    font=dict(color=TINTA_2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=BORDE_2, font=dict(family=FUENTE_CIFRAS, size=13, color=TINTA)),
        bargap=0.45,
    )
    if titulo:
        fig.update_layout(title=dict(text=titulo, x=0, xref="container", xanchor="left", y=0.98, yanchor="top",
                                     pad=dict(l=16, t=8), font=dict(family=FUENTE_CIFRAS, size=16, color=TINTA, weight=600)))
    fig.update_xaxes(automargin=True, showgrid=False, zeroline=False, linecolor=EJE, tickcolor=EJE, ticks="outside",
                     title_font=dict(color=TINTA_2, size=12), tickfont=dict(color=TINTA_3), title_standoff=10)
    fig.update_yaxes(automargin=True, gridcolor=REJILLA, gridwidth=1, zeroline=False, linecolor=EJE,
                     title_font=dict(color=TINTA_2, size=12), tickfont=dict(color=TINTA_3), title_standoff=10)
    return fig


def escalar_figura(fig, s):
    """Agranda letras, márgenes y alto de una figura por el factor de escala de la pantalla."""
    def r(v):
        return round(v * s, 1)
    L = fig.layout
    fig.update_layout(height=round((L.height or 340) * s), font_size=r(L.font.size or 14))
    if L.title.font.size:
        fig.update_layout(title_font_size=r(L.title.font.size))
    if L.hoverlabel.font.size:
        fig.update_layout(hoverlabel_font_size=r(L.hoverlabel.font.size))
    for lado in ("l", "r", "t", "b"):
        if getattr(L.margin, lado) is not None:
            setattr(L.margin, lado, round(getattr(L.margin, lado) * s))
    for nota in L.annotations:
        if nota.font.size:
            nota.font.size = r(nota.font.size)
    for eje in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        if eje.tickfont.size:
            eje.tickfont.size = r(eje.tickfont.size)
        if eje.title.font.size:
            eje.title.font.size = r(eje.title.font.size)
    for traza in fig.data:
        if getattr(traza, "textfont", None) is not None and traza.textfont.size:
            traza.textfont.size = r(traza.textfont.size)
    return fig


def encabezado(chips, lema, canales, ruta_logo=None, animar=False):
    """Papel de ECG con chips, logo, marca, trazo y lema; debajo, los canales del monitor (igual a CardioRiesgo)."""
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


# ---------------------------------------------------------------------------
# Tema y escala. El interruptor de modo oscuro queda enlazado a la URL (?oscuro=true), así que
# el modo elegido se mantiene al recargar o al compartir el enlace. El campo oculto
# "escala_pantalla" lo escribe el navegador (base tipográfica / 16 px) para agrandar los
# gráficos de Plotly en monitores grandes y televisores.
# ---------------------------------------------------------------------------
def controles():
    with st.container(horizontal_alignment="right"):
        oscuro = st.toggle(":material/dark_mode: Modo oscuro", key="oscuro", bind="query-params",
                           help="Cambia entre el tema claro (por defecto) y el oscuro.")
        st.text_input("Escala de pantalla", key="escala_pantalla", label_visibility="collapsed")
    try:
        escala = min(2.0, max(1.0, float(st.session_state.get("escala_pantalla") or 1)))
    except ValueError:
        escala = 1.0
    return oscuro, escala


def mostrar(fig, escala=1.0, **kw):
    if escala > 1:
        escalar_figura(fig, escala)
    st.plotly_chart(fig, theme=None, config={"displaylogo": False}, **kw)


PUENTE_TEMA = """
<span hidden></span>
<script>
(async () => {
  const quiero = "__MODO__";
  const clave = "stActiveTheme-" + window.location.pathname + "-v2";
  const espera = (ms) => new Promise((r) => setTimeout(r, ms));
  const actual = () => {
    const app = document.querySelector('[data-testid="stApp"]');
    return app && getComputedStyle(app).colorScheme.includes("dark") ? "oscuro" : "claro";
  };
  const recordarClaro = () => { try { localStorage.setItem(clave, JSON.stringify("Light")); } catch (e) {} };
  if (actual() === quiero) { recordarClaro(); sessionStorage.removeItem("crTemaRecarga"); return; }
  const destino = quiero === "oscuro" ? "Dark" : "Light";
  const velo = document.createElement("style");
  velo.textContent = '[data-testid="stMainMenuPopover"]{opacity:0 !important;pointer-events:none !important}';
  document.head.appendChild(velo);
  let listo = false;
  try {
    const boton = document.querySelector('[data-testid="stMainMenuButton"]');
    if (boton) {
      boton.click();
      let item = null;
      for (let i = 0; i < 40 && !item; i++) {
        await espera(25);
        item = document.querySelector('[data-testid="stMainMenuItem-theme-' + destino + '"]');
      }
      if (item) { item.click(); listo = true; }
      await espera(80);
      if (document.querySelector('[data-testid="stMainMenuPopover"]')) { boton.click(); }
    }
  } catch (e) { listo = false; }
  await espera(120);
  velo.remove();
  if (!listo || actual() !== quiero) {
    if (sessionStorage.getItem("crTemaRecarga")) { return; }  // una sola recarga, nunca en bucle
    try { localStorage.setItem(clave, JSON.stringify(destino)); sessionStorage.setItem("crTemaRecarga", "1"); } catch (e) { return; }
    window.location.reload();
    return;
  }
  sessionStorage.removeItem("crTemaRecarga");
  recordarClaro();
})();
</script>
"""


ESCALA_PANTALLA = r"""
<span hidden></span>
<script>
(() => {
  const informar = () => {
    const s = Math.round(parseFloat(getComputedStyle(document.documentElement).fontSize) / 16 * 20) / 20;
    const campo = document.querySelector('.st-key-escala_pantalla input');
    if (!campo) return;
    const actual = parseFloat(campo.value || "1");
    if (Math.abs(actual - s) < 0.01) return;
    const fijar = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    fijar.call(campo, s.toFixed(2));
    campo.dispatchEvent(new Event("input", {bubbles: true}));
    campo.dispatchEvent(new KeyboardEvent("keydown", {key: "Enter", code: "Enter", keyCode: 13, bubbles: true}));
  };
  if (!window.__crEscala) {
    window.__crEscala = true;
    let espera;
    window.addEventListener("resize", () => { clearTimeout(espera); espera = setTimeout(informar, 400); });
  }
  // Varios intentos: si la app todavía está terminando de ejecutarse, el primer aviso se puede perder
  [50, 700, 1800, 4000].forEach((t) => setTimeout(informar, t));
})();
</script>
"""


def scripts(oscuro):
    """Puente de tema y medición de escala; van al final de la página, como en CardioRiesgo."""
    st.html(PUENTE_TEMA.replace("__MODO__", "oscuro" if oscuro else "claro"), unsafe_allow_javascript=True)
    st.html(ESCALA_PANTALLA, unsafe_allow_javascript=True)

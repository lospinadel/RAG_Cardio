# -*- coding: utf-8 -*-
"""CardioGuía RAG · chatbot sobre guías clínicas cardiovasculares (ACA 2 · Computación en la Nube · Grupo 13)."""
import json
import time

import streamlit as st

st.set_page_config(page_title="CardioGuía RAG · Grupo 13", page_icon=":material/monitor_heart:", layout="wide",
                   initial_sidebar_state="auto")
T0 = time.perf_counter()

import estilo as es  # noqa: E402  (sistema de diseño de CardioRiesgo)

es.aplicar()


def registro(paso):
    """Deja en el log de Streamlit Cloud cuánto tarda cada etapa del arranque."""
    print(f"[cardioguia] {time.perf_counter() - T0:6.1f} s · {paso}", flush=True)


AVISO = st.empty()
AVISO.info("Iniciando CardioGuía RAG… la primera carga del servidor puede tardar 1 o 2 minutos.", icon=":material/hourglass_top:")
registro("inicio del script")
import numpy as np
import pandas as pd

import rag_core as rc
registro("librerías cargadas")

COLOR_DOC = es.COLOR_DOC
EJEMPLOS = ["¿Desde qué cifra se considera que uno tiene la tensión alta?",
            "Siento un dolor fuerte en el pecho que no se me quita, ¿qué debo hacer?",
            "¿Qué alimentos debo evitar si tengo el colesterol alto?",
            "¿Qué cuidados debo tener cuando me den de alta después de un infarto?"]



# --------------------------------------------------------------------------- recursos (una vez por servidor)
@st.cache_resource(show_spinner="Conectando con Qdrant Cloud…")
def obtener_base():
    registro("conectando con Qdrant")
    base = rc.BaseVectorial(rc.leer_secreto("QDRANT_URL"), rc.leer_secreto("QDRANT_API_KEY"))
    registro(f"{base.modo} conectado en {base.ms_conexion:.0f} ms · colecciones: {base.colecciones}")
    snapshot = rc.cargar_snapshot()
    acciones = {"e5": base.asegurar_coleccion("e5", snapshot)}
    if rc.leer_secreto("OPENAI_API_KEY"):
        acciones["openai"] = base.asegurar_coleccion("openai", snapshot)
    registro(f"colecciones verificadas: {acciones}")
    return base, acciones


@st.cache_resource(show_spinner="Cargando el modelo de embeddings Multilingual-E5 (la primera vez tarda un minuto)…")
def obtener_embedder(preferencia):
    registro(f"cargando embeddings ({preferencia})")
    resultado = rc.crear_embedder(preferencia, rc.leer_secreto("OPENAI_API_KEY"))
    registro(f"embeddings listos: {resultado[1]}")
    return resultado


@st.cache_resource(show_spinner="Verificando el modelo de lenguaje…")
def obtener_llm(proveedor):
    llm = rc.ClienteLLM(proveedor, rc.leer_secreto(rc.PROVEEDORES[proveedor]["clave"]))
    registro(f"LLM listo: {llm.modelo}")
    return llm


@st.cache_data(show_spinner=False)
def obtener_pacientes():
    return rc.cargar_pacientes()


@st.cache_data(ttl=120, show_spinner=False)
def listar_fragmentos(coleccion, version):
    return pd.DataFrame(BASE.fragmentos(coleccion))


try:
    BASE, ACCIONES = obtener_base()
except Exception as e:
    st.error(f"No se pudo conectar con la base vectorial ({type(e).__name__}). Revise QDRANT_URL y QDRANT_API_KEY "
             "y que el clúster esté activo en cloud.qdrant.io.")
    st.stop()

PREFERENCIA = (rc.leer_secreto("EMBEDDINGS") or "e5").lower()
try:
    EMBEDDER, MSG_EMB = obtener_embedder(PREFERENCIA)
except Exception as e:
    st.error(f"No se pudo cargar el modelo de embeddings ({type(e).__name__}).")
    st.stop()
COLECCION = rc.COLECCION[EMBEDDER.clave]

PROVEEDORES_OK = [p for p, c in rc.PROVEEDORES.items() if rc.leer_secreto(c["clave"])]
ss = st.session_state
ss.setdefault("historial", [])
ss.setdefault("version_indice", 0)

# --------------------------------------------------------------------------- barra lateral
with st.sidebar:
    es.bloque(f"""
<div class="cr-side">
  <div class="cr-side-marca">{es.ICONO_MARCA}<span>CardioGuía RAG</span></div>
  <p>Chatbot sobre las guías para pacientes del Ministerio de Salud: infarto, presión alta y colesterol.</p>
</div>""")
    LLM = None
    if PROVEEDORES_OK:
        proveedor = st.selectbox("Modelo de lenguaje", PROVEEDORES_OK, format_func=lambda p: rc.PROVEEDORES[p]["nombre"],
                                 help="Proveedor del LLM que redacta la respuesta con los fragmentos recuperados.")
        try:
            LLM = obtener_llm(proveedor)
            st.caption(f"Modelo activo: `{LLM.modelo}`")
        except Exception as e:
            st.warning(f"{rc.PROVEEDORES[proveedor]['nombre']} no respondió ({type(e).__name__}). "
                       "El chat queda en modo solo búsqueda.")
    else:
        st.warning("No hay clave de LLM configurada: el chat queda en modo solo búsqueda.")

    temperatura = st.slider("Temperatura", 0.0, 1.0, rc.HIPERPARAMETROS["temperature"], 0.1,
                            help="Más baja = respuestas más estables y apegadas a las guías.")
    top_k = st.slider("Fragmentos recuperados (top-k)", 2, 8, rc.TOP_K,
                      help="Cuántos fragmentos de Qdrant se envían al modelo como contexto.")
    guias = st.multiselect("Buscar solo en", list(rc.NOMBRE_CORTO), format_func=rc.NOMBRE_CORTO.get,
                           placeholder="Todas las guías")

    usar_perfil = st.toggle("Responder pensando en un paciente del dataset UCI", value=False)
    PERFIL, TERMINOS = None, []
    if usar_perfil:
        pacientes = obtener_pacientes()
        pid = st.number_input("Paciente (1 a 303)", 1, len(pacientes), 127)
        PERFIL, TERMINOS = rc.perfil_paciente(pacientes.loc[pid])
        st.caption(PERFIL)

    ver_prompt = st.toggle("Mostrar el prompt enviado al modelo", value=False)
    if st.button("Nueva conversación", icon=":material/refresh:", width="stretch"):
        ss.historial = []
        ss.reinicios = ss.get("reinicios", 0) + 1
        st.rerun()

    integrantes = "".join(f"<li>{n}</li>" for n in ["Daimer Cardona Rojas", "Cristhian Eduardo Cortes Martha",
                                                     "Juan Manuel Guerrero Tupaz", "Luis Alberto Ospina Delgado",
                                                     "Jhonny Mauricio Rosero Carpio"])
    es.bloque(f"""
<div class="cr-side">
  <dl>
    <div><dt>Base vectorial</dt><dd>{"Qdrant Cloud" if "Cloud" in BASE.modo else "En memoria"}</dd></div>
    <div><dt>Fragmentos</dt><dd>{es.num(BASE.estado(COLECCION).get("puntos", 0), 0)}</dd></div>
    <div><dt>Embeddings</dt><dd>{"E5" if EMBEDDER.clave == "e5" else "OpenAI"} · {rc.DIMENSION}</dd></div>
    <div><dt>Umbral de relevancia</dt><dd>{es.num(rc.UMBRAL_RELEVANCIA[EMBEDDER.clave], 3)}</dd></div>
  </dl>
  <p class="cr-side-titulo">Grupo 13</p>
  <ul>{integrantes}</ul>
  <p style="margin-top:10px">Computación en la Nube, CUN<br>Docente: Hamilton Rivera Flor</p>
  <p class="cr-side-aviso">Herramienta académica. No reemplaza la consulta médica: ante dolor en el pecho que no cede, acuda a urgencias.</p>
</div>""")
    st.caption(f"Colección `{COLECCION}` · {MSG_EMB}")

bot = rc.ChatbotRAG(BASE, EMBEDDER, LLM, top_k=top_k) if LLM else None
AVISO.empty()
registro("interfaz lista")


# --------------------------------------------------------------------------- utilidades de interfaz
def tarjeta_fuente(n, f, citada):
    texto = f["texto"] if len(f["texto"]) <= 360 else f["texto"][:360].rsplit(" ", 1)[0] + " …"
    tag = '<span class="cg-tag">citada</span>' if citada else '<span class="cg-tag no">no citada</span>'
    return (f'<div class="cg-fuente" style="--c:{COLOR_DOC.get(f["doc_id"], "#64748b")}">'
            f'<div class="cab"><span>[{n}] {rc.etiqueta_fuente(f)}{tag}</span><span class="cos">coseno {es.num(f["score"], 3)}</span></div>'
            f'<div class="txt">{texto}</div></div>')


def mostrar_detalle(turno):
    t = turno["tiempos"]
    partes = [f"Búsqueda <b>{es.num(t.get('embedding', 0) + t.get('qdrant', 0), 0)} ms</b>",
              f"LLM <b>{es.num(t.get('llm', 0) / 1000, 1)} s</b>" if t.get("llm") else "Sin llamada al LLM",
              f"Total <b>{es.num(t.get('total', 0) / 1000, 1)} s</b>"]
    if turno.get("tokens", {}).get("entrada"):
        partes.append(f"Tokens <b>{turno['tokens']['entrada']} + {turno['tokens']['salida']}</b>")
    if turno.get("pregunta_busqueda") and turno["pregunta_busqueda"] != turno["pregunta"]:
        partes.append(f"Búsqueda reformulada: «{turno['pregunta_busqueda']}»")
    st.markdown(f'<div class="cg-kpi">{"".join(f"<span>{p}</span>" for p in partes)}</div>', unsafe_allow_html=True)
    with st.expander(f"Fuentes recuperadas ({len(turno['fragmentos'])}) · citadas: "
                     f"{', '.join(f'[{c}]' for c in turno['citas']) or 'ninguna'}"):
        if turno.get("motivo"):
            st.info(turno["motivo"], icon=":material/shield:")
        st.markdown("".join(tarjeta_fuente(i, f, i in turno["citas"]) for i, f in enumerate(turno["fragmentos"], 1)),
                    unsafe_allow_html=True)
    if ver_prompt and turno.get("mensajes"):
        with st.expander("Prompt enviado al modelo"):
            for m in turno["mensajes"]:
                st.markdown(f"**{m['role']}**")
                st.code(m["content"], language=None, wrap_lines=True)


def turno_a_dict(r):
    return {"pregunta": r.pregunta, "respuesta": r.texto, "pregunta_busqueda": r.pregunta_busqueda,
            "fragmentos": r.fragmentos, "citas": r.citas, "abstencion": r.abstencion, "motivo": r.motivo,
            "tiempos": r.tiempos, "tokens": r.tokens, "modelo": r.modelo, "mensajes": getattr(r, "mensajes", None)}


# --------------------------------------------------------------------------- pestañas
RUTA_RES = rc.RUTA_BASE / "datos" / "resultados_aca2.json"
try:
    RES = json.loads(RUTA_RES.read_text(encoding="utf-8"))
except (OSError, ValueError):
    RES = None
try:
    PUNTOS = BASE.estado(COLECCION).get("puntos", 0)
except Exception:
    PUNTOS = 0
fiel = RES["resumen_dominio"]["fiel (juez)"] if RES else None
MODO_OSCURO, ESCALA = es.controles()
primera_vez = not ss.get("_intro_vista", False)
ss["_intro_vista"] = True
es.encabezado(
    ["ACA 2", "Grupo 13", "Computación en la Nube", "Especialización en IA, CUN"],
    "Pregúntele a las guías del Ministerio de Salud sobre infarto, presión alta y colesterol. "
    "Respondemos solo con lo que dicen, citamos el fragmento y la página, y si la respuesta no está en ellas, lo decimos.",
    [es.canal("Guías indexadas", "3", "infarto, hipertensión y colesterol"),
     es.canal("Fragmentos en Qdrant", es.num(PUNTOS, 0), f"vectores de {rc.DIMENSION} dimensiones"),
     es.canal("Modelo de lenguaje", LLM.modelo if LLM else "solo búsqueda", "cámbielo en la barra lateral", "modelo"),
     es.canal("Umbral de relevancia", es.num(rc.UMBRAL_RELEVANCIA[EMBEDDER.clave], 3), "por debajo, se abstiene", "umbral"),
     es.canal("Respuestas fieles", es.pct(fiel, 0) if fiel is not None else "—",
              "según el LLM juez, en la validación", "decision")],
    rc.RUTA_BASE / "assets" / "cun.png", animar=primera_vez)

tab_chat, tab_docs, tab_indice, tab_eval, tab_acerca = st.tabs(
    [":material/chat: Chat", ":material/description: Documentos", ":material/hub: Índice vectorial",
     ":material/fact_check: Evaluación", ":material/info: Acerca de"])

with tab_chat:
    zona = st.container()          # la conversación va arriba y la caja de escritura queda debajo
    escrita = st.chat_input("Escriba su pregunta sobre presión arterial, colesterol o infarto…")
    pendiente = None
    with zona:
        if not ss.historial:
            eleccion = st.pills("Ejemplos", EJEMPLOS, label_visibility="collapsed", key=f"ej_{ss.get('reinicios', 0)}")
            if eleccion:
                pendiente = eleccion
        for turno in ss.historial:
            with st.chat_message("user"):
                st.markdown(turno["pregunta"])
            with st.chat_message("assistant", avatar=":material/monitor_heart:"):
                st.markdown(turno["respuesta"])
                mostrar_detalle(turno)
    pregunta = escrita or pendiente
    if pregunta:
      with zona:
          with st.chat_message("user"):
              st.markdown(pregunta)
          with st.chat_message("assistant", avatar=":material/monitor_heart:"):
              if bot is None:
                  frag, tiempos = rc.ChatbotRAG(BASE, EMBEDDER, type("L", (), {"modelo": "—"})(), top_k).recuperar(
                      pregunta, top_k, guias or None)
                  st.info("Modo solo búsqueda: estos son los fragmentos más cercanos.")
                  st.markdown("".join(tarjeta_fuente(i, f, False) for i, f in enumerate(frag, 1)), unsafe_allow_html=True)
              else:
                  historial = [{"pregunta": t["pregunta"], "respuesta": t["respuesta"]} for t in ss.historial]
                  with st.spinner("Buscando en las guías…"):
                      r = bot.preparar(pregunta, historial, PERFIL, TERMINOS, guias or None, top_k)
                  if r.abstencion:
                      st.markdown(r.texto)
                      r.tiempos["total"] = (time.perf_counter() - r._t0) * 1000
                  else:
                      salida_llm = {}
                      try:
                          st.write_stream(LLM.transmitir(r.mensajes, salida_llm, temperature=temperatura))
                      except Exception as e:
                          salida_llm = {"texto": f"El modelo no respondió ({type(e).__name__}). Intente de nuevo.", "ms": 0}
                          st.error(salida_llm["texto"])
                      r = bot.finalizar(r, salida_llm)
                  turno = turno_a_dict(r)
                  ss.historial.append(turno)
                  mostrar_detalle(turno)

# ----------------------------------------------------------------- documentos
with tab_docs:
    frag_df = listar_fragmentos(COLECCION, ss.version_indice)
    st.markdown("#### Documentos indexados en la base vectorial")
    if frag_df.empty:
        st.warning("La colección está vacía.")
    else:
        resumen = (frag_df.groupby(["doc_id", "documento"], dropna=False)
                   .agg(fragmentos=("texto", "size"), pagina_min=("pagina_inicio", "min"), pagina_max=("pagina_fin", "max"),
                        caracteres=("caracteres", "sum"), anio=("anio", "first"), institucion=("institucion", "first"),
                        url=("url", "first")).reset_index())
        resumen["páginas"] = resumen["pagina_min"].astype(int).astype(str) + "–" + resumen["pagina_max"].astype(int).astype(str)
        st.dataframe(resumen[["doc_id", "documento", "institucion", "anio", "páginas", "fragmentos", "caracteres", "url"]],
                     hide_index=True, width="stretch",
                     column_config={"url": st.column_config.LinkColumn("fuente oficial", display_text="abrir"),
                                    "anio": st.column_config.NumberColumn("año", format="%d"),
                                    "caracteres": st.column_config.NumberColumn(format="%d")})
        cols = st.columns(3)
        etiquetas = {"sindrome": "SCA", "hipertension": "HTA", "dislipidemia": "DIS"}
        for col, pdf in zip(cols, sorted(rc.RUTA_CORPUS.glob("*.pdf"))):
            clave = next((v for k, v in etiquetas.items() if k in pdf.name), pdf.stem)
            col.download_button(f"PDF · {rc.NOMBRE_CORTO.get(clave, clave)}", pdf.read_bytes(), pdf.name,
                                "application/pdf", icon=":material/download:", width="stretch")

        st.markdown("#### Explorar fragmentos y metadatos")
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        doc_sel = c1.selectbox("Documento", ["Todos"] + resumen["doc_id"].tolist())
        variables = sorted({v for vs in frag_df["variables"] for v in (vs or [])})
        var_sel = c2.selectbox("Variable clínica mencionada", ["Todas"] + variables)
        texto_sel = c3.text_input("Contiene el texto", placeholder="por ejemplo: sal, estatinas, urgencias")
        vista = frag_df.copy()
        if doc_sel != "Todos":
            vista = vista[vista["doc_id"] == doc_sel]
        if var_sel != "Todas":
            vista = vista[vista["variables"].apply(lambda vs: var_sel in (vs or []))]
        if texto_sel:
            vista = vista[vista["texto"].map(rc.normalizar).str.contains(rc.normalizar(texto_sel), regex=False)]
        vista = vista.sort_values(["doc_id", "orden"])
        st.caption(f"{len(vista)} de {len(frag_df)} fragmentos")
        evento = st.dataframe(vista[["chunk_id", "doc_id", "pagina_inicio", "seccion", "variables", "caracteres", "texto"]],
                              hide_index=True, width="stretch", height=300, on_select="rerun", selection_mode="single-row",
                              column_config={"texto": st.column_config.TextColumn(width="large"),
                                             "pagina_inicio": st.column_config.NumberColumn("página")})
        filas = evento.selection.rows if evento else []
        if filas:
            f = vista.iloc[filas[0]].to_dict()
            st.markdown(f"**{f['chunk_id']}** · {rc.etiqueta_fuente(f)}")
            st.write(f["texto"])
            with st.expander("Metadatos (payload en Qdrant)"):
                st.json({k: v for k, v in f.items() if k != "texto"})

    st.markdown("#### Gestión documental")
    clave_admin = rc.leer_secreto("ADMIN_PASSWORD")
    if not clave_admin:
        st.caption("La carga y el retiro de documentos están desactivados: falta ADMIN_PASSWORD en los secrets.")
    elif st.text_input("Clave de administrador", type="password") != clave_admin:
        st.caption("Ingrese la clave de administrador para agregar o retirar documentos.")
    else:
        g1, g2 = st.columns(2)
        with g1:
            archivo = st.file_uploader("Agregar un documento (PDF o TXT)", type=["pdf", "txt"])
            titulo = st.text_input("Título del documento")
            if archivo and st.button("Fragmentar, vectorizar e indexar", type="primary", icon=":material/upload:"):
                with st.spinner("Indexando…"):
                    doc_id, n = rc.indexar_documento(BASE, EMBEDDER, archivo.name, archivo.getvalue(), titulo or None)
                ss.version_indice += 1
                st.success(f"{n} fragmentos indexados como {doc_id}.")
        with g2:
            propios = [d for d in (frag_df["doc_id"].unique() if not frag_df.empty else []) if str(d).startswith("DOC-")]
            if propios:
                retirar = st.selectbox("Retirar un documento agregado", propios)
                if st.button("Retirar de la base vectorial", icon=":material/delete:"):
                    n = BASE.eliminar_documento(COLECCION, retirar)
                    ss.version_indice += 1
                    st.success(f"{n} fragmentos retirados.")
            else:
                st.caption("Las tres guías base están protegidas; aquí aparecen los documentos que se agreguen.")

# ----------------------------------------------------------------- índice
with tab_indice:
    st.markdown("#### Estado del índice en Qdrant Cloud")
    estado = BASE.estado(COLECCION)
    es.fila_canales([
        es.canal("Puntos", es.num(estado.get("puntos", 0), 0), "un vector por fragmento"),
        es.canal("Estado", str(estado.get("estado", "—")), "salud de la colección", "decision"),
        es.canal("Dimensión", str(estado.get("dimensión", "—")), "tamaño de cada vector"),
        es.canal("Distancia", str(estado.get("distancia", "—")), "medida de similitud", "modelo"),
        es.canal("Conexión", f"{es.num(BASE.ms_conexion, 0)} ms", "al abrir el servidor")])
    st.caption(f"Índices de payload: {', '.join(estado.get('índices de payload', [])) or '—'} · segmentos: "
               f"{estado.get('segmentos', '—')} · al iniciar: {ACCIONES.get(EMBEDDER.clave, '—')}")
    frag_df = listar_fragmentos(COLECCION, ss.version_indice)
    if not frag_df.empty:
        conteo = frag_df.groupby("doc_id").size().reset_index(name="fragmentos")
        conteo["guía"] = conteo["doc_id"].map(lambda d: rc.NOMBRE_CORTO.get(d, d))
        import plotly.express as px
        fig = px.bar(conteo, x="fragmentos", y="guía", orientation="h", color="doc_id", text="fragmentos",
                     color_discrete_map=COLOR_DOC)
        es.estilo(fig, alto=230, leyenda=False, titulo="Fragmentos por guía")
        fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=es.TINTA_2, size=14))
        fig.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(title=None, showgrid=True, gridcolor=es.REJILLA)
        es.mostrar(fig, ESCALA)

    st.markdown("#### Búsqueda semántica directa (sin LLM)")
    c1, c2 = st.columns([3, 1])
    consulta = c1.text_input("Consulta", "¿Cuánta sal se puede comer al día?", label_visibility="collapsed")
    if c2.button("Buscar en Qdrant", type="primary", width="stretch") and consulta:
        buscador = rc.ChatbotRAG(BASE, EMBEDDER, type("L", (), {"modelo": "—"})(), top_k)
        res, tiempos = buscador.recuperar(consulta, top_k, guias or None)
        umbral = rc.UMBRAL_RELEVANCIA[EMBEDDER.clave]
        st.caption(f"Embedding de la consulta {tiempos['embedding']:.0f} ms · búsqueda en Qdrant {tiempos['qdrant']:.0f} ms · "
                   f"{'✅ sobre' if res and res[0]['score'] >= umbral else '⛔ bajo'} el umbral {umbral:.3f}")
        tabla = pd.DataFrame([{"rank": r["rank"], "coseno": round(r["score"], 4), "guía": r["doc_id"],
                               "página": r["pagina_inicio"], "sección": r.get("seccion", ""), "texto": r["texto"][:220]}
                              for r in res])
        st.dataframe(tabla, hide_index=True, width="stretch",
                     column_config={"coseno": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f")})

    if st.button("Medir latencia (5 búsquedas)", icon=":material/speed:"):
        buscador = rc.ChatbotRAG(BASE, EMBEDDER, type("L", (), {"modelo": "—"})(), top_k)
        medidas = [buscador.recuperar(p, top_k)[1] for p in EJEMPLOS + ["¿Qué es el colesterol HDL?"]][:5]
        st.dataframe(pd.DataFrame(medidas).describe().loc[["mean", "50%", "max"]].round(1).rename(
            index={"mean": "media (ms)", "50%": "mediana (ms)", "max": "máximo (ms)"}), width="content")

# ----------------------------------------------------------------- evaluación
with tab_eval:
    st.markdown("#### Validación del chatbot (notebook del ACA 2)")
    if RES is None:
        st.info("Aún no hay resultados: ejecute el notebook del ACA 2 y suba `datos/resultados_aca2.json` al repositorio.")
    else:
        res = RES
        tipos = ["decision", "decision", "modelo", "datos"]
        es.fila_canales([es.canal(etq, str(val).replace(".", ","), "", tipo)
                         for (etq, val), tipo in zip(list(res.get("kpis", {}).items())[:4], tipos)])
        for nombre, filas in res.get("tablas", {}).items():
            st.markdown(f"**{nombre}**")
            st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
        st.caption(f"Ejecución: {res.get('fecha', '—')} · modelo {res.get('modelo_llm', '—')}")

# ----------------------------------------------------------------- acerca
with tab_acerca:
    st.image(str(rc.RUTA_BASE / "assets" / "arquitectura_rag.png"), caption="Arquitectura RAG de punta a punta")
    st.markdown("#### Cómo funciona")
    es.bloque("""<ol class="cr-pasos">
<li><strong>Reformula</strong><span>Si hay conversación previa, el LLM reescribe la pregunta para que se entienda sola.</span></li>
<li><strong>Vectoriza</strong><span>Multilingual-E5 convierte la pregunta en un vector de 768 dimensiones.</span></li>
<li><strong>Busca</strong><span>Qdrant Cloud devuelve los fragmentos más parecidos por coseno.</span></li>
<li><strong>Filtra</strong><span>Si el mejor no supera el umbral calibrado en el ACA 1, se abstiene sin llamar al LLM.</span></li>
<li><strong>Responde y cita</strong><span>El LLM responde solo con esos fragmentos y cita cada afirmación con [n].</span></li>
</ol>""")
    st.markdown("""
**CUN · Especialización en Inteligencia Artificial · Computación en la Nube para Sistemas Inteligentes ·
Docente: Hamilton Rivera Flor · Grupo 13**

Daimer Cardona Rojas · Cristhian Eduardo Cortes Martha · Juan Manuel Guerrero Tupaz · Luis Alberto Ospina Delgado ·
Jhonny Mauricio Rosero Carpio

**Credenciales.** Se leen de los secrets de Streamlit o de variables de entorno; ninguna está en el código.
""")

es.scripts(MODO_OSCURO)

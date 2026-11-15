# -*- coding: utf-8 -*-
"""CardioGuía RAG · chatbot sobre guías clínicas cardiovasculares (ACA 2 · Computación en la Nube · Grupo 13)."""
import json
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import rag_core as rc

st.set_page_config(page_title="CardioGuía RAG", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

COLOR_DOC = {"SCA": "#c2410c", "HTA": "#1d4ed8", "DIS": "#0f766e"}
EJEMPLOS = ["¿Desde qué cifra se considera que uno tiene la tensión alta?",
            "Siento un dolor fuerte en el pecho que no se me quita, ¿qué debo hacer?",
            "¿Qué alimentos debo evitar si tengo el colesterol alto?",
            "¿Qué cuidados debo tener cuando me den de alta después de un infarto?"]

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.cg-fuente {border: 1px solid #dbe3ee; border-left: 5px solid var(--c); border-radius: 8px; padding: 8px 12px;
            margin: 6px 0; background: #fff; font-size: 0.88rem; line-height: 1.4;}
.cg-fuente .cab {display: flex; justify-content: space-between; gap: 10px; font-weight: 600; color: #0f172a;}
.cg-fuente .txt {color: #475569; margin-top: 4px;}
.cg-tag {display: inline-block; font-size: 0.72rem; font-weight: 700; padding: 1px 7px; border-radius: 99px;
         background: #e3f5f2; color: #0e7569; margin-left: 6px;}
.cg-tag.no {background: #f1f5f9; color: #64748b;}
.cg-kpi {font-size: 0.8rem; color: #64748b;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- recursos (una vez por servidor)
@st.cache_resource(show_spinner="Conectando con Qdrant Cloud…")
def obtener_base():
    base = rc.BaseVectorial(rc.leer_secreto("QDRANT_URL"), rc.leer_secreto("QDRANT_API_KEY"))
    snapshot = rc.cargar_snapshot()
    acciones = {"e5": base.asegurar_coleccion("e5", snapshot)}
    if rc.leer_secreto("OPENAI_API_KEY"):
        acciones["openai"] = base.asegurar_coleccion("openai", snapshot)
    return base, acciones


@st.cache_resource(show_spinner="Cargando el modelo de embeddings Multilingual-E5 (la primera vez tarda un minuto)…")
def obtener_embedder(preferencia):
    return rc.crear_embedder(preferencia, rc.leer_secreto("OPENAI_API_KEY"))


@st.cache_resource(show_spinner="Verificando el modelo de lenguaje…")
def obtener_llm(proveedor):
    return rc.ClienteLLM(proveedor, rc.leer_secreto(rc.PROVEEDORES[proveedor]["clave"]))


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
    st.image(str(rc.RUTA_BASE / "assets" / "cun.png"), width=120)
    st.markdown("### CardioGuía RAG")
    st.caption("Chatbot sobre las guías para pacientes del Ministerio de Salud: síndrome coronario agudo, "
               "hipertensión arterial y dislipidemias.")

    st.markdown("**Modelo de lenguaje**")
    LLM = None
    if PROVEEDORES_OK:
        proveedor = st.selectbox("Proveedor", PROVEEDORES_OK, format_func=lambda p: rc.PROVEEDORES[p]["nombre"],
                                 label_visibility="collapsed")
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

    st.markdown("**Perfil de paciente (opcional)**")
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

    st.divider()
    st.caption(f"🗄️ {BASE.modo} · `{COLECCION}`  \n🔤 {MSG_EMB}  \n"
               f"🎯 umbral de relevancia: coseno ≥ {rc.UMBRAL_RELEVANCIA[EMBEDDER.clave]:.3f}")

bot = rc.ChatbotRAG(BASE, EMBEDDER, LLM, top_k=top_k) if LLM else None


# --------------------------------------------------------------------------- utilidades de interfaz
def tarjeta_fuente(n, f, citada):
    texto = f["texto"] if len(f["texto"]) <= 360 else f["texto"][:360].rsplit(" ", 1)[0] + " …"
    tag = '<span class="cg-tag">citada</span>' if citada else '<span class="cg-tag no">no citada</span>'
    return (f'<div class="cg-fuente" style="--c:{COLOR_DOC.get(f["doc_id"], "#64748b")}">'
            f'<div class="cab"><span>[{n}] {rc.etiqueta_fuente(f)}{tag}</span><span>coseno {f["score"]:.3f}</span></div>'
            f'<div class="txt">{texto}</div></div>')


def mostrar_detalle(turno):
    t = turno["tiempos"]
    partes = [f"🔎 búsqueda {t.get('embedding', 0) + t.get('qdrant', 0):.0f} ms",
              f"🤖 LLM {t.get('llm', 0) / 1000:.1f} s" if t.get("llm") else "🤖 sin llamada al LLM",
              f"⏱️ total {t.get('total', 0) / 1000:.1f} s"]
    if turno.get("tokens", {}).get("entrada"):
        partes.append(f"🧾 {turno['tokens']['entrada']} + {turno['tokens']['salida']} tokens")
    if turno.get("pregunta_busqueda") and turno["pregunta_busqueda"] != turno["pregunta"]:
        partes.append(f"🔁 búsqueda reformulada: «{turno['pregunta_busqueda']}»")
    st.markdown(f'<div class="cg-kpi">{" · ".join(partes)}</div>', unsafe_allow_html=True)
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
tab_chat, tab_docs, tab_indice, tab_eval, tab_acerca = st.tabs(
    ["💬 Chat", "📚 Documentos", "🧭 Índice vectorial", "✅ Evaluación", "ℹ️ Acerca de"])

with tab_chat:
    st.markdown("#### Pregúntele a las guías")
    st.caption("Las respuestas salen solo de las guías indexadas y citan el fragmento que las respalda. "
               "Es una herramienta académica de orientación: no reemplaza la consulta médica.")
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
            with st.chat_message("assistant", avatar="🫀"):
                st.markdown(turno["respuesta"])
                mostrar_detalle(turno)
    pregunta = escrita or pendiente
    if pregunta:
      with zona:
          with st.chat_message("user"):
              st.markdown(pregunta)
          with st.chat_message("assistant", avatar="🫀"):
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
                      registro = {}
                      try:
                          st.write_stream(LLM.transmitir(r.mensajes, registro, temperature=temperatura))
                      except Exception as e:
                          registro = {"texto": f"El modelo no respondió ({type(e).__name__}). Intente de nuevo.", "ms": 0}
                          st.error(registro["texto"])
                      r = bot.finalizar(r, registro)
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
    m = st.columns(5)
    m[0].metric("Puntos", estado.get("puntos", 0))
    m[1].metric("Estado", estado.get("estado", "—"))
    m[2].metric("Dimensión", estado.get("dimensión", "—"))
    m[3].metric("Distancia", estado.get("distancia", "—"))
    m[4].metric("Conexión", f"{BASE.ms_conexion:.0f} ms")
    st.caption(f"Índices de payload: {', '.join(estado.get('índices de payload', [])) or '—'} · segmentos: "
               f"{estado.get('segmentos', '—')} · al iniciar: {ACCIONES.get(EMBEDDER.clave, '—')}")
    frag_df = listar_fragmentos(COLECCION, ss.version_indice)
    if not frag_df.empty:
        conteo = frag_df.groupby("doc_id").size().reset_index(name="fragmentos")
        conteo["guía"] = conteo["doc_id"].map(lambda d: rc.NOMBRE_CORTO.get(d, d))
        fig = px.bar(conteo, x="fragmentos", y="guía", orientation="h", color="doc_id", text="fragmentos",
                     color_discrete_map=COLOR_DOC, height=190)
        fig.update_layout(showlegend=False, margin=dict(l=0, r=10, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

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
    ruta = rc.RUTA_BASE / "datos" / "resultados_aca2.json"
    st.markdown("#### Validación del chatbot (notebook del ACA 2)")
    if not ruta.exists():
        st.info("Aún no hay resultados: ejecute el notebook del ACA 2 y suba `datos/resultados_aca2.json` al repositorio.")
    else:
        res = json.loads(ruta.read_text(encoding="utf-8"))
        k = res.get("kpis", {})
        cols = st.columns(4)
        for col, (etq, val) in zip(cols, list(k.items())[:4]):
            col.metric(etq, val)
        for nombre, filas in res.get("tablas", {}).items():
            st.markdown(f"**{nombre}**")
            st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
        st.caption(f"Ejecución: {res.get('fecha', '—')} · modelo {res.get('modelo_llm', '—')}")

# ----------------------------------------------------------------- acerca
with tab_acerca:
    st.image(str(rc.RUTA_BASE / "assets" / "arquitectura_rag.png"), caption="Arquitectura RAG de punta a punta")
    st.markdown("""
#### CardioGuía RAG · ACA 2 · Computación en la Nube para Sistemas Inteligentes
**CUN · Especialización en Inteligencia Artificial · Docente: Hamilton Rivera Flor · Grupo 13**
Daimer Cardona Rojas · Cristhian Eduardo Cortes Martha · Juan Manuel Guerrero Tupaz · Luis Alberto Ospina Delgado ·
Jhonny Mauricio Rosero Carpio

**Cómo funciona.** 1) Si hay conversación previa, el LLM reescribe la pregunta para que sea autónoma.
2) Multilingual-E5 la convierte en un vector de 768 dimensiones. 3) Qdrant Cloud devuelve los fragmentos más
parecidos por coseno. 4) Si el mejor no supera el umbral calibrado en el ACA 1, el sistema se abstiene sin llamar al LLM.
5) Si lo supera, el LLM responde solo con esos fragmentos y cita cada afirmación.

**Credenciales.** Se leen de los secrets de Streamlit o de variables de entorno; ninguna está en el código.

Proyecto académico. No reemplaza la consulta médica. Ante dolor en el pecho que no cede, acuda a urgencias.
""")

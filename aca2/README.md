# CardioGuía RAG · ACA 2 · Chatbot sobre guías clínicas cardiovasculares

Chatbot RAG que responde preguntas de pacientes con base **solo** en tres guías del Ministerio de Salud de Colombia (síndrome coronario agudo, hipertensión arterial y dislipidemias), citando la fuente de cada afirmación. Usa la base vectorial del ACA 1 en Qdrant Cloud, un LLM en la nube vía API y una interfaz en Streamlit.

**CUN · Especialización en Inteligencia Artificial · Computación en la Nube para Sistemas Inteligentes · Docente: Hamilton Rivera Flor · Grupo 13**
Daimer Cardona Rojas · Cristhian Eduardo Cortes Martha · Juan Manuel Guerrero Tupaz · Luis Alberto Ospina Delgado · Jhonny Mauricio Rosero Carpio

- **Aplicación:** https://cardioguia-rag-g13.streamlit.app
- **Notebook:** `ACA2_Nube_Grupo13_Chatbot_RAG.ipynb`

## Arquitectura

```
Pregunta ─▶ (si hay historial) LLM reescribe la pregunta ─▶ Multilingual-E5 (768 dims)
        ─▶ Qdrant Cloud: top-4 por coseno (colección guias_cardio_g13 del ACA 1)
        ─▶ ¿mejor coseno ≥ 0,822? ── no ─▶ "No encontré esa información en las guías" (sin llamar al LLM)
                                  └─ sí ─▶ prompt: reglas + fragmentos numerados + historial ─▶ gpt-4o-mini
        ─▶ respuesta con citas [n] ─▶ la interfaz muestra fuentes, cosenos, tiempos y tokens
```

| Componente | Herramienta |
|---|---|
| Base vectorial | Qdrant Cloud (capa gratuita), colección del ACA 1, distancia coseno |
| Embeddings | `intfloat/multilingual-e5-base` (Hugging Face, CPU); respaldo: OpenAI `text-embedding-3-large` |
| LLM | OpenAI `gpt-4o-mini` (temperatura 0,2 · máx. 700 tokens); alternativas: Groq Llama 3.3 y Google Gemini |
| Orquestación | Código Python nativo (`rag_core.py`), sin framework |
| Interfaz | Streamlit Community Cloud |

## Contenido

| Archivo | Qué es |
|---|---|
| `app.py` | Interfaz: chat con historial y fuentes, documentos, índice vectorial y evaluación |
| `rag_core.py` | Núcleo RAG que usan la app y el notebook |
| `estilo.py` y `.streamlit/config.toml` | Sistema de diseño compartido con CardioRiesgo (ACA 1 de Aprendizaje Automático): papel de ECG, canales de monitor, Saira y Atkinson Hyperlegible Next |
| `ACA2_Nube_Grupo13_Chatbot_RAG.ipynb` | Notebook reproducible: conexión, LLM, pipeline, diálogos con citas y validación |
| `datos/indice_aca1.json.gz` | Copia del índice del ACA 1 (131 fragmentos, vectores E5 y OpenAI, metadatos) para restaurar la colección |
| `datos/processed.cleveland.data` | Dataset UCI Heart Disease (perfiles de paciente) |
| `datos/resultados_aca2.json` | Resultados de la validación (los genera el notebook) |
| `corpus/` | Las 3 guías en PDF |
| `requirements.txt`, `secrets.toml.example`, `.env.example` | Dependencias y plantillas de credenciales |

## Ejecutar

**Notebook (Google Colab).** Cree en el panel 🔑 *Secrets* `QDRANT_URL`, `QDRANT_API_KEY` y `OPENAI_API_KEY`, abra el notebook y ejecute todo. La primera celda descarga este repositorio.

**App en un computador.**
```
cd aca2
pip install -r requirements.txt
cp secrets.toml.example .streamlit/secrets.toml   # y complete las claves
streamlit run app.py
```

**App en Streamlit Community Cloud.** *Create app* → repositorio `lospinadel/RAG_Cardio`, rama `main`, archivo `aca2/app.py`; en *Advanced settings* elija Python 3.12 y pegue los secrets de `secrets.toml.example` con sus valores.

## Seguridad

Las claves se leen de variables de entorno, de los secrets de Streamlit o de Colab Secrets; ninguna está en el código y `.env` y `secrets.toml` están en `.gitignore`. La carga y el retiro de documentos exigen `ADMIN_PASSWORD`.

Proyecto académico. No reemplaza la consulta médica.

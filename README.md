# CardioGuía RAG · ACA 1 · Computación en la Nube para Sistemas Inteligentes

Ingesta, fragmentación (chunking), embeddings y almacenamiento vectorial en la nube de tres guías clínicas cardiovasculares del Ministerio de Salud de Colombia, integradas con el dataset UCI Heart Disease.

**CUN · Especialización en Inteligencia Artificial · Docente: Hamilton Rivera Flor · Grupo 13**
Daimer Cardona Rojas · Cristhian Eduardo Cortes Martha · Juan Manuel Guerrero Tupaz · Luis Alberto Ospina Delgado · Jhonny Mauricio Rosero Carpio

## Contenido

| Ruta | Qué es |
|---|---|
| `ACA1_Nube_Grupo13_RAG_Cardio.ipynb` | Notebook completo: SQL (DuckDB), ingesta y limpieza de PDF, chunking, embeddings con Multilingual-E5 (comparado con OpenAI), carga por lotes a Qdrant Cloud y validación de la búsqueda semántica |
| `corpus/` | Las 3 guías en PDF (síndrome coronario agudo, hipertensión arterial y dislipidemias) |
| `datos/processed.cleveland.data` | Dataset UCI Heart Disease (Cleveland), 303 pacientes |
| `.env.example` | Plantilla de credenciales para ejecutar fuera de Colab |
| `requirements.txt` | Librerías necesarias |
| `cache/` | Embeddings ya calculados con E5 y con OpenAI (permiten re-ejecutar sin recalcularlos) |
| `salidas/` | Resultados de la ejecución: métricas, gráficos y metadatos de los fragmentos |
| `poster/` | Póster tipo infografía en PDF (A1) |

## Stack (todo en capa gratuita)

Google Colab · DuckDB · PyMuPDF · LangChain Text Splitters · Sentence-Transformers con `intfloat/multilingual-e5-base` (modelo abierto de Hugging Face, 768 dimensiones, sin clave) · Qdrant Cloud (Free tier, distancia coseno).

**Comparación con un modelo comercial:** la sección 8 repite la evaluación con OpenAI `text-embedding-3-large` (768 dimensiones). Sus vectores están en `cache/embeddings_text-embedding-3-large_768.npz`, calculados una sola vez por menos de US$0,01, así que la comparación se reproduce sin clave de OpenAI.

## Resultados (ejecución del 1 de noviembre de 2026 en Qdrant Cloud)

| Indicador | Valor |
|---|---|
| Fragmentos indexados | 131 (configuración pequeña: 500 caracteres, solape 75) |
| Hit@3 / Hit@1 / MRR | 15 de 15 / 14 de 15 / 0,956 |
| Umbral de relevancia (coseno E5) | 0,822: separa el 100 % de las preguntas del dominio y de fuera |
| Comparación con OpenAI text-embedding-3-large | MRR 0,967 frente a 0,956 de E5, mismo Hit@3 |
| Uso de la capa gratuita de Qdrant | 0,074 % de 1 GB · costo US$0 |

## Cómo ejecutarlo

### En Google Colab (recomendado)

1. Abra https://colab.research.google.com y suba el notebook (*Archivo → Subir notebook*).
2. En el panel de archivos (ícono de carpeta) suba `ACA1_Nube_G13.zip`. El notebook lo descomprime solo.
3. En el panel 🔑 *Secrets* cree estos secretos con los nombres exactos y active *Acceso del notebook* en cada uno:
   - `QDRANT_URL`: URL del clúster en https://cloud.qdrant.io (por ejemplo `https://xxxx.us-east4-0.gcp.cloud.qdrant.io`)
   - `QDRANT_API_KEY`: API key del clúster
   - `OPENAI_API_KEY` (opcional): solo hace falta si se borra la caché de la comparación
4. *Entorno de ejecución → Ejecutar todas*. Tarda unos 5 minutos (la primera vez descarga el modelo E5, cerca de 1 GB).

### En un computador

```
pip install -r requirements.txt
cp .env.example .env      # y escriba sus claves en .env
jupyter notebook ACA1_Nube_Grupo13_RAG_Cardio.ipynb
```

Sin `QDRANT_URL`, el notebook usa una base Qdrant en memoria (solo para pruebas).

## Seguridad

Las claves se leen de variables de entorno (Colab Secrets o `.env`). Ninguna clave está escrita en el código y `.env` está en `.gitignore`.

Proyecto académico. No reemplaza la consulta médica.

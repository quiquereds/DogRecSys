<h1 align="center">
  <br>
  🔄 Módulo de Pipeline
  <br>
</h1>

<h4 align="center">Preprocesamiento, vectorización y componentes del flujo de datos</h4>

---

## 📑 Contenido

- [Descripción](#🔍-descripción)
- [Componentes del pipeline](#📋-componentes-del-pipeline)
- [Fundamentos del procesamiento](#🧪-fundamentos-del-procesamiento)
  - [Preprocesamiento de Datos](#preprocesamiento-de-datos)
  - [Vectorización de Características](#vectorización-de-características)
  - [Simulación de Interacciones](#simulación-de-interacciones)
  - [Reranking MMR](#reranking-mmr)

---

## 🔍 Descripción

Este módulo contiene los componentes que transforman los datos crudos en formatos adecuados para los modelos de recomendación. Incluye herramientas para preprocesamiento, vectorización, simulación de interacciones y reranking de resultados.

---

## 📋 Componentes del pipeline

| Archivo              | Descripción |
|----------------------|-------------|
| **preprocessing.py** | Funciones para limpieza, normalización y transformación de datos crudos a formatos procesables. |
| **vectorizers.py**   | Implementaciones para convertir texto y variables numéricas en vectores (TF-IDF, Word2Vec, etc.). |
| **simulation.py**    | Herramientas para generar datos sintéticos de interacción usuario-raza para testeo y desarrollo. |
| **mmr_reranking.py** | Algoritmo Maximal Marginal Relevance para balancear relevancia y diversidad en recomendaciones. |
| **utils.py**         | Funciones auxiliares y helpers para las diversas etapas del pipeline. |

## 🧪 Fundamentos del procesamiento

### Preprocesamiento de datos

El preprocesamiento convierte los datos crudos en formatos limpios y estructurados para el modelado:

- **Limpieza:** Eliminación de valores nulos, duplicados y corrección de inconsistencias.
- **Normalización:** Escalado de variables numéricas mediante técnicas como:
  - **MinMaxScaler:** Comprime valores al rango [0,1].
  - **StandardScaler:** Transforma a distribución con media 0 y desviación estándar 1.
- **Codificación:** Conversión de variables categóricas mediante:
  - **One-Hot Encoding:** Crea columnas binarias para cada categoría.
  - **Label Encoding:** Asigna un valor numérico a cada categoría.

### Vectorización de características

La vectorización transforma texto y atributos en representaciones numéricas:

- **Vectorización de texto:**
  - **TF-IDF:** Pondera términos según su frecuencia en el documento e inversa en el corpus.
  - **Count Vectorization:** Representa documentos por conteo de palabras.
  - **Embeddings:** Mapea palabras a vectores densos capturando semántica.

- **Escalado numérico:**
  - Asegura que variables con diferentes escalas contribuyan equitativamente.
  - Evita que atributos con valores grandes dominen el cálculo de similitud.

### Simulación de interacciones

Cuando los datos reales son escasos, la simulación permite crear matrices de interacción:

- **Generación de perfiles sintéticos:** Crea usuarios ficticios con preferencias definidas.
- **Ratings simulados:** Asigna valoraciones según reglas heurísticas o distribuciones.
- **Escenarios de prueba:** Simula situaciones como cold-start o matrices dispersas.

### Reranking MMR

El algoritmo Maximal Marginal Relevance (MMR) diversifica resultados:

- **Fórmula:**
  ```
  MMR = λ * Rel(i) - (1-λ) * max(Sim(i,j))
  ```
  Donde λ balancea relevancia y diversidad, Rel(i) es la relevancia del ítem i, y Sim(i,j) es la similitud entre ítem i y los ya seleccionados j.

- **Proceso iterativo:**
  1. Selecciona el ítem más relevante primero.
  2. Los siguientes ítems maximizan relevancia y minimizan similitud con los ya elegidos.
  3. Continúa hasta completar el número de recomendaciones deseado.

---

Para ver la implementación de los modelos, consultar la carpeta [`models/`](../models/README.md).

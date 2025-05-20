<h1 align="center">
  <br>
  📊 Módulo de Evaluación
  <br>
</h1>

<h4 align="center">Métricas, análisis y visualización de resultados del sistema de recomendación</h4>

---

## 📑 Contenido

- [Descripción](#🔍-descripción)
- [Archivos principales](#📋-archivos-principales)
- [Fundamentos de evaluación](#🧪-fundamentos-de-evaluación)
  - [Métricas de relevancia](#métricas-de-relevancia)
  - [Métricas de ranking](#métricas-de-ranking)
  - [Métricas de diversidad](#métricas-de-diversidad)
  - [Metodologías de evaluación](#metodologías-de-evaluación)


---

## 🔍 Descripción

Este módulo contiene las herramientas necesarias para evaluar cuantitativa y cualitativamente los modelos de recomendación implementados en DogRecSys. Las métricas y visualizaciones permiten comparar objetivamente diferentes algoritmos, ajustar hiperparámetros y seleccionar el enfoque más adecuado para cada escenario de uso.

---

## 📋 Archivos principales

| Archivo      | Descripción |
|--------------|-------------|
| **metrics.py**   | Implementación de métricas estándar (Precision@K, Recall@K, NDCG@K) y avanzadas (diversidad, cobertura, serendipity). |
| **plots.py**     | Funciones para visualización de resultados, curvas de precisión, recall, NDCG, matrices de confusión, etc. |

## 🧪 Fundamentos de Evaluación

### Métricas de relevancia

- **Precisión@K:** Proporción de recomendaciones relevantes entre las K sugerencias.
  - Fórmula: `|Relevantes ∩ Recomendados| / |Recomendados|`
  - Uso: Medir exactitud cuando queremos minimizar falsos positivos.

- **Recall@K:** Proporción de ítems relevantes recuperados entre todos los relevantes.
  - Fórmula: `|Relevantes ∩ Recomendados| / |Relevantes|`
  - Uso: Evaluar cobertura cuando no queremos perder ítems relevantes.

### Métricas de ranking

- **NDCG@K (Normalized Discounted Cumulative Gain):**
  - Considera la posición de los ítems relevantes en la lista.
  - Descuenta la relevancia según la posición (ítems al inicio reciben más peso).
  - Normalizado para permitir comparaciones entre usuarios con distintos números de ítems relevantes.

- **MAP (Mean Average Precision):**
  - Promedia la precisión en cada posición donde hay un ítem relevante.
  - Captura tanto la precisión como el orden de los resultados.

### Métricas de diversidad

- **Diversidad:** Mide cuán distintas son las recomendaciones entre sí.
  - Se calcula usando distancias (coseno, Jaccard) entre ítems recomendados.
  - Importante para evitar redundancia y monotonía.

- **Cobertura del catálogo:** Porcentaje de ítems que reciben recomendaciones.
  - Evalúa si el sistema privilegia solo ítems populares o abarca todo el catálogo.

### Metodologías de evaluación

- **Evaluación Offline:**
  - **Hold-out:** Divide datos en entrenamiento (80%) y test (20%).
  - **k-fold Cross-validation:** Divide datos en k particiones para validación cruzada.
  - **Leave-one-out:** Oculta un ítem por usuario para evaluar si el sistema lo recomienda.

- **Evaluación Online:**
  - **A/B Testing:** Compara versiones del sistema con grupos de usuarios reales.
  - **Interleaving:** Mezcla resultados de dos algoritmos y analiza con cuál interactúan más los usuarios.

---

Para detalles sobre los modelos evaluados, consultar la carpeta [`models/`](../models/README.md).

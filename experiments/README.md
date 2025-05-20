<h1 align="center">
  <br>
  🧪 Módulo de Experimentación
  <br>
</h1>

<h4 align="center">Notebooks y análisis para la exploración y validación de modelos de recomendación</h4>

---

## 📑 Contenido

- [Descripción](#-🔍-descripción)
- [Notebooks principales](#📓-notebooks-principales)
- [Metodología experimental](#🔬-metodología-experimental)
  - [Diseño de experimentos](#diseño-de-experimentos)
  - [Validación de modelos](#validación-de-modelos)
  - [Análisis comparativo](#análisis-comparativo)
- [Flujo de Experimentación](#🔄-flujo-de-experimentación)

---

## 🔍 Descripción

Este módulo contiene notebooks interactivos que permiten experimentar con los diferentes modelos y técnicas de recomendación implementados en DogRecSys. Los experimentos facilitan el análisis visual, la comparación de algoritmos, el ajuste de hiperparámetros y la validación de enfoques antes de su implementación final.

---

## 📓 Notebooks principales

| Notebook                         | Descripción |
|----------------------------------|-------------|
| **01-content_based.ipynb**       | Experimenta con filtrado basado en contenido, vectorización TF-IDF, similitud coseno y análisis de resultados. |
| **02-collaborative_filtering.ipynb** | Explora técnicas de filtrado colaborativo (user-based, item-based), construcción de matrices de similitud y predicción. |
| **03-cf-visualization.ipynb**    | Visualiza matrices de interacción, clusters de usuarios similares y patrones de preferencia. |
| **04-collaborative_nmf.ipynb**   | Implementa y analiza factorización matricial no negativa (NMF), explorando factores latentes. |



---

## 🔬 Metodología experimental

### Diseño de experimentos

Los experimentos siguen una metodología estructurada para asegurar resultados comparables y reproducibles:

- **Definición de objetivos:** Cada notebook responde a preguntas específicas (¿Cómo afecta el número de factores latentes en NMF? ¿Qué algoritmo ofrece mayor diversidad?).
- **Control de variables:** Mantenemos constantes ciertos parámetros mientras variamos otros para aislar efectos.
- **Replicabilidad:** Usamos semillas aleatorias fijas para asegurar que los resultados sean reproducibles.
- **Documentación:** Cada paso experimental está documentado con explicaciones y conclusiones.

### Validación de modelos

Empleamos diversas técnicas para validar el rendimiento de los modelos:

- **Validación cruzada:** Dividimos los datos en k folds para evaluar la robustez.
- **Hold-out:** Reservamos un 20% de los datos para testing, evaluando modelos entrenados en el 80% restante.
- **Validación temporal:** En algunos casos, simulamos el paso del tiempo dividiendo datos cronológicamente.

### Análisis comparativo

Los experimentos permiten comparar múltiples enfoques bajo criterios objetivos:

- **Comparativas multimétricas:** Analizamos precisión, recall, NDCG, diversidad y tiempo de ejecución.
- **Visualización:** Exploramos las compensaciones entre métricas (p.ej., precisión vs. diversidad).
- **Segmentación de resultados:** Evaluamos el desempeño por tipos de perfil de usuario.

---

## 🔄 Flujo de experimentación

El ciclo típico de experimentación en DogRecSys sigue estos pasos:

1. **Preparación de datos:** Cargamos y preprocesamos los datasets necesarios.
2. **Exploración inicial:** Visualizamos distribuciones y patrones relevantes.
3. **Implementación de modelos:** Codificamos y ajustamos los algoritmos a probar.
4. **Entrenamiento y predicción:** Ejecutamos los modelos y generamos recomendaciones.
5. **Evaluación:** Calculamos métricas y visualizamos resultados.
6. **Análisis:** Interpretamos hallazgos y formulamos conclusiones.
7. **Iteración:** Refinamos parámetros y repetimos el proceso hasta optimizar resultados.

Los hallazgos de estos experimentos informan directamente la implementación final de los modelos en el sistema de recomendación.

---

Para detalles sobre la implementación de los modelos, consultar la carpeta [`models/`](../models/README.md).

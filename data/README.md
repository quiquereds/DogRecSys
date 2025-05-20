<h1 align="center">
  <br>
  📁 Módulo de Datos
  <br>
</h1>

<h4 align="center">Almacenamiento y gestión de datasets para el sistema de recomendación</h4>

---

## 📑 Contenido

- [Descripción](#🔍-descripción)
- [Archivos principales](#📋-archivos-principales)
- [Fundamentos de datos](#🧪-fundamentos-de-datos)
  - [Calidad de datos](#calidad-de-datos)
  - [Simulación de interacciones](#simulación-de-interacciones)
- [Ciclo de vida](#🔄-ciclo-de-vida)

---

## 🔍 Descripción

Esta carpeta contiene los datasets fundamentales para el funcionamiento de DogRecSys. Aquí se almacenan tanto los datos originales (crudos) como los datos preprocesados y las simulaciones de interacciones usuario-raza necesarias para el entrenamiento y evaluación de los modelos de recomendación.

---

## 📋 Archivos principales

| Archivo                  | Descripción |
|--------------------------|-------------|
| **akc-data-latest.csv**  | Dataset original de razas de perros (fuente AKC). Incluye características, descripciones y atributos de cada raza. |
| **preprocessed_data.csv**| Datos procesados y normalizados listos para la vectorización y modelado. |
| **user_dog_ratings.csv** | Matriz de interacciones usuario-raza (ratings explícitos o simulados). |

## 🧪 Fundamentos de datos

### Calidad de datos

La calidad de los datos es crítica en sistemas de recomendación. Datos incompletos, inconsistentes o sesgados pueden degradar el rendimiento de los modelos y generar recomendaciones poco útiles o injustas.

- **Limpieza:** Elimina duplicados, corrige valores atípicos y homogeneiza formatos.
- **Normalización:** Escala variables numéricas y codifica variables categóricas para asegurar comparabilidad.


### Simulación de interacciones

Cuando no se dispone de suficientes datos reales de usuarios, es posible simular matrices de interacción para:
- Evaluar modelos colaborativos bajo diferentes escenarios (cold start, sparsity, etc.).
- Analizar el impacto de la densidad de la matriz en la calidad de las recomendaciones.

---

## 🔄 Ciclo de vida

1. **Ingesta:** Se importa `akc-data-latest.csv` desde fuentes confiables.
2. **Exploración:** Se analizan distribuciones, correlaciones y valores atípicos.
3. **Limpieza:** Se eliminan inconsistencias y valores nulos.
4. **Transformación:** Se normalizan variables y codifican categorías, generando `preprocessed_data.csv`.
5. **Simulación:** Se generan o recolectan ratings para la matriz de interacciones en `user_dog_ratings.csv`.
6. **Alimentación:** Los datos preprocesados pasan a los modelos de recomendación.

---

Para detalles sobre el preprocesamiento, consultar la carpeta [`pipeline/`](../pipeline/README.md).

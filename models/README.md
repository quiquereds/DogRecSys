<h1 align="center">
  <br>
  🤖 Módulo de Modelos
  <br>
</h1>

<h4 align="center">Algoritmos e implementaciones de los sistemas de recomendación</h4>

---

## 📑 Contenido

- [Descripción](#-🔍-descripción)
- [Modelos implementados](#-📋-modelos-implementados)
- [Fundamentos de sistemas de recomendación](#🧪-fundamentos-de-sistemas-de-recomendación)
  - [Filtrado Basado en Contenido (FBC)](#filtrado-basado-en-contenido-fbc)
  - [Filtrado Colaborativo (FC)](#filtrado-colaborativo-fc)
  - [Factorización de Matrices (NMF)](#factorización-de-matrices-nmf)
  - [Reranking y diversificación](#reranking-y-diversificación)


---

## 🔍 Descripción

Este módulo contiene la implementación de los diferentes algoritmos de recomendación utilizados en DogRecSys. Las clases y funciones permiten construir, entrenar y usar modelos de filtrado basado en contenido, filtrado colaborativo, factorización de matrices y enfoques híbridos, así como técnicas de reranking para mejorar diversidad.

---

## 📋 Modelos implementados

| Archivo                | Descripción |
|------------------------|-------------|
| **content_based.py**   | Implementación de recomendadores basados en contenido con vectorización de características y cálculo de similitud. |
| **collaborative.py**   | Algoritmos de filtrado colaborativo en sus variantes user-based y item-based para predicción de ratings. |
| **collaborative_nmf.py** | Modelo de factorización de matrices no negativa para descubrir factores latentes y mejorar recomendaciones. |
| **hybrid.py**          | Implementación de estrategias híbridas que combinan múltiples modelos y aplican reranking MMR. |

## 🧪 Fundamentos de sistemas de recomendación

### Filtrado Basado en Contenido (FBC)

Este enfoque recomienda ítems (razas de perros) basándose en sus características intrínsecas y su similitud con las preferencias del usuario:

- **Representación vectorial:** Cada raza se codifica como un vector multidimensional que captura sus atributos (tamaño, energía, temperamento, etc.).
- **Perfil de usuario:** Se construye a partir de las preferencias explícitas o implícitas, representando sus gustos en el mismo espacio vectorial.
- **Medidas de similitud:** Se utiliza similitud coseno, distancia euclídea u otras métricas para encontrar razas similares a las preferidas.

**Ventajas:**
- No requiere datos de otros usuarios (mitiga cold-start).
- Puede recomendar ítems no populares o nuevos.
- Facilita la explicabilidad ("recomendamos esta raza por su baja caída de pelo").

**Desafíos:**
- Limitado a las características explícitas.
- Puede carecer de diversidad y serendipidad.

### Filtrado Colaborativo (FC)

Este enfoque aprovecha la "sabiduría colectiva" para recomendar en base a patrones de comportamiento similares:

- **User-based CF:** Identifica usuarios similares al actual y recomienda razas que estos han valorado positivamente.
- **Item-based CF:** Encuentra razas similares a las que el usuario ya ha mostrado interés, basándose en cómo otros usuarios las han valorado conjuntamente.
- **Matriz de interacciones:** El corazón del sistema es una matriz usuario-raza donde cada celda contiene un rating o preferencia.

**Ventajas:**
- Descubre relaciones implícitas no evidentes en las características.
- Potencialmente más personalizado y sorprendente.

**Desafíos:**
- Sufre de cold-start para nuevos usuarios o ítems.
- Requiere suficiente densidad de datos para ser efectivo.

### Factorización de Matrices (NMF)

La factorización de matrices no negativa (NMF) descompone la matriz de interacciones en factores latentes:

- **Dimensionalidad latente:** Reduce la matriz de interacciones a dos matrices de menor dimensión (usuario-factor y raza-factor).
- **Factores latentes:** Descubre conceptos implícitos como "razas familiares", "razas deportivas", aunque estos factores no sean directamente interpretables.
- **Reconstrucción:** Multiplica ambas matrices para predecir ratings desconocidos.

**Ventajas:**
- Maneja eficientemente matrices dispersas.
- Descubre relaciones no evidentes en características o interacciones directas.
- Escalable a grandes volúmenes de datos.

**Implementación:**
```
R ≈ W × H
```
Donde R es la matriz de ratings, W la matriz usuario-factor y H la matriz factor-raza.

### Reranking y diversificación

El reranking permite balancear relevancia con otros objetivos como diversidad:

- **MMR (Maximal Marginal Relevance):** Selecciona secuencialmente ítems que maximizan la combinación de relevancia y diferencia respecto a los ya seleccionados.
- **Lambda (λ):** Parámetro entre 0 y 1 que controla el equilibrio entre relevancia (λ=1) y diversidad (λ=0).
- **Matriz de similitud inter-ítem:** Permite calcular la diferencia entre ítems para asegurar diversidad.



---

Para detalles de preprocesamiento y pipeline, consultar la carpeta [`pipeline/`](../pipeline/README.md).

# Módulo MMR (Maximal Marginal Relevance) - Documentación

## ¿Qué es MMR?

El algoritmo **MMR (Maximal Marginal Relevance)** es una técnica de reordenamiento que busca balancear dos objetivos aparentemente contradictorios en sistemas de recomendación:

1. **Relevancia**: Recomendar elementos que sean altamente relevantes para el usuario
2. **Diversidad**: Evitar recomendar elementos muy similares entre sí

## ¿Por qué usar MMR?

### Problema que resuelve
Sin MMR, un sistema de recomendación podría sugerir:
- Golden Retriever (score: 0.95)
- Labrador Retriever (score: 0.92) 
- Chesapeake Bay Retriever (score: 0.90)

Aunque todos tienen scores altos, son muy similares. MMR ayuda a diversificar:
- Golden Retriever (score: 0.95)
- Beagle (score: 0.88)
- Chihuahua (score: 0.85)

### Beneficios
- **Mejor experiencia de usuario**: Recomendaciones más variadas
- **Mayor exploración**: Ayuda a descubrir opciones diferentes
- **Reduce redundancia**: Evita recomendaciones repetitivas

## ¿Cómo funciona?

### Fórmula MMR
```
MMR = λ × relevancia - (1-λ) × similitud_máxima
```

### Pasos del algoritmo:
1. **Inicialización**: Selecciona el elemento más relevante
2. **Iteración**: Para cada elemento restante:
   - Calcula su relevancia original
   - Encuentra la similitud máxima con elementos ya seleccionados
   - Aplica la fórmula MMR
   - Selecciona el elemento con mayor score MMR
3. **Repetir** hasta completar top_k elementos

## Parámetros Clave

### λ (lambda_param)
Controla el balance entre relevancia y diversidad:

| λ | Comportamiento | Cuándo usar |
|---|---|---|
| 0.9-1.0 | **Prioriza relevancia** | Cuando la precisión es crítica |
| 0.5-0.8 | **Balance equilibrado** | Caso general recomendado |
| 0.1-0.4 | **Prioriza diversidad** | Exploración y descubrimiento |
| 0.0-0.1 | **Máxima diversidad** | Casos muy específicos |

### Ejemplos de configuración:
- **Sistema médico** (λ=0.9): Precisión es crítica
- **E-commerce general** (λ=0.7): Balance entre precisión y variedad
- **Exploración musical** (λ=0.3): Descubrir nuevos géneros

## Uso del módulo

### Importación
```python
from mmr_reranking import mmr_rerank, get_mmr_explanation
```

### Ejemplo básico
```python
import pandas as pd
import numpy as np

# Preparar datos
candidates = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Golden Retriever', 'Labrador', 'Beagle', 'Poodle', 'Bulldog'],
    'score': [0.95, 0.92, 0.88, 0.85, 0.82]
})

# Embeddings (vectores de características)
embeddings = np.random.rand(5, 100)  # 5 candidatos, 100 características

# Aplicar MMR
reranked = mmr_rerank(
    candidates=candidates,
    embeddings=embeddings,
    top_k=3,
    lambda_param=0.7
)

print(reranked[['name', 'score']])
```

### Configuración recomendada por caso de uso

#### Sistema de recomendación de perros
```python
# Para familias que buscan diversidad de opciones
mmr_rerank(candidates, embeddings, top_k=5, lambda_param=0.6)

# Para expertos que priorizan características específicas
mmr_rerank(candidates, embeddings, top_k=3, lambda_param=0.8)
```

#### E-commerce
```python
# Página principal (variedad)
mmr_rerank(products, embeddings, top_k=10, lambda_param=0.5)

# Búsqueda específica (precisión)
mmr_rerank(products, embeddings, top_k=5, lambda_param=0.8)
```

## Funciones auxiliares

### `get_mmr_explanation(lambda_param)`
Explica el comportamiento esperado según el valor de λ:

```python
print(get_mmr_explanation(0.7))
# Output: "Configuración BALANCEADA: se busca un equilibrio entre relevancia y diversidad."
```

### Funciones internas
- `_validate_inputs()`: Valida parámetros de entrada
- `_calculate_diversity_penalty()`: Calcula similitud máxima
- `_calculate_mmr_score()`: Aplica fórmula MMR

## Consideraciones técnicas

### Requisitos de datos
- **candidates**: DataFrame con columna 'score' obligatoria
- **embeddings**: Matriz numpy con una fila por candidato
- **Mismo número**: len(candidates) == len(embeddings)

### Complejidad computacional
- **Tiempo**: O(k × n × d) donde k=top_k, n=candidatos, d=dimensión_embeddings
- **Espacio**: O(n × d) para almacenar embeddings

### Limitaciones
- Requiere embeddings pre-computados
- La calidad depende de la calidad de los embeddings
- Puede ser lento con muchos candidatos y alta dimensionalidad

## Testing

Para probar el módulo:
```bash
cd pipeline/
python mmr_reranking.py
```

Esto ejecutará ejemplos demostrativos con diferentes valores de λ.

## Referencias

- Carbonell, J. & Goldstein, J. (1998). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries"
- Aplicaciones en sistemas de recomendación modernos
- Variantes como MMR-MD (Maximum Marginal Relevance with Maximum Diversity)

---

**Nota**: Este módulo está diseñado específicamente para el proyecto DogRecSys, pero puede adaptarse fácilmente a otros dominios de recomendación.

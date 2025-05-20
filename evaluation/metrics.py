"""
Métricas para Evaluación de Sistemas de Recomendación
====================================================

Este módulo contiene funciones para evaluar qué tan bien un sistema de recomendación 
está funcionando. Las tres métricas principales son:

1. Precision@K: ¿Qué porcentaje de mis recomendaciones son relevantes?
2. Recall@K: ¿Qué porcentaje de todos los elementos relevantes logré recomendar?
3. NDCG@K: ¿Qué tan bien ordené las recomendaciones? (los elementos relevantes deberían estar al principio)

Todas las funciones tienen los mismos parámetros:
- recommended: Una lista con los IDs que el sistema recomendó, ordenados del más recomendado al menos
- relevant: Lista o conjunto con los IDs que son relevantes para el usuario
- k: Cuántas recomendaciones analizar (5, 10, 20, etc.)

Ejemplo:
----------------
Si a un usuario le gustan los perros con ID [5, 8, 10] y el sistema recomienda [5, 2, 8, 3, 10]:

>>> precision_at_k([5, 2, 8, 3, 10], [5, 8, 10], k=5)
0.6  # 3 de 5 recomendaciones son relevantes (60%)

>>> recall_at_k([5, 2, 8, 3, 10], [5, 8, 10], k=5)
1.0  # Se recomendaron todos los elementos relevantes (100%)

>>> ndcg_at_k([5, 2, 8, 3, 10], [5, 8, 10], k=5)
0.7655  # Hubo un bien ranking pero no es el ideal (5 está primero, pero 8 y 10 no están en 2° y 3° lugar)
"""

import numpy as np

def precision_at_k(recommended, relevant, k):
    """
    ¿Qué porcentaje de nuestras recomendaciones fueron útiles?
    
    Precision@K = (número de recomendaciones relevantes hasta k) / k
    
    Entradas:
        recommended: Lista de IDs recomendados, ordenados por relevancia (el primero es el más recomendado)
        relevant: Lista o conjunto de IDs que son realmente relevantes
        k: Número de recomendaciones a considerar
    
    Salida:
        Un valor entre 0 y 1:
        - 0 = Ninguna recomendación fue relevante
        - 1 = Todas las recomendaciones fueron relevantes
    
    Ejemplo:
        Un sistema recomienda [1, 2, 3, 4, 5] pero al usuario solo le gustan [2, 4, 7]
        >>> precision_at_k([1, 2, 3, 4, 5], [2, 4, 7], k=5)
        0.4  # 2 de 5 recomendaciones son relevantes (2 y 4)
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum([1 for item in recommended_k if item in relevant_set])
    return hits / k

def recall_at_k(recommended, relevant, k):
    """
    ¿Qué porcentaje de los elementos relevantes logramos recomendar?
    
    Recall@K = (número de elementos relevantes recomendados hasta k) / (número total de elementos relevantes)
    
    Entradas:
        recommended: Lista de IDs recomendados, ordenados por relevancia
        relevant: Lista o conjunto de IDs que son realmente relevantes
        k: Número de recomendaciones a considerar
    
    Salida:
        Un valor entre 0 y 1:
        - 0 = No se recomendó ningún elemento relevante
        - 1 = Se recomendaron todos los elementos relevantes
    
    Ejemplo:
        Un sistema recomienda [1, 2, 3] pero al usuario le gustan [2, 4, 7]
        >>> recall_at_k([1, 2, 3], [2, 4, 7], k=3)
        0.33  # Solo recomendamos 1 de 3 elementos relevantes (el 2)
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum([1 for item in recommended_k if item in relevant_set])
    return hits / len(relevant) if relevant else 0.0

def ndcg_at_k(recommended, relevant, k):
    """
    ¿Qué tan bien ordenamos las recomendaciones?
    
    NDCG valora más cuando los elementos relevantes aparecen en las primeras posiciones.
    Un elemento relevante en la posición 1 aporta más que uno en la posición 5.
    
    Entradas:
        recommended: Lista de IDs recomendados, ordenados por relevancia
        relevant: Lista o conjunto de IDs que son realmente relevantes
        k: Número de recomendaciones a considerar
    
    Salida:
        Un valor entre 0 y 1:
        - 0 = No se recomendó ningún elemento relevante
        - 1 = Orden perfecto (todos los elementos relevantes al principio, en el mejor orden posible)
    
    Ejemplo:
        Supongamos que hay 3 elementos relevantes [A, B, C], idealmente queremos recomendarlos primero:
        - Un orden perfecto: [A, B, C, D, E] → NDCG ≈ 1.0
        - Orden menos óptimo: [D, A, E, B, C] → NDCG ≈ 0.5
    """
    recommended_k = recommended[:k]
    dcg = 0.0
    relevant_set = set(relevant)
    
    # Calculamos DCG (Discounted Cumulative Gain)
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            # La fórmula da más valor a elementos relevantes que aparecen primero
            dcg += 1 / np.log2(i + 2)  # i+2 porque los índices empiezan en 0
    
    # Calculamos IDCG (el DCG del caso ideal donde todos los relevantes están al principio)
    ideal_hits = min(len(relevant), k)
    idcg = sum([1 / np.log2(i + 2) for i in range(ideal_hits)])
    
    # NDCG = DCG / IDCG
    return dcg / idcg if idcg > 0 else 0.0

"""
Módulo de reordenamiento MMR (Maximal Marginal Relevance)

Este módulo implementa el algoritmo MMR para reordenar candidatos de recomendación,
balanceando la relevancia con la diversidad para evitar recomendaciones muy similares.

El algoritmo MMR es especialmente útil en sistemas de recomendación donde se desea
mantener un equilibrio entre recomendar elementos altamente relevantes y elementos
diversos que no sean demasiado similares entre sí.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def mmr_rerank(candidates: pd.DataFrame, embeddings: np.ndarray, top_k: int = 10, lambda_param: float = 0.7):
    """
    Reordena candidatos usando el algoritmo MMR (Maximal Marginal Relevance).
    
    El algoritmo MMR selecciona iterativamente elementos que maximizan una función
    que combina relevancia y diversidad. En cada iteración:
    1. Calcula la relevancia del candidato (score original)
    2. Calcula la similitud máxima con elementos ya seleccionados (penalización por diversidad)
    3. Combina ambos factores usando lambda_param como peso
    
    Fórmula MMR: λ * relevancia - (1-λ) * max_similitud_con_seleccionados
    
    Args:
        candidates (pd.DataFrame): DataFrame con los candidatos a recomendar. 
                                 Debe contener al menos una columna 'score' con la relevancia inicial.
        embeddings (np.ndarray): Matriz de embeddings de los candidatos donde cada fila 
                                representa el vector de características de un candidato.
                                Debe tener la misma longitud que candidates.
        top_k (int, opcional): Número máximo de elementos a devolver. Por defecto 10.
        lambda_param (float, opcional): Parámetro que controla el equilibrio entre relevancia y diversidad.
                                       - Valores cercanos a 1.0: Prioriza relevancia (menos diversidad)
                                       - Valores cercanos a 0.0: Prioriza diversidad (menos relevancia)
                                       - Valor por defecto: 0.7 (balance hacia relevancia)
    
    Returns:
        pd.DataFrame: DataFrame con los candidatos reordenados según MMR.
                     Mantiene todas las columnas originales y los ordena por importancia MMR.
                     Los índices se resetean para facilitar el acceso secuencial.
    
    Raises:
        ValueError: Si candidates está vacío o si embeddings no coincide con candidates.
        KeyError: Si candidates no contiene la columna 'score'.
    
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Datos de ejemplo
        >>> candidates = pd.DataFrame({
        ...     'id': [1, 2, 3, 4, 5],
        ...     'score': [0.9, 0.8, 0.85, 0.75, 0.7],
        ...     'name': ['Dog A', 'Dog B', 'Dog C', 'Dog D', 'Dog E']
        ... })
        >>> embeddings = np.random.rand(5, 100)  # 5 perros, 100 características
        >>> 
        >>> # Reordenar con MMR
        >>> reranked = mmr_rerank(candidates, embeddings, top_k=3, lambda_param=0.7)
        >>> print(reranked[['id', 'name', 'score']])
    """
    
    # Validación de entrada
    _validate_inputs(candidates, embeddings, top_k, lambda_param)
    
    # Ajustar top_k si es mayor que el número de candidatos disponibles
    top_k = min(top_k, len(candidates))
    
    # Lista para almacenar los índices de elementos seleccionados
    selected_indices = []
    # Lista con todos los índices disponibles inicialmente
    remaining_indices = list(range(len(candidates)))

    # Algoritmo principal MMR: selección iterativa
    while len(selected_indices) < top_k and remaining_indices:
        
        # PASO 1: Si no hay elementos seleccionados, elegir el de mayor relevancia
        if not selected_indices:
            # Encontrar el índice del candidato con mayor score entre los restantes
            remaining_scores = candidates["score"].values[remaining_indices]
            best_score_pos = np.argmax(remaining_scores)
            best_idx = remaining_indices[best_score_pos]
            
            # Mover el mejor candidato de remaining a selected
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            continue

        # PASO 2: Para elementos subsecuentes, calcular MMR para cada candidato restante
        mmr_scores = []
        
        for idx in remaining_indices:
            # Obtener la relevancia original del candidato
            relevance = candidates.iloc[idx]["score"]
            
            # Calcular similitud coseno entre el candidato actual y todos los ya seleccionados
            diversity_penalty = _calculate_diversity_penalty(idx, selected_indices, embeddings)
            
            # Fórmula MMR: λ * relevancia - (1-λ) * penalización_diversidad
            mmr_score = _calculate_mmr_score(relevance, diversity_penalty, lambda_param)
            mmr_scores.append(mmr_score)

        # PASO 3: Seleccionar el candidato con el MMR más alto
        best_mmr_pos = np.argmax(mmr_scores)
        best_idx = remaining_indices[best_mmr_pos]
        
        # Mover el mejor candidato de remaining a selected
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    # Retornar los candidatos seleccionados en el orden determinado por MMR
    result = candidates.iloc[selected_indices].copy()
    return result.reset_index(drop=True)


def _validate_inputs(candidates: pd.DataFrame, embeddings: np.ndarray, top_k: int, lambda_param: float):
    """
    Valida los parámetros de entrada para la función mmr_rerank.
    
    Args:
        candidates (pd.DataFrame): DataFrame de candidatos
        embeddings (np.ndarray): Matriz de embeddings
        top_k (int): Número de elementos a retornar
        lambda_param (float): Parámetro lambda para MMR
    
    Raises:
        ValueError: Si algún parámetro no es válido
        Key
    """
    if candidates.empty:
        raise ValueError("El DataFrame de candidatos no puede estar vacío")
    
    if 'score' not in candidates.columns:
        raise KeyError("El DataFrame de candidatos debe contener una columna 'score'")
    
    if len(candidates) != len(embeddings):
        raise ValueError(f"El número de candidatos ({len(candidates)}) debe coincidir "
                        f"con el número de embeddings ({len(embeddings)})")
    
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k debe ser un entero positivo")
    
    if not (0 <= lambda_param <= 1):
        raise ValueError("lambda_param debe estar entre 0 y 1")


def _calculate_diversity_penalty(candidate_idx: int, selected_indices: list, embeddings: np.ndarray) -> float:
    """
    Calcula la penalización por diversidad para un candidato dado.
    
    La penalización es la similitud coseno máxima entre el candidato
    y cualquier elemento ya seleccionado.
    
    Args:
        candidate_idx (int): Índice del candidato a evaluar
        selected_indices (list): Lista de índices ya seleccionados
        embeddings (np.ndarray): Matriz de embeddings
    
    Returns:
        float: Penalización por diversidad (similitud máxima)
    """
    if not selected_indices:
        return 0.0
    
    # Embedding del candidato actual
    candidate_embedding = embeddings[candidate_idx].reshape(1, -1)
    
    # Embeddings de los elementos ya seleccionados
    selected_embeddings = embeddings[selected_indices]
    
    # Calcular similitudes coseno
    similarities = cosine_similarity(candidate_embedding, selected_embeddings)[0]
    
    # Retornar la similitud máxima
    return np.max(similarities)


def _calculate_mmr_score(relevance: float, diversity_penalty: float, lambda_param: float) -> float:
    """
    Calcula el score MMR para un candidato.
    
    Fórmula: λ * relevancia - (1-λ) * penalización_diversidad
    
    Args:
        relevance (float): Score de relevancia original
        diversity_penalty (float): Penalización por similitud con seleccionados
        lambda_param (float): Parámetro que controla el balance relevancia/diversidad
    
    Returns:
        float: Score MMR calculado
    """
    return lambda_param * relevance - (1 - lambda_param) * diversity_penalty


def get_mmr_explanation(lambda_param: float) -> str:
    """
    Retorna una explicación textual del comportamiento esperado según lambda_param.
    
    Args:
        lambda_param (float): Parámetro lambda utilizado
    
    Returns:
        str: Explicación del comportamiento esperado
    """
    if lambda_param >= 0.8:
        return "Configuración orientada a RELEVANCIA: se priorizan elementos con scores altos, " \
               "con poca consideración por la diversidad."
    elif lambda_param >= 0.5:
        return "Configuración BALANCEADA: se busca un equilibrio entre relevancia y diversidad."
    elif lambda_param >= 0.2:
        return "Configuración orientada a DIVERSIDAD: se priorizan elementos diversos, " \
               "aunque tengan scores menores."
    else:
        return "Configuración de MÁXIMA DIVERSIDAD: se minimizan las similitudes, " \
               "con muy poca consideración por los scores originales."


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del algoritmo MMR para reordenamiento de candidatos.
    """
    
    # Crear datos de ejemplo
    print("=== EJEMPLO DE USO: MMR RERANKING ===\n")
    
    candidates_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Golden Retriever', 'Labrador', 'Beagle', 'Poodle', 'Bulldog'],
        'score': [0.95, 0.92, 0.88, 0.85, 0.82],
        'size': ['Large', 'Large', 'Medium', 'Medium', 'Medium']
    }
    
    candidates = pd.DataFrame(candidates_data)
    
    # Simular embeddings
    np.random.seed(42)
    embeddings = np.random.rand(len(candidates), 50)
    
    print("Candidatos originales:")
    print(candidates[['name', 'score']].to_string(index=False))
    print()
    
    # Probar diferentes configuraciones de lambda
    for lambda_param in [0.9, 0.5, 0.1]:
        print(f"--- λ = {lambda_param} ---")
        print(get_mmr_explanation(lambda_param))
        
        reranked = mmr_rerank(candidates, embeddings, top_k=3, lambda_param=lambda_param)
        print("Top 3:")
        print(reranked[['name', 'score']].to_string(index=False))
        print()
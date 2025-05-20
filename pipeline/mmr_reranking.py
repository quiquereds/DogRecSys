import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def mmr_rerank(candidates: pd.DataFrame, embeddings: np.ndarray, top_k: int = 10, lambda_param: float = 0.7):
    """    
    Args:
        candidates (pd.DataFrame): DataFrame con los candidatos a recomendar. Debe contener una columna 'id' y una columna 'score'.
        embeddings (np.ndarray): Matriz de embeddings de los candidatos.
        top_k (int): Número de elementos a devolver.
        lambda_param (float): Parámetro que controla el equilibrio entre relevancia y diversidad. 
                            Un valor cercano a 1 prioriza la relevancia, mientras que un valor cercano a 0 prioriza la diversidad.
    Returns:
        pd.DataFrame: DataFrame con los candidatos reordenados, incluyendo las columnas 'id', 'score' y 'mmr_score'.
    """
    
    selected_indices = []
    remaining_indices = list(range(len(candidates)))

    while len(selected_indices) < top_k and remaining_indices:
        if not selected_indices:
            # Seleccionar el de mayor relevancia al inicio
            idx = np.argmax(candidates["score"].values[remaining_indices])
            selected_indices.append(remaining_indices.pop(idx))
            continue

        mmr_scores = []
        for idx in remaining_indices:
            rel = candidates.iloc[idx]["score"]
            sims = cosine_similarity(
                [embeddings[idx]],
                [embeddings[j] for j in selected_indices]
            )[0]
            diversity_penalty = np.max(sims)
            mmr = lambda_param * rel - (1 - lambda_param) * diversity_penalty
            mmr_scores.append(mmr)

        best_idx = remaining_indices[np.argmax(mmr_scores)]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return candidates.iloc[selected_indices].reset_index(drop=True)
# models/cf.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_matrix(ratings_df: pd.DataFrame):
    """
    Calcula la matriz de similitud coseno entre razas (columnas).
    """
    item_similarity = cosine_similarity(ratings_df.T)
    sim_df = pd.DataFrame(item_similarity, index=ratings_df.columns, columns=ratings_df.columns)
    return sim_df

def recommend_by_cf(user_likes: list[str], similarity_df: pd.DataFrame, top_k=5) -> pd.DataFrame:
    """
    Recomienda ítems (razas) basándose en la similitud con ítems que el usuario ya valoró.
    
    Args:
        user_likes (list[str]): Lista de nombres de razas que el usuario "valoró positivamente".
        similarity_df (pd.DataFrame): Matriz de similitud entre razas.
    
    Returns:
        pd.DataFrame: Lista ordenada de recomendaciones con puntaje de similitud promedio.
    """
    scores = similarity_df[user_likes].mean(axis=1)
    scores = scores.drop(labels=user_likes, errors="ignore")  # Excluir razas ya conocidas
    return scores.sort_values(ascending=False).head(top_k).reset_index(name="cf_score")

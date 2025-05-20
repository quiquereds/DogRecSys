import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de similitud coseno entre ítems (razas).
    
    Args:
        ratings_df (pd.DataFrame): DataFrame con las calificaciones de los usuarios (filas) para cada raza (columnas).
        Las calificaciones deben ser numéricas y no contener valores nulos.
    Returns:
        pd.DataFrame: Matriz de similitud coseno entre razas.
    """
    similarity = cosine_similarity(ratings_df.T)
    return pd.DataFrame(similarity, index=ratings_df.columns, columns=ratings_df.columns)

def recommend_by_cf(user_likes: list[str], similarity_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Recomienda razas basadas en la similitud a las razas favoritas del usuario.
    
    Args:
        user_likes (list): Lista de razas que le gustaron al usuario.
        similarity_df (pd.DataFrame): Matriz de similitud entre razas.
        top_k (int): Número de recomendaciones a devolver.

    Returns:
        pd.DataFrame: Top-K recomendaciones con puntajes de similitud promedio.
    """
    scores = similarity_df[user_likes].mean(axis=1)
    scores = scores.drop(labels=user_likes, errors="ignore")
    return scores.sort_values(ascending=False).head(top_k).reset_index(name="cf_score")


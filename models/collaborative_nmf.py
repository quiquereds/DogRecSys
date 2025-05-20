import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

def factorize_with_nmf(ratings_df: pd.DataFrame, n_components: int = 20, return_model: bool = False):
    """    
    Args:
        ratings_df (pd.DataFrame): DataFrame con las calificaciones de los usuarios (filas) para cada ítem (columnas).
        n_components (int): Número de componentes a usar en la factorización.
        return_model (bool): Si True, devuelve el modelo NMF además de la matriz reconstruida.
    
    Returns:
        pd.DataFrame: Matriz reconstruida con predicciones (mismos índices y columnas).
        np.ndarray (opcional): Si return_model es True, devuelve también la matriz W (usuarios).
        np.ndarray (opcional): Si return_model es True, devuelve también la matriz H (ítems).
    """
    model = NMF(n_components=n_components, init="random", random_state=42)
    W = model.fit_transform(ratings_df.values)
    H = model.components_
    R_hat = np.dot(W, H)
    
    if return_model:
        return pd.DataFrame(R_hat, index=ratings_df.index, columns=ratings_df.columns), W, H
    else:
        return pd.DataFrame(R_hat, index=ratings_df.index, columns=ratings_df.columns)


def recommend_from_nmf(predicted_matrix: pd.DataFrame, user_id: str, known_items: list[str], top_k: int = 5) -> pd.DataFrame:
    """
    Recomienda ítems a un usuario con base en la matriz predicha, excluyendo los ya conocidos.
    
    Args:
        predicted_matrix (pd.DataFrame): Matriz predicha por NMF.
        user_id (str): ID del usuario para el que se hacen las recomendaciones.
        known_items (list): Lista de ítems que el usuario ya conoce.
        top_k (int): Número de recomendaciones a devolver.
    Returns:
        pd.DataFrame: Top-K recomendaciones con puntajes.
    """
    # Verificar que el usuario existe en la matriz
    if user_id not in predicted_matrix.index:
        print(f"Error: El usuario {user_id} no existe en la matriz de predicciones")
        return pd.DataFrame(columns=["index", "nmf_score"])
    
    # Obtener las puntuaciones predichas para el usuario
    scores = predicted_matrix.loc[user_id]
    
    # Verificar cuántos items están disponibles antes de excluir los conocidos
    total_items = len(scores)
    print(f"Total de items disponibles para recomendar (antes de filtrar): {total_items}")
    
    # Excluir items que el usuario ya conoce
    scores = scores.drop(labels=known_items, errors="ignore")
    
    # Verificar cuántos items quedan después de filtrar
    remaining_items = len(scores)
    print(f"Items disponibles después de excluir los conocidos: {remaining_items}")
    
    # Verificar si quedan suficientes items para recomendar
    if remaining_items == 0:
        print(f"No quedan items para recomendar al usuario {user_id} después de excluir los {len(known_items)} items conocidos")
        return pd.DataFrame(columns=["index", "nmf_score"])
    
    # Ordenar y devolver las top-k recomendaciones
    top_recommendations = scores.sort_values(ascending=False).head(top_k)
    print(f"Top {len(top_recommendations)} recomendaciones generadas con éxito")
    
    return top_recommendations.reset_index(name="nmf_score")

# models/cbf.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def generate_user_vector(user_preferences: dict, scaler, feature_order: list[str]):
    """
    Genera un vector normalizado a partir del diccionario de preferencias del usuario.
    
    Args:
        user_preferences (dict): Preferencias del usuario con claves iguales a las columnas de entrenamiento.
        scaler (StandardScaler): Escalador entrenado con los datos originales.
        feature_order (list[str]): Orden de las columnas estructurales.
    
    Returns:
        np.ndarray: Vector normalizado (1 x N) para el usuario.
    """
    user_df = pd.DataFrame([user_preferences])[feature_order]
    user_vector = scaler.transform(user_df)
    return user_vector

def recommend_by_content(user_vector, item_vectors, df_original: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Calcula similitud coseno entre el perfil del usuario y las razas.
    
    Args:
        user_vector (np.ndarray): Vector del usuario.
        item_vectors (np.ndarray): Matriz con vectores de las razas.
        df_original (pd.DataFrame): DataFrame original con información de las razas.
        top_k (int): Número de recomendaciones a devolver.
    
    Returns:
        pd.DataFrame: Top-k recomendaciones ordenadas por score de similitud.
    """
    similarities = cosine_similarity(user_vector, item_vectors)[0]
    df_result = df_original.copy()
    df_result["cbf_score"] = similarities
    return df_result.sort_values(by="cbf_score", ascending=False).head(top_k)


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

def generate_text_query_vector(query_text: str, tfidf_vectorizer):
    """
    Genera un vector TF-IDF a partir de una consulta de texto del usuario.
    
    Args:
        query_text (str): Texto de consulta del usuario.
        tfidf_vectorizer: Vectorizador TF-IDF entrenado con los datos originales.
    
    Returns:
        np.ndarray: Vector TF-IDF para la consulta del usuario.
    """
    return tfidf_vectorizer.transform([query_text])

def combine_similarities(num_sim: np.ndarray, text_sim: np.ndarray, alpha: float = 0.5):
    """
    Combina las similitudes numéricas y textuales con un factor de ponderación.
    
    Args:
        num_sim (np.ndarray): Similitudes calculadas con vectores numéricos.
        text_sim (np.ndarray): Similitudes calculadas con vectores de texto.
        alpha (float): Factor de ponderación entre 0 y 1.
            - alpha = 1: Solo considera similitudes numéricas.
            - alpha = 0: Solo considera similitudes textuales.
            - alpha = 0.5: Pondera por igual ambas similitudes.
    
    Returns:
        np.ndarray: Similitudes combinadas.
    """
    return alpha * num_sim + (1 - alpha) * text_sim

def recommend_by_content(
    user_vector, 
    item_vectors, 
    df_original: pd.DataFrame, 
    top_k: int = 5,
    text_query: str = None,
    text_vectors = None,
    tfidf_vectorizer = None,
    alpha: float = 1.0
) -> pd.DataFrame:
    """
    Calcula similitud entre el perfil del usuario y las razas.
    Opcionalmente combina similitudes numéricas y textuales.
    
    Args:
        user_vector (np.ndarray): Vector numérico del usuario.
        item_vectors (np.ndarray): Matriz con vectores numéricos de las razas.
        df_original (pd.DataFrame): DataFrame original con información de las razas.
        top_k (int): Número de recomendaciones a devolver.
        text_query (str, opcional): Consulta de texto del usuario.
        text_vectors (matriz dispersa, opcional): Matriz con vectores de texto de las razas.
        tfidf_vectorizer (opcional): Vectorizador TF-IDF entrenado.
        alpha (float): Factor de ponderación entre similitudes (1 = solo numéricas, 0 = solo textuales).
    
    Returns:
        pd.DataFrame: Top-k recomendaciones ordenadas por score de similitud.
    """
    # Calculamos similitud con vectores numéricos
    num_similarities = cosine_similarity(user_vector, item_vectors)[0]
    
    # Si tenemos todos los elementos necesarios para similitud textual
    if text_query and text_vectors is not None and tfidf_vectorizer is not None:
        query_vector = generate_text_query_vector(text_query, tfidf_vectorizer)
        text_similarities = cosine_similarity(query_vector, text_vectors)[0]
        
        # Combinamos ambas similitudes
        similarities = combine_similarities(num_similarities, text_similarities, alpha)
    else:
        # Si no tenemos los elementos para similitud textual, usamos solo la numérica
        similarities = num_similarities
    
    # Creamos el DataFrame de resultado
    df_result = df_original.copy()
    df_result["cbf_score"] = similarities
    return df_result.sort_values(by="cbf_score", ascending=False).head(top_k)


def recommend_hybrid_content(
    user_numeric_preferences: dict,
    user_text_query: str,
    item_numeric_vectors,
    item_text_vectors,
    scaler,
    tfidf_vectorizer,
    feature_order: list[str],
    df_original: pd.DataFrame,
    alpha: float = 0.5,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Función de conveniencia que combina todo el proceso de recomendación híbrida.
    
    Args:
        user_numeric_preferences (dict): Preferencias numéricas del usuario.
        user_text_query (str): Consulta de texto del usuario.
        item_numeric_vectors (np.ndarray): Matriz con vectores numéricos de las razas.
        item_text_vectors (matriz dispersa): Matriz con vectores de texto de las razas.
        scaler: Escalador entrenado con los datos numéricos.
        tfidf_vectorizer: Vectorizador TF-IDF entrenado con los datos textuales.
        feature_order (list[str]): Orden de las columnas estructurales numéricas.
        df_original (pd.DataFrame): DataFrame original con información de las razas.
        alpha (float): Factor de ponderación (1 = solo numéricas, 0 = solo textuales).
        top_k (int): Número de recomendaciones a devolver.
        
    Returns:
        pd.DataFrame: Top-k recomendaciones ordenadas por score combinado.
    """
    # Generamos vector numérico del usuario
    user_vector = generate_user_vector(user_numeric_preferences, scaler, feature_order)
    
    # Recomendamos con ambos tipos de vectores
    return recommend_by_content(
        user_vector=user_vector,
        item_vectors=item_numeric_vectors,
        df_original=df_original,
        top_k=top_k,
        text_query=user_text_query,
        text_vectors=item_text_vectors,
        tfidf_vectorizer=tfidf_vectorizer,
        alpha=alpha
    )

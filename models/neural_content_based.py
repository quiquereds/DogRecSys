"""
Módulo para recomendaciones basadas en contenido que incorpora embeddings contextuales.
Extiende la funcionalidad de enhanced_content_based.py para soportar BERT.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple, Union

# Función existente para generar vector de usuario (igual al módulo original)
def generate_user_vector(preferences: Dict[str, float], 
                        scaler, 
                        feature_order: List[str]) -> np.ndarray:
    """
    Genera un vector numérico de preferencias de usuario.
    
    Args:
        preferences: Diccionario con preferencias numéricas
        scaler: Scaler usado para normalizar características
        feature_order: Orden de características 
        
    Returns:
        Vector de preferencias de usuario
    """
    # Crear vector de preferencias en el orden correcto
    user_preferences = np.array([[preferences[feature] for feature in feature_order]])
    
    # Aplicar scaling
    return scaler.transform(user_preferences)

def recommend_by_content_bert(
    user_vector: np.ndarray,
    item_vectors_num: np.ndarray,
    df: pd.DataFrame,
    text_query: Optional[str] = None,
    text_embeddings: Optional[np.ndarray] = None,
    bert_model = None,
    alpha: float = 0.5,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Recomienda items combinando similitud numérica y textual con embeddings BERT.
    
    Args:
        user_vector: Vector numérico de preferencias de usuario
        item_vectors_num: Matriz de vectores numéricos de items
        df: DataFrame con información de los items
        text_query: Consulta de texto del usuario (opcional)
        text_embeddings: Matriz de embeddings BERT para los items
        bert_model: Modelo para generar embeddings de la consulta
        alpha: Peso relativo para características numéricas vs textuales (1 = solo numéricas)
        top_k: Número de recomendaciones a devolver
        
    Returns:
        DataFrame con las recomendaciones ordenadas por score
    """
    # Calcular similitud con vectores numéricos
    num_similarities = cosine_similarity(user_vector, item_vectors_num)[0]
    
    # Si hay consulta de texto y embeddings, calcular similitud textual
    if text_query and text_embeddings is not None and bert_model is not None:
        from pipeline.embedding_vectorizers import vectorize_query_bert
        
        # Generar embedding para la consulta
        query_embedding = vectorize_query_bert(text_query, bert_model)
        
        # Calcular similitud con todos los items
        text_similarities = cosine_similarity(query_embedding, text_embeddings)[0]
        
        # Normalizar ambas similitudes (importante para combinar espacios diferentes)
        num_similarities = (num_similarities - np.min(num_similarities)) / (np.max(num_similarities) - np.min(num_similarities) + 1e-10)
        text_similarities = (text_similarities - np.min(text_similarities)) / (np.max(text_similarities) - np.min(text_similarities) + 1e-10)
        
        # Combinar similitudes con el peso alpha
        combined_similarities = alpha * num_similarities + (1 - alpha) * text_similarities
    else:
        combined_similarities = num_similarities
    
    # Obtener índices de los top-k items
    top_indices = np.argsort(combined_similarities)[::-1][:top_k]
    
    # Crear dataframe con recomendaciones
    recommendations = df.iloc[top_indices].copy()
    recommendations['cbf_score'] = combined_similarities[top_indices]
    
    return recommendations.sort_values('cbf_score', ascending=False)

# Mantener la función original para compatibilidad
def recommend_by_content(
    user_vector: np.ndarray,
    item_vectors_num: np.ndarray,
    df: pd.DataFrame,
    text_query: Optional[str] = None,
    text_vectors: Optional[np.ndarray] = None,
    tfidf_vectorizer = None,
    alpha: float = 0.5,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Función original para recomendar items combinando similitud numérica y textual con TF-IDF.
    """
    # Calcular similitud con vectores numéricos
    num_similarities = cosine_similarity(user_vector, item_vectors_num)[0]
    
    # Si hay consulta de texto, calcular similitud textual
    if text_query and text_vectors is not None and tfidf_vectorizer is not None:
        # Transformar consulta en vector TF-IDF
        text_vec = tfidf_vectorizer.transform([text_query])
        
        # Calcular similitud textual
        text_similarities = cosine_similarity(text_vec, text_vectors)[0]
        
        # Combinar similitudes con el peso alpha
        combined_similarities = alpha * num_similarities + (1 - alpha) * text_similarities
    else:
        combined_similarities = num_similarities
    
    # Obtener índices de los top-k items
    top_indices = np.argsort(combined_similarities)[::-1][:top_k]
    
    # Crear dataframe con recomendaciones
    recommendations = df.iloc[top_indices].copy()
    recommendations['cbf_score'] = combined_similarities[top_indices]
    
    return recommendations.sort_values('cbf_score', ascending=False)

def recommend_hybrid_content(
    user_vector: np.ndarray,
    item_vectors_num: np.ndarray,
    df: pd.DataFrame,
    text_query: Optional[str] = None,
    text_vectors_tfidf: Optional[np.ndarray] = None,
    tfidf_vectorizer = None,
    text_embeddings_bert: Optional[np.ndarray] = None,
    bert_model = None,
    alpha_num_text: float = 0.5,
    alpha_tfidf_bert: float = 0.5,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Recomienda items combinando similitud numérica, TF-IDF y BERT.
    
    Args:
        user_vector: Vector numérico de preferencias de usuario
        item_vectors_num: Matriz de vectores numéricos de items
        df: DataFrame con información de los items
        text_query: Consulta de texto del usuario (opcional)
        text_vectors_tfidf: Matriz de vectores TF-IDF para los items
        tfidf_vectorizer: Vectorizador TF-IDF entrenado
        text_embeddings_bert: Matriz de embeddings BERT para los items
        bert_model: Modelo para generar embeddings de la consulta
        alpha_num_text: Peso entre componentes numéricos vs textuales (1 = solo numéricos)
        alpha_tfidf_bert: Peso entre TF-IDF vs BERT (1 = solo TF-IDF)
        top_k: Número de recomendaciones a devolver
        
    Returns:
        DataFrame con las recomendaciones ordenadas por score
    """
    # Calcular similitud con vectores numéricos
    num_similarities = cosine_similarity(user_vector, item_vectors_num)[0]
    
    # Inicializar similitudes textuales
    text_similarities = np.zeros(len(df))
    
    if text_query:
        # Si tenemos TF-IDF, calcular similitud
        if text_vectors_tfidf is not None and tfidf_vectorizer is not None:
            text_vec_tfidf = tfidf_vectorizer.transform([text_query])
            tfidf_similarities = cosine_similarity(text_vec_tfidf, text_vectors_tfidf)[0]
        else:
            tfidf_similarities = np.zeros(len(df))
        
        # Si tenemos BERT, calcular similitud
        if text_embeddings_bert is not None and bert_model is not None:
            from pipeline.embedding_vectorizers import vectorize_query_bert
            
            query_embedding = vectorize_query_bert(text_query, bert_model)
            bert_similarities = cosine_similarity(query_embedding, text_embeddings_bert)[0]
        else:
            bert_similarities = np.zeros(len(df))
        
        # Normalizar similitudes
        tfidf_similarities = (tfidf_similarities - np.min(tfidf_similarities)) / (np.max(tfidf_similarities) - np.min(tfidf_similarities) + 1e-10)
        bert_similarities = (bert_similarities - np.min(bert_similarities)) / (np.max(bert_similarities) - np.min(bert_similarities) + 1e-10)
        
        # Combinar similitudes textuales
        text_similarities = alpha_tfidf_bert * tfidf_similarities + (1 - alpha_tfidf_bert) * bert_similarities
    
    # Normalizar similitudes numéricas
    num_similarities = (num_similarities - np.min(num_similarities)) / (np.max(num_similarities) - np.min(num_similarities) + 1e-10)
    
    # Combinar similitudes numéricas y textuales
    if text_query:
        combined_similarities = alpha_num_text * num_similarities + (1 - alpha_num_text) * text_similarities
    else:
        combined_similarities = num_similarities
    
    # Obtener índices de los top-k items
    top_indices = np.argsort(combined_similarities)[::-1][:top_k]
    
    # Crear dataframe con recomendaciones
    recommendations = df.iloc[top_indices].copy()
    recommendations['cbf_score'] = combined_similarities[top_indices]
    
    return recommendations.sort_values('cbf_score', ascending=False)

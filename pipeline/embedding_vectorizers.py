"""
Módulo para la generación de embeddings contextuales usando modelos transformers.
Implementa funciones para vectorizar texto utilizando modelos como BERT.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def vectorize_text_bert(df: pd.DataFrame, 
                       text_column: str = 'text_combined', 
                       model_name: str = 'all-MiniLM-L6-v2') -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Genera embeddings contextuales (BERT) para los textos de un dataframe.
    
    Args:
        df: DataFrame con los textos a vectorizar
        text_column: Nombre de la columna que contiene los textos
        model_name: Nombre del modelo de SentenceTransformer a utilizar
        
    Returns:
        Tuple con embeddings generados y el modelo utilizado
    """
    logger.info(f"Generando embeddings con el modelo: {model_name}")
    
    try:
        # Cargar modelo preentrenado de SentenceTransformer (HuggingFace)
        model = SentenceTransformer(model_name)
        
        # Extraer textos del dataframe
        texts = df[text_column].tolist()
        logger.info(f"Procesando {len(texts)} textos")
        
        # Generar embeddings
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        logger.info(f"Embeddings generados con forma: {embeddings.shape}")
        
        return embeddings, model
    
    except Exception as e:
        logger.error(f"Error al generar embeddings: {e}")
        raise
        
def vectorize_query_bert(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Genera embedding para una consulta de usuario.
    
    Args:
        query: Texto de consulta
        model: Modelo SentenceTransformer previamente cargado
        
    Returns:
        Vector embedding para la consulta
    """
    if not query:
        return None
        
    # Generar embedding
    query_embedding = model.encode([query])[0]
    return query_embedding.reshape(1, -1)

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normaliza embeddings para que tengan norma unitaria.
    Esto es útil para cálculos de similitud coseno.
    
    Args:
        embeddings: Matriz de embeddings
        
    Returns:
        Embeddings normalizados
    """
    # Normalizar cada embedding a magnitud = 1
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

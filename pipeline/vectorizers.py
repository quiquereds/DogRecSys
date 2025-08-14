"""
Pipeline para vectorizar los datos de entrada
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def vectorize_text_tfidf(df: pd.DataFrame, column: str = "text_combined", max_features: int = 500, 
                     min_df: float = 0.01, max_df: float = 0.95, ngram_range: tuple = (1, 1)):
    """
    Vectoriza una columna de texto utilizando TF-IDF con parámetros optimizados.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene la columna a vectorizar.
        column (str): Nombre de la columna de texto a vectorizar.
        max_features (int): Número máximo de características a extraer.
        min_df (float): Frecuencia mínima de documento para incluir un término (0.01 = 1%).
                       Útil para eliminar términos demasiado raros.
        max_df (float): Frecuencia máxima de documento para incluir un término (0.95 = 95%).
                       Útil para eliminar términos demasiado comunes.
        ngram_range (tuple): Rango de n-gramas a extraer. Por defecto (1, 1) para usar solo palabras individuales.
    
    Returns:
        X_text (scipy.sparse.csr.csr_matrix): Matriz dispersa de características vectorizadas.
        tfidf (TfidfVectorizer): Objeto TfidfVectorizer utilizado para la vectorización.
    """
    # Instanciamos el vectorizador TF-IDF con parámetros optimizados
    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,   # Por defecto (1,1) - solo palabras individuales
        stop_words="english",      # Ya no es tan necesario debido al preprocesamiento avanzado
        use_idf=True,              # Usar IDF para ponderar términos
        smooth_idf=True,           # Suavizar IDF para evitar divisiones por cero
        sublinear_tf=True          # Aplicar escala sublinear a TF (1+log(tf))
    )
    
    # Vectorizamos la columna de texto combinada
    X_text = tfidf.fit_transform(df[column].fillna(""))
    return X_text, tfidf

def vectorize_numerical(df: pd.DataFrame, columns: list[str]):
    """
    Vectoriza las columnas numéricas utilizando StandardScaler.
    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas a vectorizar.
        columns (list): Lista de nombres de columnas numéricas a vectorizar.
    Returns:
        X_num (np.ndarray): Matriz de características numéricas vectorizadas.
        scaler (StandardScaler): Objeto StandardScaler utilizado para la vectorización.
    """
    # Instanciamos el escalador
    scaler = StandardScaler()
    # Escalamos las columnas numéricas
    X_num = scaler.fit_transform(df[columns].fillna(0))
    return X_num, scaler
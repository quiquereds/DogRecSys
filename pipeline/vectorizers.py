"""

Pipeline para vectorizar los datos de entrada

"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def vectorize_text_tfidf(df: pd.DataFrame, column: str = "text_combined", max_features: int = 500):
    """
    Vectoriza una columna de texto utilizando TF-IDF.
    Args:
        df (pd.DataFrame): DataFrame que contiene la columna a vectorizar.
        column (str): Nombre de la columna de texto a vectorizar.
        max_features (int): Número máximo de características a extraer.
    Returns:
        X_text (scipy.sparse.csr.csr_matrix): Matriz dispersa de características vectorizadas.
        tfidf (TfidfVectorizer): Objeto TfidfVectorizer utilizado para la vectorización.
    """
    # Instanciamos el vectorizador TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
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
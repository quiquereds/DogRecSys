"""

Pipeline para preprocesar los datos de entrada para el sistema de recomendación

"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from .text_processing import normalize_text, enrich_text_with_ngrams

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes según el tipo de columna:
    - Columnas categóricas: imputa con la moda (valor más frecuente)
    - Columnas numéricas: imputa con la mediana
    - Columnas de texto: imputa con cadena vacía
    
    Args:
        df (pd.DataFrame): DataFrame con valores faltantes
        
    Returns:
        pd.DataFrame: DataFrame con valores imputados
    """
    df_imputed = df.copy()
    
    # Obtener columnas por tipo
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = ['group', 'grooming_frequency_category', 'shedding_category', 
                        'energy_level_category', 'trainability_category', 'demeanor_category']
    text_cols = ['description', 'temperament']
    
    # Imputar columnas numéricas con la mediana
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df_imputed[col].fillna(median_value, inplace=True)
    
    # Imputar columnas categóricas con la moda
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]  # La moda puede tener múltiples valores, tomamos el primero
            df_imputed[col].fillna(mode_value, inplace=True)
    
    # Imputar columnas de texto con cadena vacía
    for col in text_cols:
        if col in df.columns:
            df_imputed[col].fillna("", inplace=True)
    
    return df_imputed

def normalize_breed_name(name: str) -> str:
    """
    Normaliza el nombre de la raza de un perro (limpia espacios, convierte a minúsculas y reemplaza espacios por guiones bajos).
    Args:
        name (str): Nombre de la raza de perro a normalizar.
    Returns:
        str: Nombre de la raza normalizado.
    """
    return (
        str(name)
            .lower()
            .replace(" ", "_")
            .replace("´", "'")
            .replace("`", "'")
            .replace("'", "'")
            .replace("'", "'")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .strip()
    )
    
def normalize_categorical_text(text: str) -> str:
    """
    Normaliza texto de categorías (limpia espacios, convierte a minúsculas y reemplaza espacios por guiones bajos).
    Similar a normalize_breed_name pero más general para otras columnas categóricas.
    
    Args:
        text (str): Texto categórico a normalizar.
    Returns:
        str: Texto categórico normalizado.
    """
    return (
        str(text)
            .lower()
            .replace(" ", "_")
            .strip()
    )
    
def process_free_text(text: str, enrich_with_ngrams: bool = False) -> str:
    """
    Procesa texto libre para análisis TF-IDF, aplicando normalización y limpieza.
    
    Args:
        text (str): Texto a procesar.
        enrich_with_ngrams (bool): Parámetro mantenido por compatibilidad, pero ignorado.
                                 (ahora siempre es False)
        
    Returns:
        str: Texto procesado listo para TF-IDF.
    """
    # Aplicamos normalización avanzada (elimina stopwords, aplica lematización)
    normalized = normalize_text(text, remove_stopwords=True, apply_lemmatization=True)
    
    # Ya no usamos n-gramas en este procesamiento
    return normalized

def combine_description_and_temperament(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Combina las columnas 'description' y 'temperament' en una nueva columna 'text_combined'.
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos de entrada.
        columns (list[str]): Lista de nombres de columnas de texto a combinar.
    Returns:
        pd.DataFrame: DataFrame con la nueva columna 'description_and_temperament'.
    """
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df["text_combined"] = df[columns].fillna("").agg(" ".join, axis=1)
    return df

def preprocess_df(df: pd.DataFrame, text_columns: list[str], id_column: str = "breed") -> pd.DataFrame:
    """
    Preprocesa el DataFrame de entrada aplicando imputación de valores faltantes y 
    normalización específica por tipo de columna.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos de entrada.
        text_columns (list[str]): Lista de nombres de columnas de texto a normalizar y combinar.
        id_column (str): Nombre de la columna que contiene los identificadores únicos.
    Returns:
        pd.DataFrame: DataFrame preprocesado.
    """
    # Creamos una copia para no modificar el original
    df_processed = df.copy()
    
    # Paso 1: Imputamos valores faltantes según el tipo de columna
    df_processed = impute_missing_values(df_processed)
    
    # Paso 2: Normaliza los nombres de las razas
    df_processed["breed"] = df_processed[id_column].apply(normalize_breed_name)
    
    # Paso 3: Procesamos cada columna según su tipo
    for col in text_columns:
        if col in ["description", "temperament"]:
            # Texto libre: normalizamos sin n-gramas para todas las columnas de texto
            # (ya no es necesario fillna aquí porque se hizo en la imputación)
            df_processed[col] = df_processed[col].apply(
                lambda x: process_free_text(x, enrich_with_ngrams=False)
            )
        elif col in ["group", "grooming_frequency_category", "shedding_category", 
                    "energy_level_category", "trainability_category", "demeanor_category"]:
            # Columnas categóricas: normalizamos formato
            df_processed[col] = df_processed[col].apply(normalize_categorical_text)
    
    # Paso 4: Combinamos las columnas procesadas
    df_processed = combine_description_and_temperament(df_processed, text_columns)
    
    return df_processed


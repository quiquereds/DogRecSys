"""

Pipeline para preprocessar los dadts de entrada

"""

import pandas as pd

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
            .replace("’", "'")
            .replace("‘", "'")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .strip()
    )
    
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
    Preprocesa el DataFrame de entrada.
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos de entrada.
        text_columns (list[str]): Lista de nombres de columnas de texto a combinar.
        id_column (str): Nombre de la columna que contiene los identificadores únicos.
    Returns:
        pd.DataFrame: DataFrame preprocesado.
    """
    # Normaliza los nombres de las razas
    df["breed"] = df[id_column].apply(normalize_breed_name)
    
    # Combina las columnas
    df = combine_description_and_temperament(df, text_columns)
    
    return df


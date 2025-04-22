import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def simulate_user_item_matrix(df: pd.DataFrame, n_users: int = 100, min_ratings: int = 5, max_ratings: int = 15, popularity_col: str = "popularity"):
    """
    Simula una matriz usuario-item donde las razas más populares tienen mayor probabilidad de ser valoradas.

    Args:
        df (pd.DataFrame): DataFrame con al menos columnas 'breed' y 'popularity'.
        n_users (int): Número de usuarios simulados.
        min_ratings (int): Número mínimo de razas valoradas por usuario.
        max_ratings (int): Número máximo de razas valoradas por usuario.
        popularity_col (str): Nombre de la columna de popularidad.

    Returns:
        pd.DataFrame: Matriz usuario-item (usuarios como filas, razas como columnas, valores de 1 a 5).
    """
    # Usamos popularidad como probabilidad
    df = df.copy()
    df["popularity_weight"] = pd.to_numeric(df[popularity_col], errors="coerce").fillna(0) + 1
    # Para evitar pesos cero
    weights = df["popularity_weight"].values
    weights = weights / weights.sum()

    all_breeds = df["breed"].values
    ratings = {}

    for i in range(n_users):
        user_id = f"User{i+1}"
        ratings[user_id] = {}

        n_ratings = np.random.randint(min_ratings, max_ratings + 1)
        selected_breeds = np.random.choice(all_breeds, size=n_ratings, replace=False, p=weights)

        for breed in selected_breeds:
            score = np.random.choice([2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.3])  # Ligero sesgo a positivos
            ratings[user_id][breed] = score

    # Convertir a matriz
    rating_df = pd.DataFrame(index=ratings.keys(), columns=all_breeds).fillna(0)
    for user, user_ratings in ratings.items():
        for breed, score in user_ratings.items():
            rating_df.loc[user, breed] = score

    return rating_df


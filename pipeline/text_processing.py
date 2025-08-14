"""
Módulo para procesamiento avanzado de texto
Incluye funciones para normalizar y limpiar texto para análisis de NLP y TF-IDF
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Descargamos recursos de NLTK si no existen
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

def normalize_text(text: str, 
                  remove_stopwords: bool = True, 
                  apply_stemming: bool = False,
                  apply_lemmatization: bool = True) -> str:
    """
    Normaliza texto libre para análisis NLP (elimina caracteres especiales,
    puntuación, stopwords, normaliza espacios y convierte a minúsculas).
    
    Args:
        text (str): Texto a normalizar.
        remove_stopwords (bool): Si se deben eliminar las stopwords.
        apply_stemming (bool): Si se debe aplicar stemming (no recomendado si se aplica lemmatización).
        apply_lemmatization (bool): Si se debe aplicar lemmatización.
    
    Returns:
        str: Texto normalizado y limpio.
    """
    # Convertir a string por seguridad
    text = str(text)
    if not text.strip():
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar puntuación y caracteres especiales 
    # (mantiene espacios y letras/números)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Eliminar números si no son relevantes
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenizar el texto
    words = nltk.word_tokenize(text)
    
    # Eliminar stopwords si está configurado
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    
    # Aplicar stemming si está configurado
    if apply_stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    
    # Aplicar lemmatización si está configurado
    if apply_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    
    # Unir las palabras nuevamente
    text = ' '.join(words)
    
    # Normalizar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def create_ngrams(text: str, n: int = 2) -> List[str]:
    """
    Crea n-gramas a partir de un texto.
    
    Args:
        text (str): Texto de entrada.
        n (int): Tamaño del n-grama (bigrama=2, trigrama=3, etc).
    
    Returns:
        List[str]: Lista de n-gramas.
    """
    words = text.split()
    ngrams = []
    
    for i in range(len(words) - n + 1):
        ngrams.append('_'.join(words[i:i+n]))
        
    return ngrams

def enrich_text_with_ngrams(text: str, max_ngram: int = 3) -> str:
    """
    Enriquece un texto con n-gramas para mejorar la representación TF-IDF.
    
    Args:
        text (str): Texto normalizado de entrada.
        max_ngram (int): Tamaño máximo del n-grama.
    
    Returns:
        str: Texto enriquecido con n-gramas.
    """
    result = text
    
    # Agregamos bigramas y trigramas significativos
    for n in range(2, max_ngram + 1):
        ngrams = create_ngrams(text, n)
        # Agregamos los n-gramas como términos adicionales
        if ngrams:
            result += ' ' + ' '.join(ngrams)
            
    return result

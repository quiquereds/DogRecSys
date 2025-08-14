"""
Módulo de evaluación avanzada para sistemas de recomendación

Este módulo extiende las métricas básicas con:
- Métricas de diversidad
- Análisis de cobertura
- Evaluación de explicabilidad
- Comparación entre sistemas
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def diversity_intra_list(recommendations: List[str], 
                        item_features: np.ndarray, 
                        item_names: List[str]) -> float:
    """
    Calcula la diversidad intra-lista de recomendaciones.
    
    Args:
        recommendations: Lista de elementos recomendados
        item_features: Matriz de características de todos los elementos
        item_names: Lista de nombres de todos los elementos
        
    Returns:
        Puntuación de diversidad (0-1, mayor es más diverso)
    """
    if len(recommendations) <= 1:
        return 0.0
    
    # Obtener índices de elementos recomendados
    rec_indices = [item_names.index(item) for item in recommendations if item in item_names]
    
    if len(rec_indices) <= 1:
        return 0.0
    
    # Calcular similitudes entre elementos recomendados
    rec_features = item_features[rec_indices]
    similarity_matrix = cosine_similarity(rec_features)
    
    # Calcular diversidad como 1 - promedio de similitudes (excluyendo diagonal)
    n = len(rec_indices)
    total_similarity = np.sum(similarity_matrix) - np.trace(similarity_matrix)
    avg_similarity = total_similarity / (n * (n - 1))
    
    return 1 - avg_similarity


def coverage_analysis(all_recommendations: Dict[str, List[str]], 
                     total_items: List[str]) -> Dict[str, float]:
    """
    Analiza la cobertura del catálogo por las recomendaciones.
    
    Args:
        all_recommendations: Diccionario {usuario: [recomendaciones]}
        total_items: Lista de todos los elementos disponibles
        
    Returns:
        Diccionario con métricas de cobertura
    """
    # Obtener todos los elementos recomendados
    all_recommended = set()
    for recs in all_recommendations.values():
        all_recommended.update(recs)
    
    # Calcular métricas
    catalog_coverage = len(all_recommended) / len(total_items)
    
    # Distribución de popularidad de recomendaciones
    rec_counts = {}
    for recs in all_recommendations.values():
        for item in recs:
            rec_counts[item] = rec_counts.get(item, 0) + 1
    
    # Gini coefficient para medir concentración
    if rec_counts:
        values = sorted(rec_counts.values())
        n = len(values)
        gini = (2 * sum((i + 1) * v for i, v in enumerate(values))) / (n * sum(values)) - (n + 1) / n
    else:
        gini = 0
    
    return {
        'catalog_coverage': catalog_coverage,
        'unique_items_recommended': len(all_recommended),
        'total_items': len(total_items),
        'gini_coefficient': gini,
        'recommendation_concentration': 1 - catalog_coverage
    }


def novelty_score(recommendations: List[str], 
                 popularity_scores: Dict[str, float]) -> float:
    """
    Calcula el score de novedad basado en popularidad inversa.
    
    Args:
        recommendations: Lista de elementos recomendados
        popularity_scores: Diccionario {item: popularity_score}
        
    Returns:
        Score de novedad promedio
    """
    if not recommendations:
        return 0.0
    
    novelty_scores = []
    max_popularity = max(popularity_scores.values()) if popularity_scores else 1
    
    for item in recommendations:
        if item in popularity_scores:
            # Novedad = 1 - (popularidad normalizada)
            novelty = 1 - (popularity_scores[item] / max_popularity)
            novelty_scores.append(novelty)
    
    return np.mean(novelty_scores) if novelty_scores else 0.0


def serendipity_score(recommendations: List[str],
                     expected_categories: List[str],
                     item_categories: Dict[str, str],
                     quality_threshold: float = 0.5) -> float:
    """
    Calcula el score de serendipity (recomendaciones inesperadas pero relevantes).
    
    Args:
        recommendations: Lista de elementos recomendados
        expected_categories: Categorías que el usuario esperaría
        item_categories: Diccionario {item: category}
        quality_threshold: Umbral mínimo de calidad para considerar serendipity
        
    Returns:
        Score de serendipity
    """
    if not recommendations:
        return 0.0
    
    serendipitous_items = 0
    total_items = 0
    
    for item in recommendations:
        if item in item_categories:
            item_category = item_categories[item]
            # Si el item es de una categoría inesperada, es serendipitous
            if item_category not in expected_categories:
                serendipitous_items += 1
            total_items += 1
    
    return serendipitous_items / total_items if total_items > 0 else 0.0


def explanatory_diversity(explanations: List[Dict]) -> Dict[str, float]:
    """
    Analiza la diversidad de las explicaciones proporcionadas.
    
    Args:
        explanations: Lista de explicaciones de recomendaciones
        
    Returns:
        Métricas de diversidad de explicaciones
    """
    if not explanations:
        return {'feature_diversity': 0.0, 'reason_diversity': 0.0}
    
    # Extraer características mencionadas en explicaciones
    all_features = set()
    feature_counts = {}
    
    for exp in explanations:
        if 'matching_attributes' in exp:
            for attr in exp['matching_attributes']:
                feature = attr.split(':')[0]  # Extraer nombre de característica
                all_features.add(feature)
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # Calcular diversidad de características
    n_features = len(all_features)
    feature_diversity = n_features / len(explanations) if explanations else 0
    
    # Calcular entropía de distribución de características
    if feature_counts:
        total_mentions = sum(feature_counts.values())
        entropy = -sum((count/total_mentions) * np.log2(count/total_mentions) 
                      for count in feature_counts.values())
        reason_diversity = entropy / np.log2(len(feature_counts))  # Normalizar
    else:
        reason_diversity = 0.0
    
    return {
        'feature_diversity': feature_diversity,
        'reason_diversity': reason_diversity,
        'unique_features_used': n_features,
        'total_explanations': len(explanations)
    }


def compare_recommendation_systems(system1_recs: Dict[str, List[str]],
                                 system2_recs: Dict[str, List[str]],
                                 system1_name: str = "Sistema 1",
                                 system2_name: str = "Sistema 2") -> Dict:
    """
    Compara dos sistemas de recomendación.
    
    Args:
        system1_recs: Recomendaciones del primer sistema
        system2_recs: Recomendaciones del segundo sistema
        system1_name: Nombre del primer sistema
        system2_name: Nombre del segundo sistema
        
    Returns:
        Diccionario con comparaciones detalladas
    """
    comparison = {
        'systems': [system1_name, system2_name],
        'user_overlap': {},
        'recommendation_overlap': {},
        'unique_recommendations': {}
    }
    
    # Análisis por usuario
    common_users = set(system1_recs.keys()) & set(system2_recs.keys())
    
    for user in common_users:
        recs1 = set(system1_recs[user])
        recs2 = set(system2_recs[user])
        
        overlap = len(recs1 & recs2)
        union = len(recs1 | recs2)
        
        comparison['user_overlap'][user] = {
            'jaccard_similarity': overlap / union if union > 0 else 0,
            'overlap_count': overlap,
            'total_unique': union,
            f'{system1_name}_unique': len(recs1 - recs2),
            f'{system2_name}_unique': len(recs2 - recs1)
        }
    
    # Análisis global
    all_recs1 = set()
    all_recs2 = set()
    
    for recs in system1_recs.values():
        all_recs1.update(recs)
    for recs in system2_recs.values():
        all_recs2.update(recs)
    
    global_overlap = len(all_recs1 & all_recs2)
    global_union = len(all_recs1 | all_recs2)
    
    comparison['global_analysis'] = {
        'jaccard_similarity': global_overlap / global_union if global_union > 0 else 0,
        'overlap_items': global_overlap,
        'total_unique_items': global_union,
        f'{system1_name}_catalog_size': len(all_recs1),
        f'{system2_name}_catalog_size': len(all_recs2),
        f'{system1_name}_unique_items': len(all_recs1 - all_recs2),
        f'{system2_name}_unique_items': len(all_recs2 - all_recs1)
    }
    
    return comparison


def generate_evaluation_report(original_recs: Dict,
                             enhanced_recs: Dict,
                             df: pd.DataFrame,
                             item_features: np.ndarray) -> Dict:
    """
    Genera un reporte completo de evaluación comparando dos sistemas.
    
    Args:
        original_recs: Recomendaciones del sistema original
        enhanced_recs: Recomendaciones del sistema mejorado
        df: DataFrame con información de items
        item_features: Matriz de características de items
        
    Returns:
        Reporte completo de evaluación
    """
    print("📊 Generando reporte de evaluación completo...\n")
    
    # Preparar datos
    all_breeds = df['breed'].tolist()
    popularity_scores = dict(zip(df['breed'], df['popularity']))
    category_mapping = dict(zip(df['breed'], df['group']))
    
    report = {
        'diversity_analysis': {},
        'coverage_analysis': {},
        'novelty_analysis': {},
        'system_comparison': {}
    }
    
    # 1. Análisis de diversidad
    print("1. Analizando diversidad...")
    for system_name, recs_dict in [("Original", original_recs), ("Mejorado", enhanced_recs)]:
        diversities = []
        for user, recs in recs_dict.items():
            # Convertir nombres de breed si es un DataFrame
            if hasattr(recs, 'breed'):
                breed_list = recs['breed'].head(10).tolist()
            else:
                breed_list = recs[:10] if isinstance(recs, list) else recs.head(10).tolist()
            
            diversity = diversity_intra_list(breed_list, item_features, all_breeds)
            diversities.append(diversity)
        
        report['diversity_analysis'][system_name] = {
            'avg_diversity': np.mean(diversities),
            'std_diversity': np.std(diversities),
            'min_diversity': np.min(diversities),
            'max_diversity': np.max(diversities)
        }
    
    # 2. Análisis de cobertura
    print("2. Analizando cobertura...")
    for system_name, recs_dict in [("Original", original_recs), ("Mejorado", enhanced_recs)]:
        # Convertir recomendaciones a formato uniforme
        uniform_recs = {}
        for user, recs in recs_dict.items():
            if hasattr(recs, 'breed'):
                uniform_recs[user] = recs['breed'].head(10).tolist()
            else:
                uniform_recs[user] = recs[:10] if isinstance(recs, list) else recs.head(10).tolist()
        
        coverage = coverage_analysis(uniform_recs, all_breeds)
        report['coverage_analysis'][system_name] = coverage
    
    # 3. Análisis de novedad
    print("3. Analizando novedad...")
    for system_name, recs_dict in [("Original", original_recs), ("Mejorado", enhanced_recs)]:
        novelties = []
        for user, recs in recs_dict.items():
            if hasattr(recs, 'breed'):
                breed_list = recs['breed'].head(10).tolist()
            else:
                breed_list = recs[:10] if isinstance(recs, list) else recs.head(10).tolist()
            
            novelty = novelty_score(breed_list, popularity_scores)
            novelties.append(novelty)
        
        report['novelty_analysis'][system_name] = {
            'avg_novelty': np.mean(novelties),
            'std_novelty': np.std(novelties)
        }
    
    # 4. Comparación entre sistemas
    print("4. Comparando sistemas...")
    # Preparar datos para comparación
    original_uniform = {}
    enhanced_uniform = {}
    
    common_users = set(original_recs.keys()) & set(enhanced_recs.keys())
    
    for user in common_users:
        # Original
        orig_recs = original_recs[user]
        if hasattr(orig_recs, 'breed'):
            original_uniform[user] = orig_recs['breed'].head(10).tolist()
        else:
            original_uniform[user] = orig_recs[:10] if isinstance(orig_recs, list) else orig_recs.head(10).tolist()
        
        # Enhanced
        enh_recs = enhanced_recs[user]
        if hasattr(enh_recs, 'breed'):
            enhanced_uniform[user] = enh_recs['breed'].head(10).tolist()
        else:
            enhanced_uniform[user] = enh_recs[:10] if isinstance(enh_recs, list) else enh_recs.head(10).tolist()
    
    comparison = compare_recommendation_systems(
        original_uniform, enhanced_uniform, "Original", "Mejorado"
    )
    report['system_comparison'] = comparison
    
    print("✅ Reporte generado exitosamente!")
    return report


def visualize_evaluation_report(report: Dict):
    """
    Visualiza el reporte de evaluación.
    
    Args:
        report: Reporte generado por generate_evaluation_report
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Evaluación Comparativa de Sistemas de Recomendación', fontsize=16)
    
    # 1. Diversidad
    ax1 = axes[0, 0]
    systems = list(report['diversity_analysis'].keys())
    diversities = [report['diversity_analysis'][s]['avg_diversity'] for s in systems]
    diversity_std = [report['diversity_analysis'][s]['std_diversity'] for s in systems]
    
    bars1 = ax1.bar(systems, diversities, yerr=diversity_std, capsize=5, 
                    color=['lightcoral', 'lightblue'])
    ax1.set_title('Diversidad Promedio')
    ax1.set_ylabel('Score de Diversidad')
    ax1.set_ylim(0, 1)
    
    # Añadir valores en las barras
    for bar, val in zip(bars1, diversities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 2. Cobertura
    ax2 = axes[0, 1]
    coverages = [report['coverage_analysis'][s]['catalog_coverage'] for s in systems]
    
    bars2 = ax2.bar(systems, coverages, color=['lightcoral', 'lightblue'])
    ax2.set_title('Cobertura del Catálogo')
    ax2.set_ylabel('% del Catálogo Cubierto')
    ax2.set_ylim(0, 1)
    
    for bar, val in zip(bars2, coverages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom')
    
    # 3. Novedad
    ax3 = axes[1, 0]
    novelties = [report['novelty_analysis'][s]['avg_novelty'] for s in systems]
    novelty_std = [report['novelty_analysis'][s]['std_novelty'] for s in systems]
    
    bars3 = ax3.bar(systems, novelties, yerr=novelty_std, capsize=5,
                    color=['lightcoral', 'lightblue'])
    ax3.set_title('Novedad Promedio')
    ax3.set_ylabel('Score de Novedad')
    ax3.set_ylim(0, 1)
    
    for bar, val in zip(bars3, novelties):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 4. Similitud entre sistemas
    ax4 = axes[1, 1]
    if 'global_analysis' in report['system_comparison']:
        global_sim = report['system_comparison']['global_analysis']['jaccard_similarity']
        user_sims = [overlap['jaccard_similarity'] 
                    for overlap in report['system_comparison']['user_overlap'].values()]
        
        ax4.hist(user_sims, bins=10, alpha=0.7, color='lightgreen', 
                label='Similitud por Usuario')
        ax4.axvline(global_sim, color='red', linestyle='--', linewidth=2,
                   label=f'Similitud Global: {global_sim:.3f}')
        ax4.axvline(np.mean(user_sims), color='blue', linestyle='--', linewidth=2,
                   label=f'Promedio: {np.mean(user_sims):.3f}')
        
        ax4.set_title('Similitud entre Sistemas')
        ax4.set_xlabel('Similitud de Jaccard')
        ax4.set_ylabel('Frecuencia')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()


def print_evaluation_summary(report: Dict):
    """
    Imprime un resumen textual del reporte de evaluación.
    
    Args:
        report: Reporte generado por generate_evaluation_report
    """
    print("📋 RESUMEN DE EVALUACIÓN\n")
    print("=" * 50)
    
    # Diversidad
    print("\n🎨 DIVERSIDAD:")
    for system, metrics in report['diversity_analysis'].items():
        print(f"  {system}:")
        print(f"    • Promedio: {metrics['avg_diversity']:.3f}")
        print(f"    • Rango: {metrics['min_diversity']:.3f} - {metrics['max_diversity']:.3f}")
    
    # Cobertura
    print("\n📊 COBERTURA:")
    for system, metrics in report['coverage_analysis'].items():
        print(f"  {system}:")
        print(f"    • Catálogo cubierto: {metrics['catalog_coverage']:.1%}")
        print(f"    • Items únicos: {metrics['unique_items_recommended']}/{metrics['total_items']}")
        print(f"    • Gini coefficient: {metrics['gini_coefficient']:.3f}")
    
    # Novedad
    print("\n✨ NOVEDAD:")
    for system, metrics in report['novelty_analysis'].items():
        print(f"  {system}:")
        print(f"    • Novedad promedio: {metrics['avg_novelty']:.3f}")
        print(f"    • Desviación estándar: {metrics['std_novelty']:.3f}")
    
    # Comparación
    if 'global_analysis' in report['system_comparison']:
        print("\n🔄 COMPARACIÓN ENTRE SISTEMAS:")
        global_analysis = report['system_comparison']['global_analysis']
        print(f"  • Similitud global: {global_analysis['jaccard_similarity']:.3f}")
        print(f"  • Items únicos del mejorado: {global_analysis['Mejorado_unique_items']}")
        print(f"  • Items en común: {global_analysis['overlap_items']}")
    
    print("\n" + "=" * 50)

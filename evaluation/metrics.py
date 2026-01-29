"""
Métricas de avaliação para imputação de valores em falta.

Este módulo contém funções para calcular métricas de qualidade de imputação.
Todas as funções são simples, bem documentadas e testáveis.

Métricas para variáveis numéricas:
    - RMSE: Root Mean Squared Error
    - NRMSE: Normalized RMSE (pelo range)
    - MAE: Mean Absolute Error
    - R2: Coeficiente de determinação
    - Pearson: Correlação de Pearson

Métricas para variáveis categóricas:
    - Accuracy: Proporção de valores correctamente imputados
    - F1: F1-score (para binário)

Autor: Ricardo Vicente
Data: Janeiro 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt(mean((y_true - y_pred)^2))

    Args:
        y_true: Valores verdadeiros (1D array)
        y_pred: Valores previstos/imputados (1D array)

    Returns:
        RMSE value (float). Menor é melhor. Mínimo = 0.

    Raises:
        ValueError: Se arrays vazios ou tamanhos diferentes
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Arrays não podem estar vazios")

    if len(y_true) != len(y_pred):
        raise ValueError(f"Arrays devem ter o mesmo tamanho: {len(y_true)} vs {len(y_pred)}")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Root Mean Squared Error.

    NRMSE = RMSE / (max(y_true) - min(y_true))

    Normaliza o RMSE pelo range dos valores verdadeiros,
    permitindo comparação entre variáveis com escalas diferentes.

    Args:
        y_true: Valores verdadeiros (1D array)
        y_pred: Valores previstos/imputados (1D array)

    Returns:
        NRMSE value (float). Menor é melhor.
        Tipicamente entre 0 e 1, mas pode ser > 1 para erros muito grandes.

    Raises:
        ValueError: Se arrays vazios, tamanhos diferentes, ou range = 0
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Arrays não podem estar vazios")

    if len(y_true) != len(y_pred):
        raise ValueError(f"Arrays devem ter o mesmo tamanho: {len(y_true)} vs {len(y_pred)}")

    value_range = float(np.max(y_true) - np.min(y_true))

    if value_range == 0:
        raise ValueError("Range dos valores verdadeiros é zero (todos iguais)")

    return rmse(y_true, y_pred) / value_range


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    MAE = mean(|y_true - y_pred|)

    Args:
        y_true: Valores verdadeiros (1D array)
        y_pred: Valores previstos/imputados (1D array)

    Returns:
        MAE value (float). Menor é melhor. Mínimo = 0.

    Raises:
        ValueError: Se arrays vazios ou tamanhos diferentes
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Arrays não podem estar vazios")

    if len(y_true) != len(y_pred):
        raise ValueError(f"Arrays devem ter o mesmo tamanho: {len(y_true)} vs {len(y_pred)}")

    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coeficiente de determinação (R²).

    R² = 1 - SS_res / SS_tot

    onde:
        SS_res = sum((y_true - y_pred)^2)  # soma dos quadrados dos resíduos
        SS_tot = sum((y_true - mean(y_true))^2)  # soma total dos quadrados

    Args:
        y_true: Valores verdadeiros (1D array)
        y_pred: Valores previstos/imputados (1D array)

    Returns:
        R² value (float).
        - R² = 1: previsão perfeita
        - R² = 0: equivalente a prever a média
        - R² < 0: pior que prever a média

    Raises:
        ValueError: Se arrays vazios, tamanhos diferentes, ou variância = 0
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Arrays não podem estar vazios")

    if len(y_true) != len(y_pred):
        raise ValueError(f"Arrays devem ter o mesmo tamanho: {len(y_true)} vs {len(y_pred)}")

    if len(y_true) < 2:
        raise ValueError("Necessário pelo menos 2 valores para calcular R²")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        raise ValueError("Variância dos valores verdadeiros é zero (todos iguais)")

    return float(1 - ss_res / ss_tot)


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Correlação de Pearson.

    r = cov(y_true, y_pred) / (std(y_true) * std(y_pred))

    Args:
        y_true: Valores verdadeiros (1D array)
        y_pred: Valores previstos/imputados (1D array)

    Returns:
        Correlação de Pearson (float).
        - r = 1: correlação positiva perfeita
        - r = 0: sem correlação linear
        - r = -1: correlação negativa perfeita

    Raises:
        ValueError: Se arrays vazios, tamanhos diferentes, ou desvio padrão = 0
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Arrays não podem estar vazios")

    if len(y_true) != len(y_pred):
        raise ValueError(f"Arrays devem ter o mesmo tamanho: {len(y_true)} vs {len(y_pred)}")

    if len(y_true) < 2:
        raise ValueError("Necessário pelo menos 2 valores para calcular correlação")

    std_true = np.std(y_true, ddof=0)
    std_pred = np.std(y_pred, ddof=0)

    if std_true == 0:
        raise ValueError("Desvio padrão dos valores verdadeiros é zero")

    if std_pred == 0:
        raise ValueError("Desvio padrão dos valores previstos é zero")

    # Correlação de Pearson usando fórmula directa
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    correlation = covariance / (std_true * std_pred)

    return float(correlation)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy para variáveis categóricas.

    Accuracy = (número de acertos) / (total de previsões)

    Args:
        y_true: Valores verdadeiros (1D array, qualquer tipo)
        y_pred: Valores previstos/imputados (1D array, qualquer tipo)

    Returns:
        Accuracy (float entre 0 e 1). Maior é melhor.

    Raises:
        ValueError: Se arrays vazios ou tamanhos diferentes
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Arrays não podem estar vazios")

    if len(y_true) != len(y_pred):
        raise ValueError(f"Arrays devem ter o mesmo tamanho: {len(y_true)} vs {len(y_pred)}")

    # Comparação element-wise (funciona para strings, números, etc.)
    matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    return float(matches / len(y_true))


def compute_imputation_metrics(
    original: np.ndarray,
    imputed: np.ndarray,
    missing_mask: np.ndarray,
    is_categorical: bool = False
) -> Dict[str, float]:
    """
    Calcula todas as métricas relevantes para uma coluna imputada.

    Compara APENAS os valores que estavam em falta (definidos pela mask).

    Args:
        original: Valores originais completos da coluna (antes de introduzir missings)
        imputed: Valores após imputação
        missing_mask: Máscara booleana (True = era missing)
        is_categorical: Se True, calcula accuracy; se False, métricas numéricas

    Returns:
        Dict com métricas:
            - Se numérico: {'rmse', 'nrmse', 'mae', 'r2', 'pearson', 'n_missing'}
            - Se categórico: {'accuracy', 'n_missing'}

    Raises:
        ValueError: Se não houver valores missing ou arrays inválidos
    """
    original = np.asarray(original)
    imputed = np.asarray(imputed)
    missing_mask = np.asarray(missing_mask, dtype=bool)

    if len(original) != len(imputed) or len(original) != len(missing_mask):
        raise ValueError("Todos os arrays devem ter o mesmo tamanho")

    n_missing = int(np.sum(missing_mask))

    if n_missing == 0:
        raise ValueError("Não há valores missing (mask é toda False)")

    # Extrair apenas os valores que estavam em falta
    y_true = original[missing_mask]
    y_pred = imputed[missing_mask]

    if is_categorical:
        return {
            'accuracy': accuracy(y_true, y_pred),
            'n_missing': n_missing
        }
    else:
        # Converter para float para métricas numéricas
        y_true = y_true.astype(np.float64)
        y_pred = y_pred.astype(np.float64)

        metrics = {'n_missing': n_missing}

        try:
            metrics['rmse'] = rmse(y_true, y_pred)
        except ValueError:
            metrics['rmse'] = np.nan

        try:
            metrics['nrmse'] = nrmse(y_true, y_pred)
        except ValueError:
            metrics['nrmse'] = np.nan

        try:
            metrics['mae'] = mae(y_true, y_pred)
        except ValueError:
            metrics['mae'] = np.nan

        try:
            metrics['r2'] = r2_score(y_true, y_pred)
        except ValueError:
            metrics['r2'] = np.nan

        try:
            metrics['pearson'] = pearson_correlation(y_true, y_pred)
        except ValueError:
            metrics['pearson'] = np.nan

        return metrics


def aggregate_metrics(metrics_list: list) -> Dict[str, float]:
    """
    Agrega métricas de múltiplas colunas/experiências.

    Calcula média e desvio padrão para cada métrica.

    Args:
        metrics_list: Lista de dicts com métricas

    Returns:
        Dict com métricas agregadas: {metric_name: mean, metric_name_std: std}
    """
    if not metrics_list:
        return {}

    # Identificar todas as métricas presentes
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    result = {}

    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]

        if values:
            result[key] = float(np.mean(values))
            if len(values) > 1:
                result[f'{key}_std'] = float(np.std(values, ddof=1))
            else:
                result[f'{key}_std'] = 0.0
        else:
            result[key] = np.nan
            result[f'{key}_std'] = np.nan

    return result

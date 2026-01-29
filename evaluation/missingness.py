"""
Funções para introduzir valores em falta em datasets.

Este módulo implementa os três padrões principais de missingness:
- MCAR (Missing Completely At Random): probabilidade independente dos dados
- MAR (Missing At Random): probabilidade depende de valores observados
- MNAR (Missing Not At Random): probabilidade depende do próprio valor

Todas as funções:
- São determinísticas (controladas por random_state)
- Retornam o dataset modificado E a máscara de missings
- Garantem que nenhuma linha/coluna fica 100% vazia
- São simples e testáveis

Autor: Ricardo Vicente
Data: Janeiro 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional


def introduce_mcar(
    data: Union[np.ndarray, pd.DataFrame],
    missing_rate: float,
    random_state: int = 42
) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
    """
    Introduz valores em falta com padrão MCAR (Missing Completely At Random).

    Em MCAR, a probabilidade de um valor estar em falta é igual para todos
    os valores, independentemente de qualquer outra variável.

    Args:
        data: Dataset original (numpy array ou DataFrame)
        missing_rate: Proporção de valores a remover (entre 0 e 1)
        random_state: Seed para reprodutibilidade

    Returns:
        Tuple contendo:
            - data_missing: Dataset com valores em falta (NaN)
            - mask: Máscara booleana (True = valor foi removido)

    Raises:
        ValueError: Se missing_rate não está entre 0 e 1, ou data está vazio

    Exemplo:
        >>> data = np.array([[1, 2], [3, 4], [5, 6]])
        >>> data_missing, mask = introduce_mcar(data, 0.3, random_state=42)
        >>> print(mask.sum() / mask.size)  # ~0.3
    """
    # Validação
    if not 0 <= missing_rate < 1:
        raise ValueError(f"missing_rate deve estar entre 0 e 1, recebido: {missing_rate}")

    is_dataframe = isinstance(data, pd.DataFrame)

    if is_dataframe:
        values = data.values.copy()
        original_columns = data.columns
        original_index = data.index
    else:
        values = np.array(data, dtype=np.float64).copy()

    n_rows, n_cols = values.shape

    if n_rows == 0 or n_cols == 0:
        raise ValueError("Dataset não pode estar vazio")

    # Gerar máscara MCAR
    rng = np.random.RandomState(random_state)
    mask = rng.random((n_rows, n_cols)) < missing_rate

    # Garantir pelo menos 1 valor por linha
    for i in range(n_rows):
        if mask[i].all():
            # Escolher uma coluna aleatória para manter
            keep_idx = rng.randint(n_cols)
            mask[i, keep_idx] = False

    # Garantir pelo menos 1 valor por coluna
    for j in range(n_cols):
        if mask[:, j].all():
            # Escolher uma linha aleatória para manter
            keep_idx = rng.randint(n_rows)
            mask[keep_idx, j] = False

    # Aplicar missings
    values = values.astype(np.float64)
    values[mask] = np.nan

    if is_dataframe:
        data_missing = pd.DataFrame(values, columns=original_columns, index=original_index)
    else:
        data_missing = values

    return data_missing, mask


def introduce_mar(
    data: Union[np.ndarray, pd.DataFrame],
    missing_rate: float,
    random_state: int = 42,
    driver_col: int = 0
) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
    """
    Introduz valores em falta com padrão MAR (Missing At Random).

    Em MAR, a probabilidade de um valor estar em falta depende de outros
    valores OBSERVADOS. Aqui, usamos uma coluna "driver" para determinar
    a probabilidade: valores acima da mediana do driver têm maior probabilidade
    de ter missings nas outras colunas.

    Args:
        data: Dataset original (numpy array ou DataFrame)
        missing_rate: Proporção alvo de valores a remover (entre 0 e 1)
        random_state: Seed para reprodutibilidade
        driver_col: Índice da coluna que determina a probabilidade (não terá missings)

    Returns:
        Tuple contendo:
            - data_missing: Dataset com valores em falta (NaN)
            - mask: Máscara booleana (True = valor foi removido)

    Raises:
        ValueError: Se parâmetros inválidos

    Nota:
        A taxa real de missings pode diferir ligeiramente da taxa alvo
        devido ao mecanismo MAR e às garantias de não ter linhas/colunas vazias.
    """
    # Validação
    if not 0 <= missing_rate < 1:
        raise ValueError(f"missing_rate deve estar entre 0 e 1, recebido: {missing_rate}")

    is_dataframe = isinstance(data, pd.DataFrame)

    if is_dataframe:
        values = data.values.copy()
        original_columns = data.columns
        original_index = data.index
    else:
        values = np.array(data, dtype=np.float64).copy()

    n_rows, n_cols = values.shape

    if n_rows == 0 or n_cols == 0:
        raise ValueError("Dataset não pode estar vazio")

    if not 0 <= driver_col < n_cols:
        raise ValueError(f"driver_col deve estar entre 0 e {n_cols-1}, recebido: {driver_col}")

    # Usar driver_col para determinar probabilidades
    driver_values = values[:, driver_col].astype(np.float64)
    driver_median = np.nanmedian(driver_values)

    # Probabilidades: alta (1.5x) para valores acima da mediana, baixa (0.5x) para abaixo
    # Isto mantém a taxa média aproximadamente igual a missing_rate
    prob_high = missing_rate * 1.5
    prob_low = missing_rate * 0.5

    # Limitar probabilidades a [0, 0.95]
    prob_high = min(prob_high, 0.95)
    prob_low = max(prob_low, 0.0)

    rng = np.random.RandomState(random_state)
    mask = np.zeros((n_rows, n_cols), dtype=bool)

    for i in range(n_rows):
        for j in range(n_cols):
            if j == driver_col:
                # Nunca introduzir missing no driver
                continue

            # Probabilidade depende do valor do driver
            if driver_values[i] > driver_median:
                prob = prob_high
            else:
                prob = prob_low

            if rng.random() < prob:
                mask[i, j] = True

    # Garantir pelo menos 1 valor por linha
    for i in range(n_rows):
        if mask[i].all():
            keep_idx = rng.randint(n_cols)
            mask[i, keep_idx] = False

    # Garantir pelo menos 1 valor por coluna (excepto driver que já não tem missings)
    for j in range(n_cols):
        if mask[:, j].all():
            keep_idx = rng.randint(n_rows)
            mask[keep_idx, j] = False

    # Aplicar missings
    values = values.astype(np.float64)
    values[mask] = np.nan

    if is_dataframe:
        data_missing = pd.DataFrame(values, columns=original_columns, index=original_index)
    else:
        data_missing = values

    return data_missing, mask


def introduce_mnar(
    data: Union[np.ndarray, pd.DataFrame],
    missing_rate: float,
    random_state: int = 42
) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
    """
    Introduz valores em falta com padrão MNAR (Missing Not At Random).

    Em MNAR, a probabilidade de um valor estar em falta depende do próprio
    valor que está em falta. Aqui, valores acima da mediana de cada coluna
    têm maior probabilidade de estar em falta.

    Args:
        data: Dataset original (numpy array ou DataFrame)
        missing_rate: Proporção alvo de valores a remover (entre 0 e 1)
        random_state: Seed para reprodutibilidade

    Returns:
        Tuple contendo:
            - data_missing: Dataset com valores em falta (NaN)
            - mask: Máscara booleana (True = valor foi removido)

    Raises:
        ValueError: Se parâmetros inválidos

    Nota:
        Este padrão é o mais difícil de imputar porque a informação
        sobre os valores em falta está relacionada com os próprios valores.
    """
    # Validação
    if not 0 <= missing_rate < 1:
        raise ValueError(f"missing_rate deve estar entre 0 e 1, recebido: {missing_rate}")

    is_dataframe = isinstance(data, pd.DataFrame)

    if is_dataframe:
        values = data.values.copy()
        original_columns = data.columns
        original_index = data.index
    else:
        values = np.array(data, dtype=np.float64).copy()

    n_rows, n_cols = values.shape

    if n_rows == 0 or n_cols == 0:
        raise ValueError("Dataset não pode estar vazio")

    # Probabilidades baseadas no valor
    prob_high = min(missing_rate * 1.5, 0.95)
    prob_low = max(missing_rate * 0.5, 0.0)

    rng = np.random.RandomState(random_state)
    mask = np.zeros((n_rows, n_cols), dtype=bool)

    for j in range(n_cols):
        col_values = values[:, j].astype(np.float64)
        col_median = np.nanmedian(col_values)

        for i in range(n_rows):
            # Probabilidade depende do próprio valor
            if col_values[i] > col_median:
                prob = prob_high
            else:
                prob = prob_low

            if rng.random() < prob:
                mask[i, j] = True

    # Garantir pelo menos 1 valor por linha
    for i in range(n_rows):
        if mask[i].all():
            keep_idx = rng.randint(n_cols)
            mask[i, keep_idx] = False

    # Garantir pelo menos 1 valor por coluna
    for j in range(n_cols):
        if mask[:, j].all():
            keep_idx = rng.randint(n_rows)
            mask[keep_idx, j] = False

    # Aplicar missings
    values = values.astype(np.float64)
    values[mask] = np.nan

    if is_dataframe:
        data_missing = pd.DataFrame(values, columns=original_columns, index=original_index)
    else:
        data_missing = values

    return data_missing, mask


def get_missing_stats(mask: np.ndarray) -> dict:
    """
    Calcula estatísticas sobre os valores em falta.

    Args:
        mask: Máscara booleana (True = missing)

    Returns:
        Dict com estatísticas:
            - total_missing: número total de missings
            - total_cells: número total de células
            - missing_rate: proporção de missings
            - missing_per_row: array com contagem por linha
            - missing_per_col: array com contagem por coluna
            - rows_with_missing: número de linhas com pelo menos 1 missing
            - cols_with_missing: número de colunas com pelo menos 1 missing
    """
    mask = np.asarray(mask, dtype=bool)
    n_rows, n_cols = mask.shape

    total_missing = int(mask.sum())
    total_cells = mask.size

    return {
        'total_missing': total_missing,
        'total_cells': total_cells,
        'missing_rate': total_missing / total_cells if total_cells > 0 else 0.0,
        'missing_per_row': mask.sum(axis=1),
        'missing_per_col': mask.sum(axis=0),
        'rows_with_missing': int((mask.sum(axis=1) > 0).sum()),
        'cols_with_missing': int((mask.sum(axis=0) > 0).sum()),
        'n_rows': n_rows,
        'n_cols': n_cols
    }

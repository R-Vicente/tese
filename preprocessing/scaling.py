import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import hashlib


def _compute_data_hash(data: pd.DataFrame) -> str:
    """Calcula hash do conteúdo do DataFrame para cache key."""
    # Usar uma combinação de shape, columns e amostra dos dados
    # Não usar todos os dados para performance
    hash_input = f"{data.shape}_{list(data.columns)}_{data.iloc[:min(10, len(data))].values.tobytes()}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_scaled_data(data: pd.DataFrame, mixed_handler, cache=None, force_refit: bool = False,
                    scaling_method: str = "standard"):
    """
    Escala os dados numéricos e normaliza categóricas.

    Args:
        data: DataFrame com dados (pode conter NaN)
        mixed_handler: MixedDataHandler com informação de tipos
        cache: Dicionário para cache (opcional)
        force_refit: Se True, ignora cache e recalcula
        scaling_method: Método de scaling para numéricas:
            - "standard": StandardScaler (z-score) - default
            - "minmax": MinMaxScaler [0, 1]
            - "robust": RobustScaler (usa mediana e IQR)
            - "none": Sem scaling (dados já normalizados)

    Returns:
        DataFrame com dados escalados
    """
    # Cache key baseada em hash do conteúdo, não em id()
    cache_key = (_compute_data_hash(data), scaling_method)

    if cache is not None and cache.get("key") == cache_key and not force_refit:
        return cache["data"]

    scaled_data = data.copy()

    # Escalar colunas numéricas (se não for 'none')
    if len(mixed_handler.numeric_cols) > 0 and scaling_method != "none":
        numeric_data = data[mixed_handler.numeric_cols].copy()

        # Criar scaler apropriado
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"scaling_method inválido: {scaling_method}. "
                           f"Usar 'standard', 'minmax', 'robust' ou 'none'.")

        # Fit apenas nos valores observados (sem NaN)
        # Para cada coluna, calcular estatísticas apenas nos valores não-NaN
        for col in mixed_handler.numeric_cols:
            col_data = numeric_data[col].values
            observed_mask = ~np.isnan(col_data)

            if observed_mask.sum() > 0:
                observed_values = col_data[observed_mask].reshape(-1, 1)

                # Fit no subconjunto observado
                col_scaler = type(scaler)()
                col_scaler.fit(observed_values)

                # Transform todos os valores (NaN permanece NaN)
                scaled_col = col_data.copy()
                scaled_col[observed_mask] = col_scaler.transform(observed_values).ravel()
                scaled_data[col] = scaled_col

    # Normalizar colunas nominais para [0, 1]
    for col in mixed_handler.nominal_cols:
        n_categories = mixed_handler.nominal_mappings[col]['n_categories']
        if n_categories > 1:
            # Os dados já estão em códigos inteiros (0, 1, 2, ...)
            # Normalizar para [0, 1]
            col_data = data[col].values.astype(float)
            scaled_data[col] = col_data / (n_categories - 1)
        else:
            scaled_data[col] = 0.0

    # Actualizar cache
    if cache is not None:
        cache["data"] = scaled_data
        cache["key"] = cache_key

    return scaled_data


def compute_range_factors(data: pd.DataFrame, scaled_data: pd.DataFrame, mixed_handler, verbose: bool = False):
    """
    Calcula factores de range para normalização de distâncias.

    Para numéricas: 1/range empírico (nos dados scaled)
    Para categóricas: 1.0 (já normalizadas)
    """
    n_features = len(data.columns)
    range_factors = np.ones(n_features)

    if verbose:
        print("\n=== DIAGNÓSTICO: RANGE FACTORS (EMPÍRICO) ===")

    for idx, col in enumerate(data.columns):
        if col in mixed_handler.numeric_cols:
            col_values = scaled_data[col].dropna()
            if len(col_values) > 1:
                min_val = col_values.min()
                max_val = col_values.max()
                empirical_range = max_val - min_val
                if empirical_range > 1e-6:
                    range_factors[idx] = 1.0 / empirical_range
                    if verbose:
                        print(f"  {col}: range=[{min_val:.2f}, {max_val:.2f}] = {empirical_range:.2f}")
                else:
                    # Coluna constante: não contribui para distância
                    range_factors[idx] = 0.0
                    if verbose:
                        print(f"  {col}: CONSTANTE (range_factor=0.0, ignorada)")
            else:
                # Apenas 1 valor: não contribui para distância
                range_factors[idx] = 0.0
                if verbose:
                    print(f"  {col}: APENAS 1 VALOR (range_factor=0.0, ignorada)")
        else:
            if verbose:
                tipo = 'binary' if col in mixed_handler.binary_cols else 'ordinal' if col in mixed_handler.ordinal_cols else 'nominal'
                print(f"  {col}: {tipo} (range_factor=1.0)")

    if verbose:
        print("=== FIM DIAGNÓSTICO ===\n")

    return range_factors

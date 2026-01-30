"""
ISCA-k v2 - Imputador com pesos baseados em Mutual Information.

Versão actual: KNN com pesos MI
- Calcula MI entre todas as colunas
- Usa MI como peso para distâncias
- Imputa usando vizinhos mais próximos
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Tuple


class ISCAkImputer:
    """
    Imputador ISCA-k com pesos baseados em Mutual Information.

    Versão actual:
    - [x] Pesos MI (via correlação Spearman)
    - [ ] PDS (Partial Distance Strategy)
    - [ ] k adaptativo
    """

    def __init__(self,
                 n_neighbors: int = 5,
                 use_mi_weights: bool = True,
                 min_samples_mi: int = 10,
                 verbose: bool = False):
        """
        Args:
            n_neighbors: Número de vizinhos para imputação
            use_mi_weights: Se True, usa pesos MI nas distâncias
            min_samples_mi: Mínimo de amostras para calcular MI
            verbose: Mostrar progresso
        """
        self.n_neighbors = n_neighbors
        self.use_mi_weights = use_mi_weights
        self.min_samples_mi = min_samples_mi
        self.verbose = verbose

        self.mi_matrix_: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None
        self.columns_: Optional[list] = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[ISCA-k] {msg}")

    def _calculate_mi_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de MI usando correlação de Spearman.

        Spearman é uma aproximação rápida de MI para relações monótonas.
        MI_approx = |rho_spearman|

        Args:
            data: Array (n_samples, n_features)

        Returns:
            Matriz (n_features, n_features) com MI aproximado
        """
        n_features = data.shape[1]
        mi_matrix = np.eye(n_features)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Máscara para valores válidos em ambas as colunas
                mask = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
                n_valid = mask.sum()

                if n_valid >= self.min_samples_mi:
                    try:
                        corr, _ = spearmanr(data[mask, i], data[mask, j])
                        mi_approx = abs(corr) if np.isfinite(corr) else 0.0
                    except:
                        mi_approx = 0.0
                else:
                    mi_approx = 0.0

                mi_matrix[i, j] = mi_approx
                mi_matrix[j, i] = mi_approx

        return mi_matrix

    def _weighted_distance(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           weights: np.ndarray) -> float:
        """
        Calcula distância euclidiana ponderada.

        d(x, y) = sqrt(sum(w_i * (x_i - y_i)^2))

        Apenas considera features onde ambos têm valores.

        Args:
            x, y: Vectores de features
            weights: Pesos por feature

        Returns:
            Distância (ou np.inf se não há overlap)
        """
        # Máscara de valores válidos
        valid = ~(np.isnan(x) | np.isnan(y))

        if valid.sum() == 0:
            return np.inf

        diff_sq = (x[valid] - y[valid]) ** 2
        w = weights[valid]

        # Normalizar pesos
        w_sum = w.sum()
        if w_sum == 0:
            w = np.ones_like(w)
            w_sum = w.sum()

        return np.sqrt(np.sum(w * diff_sq) / w_sum * len(weights))

    def _find_neighbors(self,
                        target_row: int,
                        target_col: int,
                        data_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encontra os k vizinhos mais próximos para imputar uma célula.

        Args:
            target_row: Índice da linha com missing
            target_col: Índice da coluna com missing
            data_scaled: Dados normalizados

        Returns:
            (indices, distances) dos k vizinhos
        """
        n_samples = data_scaled.shape[0]
        target = data_scaled[target_row]

        # Pesos MI para a coluna target
        if self.use_mi_weights and self.mi_matrix_ is not None:
            weights = self.mi_matrix_[target_col]
        else:
            weights = np.ones(data_scaled.shape[1])

        distances = []
        indices = []

        for i in range(n_samples):
            if i == target_row:
                continue

            # Só considerar linhas que têm valor na coluna target
            if np.isnan(data_scaled[i, target_col]):
                continue

            d = self._weighted_distance(target, data_scaled[i], weights)

            if np.isfinite(d):
                distances.append(d)
                indices.append(i)

        if len(distances) == 0:
            return np.array([]), np.array([])

        # Ordenar por distância
        distances = np.array(distances)
        indices = np.array(indices)
        order = np.argsort(distances)

        # Retornar top k
        k = min(self.n_neighbors, len(order))
        return indices[order[:k]], distances[order[:k]]

    def _impute_cell(self,
                     target_row: int,
                     target_col: int,
                     data_scaled: np.ndarray,
                     data_original: np.ndarray) -> float:
        """
        Imputa uma célula usando média ponderada dos vizinhos.

        Peso do vizinho = 1 / (distance + epsilon)

        Args:
            target_row, target_col: Posição da célula
            data_scaled: Dados normalizados (para calcular distâncias)
            data_original: Dados originais (para obter valores)

        Returns:
            Valor imputado na escala original
        """
        neighbor_idx, neighbor_dist = self._find_neighbors(
            target_row, target_col, data_scaled
        )

        if len(neighbor_idx) == 0:
            # Fallback: média da coluna
            col_values = data_original[:, target_col]
            return np.nanmean(col_values)

        # Valores dos vizinhos na coluna target
        neighbor_values = data_original[neighbor_idx, target_col]

        # Pesos inversamente proporcionais à distância
        epsilon = 1e-10
        weights = 1.0 / (neighbor_dist + epsilon)
        weights = weights / weights.sum()

        return np.sum(weights * neighbor_values)

    def fit(self, data: np.ndarray) -> 'ISCAkImputer':
        """
        Ajusta o imputador (calcula MI e scaler).

        Args:
            data: Array (n_samples, n_features)
        """
        self._log(f"Fitting com {data.shape[0]} amostras, {data.shape[1]} features")

        # Normalizar dados (ignorando NaN)
        self.scaler_ = StandardScaler()

        # Calcular média e std ignorando NaN
        col_means = np.nanmean(data, axis=0)
        col_stds = np.nanstd(data, axis=0)
        col_stds[col_stds == 0] = 1.0  # Evitar divisão por zero

        self.scaler_.mean_ = col_means
        self.scaler_.scale_ = col_stds
        self.scaler_.var_ = col_stds ** 2
        self.scaler_.n_features_in_ = data.shape[1]

        # Calcular MI
        if self.use_mi_weights:
            self._log("Calculando matriz MI...")
            self.mi_matrix_ = self._calculate_mi_matrix(data)
            self._log(f"MI matrix shape: {self.mi_matrix_.shape}")

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Imputa valores em falta.

        Args:
            data: Array (n_samples, n_features) com NaN

        Returns:
            Array imputado
        """
        data = data.copy()
        n_samples, n_features = data.shape

        # Normalizar para cálculo de distâncias
        data_scaled = (data - self.scaler_.mean_) / self.scaler_.scale_

        # Encontrar células com missing
        missing_mask = np.isnan(data)
        missing_rows, missing_cols = np.where(missing_mask)
        n_missing = len(missing_rows)

        self._log(f"Imputando {n_missing} valores em falta...")

        # Ordenar por coluna (imputar colunas com menos missing primeiro)
        col_missing_count = missing_mask.sum(axis=0)
        col_order = np.argsort(col_missing_count)

        imputed_count = 0
        for col in col_order:
            col_missing_rows = np.where(missing_mask[:, col])[0]

            for row in col_missing_rows:
                imputed_value = self._impute_cell(
                    row, col, data_scaled, data
                )
                data[row, col] = imputed_value

                # Actualizar dados normalizados
                data_scaled[row, col] = (imputed_value - self.scaler_.mean_[col]) / self.scaler_.scale_[col]

                imputed_count += 1
                if self.verbose and imputed_count % 100 == 0:
                    self._log(f"  {imputed_count}/{n_missing} imputados")

        self._log(f"Imputação concluída: {imputed_count} valores")
        return data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit e transform em um passo."""
        return self.fit(data).transform(data)

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Interface para DataFrames.

        Args:
            data: DataFrame com NaN

        Returns:
            DataFrame imputado
        """
        self.columns_ = list(data.columns)

        # Converter para array
        arr = data.values.astype(float)

        # Imputar
        arr_imputed = self.fit_transform(arr)

        # Reconstruir DataFrame
        return pd.DataFrame(arr_imputed, index=data.index, columns=data.columns)


# =============================================================================
# TESTE
# =============================================================================
if __name__ == "__main__":
    print("Testando ISCAkImputer...\n")

    np.random.seed(42)

    # Criar dados de teste
    n = 100
    data = np.column_stack([
        np.random.randn(n),           # x1
        np.random.randn(n) * 2 + 1,   # x2
        np.random.randn(n) * 0.5,     # x3
    ])

    # Adicionar correlação
    data[:, 1] = data[:, 0] * 0.8 + data[:, 1] * 0.2  # x2 correlacionado com x1

    # Guardar cópia original
    data_original = data.copy()

    # Introduzir missings
    missing_rate = 0.2
    mask = np.random.random(data.shape) < missing_rate
    for i in range(n):
        if mask[i].all():
            mask[i, np.random.randint(3)] = False
    data[mask] = np.nan

    print(f"Dados: {data.shape}")
    print(f"Missings: {np.isnan(data).sum()} ({np.isnan(data).mean():.1%})")

    # Testar com MI
    print("\n--- COM pesos MI ---")
    imputer_mi = ISCAkImputer(n_neighbors=5, use_mi_weights=True, verbose=True)
    data_imputed_mi = imputer_mi.fit_transform(data.copy())

    # Calcular erro
    error_mi = np.sqrt(np.mean((data_original[mask] - data_imputed_mi[mask]) ** 2))
    print(f"RMSE (com MI): {error_mi:.4f}")

    # Testar sem MI
    print("\n--- SEM pesos MI ---")
    imputer_no_mi = ISCAkImputer(n_neighbors=5, use_mi_weights=False, verbose=True)
    data_imputed_no_mi = imputer_no_mi.fit_transform(data.copy())

    error_no_mi = np.sqrt(np.mean((data_original[mask] - data_imputed_no_mi[mask]) ** 2))
    print(f"RMSE (sem MI): {error_no_mi:.4f}")

    # Comparação
    print(f"\nDiferença: {error_no_mi - error_mi:+.4f}")
    if error_mi < error_no_mi:
        print("✓ MI melhorou a imputação")
    else:
        print("✗ MI não melhorou (ou igual)")

    # Mostrar matriz MI
    print("\nMatriz MI:")
    print(imputer_mi.mi_matrix_.round(3))

"""
Testes unitários para evaluation/metrics.py

Cada métrica é testada com:
1. Casos conhecidos (valores calculados manualmente)
2. Casos limite (edge cases)
3. Tratamento de erros
4. Comparação com implementações de referência (sklearn/scipy)

Execução: python -m pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from evaluation.metrics import (
    rmse,
    nrmse,
    mae,
    r2_score,
    pearson_correlation,
    accuracy,
    compute_imputation_metrics,
    aggregate_metrics
)


class TestRMSE:
    """Testes para Root Mean Squared Error."""

    def test_perfect_prediction(self):
        """RMSE deve ser 0 para previsões perfeitas."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rmse(y_true, y_pred) == 0.0

    def test_known_value(self):
        """RMSE com valor calculado manualmente."""
        # Erros: [1, 1, 1, 1] -> MSE = 1 -> RMSE = 1
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0])
        assert rmse(y_true, y_pred) == 1.0

    def test_known_value_2(self):
        """RMSE com valor mais complexo."""
        # Erros: [0, 2, 0, 2] -> Erros²: [0, 4, 0, 4] -> MSE = 2 -> RMSE = sqrt(2)
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 4.0, 3.0, 6.0])
        expected = np.sqrt(2.0)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10

    def test_single_value(self):
        """RMSE com um único valor."""
        y_true = np.array([5.0])
        y_pred = np.array([3.0])
        assert rmse(y_true, y_pred) == 2.0

    def test_empty_array_raises(self):
        """RMSE deve levantar erro para arrays vazios."""
        with pytest.raises(ValueError, match="vazios"):
            rmse(np.array([]), np.array([]))

    def test_different_lengths_raises(self):
        """RMSE deve levantar erro para arrays de tamanhos diferentes."""
        with pytest.raises(ValueError, match="mesmo tamanho"):
            rmse(np.array([1, 2, 3]), np.array([1, 2]))

    def test_matches_sklearn(self):
        """RMSE deve corresponder à implementação do sklearn."""
        from sklearn.metrics import mean_squared_error
        y_true = np.array([1.5, 2.3, 3.7, 4.2, 5.1])
        y_pred = np.array([1.4, 2.5, 3.5, 4.0, 5.5])
        sklearn_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        assert abs(rmse(y_true, y_pred) - sklearn_rmse) < 1e-10


class TestNRMSE:
    """Testes para Normalized Root Mean Squared Error."""

    def test_perfect_prediction(self):
        """NRMSE deve ser 0 para previsões perfeitas."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nrmse(y_true, y_pred) == 0.0

    def test_known_value(self):
        """NRMSE com valor calculado manualmente."""
        # Range = 5 - 1 = 4
        # RMSE = 1 (do teste anterior)
        # NRMSE = 1/4 = 0.25
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        expected = 1.0 / 4.0  # RMSE=1, range=4
        assert abs(nrmse(y_true, y_pred) - expected) < 1e-10

    def test_zero_range_raises(self):
        """NRMSE deve levantar erro quando todos os valores são iguais."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="Range.*zero"):
            nrmse(y_true, y_pred)

    def test_empty_array_raises(self):
        """NRMSE deve levantar erro para arrays vazios."""
        with pytest.raises(ValueError, match="vazios"):
            nrmse(np.array([]), np.array([]))


class TestMAE:
    """Testes para Mean Absolute Error."""

    def test_perfect_prediction(self):
        """MAE deve ser 0 para previsões perfeitas."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mae(y_true, y_pred) == 0.0

    def test_known_value(self):
        """MAE com valor calculado manualmente."""
        # Erros absolutos: [1, 2, 3] -> MAE = 2
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 4.0, 6.0])
        assert mae(y_true, y_pred) == 2.0

    def test_symmetric(self):
        """MAE deve ser simétrico (erro positivo = erro negativo)."""
        y_true = np.array([5.0, 5.0])
        y_pred1 = np.array([3.0, 7.0])  # Erros: -2, +2
        y_pred2 = np.array([7.0, 3.0])  # Erros: +2, -2
        assert mae(y_true, y_pred1) == mae(y_true, y_pred2) == 2.0

    def test_matches_sklearn(self):
        """MAE deve corresponder à implementação do sklearn."""
        from sklearn.metrics import mean_absolute_error
        y_true = np.array([1.5, 2.3, 3.7, 4.2, 5.1])
        y_pred = np.array([1.4, 2.5, 3.5, 4.0, 5.5])
        assert abs(mae(y_true, y_pred) - mean_absolute_error(y_true, y_pred)) < 1e-10


class TestR2Score:
    """Testes para coeficiente de determinação R²."""

    def test_perfect_prediction(self):
        """R² deve ser 1 para previsões perfeitas."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2_score(y_true, y_pred) == 1.0

    def test_mean_prediction(self):
        """R² deve ser 0 quando previsão é a média."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # média de y_true
        assert abs(r2_score(y_true, y_pred)) < 1e-10

    def test_worse_than_mean(self):
        """R² deve ser negativo quando pior que prever a média."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])  # muito longe
        assert r2_score(y_true, y_pred) < 0

    def test_constant_true_raises(self):
        """R² deve levantar erro quando y_true é constante."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="[Vv]ariância.*zero"):
            r2_score(y_true, y_pred)

    def test_single_value_raises(self):
        """R² deve levantar erro com apenas 1 valor."""
        with pytest.raises(ValueError, match="pelo menos 2"):
            r2_score(np.array([1.0]), np.array([1.0]))

    def test_matches_sklearn(self):
        """R² deve corresponder à implementação do sklearn."""
        from sklearn.metrics import r2_score as sklearn_r2
        y_true = np.array([1.5, 2.3, 3.7, 4.2, 5.1])
        y_pred = np.array([1.4, 2.5, 3.5, 4.0, 5.5])
        assert abs(r2_score(y_true, y_pred) - sklearn_r2(y_true, y_pred)) < 1e-10


class TestPearsonCorrelation:
    """Testes para correlação de Pearson."""

    def test_perfect_positive_correlation(self):
        """Correlação deve ser 1 para relação linear positiva perfeita."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x
        assert abs(pearson_correlation(y_true, y_pred) - 1.0) < 1e-10

    def test_perfect_negative_correlation(self):
        """Correlação deve ser -1 para relação linear negativa perfeita."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # inverso
        assert abs(pearson_correlation(y_true, y_pred) - (-1.0)) < 1e-10

    def test_no_correlation(self):
        """Correlação deve ser ~0 para dados não correlacionados."""
        # Construir dados ortogonais
        y_true = np.array([1.0, -1.0, 1.0, -1.0])
        y_pred = np.array([1.0, 1.0, -1.0, -1.0])
        assert abs(pearson_correlation(y_true, y_pred)) < 1e-10

    def test_constant_true_raises(self):
        """Correlação deve levantar erro quando y_true é constante."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="[Dd]esvio padrão.*verdadeiros.*zero"):
            pearson_correlation(y_true, y_pred)

    def test_constant_pred_raises(self):
        """Correlação deve levantar erro quando y_pred é constante."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match="[Dd]esvio padrão.*previstos.*zero"):
            pearson_correlation(y_true, y_pred)

    def test_matches_scipy(self):
        """Correlação deve corresponder à implementação do scipy."""
        from scipy.stats import pearsonr
        y_true = np.array([1.5, 2.3, 3.7, 4.2, 5.1])
        y_pred = np.array([1.4, 2.5, 3.5, 4.0, 5.5])
        scipy_corr, _ = pearsonr(y_true, y_pred)
        assert abs(pearson_correlation(y_true, y_pred) - scipy_corr) < 1e-10


class TestAccuracy:
    """Testes para accuracy de variáveis categóricas."""

    def test_perfect_prediction(self):
        """Accuracy deve ser 1 para previsões perfeitas."""
        y_true = np.array(['a', 'b', 'c', 'a', 'b'])
        y_pred = np.array(['a', 'b', 'c', 'a', 'b'])
        assert accuracy(y_true, y_pred) == 1.0

    def test_all_wrong(self):
        """Accuracy deve ser 0 quando todas erradas."""
        y_true = np.array(['a', 'b', 'c'])
        y_pred = np.array(['b', 'c', 'a'])
        assert accuracy(y_true, y_pred) == 0.0

    def test_half_correct(self):
        """Accuracy com 50% correcto."""
        y_true = np.array(['a', 'b', 'c', 'd'])
        y_pred = np.array(['a', 'b', 'x', 'y'])
        assert accuracy(y_true, y_pred) == 0.5

    def test_numeric_values(self):
        """Accuracy deve funcionar com valores numéricos."""
        y_true = np.array([1, 2, 3, 1, 2])
        y_pred = np.array([1, 2, 3, 1, 2])
        assert accuracy(y_true, y_pred) == 1.0

    def test_empty_raises(self):
        """Accuracy deve levantar erro para arrays vazios."""
        with pytest.raises(ValueError, match="vazios"):
            accuracy(np.array([]), np.array([]))

    def test_matches_sklearn(self):
        """Accuracy deve corresponder à implementação do sklearn."""
        from sklearn.metrics import accuracy_score as sklearn_acc
        y_true = np.array(['a', 'b', 'c', 'a', 'b', 'c'])
        y_pred = np.array(['a', 'b', 'x', 'a', 'x', 'c'])
        assert abs(accuracy(y_true, y_pred) - sklearn_acc(y_true, y_pred)) < 1e-10


class TestComputeImputationMetrics:
    """Testes para compute_imputation_metrics."""

    def test_numeric_all_metrics(self):
        """Deve calcular todas as métricas para dados numéricos."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        imputed = np.array([1.0, 2.5, 3.0, 3.5, 5.0])
        mask = np.array([False, True, False, True, False])  # posições 1 e 3 eram missing

        metrics = compute_imputation_metrics(original, imputed, mask, is_categorical=False)

        assert 'rmse' in metrics
        assert 'nrmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'pearson' in metrics
        assert metrics['n_missing'] == 2

    def test_categorical_accuracy(self):
        """Deve calcular accuracy para dados categóricos."""
        original = np.array(['a', 'b', 'c', 'd', 'e'])
        imputed = np.array(['a', 'b', 'x', 'd', 'e'])
        mask = np.array([False, True, True, False, False])

        metrics = compute_imputation_metrics(original, imputed, mask, is_categorical=True)

        assert 'accuracy' in metrics
        assert metrics['n_missing'] == 2
        assert metrics['accuracy'] == 0.5  # 'b' certo, 'c' errado

    def test_no_missing_raises(self):
        """Deve levantar erro se não houver valores missing."""
        original = np.array([1.0, 2.0, 3.0])
        imputed = np.array([1.0, 2.0, 3.0])
        mask = np.array([False, False, False])

        with pytest.raises(ValueError, match="[Nn]ão há valores missing"):
            compute_imputation_metrics(original, imputed, mask)


class TestAggregateMetrics:
    """Testes para aggregate_metrics."""

    def test_simple_aggregation(self):
        """Deve calcular média e desvio padrão."""
        metrics_list = [
            {'rmse': 1.0, 'r2': 0.8},
            {'rmse': 2.0, 'r2': 0.6},
            {'rmse': 3.0, 'r2': 0.7}
        ]

        agg = aggregate_metrics(metrics_list)

        assert abs(agg['rmse'] - 2.0) < 1e-10  # média de [1, 2, 3]
        assert abs(agg['r2'] - 0.7) < 1e-10   # média de [0.8, 0.6, 0.7]
        assert 'rmse_std' in agg
        assert 'r2_std' in agg

    def test_handles_nan(self):
        """Deve ignorar NaN nos cálculos."""
        metrics_list = [
            {'rmse': 1.0},
            {'rmse': np.nan},
            {'rmse': 3.0}
        ]

        agg = aggregate_metrics(metrics_list)

        assert agg['rmse'] == 2.0  # média de [1, 3], ignorando nan

    def test_empty_list(self):
        """Deve retornar dict vazio para lista vazia."""
        assert aggregate_metrics([]) == {}


# ============================================================================
# Testes de integração e edge cases
# ============================================================================

class TestIntegration:
    """Testes de integração verificando consistência entre métricas."""

    def test_perfect_imputation_all_metrics(self):
        """Imputação perfeita deve dar métricas ótimas."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        imputed = original.copy()
        mask = np.array([True, True, True, True, True])

        metrics = compute_imputation_metrics(original, imputed, mask, is_categorical=False)

        assert metrics['rmse'] == 0.0
        assert metrics['nrmse'] == 0.0
        assert metrics['mae'] == 0.0
        assert abs(metrics['r2'] - 1.0) < 1e-10
        assert abs(metrics['pearson'] - 1.0) < 1e-10

    def test_metrics_scale_independence(self):
        """NRMSE deve ser independente da escala."""
        y_true1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred1 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        # Mesmos dados multiplicados por 100
        y_true2 = y_true1 * 100
        y_pred2 = y_pred1 * 100

        # NRMSE deve ser igual (normalizado)
        assert abs(nrmse(y_true1, y_pred1) - nrmse(y_true2, y_pred2)) < 1e-10

        # RMSE deve ser diferente (não normalizado)
        assert rmse(y_true1, y_pred1) != rmse(y_true2, y_pred2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

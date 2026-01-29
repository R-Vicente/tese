"""
Testes unitários para evaluation/missingness.py

Testa:
1. MCAR: distribuição uniforme de missings
2. MAR: missings correlacionados com driver
3. MNAR: missings correlacionados com próprio valor
4. Garantias (sem linhas/colunas 100% vazias)
5. Reprodutibilidade (determinismo com seed)

Execução: python -m pytest tests/test_missingness.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(0, '.')

from evaluation.missingness import (
    introduce_mcar,
    introduce_mar,
    introduce_mnar,
    get_missing_stats
)


class TestMCAR:
    """Testes para Missing Completely At Random."""

    def test_returns_correct_shapes(self):
        """Deve retornar dados e máscara com shapes correctos."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mcar(data, 0.2)

        assert data_missing.shape == data.shape
        assert mask.shape == data.shape

    def test_missing_rate_approximate(self):
        """Taxa de missings deve ser aproximadamente a solicitada."""
        data = np.random.randn(1000, 10)  # Grande para melhor aproximação
        data_missing, mask = introduce_mcar(data, 0.3)

        actual_rate = mask.sum() / mask.size
        assert 0.25 < actual_rate < 0.35  # Tolerância de ±0.05

    def test_reproducibility(self):
        """Mesmo seed deve produzir mesmos resultados."""
        data = np.random.randn(50, 5)

        data_missing1, mask1 = introduce_mcar(data, 0.2, random_state=42)
        data_missing2, mask2 = introduce_mcar(data, 0.2, random_state=42)

        assert np.array_equal(mask1, mask2)
        # NaN == NaN é False, então usamos isnan
        assert np.array_equal(np.isnan(data_missing1), np.isnan(data_missing2))

    def test_different_seeds_different_results(self):
        """Seeds diferentes devem produzir resultados diferentes."""
        data = np.random.randn(50, 5)

        _, mask1 = introduce_mcar(data, 0.2, random_state=42)
        _, mask2 = introduce_mcar(data, 0.2, random_state=123)

        assert not np.array_equal(mask1, mask2)

    def test_no_completely_empty_rows(self):
        """Nenhuma linha deve estar 100% vazia."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mcar(data, 0.8)  # Taxa alta

        for i in range(mask.shape[0]):
            assert not mask[i].all(), f"Linha {i} está completamente vazia"

    def test_no_completely_empty_cols(self):
        """Nenhuma coluna deve estar 100% vazia."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mcar(data, 0.8)  # Taxa alta

        for j in range(mask.shape[1]):
            assert not mask[:, j].all(), f"Coluna {j} está completamente vazia"

    def test_zero_missing_rate(self):
        """Taxa 0 não deve introduzir missings."""
        data = np.random.randn(50, 5)
        data_missing, mask = introduce_mcar(data, 0.0)

        assert mask.sum() == 0
        assert np.array_equal(data, data_missing)

    def test_works_with_dataframe(self):
        """Deve funcionar com pandas DataFrame."""
        df = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])
        df_missing, mask = introduce_mcar(df, 0.2)

        assert isinstance(df_missing, pd.DataFrame)
        assert list(df_missing.columns) == ['a', 'b', 'c']
        assert mask.shape == (50, 3)

    def test_invalid_missing_rate_raises(self):
        """Taxa fora de [0, 1) deve levantar erro."""
        data = np.random.randn(10, 3)

        with pytest.raises(ValueError):
            introduce_mcar(data, 1.0)

        with pytest.raises(ValueError):
            introduce_mcar(data, -0.1)

        with pytest.raises(ValueError):
            introduce_mcar(data, 1.5)

    def test_empty_data_raises(self):
        """Dataset vazio deve levantar erro."""
        with pytest.raises(ValueError):
            introduce_mcar(np.array([]).reshape(0, 0), 0.2)

    def test_mask_matches_nans(self):
        """Máscara deve corresponder exactamente aos NaNs."""
        data = np.random.randn(50, 5)
        data_missing, mask = introduce_mcar(data, 0.3)

        nan_locations = np.isnan(data_missing)
        assert np.array_equal(mask, nan_locations)


class TestMAR:
    """Testes para Missing At Random."""

    def test_returns_correct_shapes(self):
        """Deve retornar dados e máscara com shapes correctos."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mar(data, 0.2)

        assert data_missing.shape == data.shape
        assert mask.shape == data.shape

    def test_driver_col_has_no_missing(self):
        """Coluna driver não deve ter missings."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mar(data, 0.3, driver_col=0)

        assert mask[:, 0].sum() == 0  # Coluna 0 sem missings

    def test_driver_col_parameter(self):
        """Deve respeitar o parâmetro driver_col."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mar(data, 0.3, driver_col=2)

        assert mask[:, 2].sum() == 0  # Coluna 2 sem missings

    def test_mar_correlation(self):
        """Missings devem estar correlacionados com driver."""
        # Criar dados onde coluna 0 tem valores claramente separados
        n = 500
        data = np.random.randn(n, 5)
        data[:n//2, 0] = -10  # Primeira metade: valores baixos
        data[n//2:, 0] = 10   # Segunda metade: valores altos

        data_missing, mask = introduce_mar(data, 0.3, driver_col=0)

        # Calcular taxa de missing para cada grupo
        low_group_missing = mask[:n//2, 1:].mean()
        high_group_missing = mask[n//2:, 1:].mean()

        # Grupo com valores altos deve ter mais missings
        assert high_group_missing > low_group_missing

    def test_no_completely_empty_rows(self):
        """Nenhuma linha deve estar 100% vazia."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mar(data, 0.8)

        for i in range(mask.shape[0]):
            assert not mask[i].all()

    def test_reproducibility(self):
        """Mesmo seed deve produzir mesmos resultados."""
        data = np.random.randn(50, 5)

        _, mask1 = introduce_mar(data, 0.2, random_state=42)
        _, mask2 = introduce_mar(data, 0.2, random_state=42)

        assert np.array_equal(mask1, mask2)

    def test_invalid_driver_col_raises(self):
        """driver_col inválido deve levantar erro."""
        data = np.random.randn(10, 3)

        with pytest.raises(ValueError):
            introduce_mar(data, 0.2, driver_col=5)

        with pytest.raises(ValueError):
            introduce_mar(data, 0.2, driver_col=-1)


class TestMNAR:
    """Testes para Missing Not At Random."""

    def test_returns_correct_shapes(self):
        """Deve retornar dados e máscara com shapes correctos."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mnar(data, 0.2)

        assert data_missing.shape == data.shape
        assert mask.shape == data.shape

    def test_mnar_correlation(self):
        """Missings devem estar correlacionados com próprio valor."""
        # Criar dados com distribuição clara
        n = 500
        data = np.zeros((n, 3))
        data[:n//2, :] = -10  # Primeira metade: valores baixos
        data[n//2:, :] = 10   # Segunda metade: valores altos

        data_missing, mask = introduce_mnar(data, 0.3)

        # Calcular taxa de missing para cada grupo
        low_group_missing = mask[:n//2, :].mean()
        high_group_missing = mask[n//2:, :].mean()

        # Valores altos devem ter mais missings (por design do MNAR)
        assert high_group_missing > low_group_missing

    def test_no_completely_empty_rows(self):
        """Nenhuma linha deve estar 100% vazia."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mnar(data, 0.8)

        for i in range(mask.shape[0]):
            assert not mask[i].all()

    def test_no_completely_empty_cols(self):
        """Nenhuma coluna deve estar 100% vazia."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mnar(data, 0.8)

        for j in range(mask.shape[1]):
            assert not mask[:, j].all()

    def test_reproducibility(self):
        """Mesmo seed deve produzir mesmos resultados."""
        data = np.random.randn(50, 5)

        _, mask1 = introduce_mnar(data, 0.2, random_state=42)
        _, mask2 = introduce_mnar(data, 0.2, random_state=42)

        assert np.array_equal(mask1, mask2)


class TestGetMissingStats:
    """Testes para get_missing_stats."""

    def test_correct_counts(self):
        """Deve contar correctamente os missings."""
        mask = np.array([
            [True, False, False],
            [True, True, False],
            [False, False, False]
        ])

        stats = get_missing_stats(mask)

        assert stats['total_missing'] == 3
        assert stats['total_cells'] == 9
        assert abs(stats['missing_rate'] - 3/9) < 1e-10

    def test_per_row_counts(self):
        """Deve contar missings por linha."""
        mask = np.array([
            [True, False, False],  # 1 missing
            [True, True, True],    # 3 missings
            [False, False, False]  # 0 missings
        ])

        stats = get_missing_stats(mask)

        assert list(stats['missing_per_row']) == [1, 3, 0]

    def test_per_col_counts(self):
        """Deve contar missings por coluna."""
        mask = np.array([
            [True, False, False],
            [True, True, False],
            [True, False, False]
        ])

        stats = get_missing_stats(mask)

        assert list(stats['missing_per_col']) == [3, 1, 0]

    def test_rows_cols_with_missing(self):
        """Deve contar linhas/colunas afectadas."""
        mask = np.array([
            [True, False, False],
            [False, False, False],
            [True, True, False]
        ])

        stats = get_missing_stats(mask)

        assert stats['rows_with_missing'] == 2  # Linhas 0 e 2
        assert stats['cols_with_missing'] == 2  # Colunas 0 e 1

    def test_no_missing(self):
        """Deve funcionar quando não há missings."""
        mask = np.zeros((5, 3), dtype=bool)
        stats = get_missing_stats(mask)

        assert stats['total_missing'] == 0
        assert stats['missing_rate'] == 0.0
        assert stats['rows_with_missing'] == 0
        assert stats['cols_with_missing'] == 0


class TestIntegration:
    """Testes de integração entre funções."""

    def test_stats_match_mask(self):
        """Estatísticas devem corresponder à máscara."""
        data = np.random.randn(100, 5)
        data_missing, mask = introduce_mcar(data, 0.3)

        stats = get_missing_stats(mask)

        assert stats['total_missing'] == mask.sum()
        assert stats['missing_rate'] == mask.sum() / mask.size

    def test_all_patterns_produce_valid_output(self):
        """Todos os padrões devem produzir output válido."""
        data = np.random.randn(100, 5)

        for pattern, func in [
            ('MCAR', introduce_mcar),
            ('MAR', introduce_mar),
            ('MNAR', introduce_mnar)
        ]:
            data_missing, mask = func(data, 0.2)

            # Verificar que NaNs correspondem à máscara
            assert np.array_equal(np.isnan(data_missing), mask), f"Falhou para {pattern}"

            # Verificar que valores não-missing são preservados
            preserved = data[~mask]
            original = data_missing[~mask]
            # Converter para array e remover NaN que possa existir
            preserved_clean = preserved[~np.isnan(preserved)]
            original_clean = original[~np.isnan(original)]
            assert np.allclose(preserved_clean, original_clean), f"Valores alterados em {pattern}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Análise de datasets para ISCA-k.

Fase 0 - Diagnóstico do dataset:
- Taxa de missings (global, por coluna, por linha)
- Detectar padrão de missings (MCAR vs MAR vs MNAR)
- Avaliar dimensionalidade (n, p, ratio n/p)
- Detectar estrutura natural (clusters, densidade)
- Detectar outliers e distribuições
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class MissingPattern(Enum):
    """Padrão de valores em falta."""
    MCAR = "mcar"  # Missing Completely At Random
    MAR = "mar"    # Missing At Random
    MNAR = "mnar"  # Missing Not At Random
    UNKNOWN = "unknown"


@dataclass
class MissingnessReport:
    """Relatório de análise de missingness."""
    global_rate: float
    col_rates: Dict[str, float]
    row_rates: np.ndarray
    pattern: MissingPattern
    pattern_confidence: float
    little_mcar_pvalue: Optional[float] = None


@dataclass
class DimensionalityReport:
    """Relatório de dimensionalidade."""
    n_samples: int
    n_features: int
    ratio: float
    is_high_dimensional: bool
    complete_cases: int
    complete_case_ratio: float


@dataclass
class StructureReport:
    """Relatório de estrutura do dataset."""
    n_clusters_suggested: int
    cluster_labels: Optional[np.ndarray]
    silhouette_score: Optional[float]
    avg_density: float
    density_std: float


@dataclass
class DistributionReport:
    """Relatório de distribuições e outliers."""
    n_outliers_per_col: Dict[str, int]
    total_outliers: int
    outlier_rate: float
    skewness: Dict[str, float]
    normality_pvalues: Dict[str, float]


# =============================================================================
# ANÁLISE DE MISSINGNESS
# =============================================================================

def compute_missing_rates(data: np.ndarray,
                          col_names: Optional[List[str]] = None) -> MissingnessReport:
    """
    Calcula taxas de missing global, por coluna e por linha.

    Args:
        data: Dataset com possíveis NaN
        col_names: Nomes das colunas

    Returns:
        MissingnessReport com estatísticas
    """
    data = np.asarray(data, dtype=float)
    n_rows, n_cols = data.shape

    if col_names is None:
        col_names = [f"var_{i}" for i in range(n_cols)]

    # Máscara de missings
    missing_mask = np.isnan(data)

    # Taxa global
    global_rate = missing_mask.sum() / missing_mask.size

    # Taxa por coluna
    col_rates = {}
    for j, name in enumerate(col_names):
        col_rates[name] = missing_mask[:, j].sum() / n_rows

    # Taxa por linha
    row_rates = missing_mask.sum(axis=1) / n_cols

    # Detectar padrão (análise simplificada)
    pattern, confidence, pvalue = _detect_missing_pattern(data, missing_mask)

    return MissingnessReport(
        global_rate=global_rate,
        col_rates=col_rates,
        row_rates=row_rates,
        pattern=pattern,
        pattern_confidence=confidence,
        little_mcar_pvalue=pvalue
    )


def _detect_missing_pattern(data: np.ndarray,
                            missing_mask: np.ndarray) -> Tuple[MissingPattern, float, Optional[float]]:
    """
    Detecta padrão de missingness usando teste de Little (simplificado).

    Heurística:
    - MCAR: missings uniformemente distribuídos
    - MAR: missings correlacionados com outras variáveis observadas
    - MNAR: missings correlacionados com a própria variável

    Returns:
        (pattern, confidence, p_value)
    """
    n_rows, n_cols = data.shape

    # Se não há missings
    if not missing_mask.any():
        return MissingPattern.UNKNOWN, 1.0, None

    # Teste simplificado de Little para MCAR
    # Ideia: comparar médias das variáveis observadas entre grupos com/sem missing

    p_values = []

    for j in range(n_cols):
        col_missing = missing_mask[:, j]

        # Se coluna não tem missings ou tem muitos missings, ignorar
        if col_missing.sum() == 0 or col_missing.sum() > n_rows * 0.9:
            continue

        # Comparar médias de outras variáveis entre grupos
        for k in range(n_cols):
            if k == j:
                continue

            other_col = data[:, k]

            # Valores da outra coluna quando j está missing vs não missing
            group_missing = other_col[col_missing & ~np.isnan(other_col)]
            group_observed = other_col[~col_missing & ~np.isnan(other_col)]

            if len(group_missing) < 5 or len(group_observed) < 5:
                continue

            # T-test
            try:
                _, pval = stats.ttest_ind(group_missing, group_observed)
                if not np.isnan(pval):
                    p_values.append(pval)
            except Exception:
                continue

    if not p_values:
        return MissingPattern.UNKNOWN, 0.5, None

    # Combinação de p-values (método de Fisher)
    chi2_stat = -2 * sum(np.log(p) for p in p_values if p > 0)
    df = 2 * len(p_values)
    combined_pvalue = 1 - stats.chi2.cdf(chi2_stat, df)

    # Interpretar resultado
    # Se p-value alto (> 0.05), não rejeitamos MCAR
    if combined_pvalue > 0.05:
        pattern = MissingPattern.MCAR
        confidence = min(combined_pvalue * 2, 1.0)  # Mais alto = mais confiança
    else:
        # Missings não são aleatórios - pode ser MAR ou MNAR
        # Para distinguir, precisaríamos de análise mais sofisticada
        pattern = MissingPattern.MAR
        confidence = 1 - combined_pvalue

    return pattern, confidence, combined_pvalue


# =============================================================================
# ANÁLISE DE DIMENSIONALIDADE
# =============================================================================

def analyze_dimensionality(data: np.ndarray) -> DimensionalityReport:
    """
    Analisa dimensionalidade do dataset.

    Alta dimensionalidade: p > n ou ratio n/p < 10
    """
    data = np.asarray(data, dtype=float)
    n_samples, n_features = data.shape

    ratio = n_samples / n_features if n_features > 0 else float('inf')

    # Alta dimensionalidade se ratio < 10 ou p > n
    is_high_dim = ratio < 10 or n_features > n_samples

    # Casos completos (sem nenhum missing)
    complete_mask = ~np.isnan(data).any(axis=1)
    complete_cases = complete_mask.sum()
    complete_ratio = complete_cases / n_samples

    return DimensionalityReport(
        n_samples=n_samples,
        n_features=n_features,
        ratio=ratio,
        is_high_dimensional=is_high_dim,
        complete_cases=complete_cases,
        complete_case_ratio=complete_ratio
    )


# =============================================================================
# ANÁLISE DE ESTRUTURA (CLUSTERS E DENSIDADE)
# =============================================================================

def analyze_structure(data: np.ndarray,
                      max_clusters: int = 10) -> StructureReport:
    """
    Detecta estrutura natural do dataset (clusters, densidade).

    Usa apenas casos completos para análise.
    """
    data = np.asarray(data, dtype=float)

    # Usar apenas casos completos
    complete_mask = ~np.isnan(data).any(axis=1)
    data_complete = data[complete_mask]

    if len(data_complete) < 10:
        return StructureReport(
            n_clusters_suggested=1,
            cluster_labels=None,
            silhouette_score=None,
            avg_density=0.0,
            density_std=0.0
        )

    # Normalizar para análise
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_complete)

    # Encontrar número óptimo de clusters (método do cotovelo simplificado)
    n_clusters_suggested = _find_optimal_clusters(data_scaled, max_clusters)

    # Aplicar KMeans com clusters sugeridos
    if n_clusters_suggested > 1:
        kmeans = KMeans(n_clusters=n_clusters_suggested, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)

        # Silhouette score
        from sklearn.metrics import silhouette_score as sklearn_silhouette
        try:
            sil_score = sklearn_silhouette(data_scaled, labels)
        except Exception:
            sil_score = None
    else:
        labels = np.zeros(len(data_complete), dtype=int)
        sil_score = None

    # Criar array completo de labels (NaN para casos incompletos)
    full_labels = np.full(len(data), -1, dtype=int)
    full_labels[complete_mask] = labels

    # Calcular densidade local (distância média aos k vizinhos mais próximos)
    avg_density, density_std = _compute_local_density(data_scaled)

    return StructureReport(
        n_clusters_suggested=n_clusters_suggested,
        cluster_labels=full_labels,
        silhouette_score=sil_score,
        avg_density=avg_density,
        density_std=density_std
    )


def _find_optimal_clusters(data: np.ndarray, max_k: int = 10) -> int:
    """
    Encontra número óptimo de clusters usando método do cotovelo.
    """
    n_samples = len(data)
    max_k = min(max_k, n_samples - 1)

    if max_k < 2:
        return 1

    inertias = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Encontrar "cotovelo" - ponto onde ganho marginal diminui muito
    # Usando diferenças de segunda ordem
    if len(inertias) < 3:
        return 1

    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)

    # O cotovelo é onde a segunda derivada é máxima
    if len(diffs2) > 0:
        elbow_idx = np.argmax(diffs2) + 1  # +1 porque diff reduz tamanho
        return elbow_idx + 1  # +1 porque range começa em 1
    else:
        return 1


def _compute_local_density(data: np.ndarray, k: int = 5) -> Tuple[float, float]:
    """
    Calcula densidade local média usando k vizinhos mais próximos.

    Densidade alta = distâncias pequenas aos vizinhos.
    """
    n_samples = len(data)
    k = min(k, n_samples - 1)

    if k < 1:
        return 0.0, 0.0

    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 porque inclui o próprio ponto
    nn.fit(data)
    distances, _ = nn.kneighbors(data)

    # Média das distâncias (excluindo distância a si mesmo = 0)
    avg_distances = distances[:, 1:].mean(axis=1)

    # Densidade como inverso da distância média
    # Para evitar divisão por zero, adicionamos pequeno epsilon
    densities = 1 / (avg_distances + 1e-10)

    return float(densities.mean()), float(densities.std())


# =============================================================================
# ANÁLISE DE DISTRIBUIÇÕES E OUTLIERS
# =============================================================================

def analyze_distributions(data: np.ndarray,
                          col_names: Optional[List[str]] = None,
                          outlier_threshold: float = 3.0) -> DistributionReport:
    """
    Analisa distribuições e detecta outliers.

    Outliers: valores com |z-score| > threshold
    """
    data = np.asarray(data, dtype=float)
    n_rows, n_cols = data.shape

    if col_names is None:
        col_names = [f"var_{i}" for i in range(n_cols)]

    n_outliers_per_col = {}
    skewness = {}
    normality_pvalues = {}
    total_outliers = 0

    for j, name in enumerate(col_names):
        col = data[:, j]
        valid = col[~np.isnan(col)]

        if len(valid) < 3:
            n_outliers_per_col[name] = 0
            skewness[name] = np.nan
            normality_pvalues[name] = np.nan
            continue

        # Outliers via z-score
        mean = np.mean(valid)
        std = np.std(valid)
        if std > 0:
            z_scores = np.abs((valid - mean) / std)
            n_outliers = (z_scores > outlier_threshold).sum()
        else:
            n_outliers = 0

        n_outliers_per_col[name] = int(n_outliers)
        total_outliers += n_outliers

        # Skewness
        skewness[name] = float(stats.skew(valid))

        # Teste de normalidade (Shapiro-Wilk para n < 5000)
        if len(valid) < 5000:
            try:
                _, pval = stats.shapiro(valid[:5000])
                normality_pvalues[name] = float(pval)
            except Exception:
                normality_pvalues[name] = np.nan
        else:
            # Para datasets grandes, usar D'Agostino
            try:
                _, pval = stats.normaltest(valid)
                normality_pvalues[name] = float(pval)
            except Exception:
                normality_pvalues[name] = np.nan

    # Taxa de outliers
    total_valid = (~np.isnan(data)).sum()
    outlier_rate = total_outliers / total_valid if total_valid > 0 else 0.0

    return DistributionReport(
        n_outliers_per_col=n_outliers_per_col,
        total_outliers=total_outliers,
        outlier_rate=outlier_rate,
        skewness=skewness,
        normality_pvalues=normality_pvalues
    )


# =============================================================================
# CLASSE PRINCIPAL: DatasetAnalyzer
# =============================================================================

class DatasetAnalyzer:
    """
    Analisador completo de datasets para Fase 0 do ISCA-k.

    Combina todas as análises:
    - Tipos de variáveis
    - Missingness
    - Dimensionalidade
    - Estrutura
    - Distribuições
    """

    def __init__(self, data: Union[np.ndarray, pd.DataFrame],
                 col_names: Optional[List[str]] = None):
        """
        Inicializa o analisador.

        Args:
            data: Dataset (numpy array ou pandas DataFrame)
            col_names: Nomes das colunas (opcional se DataFrame)
        """
        if isinstance(data, pd.DataFrame):
            self.col_names = list(data.columns) if col_names is None else col_names
            self.data = data.values.astype(float)
            self._df = data
        else:
            self.data = np.asarray(data, dtype=float)
            n_cols = self.data.shape[1] if self.data.ndim > 1 else 1
            self.col_names = col_names if col_names else [f"var_{i}" for i in range(n_cols)]
            self._df = None

        self._missingness: Optional[MissingnessReport] = None
        self._dimensionality: Optional[DimensionalityReport] = None
        self._structure: Optional[StructureReport] = None
        self._distributions: Optional[DistributionReport] = None

    def analyze_all(self) -> Dict:
        """
        Executa todas as análises.

        Returns:
            Dicionário com todos os relatórios
        """
        return {
            'missingness': self.analyze_missingness(),
            'dimensionality': self.analyze_dimensionality(),
            'structure': self.analyze_structure(),
            'distributions': self.analyze_distributions()
        }

    def analyze_missingness(self) -> MissingnessReport:
        """Analisa padrões de missingness."""
        if self._missingness is None:
            self._missingness = compute_missing_rates(self.data, self.col_names)
        return self._missingness

    def analyze_dimensionality(self) -> DimensionalityReport:
        """Analisa dimensionalidade."""
        if self._dimensionality is None:
            self._dimensionality = analyze_dimensionality(self.data)
        return self._dimensionality

    def analyze_structure(self) -> StructureReport:
        """Analisa estrutura (clusters, densidade)."""
        if self._structure is None:
            self._structure = analyze_structure(self.data)
        return self._structure

    def analyze_distributions(self) -> DistributionReport:
        """Analisa distribuições e outliers."""
        if self._distributions is None:
            self._distributions = analyze_distributions(self.data, self.col_names)
        return self._distributions

    def print_report(self) -> None:
        """Imprime relatório completo."""
        results = self.analyze_all()

        print("=" * 60)
        print("RELATÓRIO DE ANÁLISE DO DATASET")
        print("=" * 60)

        # Dimensionalidade
        dim = results['dimensionality']
        print(f"\n📐 DIMENSIONALIDADE")
        print(f"   Amostras: {dim.n_samples}")
        print(f"   Features: {dim.n_features}")
        print(f"   Ratio n/p: {dim.ratio:.1f}")
        print(f"   Alta dimensionalidade: {'SIM' if dim.is_high_dimensional else 'NÃO'}")
        print(f"   Casos completos: {dim.complete_cases} ({dim.complete_case_ratio:.1%})")

        # Missingness
        miss = results['missingness']
        print(f"\n❓ MISSINGNESS")
        print(f"   Taxa global: {miss.global_rate:.1%}")
        print(f"   Padrão detectado: {miss.pattern.value.upper()}")
        print(f"   Confiança: {miss.pattern_confidence:.1%}")
        if miss.col_rates:
            max_col = max(miss.col_rates.items(), key=lambda x: x[1])
            print(f"   Coluna com mais missings: {max_col[0]} ({max_col[1]:.1%})")

        # Estrutura
        struct = results['structure']
        print(f"\n🔷 ESTRUTURA")
        print(f"   Clusters sugeridos: {struct.n_clusters_suggested}")
        if struct.silhouette_score is not None:
            print(f"   Silhouette score: {struct.silhouette_score:.3f}")
        print(f"   Densidade média: {struct.avg_density:.3f} (±{struct.density_std:.3f})")

        # Distribuições
        dist = results['distributions']
        print(f"\n📊 DISTRIBUIÇÕES")
        print(f"   Total de outliers: {dist.total_outliers} ({dist.outlier_rate:.1%})")
        if dist.skewness:
            most_skewed = max(dist.skewness.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)
            print(f"   Variável mais assimétrica: {most_skewed[0]} (skew={most_skewed[1]:.2f})")

        print("\n" + "=" * 60)


# =============================================================================
# TESTES
# =============================================================================
if __name__ == "__main__":
    print("Testando análise de datasets...\n")

    # Criar dados de teste
    np.random.seed(42)
    n = 200
    p = 10

    # Dataset com estrutura clara (2 clusters)
    data1 = np.random.randn(n // 2, p) + 2
    data2 = np.random.randn(n // 2, p) - 2
    data = np.vstack([data1, data2])

    # Adicionar alguns missings MCAR
    missing_mask = np.random.random((n, p)) < 0.1
    data[missing_mask] = np.nan

    # Adicionar outliers
    for i in range(5):
        j = np.random.randint(p)
        data[i, j] = 50  # Outlier extremo

    print(f"Dataset: {data.shape}")
    print(f"Missings: {np.isnan(data).sum()} ({np.isnan(data).mean():.1%})")

    # Analisar
    analyzer = DatasetAnalyzer(data)
    analyzer.print_report()

    # Testes básicos
    results = analyzer.analyze_all()

    assert results['dimensionality'].n_samples == n
    assert results['dimensionality'].n_features == p
    assert 0 <= results['missingness'].global_rate <= 1
    assert results['structure'].n_clusters_suggested >= 1

    print("\n✓ Todos os testes passaram!")

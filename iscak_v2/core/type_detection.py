"""
Detecção de tipos de variáveis para ISCA-k.

Tipos suportados:
- BINARY: exactamente 2 valores únicos
- NOMINAL: categórico sem ordem natural (strings ou poucos valores únicos)
- ORDINAL: categórico com ordem natural (inteiros com poucos valores)
- CONTINUOUS: numérico contínuo (floats ou muitos valores únicos)
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import List, Dict, Union, Optional
from dataclasses import dataclass


class VariableType(Enum):
    """Tipos de variáveis suportados."""
    BINARY = "binary"
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    CONTINUOUS = "continuous"
    UNKNOWN = "unknown"


@dataclass
class VariableInfo:
    """Informação sobre uma variável."""
    name: str
    var_type: VariableType
    n_unique: int
    n_missing: int
    missing_rate: float
    unique_values: Optional[np.ndarray] = None

    def __repr__(self):
        return f"VariableInfo({self.name}: {self.var_type.value}, unique={self.n_unique}, missing={self.missing_rate:.1%})"


def _is_string_dtype(dtype: np.dtype) -> bool:
    """Verifica se dtype é string/object."""
    # dtype.kind: 'O' = object, 'U' = unicode string, 'S' = byte string
    return dtype.kind in ('O', 'U', 'S')


def _values_are_integers(values: np.ndarray) -> bool:
    """Verifica se valores numéricos são efectivamente inteiros."""
    try:
        values_float = values.astype(float)
        return np.allclose(values_float, np.round(values_float), equal_nan=True)
    except (ValueError, TypeError):
        return False


def detect_single_variable(values: np.ndarray, name: str = "var",
                           ordinal_threshold: int = 10) -> VariableInfo:
    """
    Detecta o tipo de uma única variável.

    Args:
        values: Array de valores da variável
        name: Nome da variável
        ordinal_threshold: Número máximo de valores únicos para considerar ordinal

    Returns:
        VariableInfo com tipo detectado e estatísticas
    """
    values = np.asarray(values)
    dtype = values.dtype
    is_string = _is_string_dtype(dtype)

    # Contar missings
    if is_string:
        missing_mask = pd.isna(values)
    else:
        missing_mask = np.isnan(values.astype(float))

    n_total = len(values)
    n_missing = int(missing_mask.sum())
    n_valid = n_total - n_missing
    missing_rate = n_missing / n_total if n_total > 0 else 0.0

    # Valores válidos
    valid_values = values[~missing_mask]

    if n_valid == 0:
        return VariableInfo(
            name=name,
            var_type=VariableType.UNKNOWN,
            n_unique=0,
            n_missing=n_missing,
            missing_rate=missing_rate
        )

    # Contar únicos
    unique_values = np.unique(valid_values)
    n_unique = len(unique_values)

    # === DETECTAR TIPO ===

    # 1. BINARY: exactamente 2 valores únicos
    if n_unique == 2:
        var_type = VariableType.BINARY

    # 2. NOMINAL: strings são sempre nominais (excepto binário já tratado acima)
    elif is_string:
        var_type = VariableType.NOMINAL

    # 3. Para numéricos, decidir entre ORDINAL e CONTINUOUS
    else:
        # Verificar se são inteiros
        is_integer = _values_are_integers(valid_values)

        # ORDINAL: poucos valores únicos E são inteiros
        if n_unique <= ordinal_threshold and is_integer:
            var_type = VariableType.ORDINAL

        # CONTINUOUS: muitos valores únicos OU não são inteiros
        else:
            var_type = VariableType.CONTINUOUS

    return VariableInfo(
        name=name,
        var_type=var_type,
        n_unique=n_unique,
        n_missing=n_missing,
        missing_rate=missing_rate,
        unique_values=unique_values if n_unique <= 20 else None
    )


def detect_variable_types(data: Union[np.ndarray, pd.DataFrame],
                          column_names: Optional[List[str]] = None,
                          ordinal_threshold: int = 10) -> Dict[str, VariableInfo]:
    """
    Detecta tipos de todas as variáveis num dataset.

    Args:
        data: Dataset (numpy array ou pandas DataFrame)
        column_names: Nomes das colunas (opcional se DataFrame)
        ordinal_threshold: Número máximo de valores únicos para considerar ordinal

    Returns:
        Dicionário {nome: VariableInfo}
    """
    # Converter para array se necessário
    if isinstance(data, pd.DataFrame):
        if column_names is None:
            column_names = list(data.columns)
        # Processar coluna a coluna para preservar dtypes
        results = {}
        for name in column_names:
            results[name] = detect_single_variable(
                data[name].values,
                name=name,
                ordinal_threshold=ordinal_threshold
            )
        return results

    # Numpy array
    data_array = np.asarray(data)
    if column_names is None:
        column_names = [f"var_{i}" for i in range(data_array.shape[1])]

    if data_array.ndim == 1:
        data_array = data_array.reshape(-1, 1)
        column_names = ["var_0"]

    results = {}
    for i, name in enumerate(column_names):
        results[name] = detect_single_variable(
            data_array[:, i],
            name=name,
            ordinal_threshold=ordinal_threshold
        )

    return results


def get_type_summary(type_info: Dict[str, VariableInfo]) -> Dict[str, List[str]]:
    """
    Agrupa variáveis por tipo.

    Returns:
        Dicionário {tipo: [lista de nomes]}
    """
    summary = {t.value: [] for t in VariableType}

    for name, info in type_info.items():
        summary[info.var_type.value].append(name)

    # Remover tipos vazios
    return {k: v for k, v in summary.items() if v}


def print_type_report(type_info: Dict[str, VariableInfo]) -> None:
    """Imprime relatório de tipos detectados."""
    print("=" * 50)
    print("RELATÓRIO DE TIPOS DE VARIÁVEIS")
    print("=" * 50)

    summary = get_type_summary(type_info)

    for var_type, names in summary.items():
        print(f"\n{var_type.upper()} ({len(names)} variáveis):")
        for name in names:
            info = type_info[name]
            print(f"  • {name}: {info.n_unique} únicos, {info.missing_rate:.1%} missing")

    print("\n" + "=" * 50)


# =============================================================================
# TESTES
# =============================================================================
if __name__ == "__main__":
    print("Testando detecção de tipos...\n")

    # Criar dados de teste
    np.random.seed(42)
    n = 100

    test_data = pd.DataFrame({
        'binario': np.random.choice([0, 1], n),
        'ordinal': np.random.choice([1, 2, 3, 4, 5], n),
        'continuo': np.random.randn(n),
        'nominal': np.random.choice(['A', 'B', 'C', 'D'], n),
        'binario_str': np.random.choice(['Sim', 'Não'], n),
    })

    # Adicionar missings
    test_data.loc[np.random.choice(n, 10, replace=False), 'continuo'] = np.nan
    test_data.loc[np.random.choice(n, 5, replace=False), 'ordinal'] = np.nan

    print("Dados de teste:")
    print(test_data.head())
    print(f"\ndtypes: {dict(test_data.dtypes)}")
    print()

    # Detectar tipos
    types = detect_variable_types(test_data)

    # Mostrar resultados
    print("Resultados da detecção:")
    for name, info in types.items():
        print(f"  {name}: {info.var_type.value} (n_unique={info.n_unique})")

    # Testes de validação
    errors = []

    if types['binario'].var_type != VariableType.BINARY:
        errors.append(f"binario: esperado BINARY, obtido {types['binario'].var_type}")

    if types['ordinal'].var_type != VariableType.ORDINAL:
        errors.append(f"ordinal: esperado ORDINAL, obtido {types['ordinal'].var_type}")

    if types['continuo'].var_type != VariableType.CONTINUOUS:
        errors.append(f"continuo: esperado CONTINUOUS, obtido {types['continuo'].var_type}")

    if types['nominal'].var_type != VariableType.NOMINAL:
        errors.append(f"nominal: esperado NOMINAL, obtido {types['nominal'].var_type}")

    if types['binario_str'].var_type != VariableType.BINARY:
        errors.append(f"binario_str: esperado BINARY, obtido {types['binario_str'].var_type}")

    if errors:
        print("\n❌ ERROS:")
        for e in errors:
            print(f"  {e}")
        exit(1)
    else:
        print("\n✓ Todos os testes passaram!")

    # Relatório completo
    print()
    print_type_report(types)

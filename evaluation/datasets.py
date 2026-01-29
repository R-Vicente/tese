"""
Carregamento de datasets standard para benchmarking.

Este módulo fornece funções para carregar datasets conhecidos e bem documentados
que são comummente usados para avaliar métodos de imputação.

Datasets numéricos:
    - Iris: 150 samples, 4 features (pequeno, baixa dimensionalidade)
    - Diabetes: 442 samples, 10 features (médio, já normalizado!)
    - Wine: 178 samples, 13 features (pequeno/médio)

Datasets mistos (numérico + categórico):
    - Titanic: ~700 samples, 7 features (após limpeza)

Autor: Ricardo Vicente
Data: Janeiro 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.datasets import load_iris, load_diabetes, load_wine


def load_iris_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Carrega o dataset Iris.

    - 150 amostras
    - 4 features numéricas: sepal length, sepal width, petal length, petal width
    - Não tem missings originais
    - Dados não normalizados (escalas diferentes por feature)

    Returns:
        Tuple contendo:
            - DataFrame com os dados
            - Dict com tipos de coluna {'col_name': 'numeric' ou 'categorical'}
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    col_types = {col: 'numeric' for col in df.columns}

    return df, col_types


def load_diabetes_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Carrega o dataset Diabetes.

    ATENÇÃO: Os dados já estão normalizados (mean=0, std ≈ 0.05)!
    Isto é importante para métodos que aplicam scaling - não devem re-escalar.

    - 442 amostras
    - 10 features numéricas (age, sex, bmi, bp, s1-s6)
    - Não tem missings originais
    - Dados JÁ NORMALIZADOS

    Returns:
        Tuple contendo:
            - DataFrame com os dados
            - Dict com tipos de coluna
    """
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    col_types = {col: 'numeric' for col in df.columns}

    return df, col_types


def load_wine_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Carrega o dataset Wine.

    - 178 amostras
    - 13 features numéricas (características químicas de vinhos)
    - Não tem missings originais
    - Dados não normalizados (escalas muito diferentes por feature)

    Returns:
        Tuple contendo:
            - DataFrame com os dados
            - Dict com tipos de coluna
    """
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)

    col_types = {col: 'numeric' for col in df.columns}

    return df, col_types


def load_titanic_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Carrega o dataset Titanic (versão limpa).

    Dataset misto com variáveis numéricas e categóricas.
    Linhas com missings originais são removidas para ter ground truth limpo.

    - ~700 amostras (após limpeza)
    - 7 features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    - Misto: 4 numéricas + 3 categóricas

    Returns:
        Tuple contendo:
            - DataFrame com os dados
            - Dict com tipos de coluna

    Raises:
        ConnectionError: Se não conseguir baixar os dados
    """
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

    try:
        df_raw = pd.read_csv(url)
    except Exception as e:
        raise ConnectionError(f"Não foi possível baixar Titanic dataset: {e}")

    # Seleccionar colunas úteis
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df_raw[cols].dropna()  # Remover linhas com missings originais

    col_types = {
        'Pclass': 'categorical',   # 1, 2, 3 (classes)
        'Sex': 'categorical',      # male, female
        'Age': 'numeric',
        'SibSp': 'numeric',        # número de siblings/spouses
        'Parch': 'numeric',        # número de parents/children
        'Fare': 'numeric',
        'Embarked': 'categorical'  # C, Q, S (portos)
    }

    return df, col_types


def get_dataset_info(name: str) -> Dict:
    """
    Retorna informações sobre um dataset.

    Args:
        name: Nome do dataset ('iris', 'diabetes', 'wine', 'titanic')

    Returns:
        Dict com informações:
            - n_samples: número de amostras
            - n_features: número de features
            - n_numeric: número de features numéricas
            - n_categorical: número de features categóricas
            - is_normalized: se os dados já estão normalizados
            - description: descrição breve
    """
    info = {
        'iris': {
            'n_samples': 150,
            'n_features': 4,
            'n_numeric': 4,
            'n_categorical': 0,
            'is_normalized': False,
            'description': 'Medições de flores Iris (pequeno, baixa-dim)'
        },
        'diabetes': {
            'n_samples': 442,
            'n_features': 10,
            'n_numeric': 10,
            'n_categorical': 0,
            'is_normalized': True,  # IMPORTANTE!
            'description': 'Indicadores de diabetes (médio, JÁ NORMALIZADO)'
        },
        'wine': {
            'n_samples': 178,
            'n_features': 13,
            'n_numeric': 13,
            'n_categorical': 0,
            'is_normalized': False,
            'description': 'Características químicas de vinhos (médio)'
        },
        'titanic': {
            'n_samples': 714,  # aproximado após limpeza
            'n_features': 7,
            'n_numeric': 4,
            'n_categorical': 3,
            'is_normalized': False,
            'description': 'Passageiros do Titanic (misto: num + cat)'
        }
    }

    if name.lower() not in info:
        raise ValueError(f"Dataset desconhecido: {name}. Disponíveis: {list(info.keys())}")

    return info[name.lower()]


def load_dataset(name: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Carrega um dataset pelo nome.

    Args:
        name: Nome do dataset ('iris', 'diabetes', 'wine', 'titanic')

    Returns:
        Tuple contendo:
            - DataFrame com os dados
            - Dict com tipos de coluna

    Raises:
        ValueError: Se nome desconhecido
    """
    loaders = {
        'iris': load_iris_data,
        'diabetes': load_diabetes_data,
        'wine': load_wine_data,
        'titanic': load_titanic_data
    }

    if name.lower() not in loaders:
        raise ValueError(f"Dataset desconhecido: {name}. Disponíveis: {list(loaders.keys())}")

    return loaders[name.lower()]()


def list_datasets() -> list:
    """
    Lista todos os datasets disponíveis.

    Returns:
        Lista de nomes de datasets
    """
    return ['iris', 'diabetes', 'wine', 'titanic']

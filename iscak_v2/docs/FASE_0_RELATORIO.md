# FASE 0: Diagnóstico do Dataset - Relatório

## 1. Detecção de Tipos de Variáveis

### 1.1 Algoritmo Original (v1) - `preprocessing/type_detection.py`

A versão original usa a classe `MixedDataHandler` com a seguinte lógica:

```
Para cada coluna:
1. Se dtype é object/string/category → NOMINAL
2. Se dtype é bool → BINARY
3. Se numérico com exactamente 2 valores {0,1} → BINARY
4. Se numérico com 2-10 valores CONSECUTIVOS → AMBIGUOUS
5. Senão → NUMERIC
```

**Tratamento de AMBIGUOUS (parte interactiva):**
- Pergunta ao utilizador: "Numérica? Nominal? Ordinal?"
- Para ordinal: solicita a ordem dos valores
- Valida a ordem contra valores únicos
- Fallback para numérica se ordem incorrecta

**Funcionalidades extra:**
- `save_config()` / `load_config()` para persistência
- `inverse_transform()` para reverter encoding
- Encoding: Nominal→integer codes, Ordinal→[0,1] uniforme

### 1.2 Algoritmo Novo (v2) - `iscak_v2/core/type_detection.py`

Versão simplificada:

```
Para cada coluna:
1. Se exactamente 2 valores únicos → BINARY
2. Se dtype é string/object → NOMINAL
3. Se numérico com ≤10 valores únicos E são inteiros → ORDINAL
4. Senão → CONTINUOUS
```

**Limitações da v2:**
- ❌ Sem modo interactivo
- ❌ Sem detecção de AMBIGUOUS
- ❌ Sem persistência de configuração
- ❌ Sem inverse_transform

### 1.3 Comparação e Decisão

| Aspecto | v1 | v2 | Decisão |
|---------|----|----|---------|
| Detecção automática | ✅ | ✅ | OK |
| Modo interactivo | ✅ | ❌ | **Recuperar v1** |
| Casos ambíguos | ✅ | ❌ | **Recuperar v1** |
| Persistência | ✅ | ❌ | **Recuperar v1** |
| Inverse transform | ✅ | ❌ | **Recuperar v1** |
| Código limpo | ❌ | ✅ | Manter v2 |

**CONCLUSÃO:** Usar v1 (`MixedDataHandler`) como base, mas integrar no novo pacote.

---

## 2. Análise de Missingness

### 2.1 Cálculo de Taxas (implementado em v2)

```python
# Taxa global
global_rate = n_missing / n_total

# Taxa por coluna
col_rate[j] = n_missing_col[j] / n_rows

# Taxa por linha
row_rate[i] = n_missing_row[i] / n_cols
```

**Status:** ✅ Implementado correctamente

### 2.2 Detecção de Padrão MCAR/MAR/MNAR

**Algoritmo usado (v2):**
1. Para cada coluna j com missings
2. Para cada outra coluna k
3. Comparar médias de k entre grupos "j missing" vs "j observado"
4. T-test para testar diferença
5. Combinar p-values (método de Fisher)
6. Se p-value combinado > 0.05 → MCAR (não rejeitamos aleatoriedade)
7. Se p-value ≤ 0.05 → MAR (missings correlacionados com observados)

**Limitações:**
- Não distingue MAR de MNAR (requer análise mais sofisticada)
- Teste simplificado de Little (não é o teste completo)

**NOTA:** A detecção de MNAR é teoricamente impossível sem informação externa, pois depende dos valores que faltam.

---

## 3. Funções de Inserção de Missings (para benchmark)

### 3.1 MCAR - Missing Completely At Random

```python
def introduce_mcar(data, missing_rate):
    # Máscara aleatória uniforme
    mask = random() < missing_rate

    # Constraints:
    # - Cada linha tem ≥1 valor observado
    # - Cada coluna tem ≥1 valor observado
```

**Hipótese estatística:** P(missing) = constante, independente de X e Y

### 3.2 MAR - Missing At Random

```python
def introduce_mar(data, missing_rate):
    # Escolhe 1ª coluna numérica como "driver"
    driver = data[:, 0]
    median = np.median(driver)

    # Se driver > mediana: P(missing) = rate * 1.5
    # Se driver ≤ mediana: P(missing) = rate * 0.5

    # Driver nunca tem missings (para manter dependência observável)
```

**Hipótese estatística:** P(missing|X_obs) ≠ P(missing), mas P(missing|X_obs, X_miss) = P(missing|X_obs)

### 3.3 MNAR - Missing Not At Random

```python
def introduce_mnar(data, missing_rate):
    # Para cada coluna independentemente:
    # Se valor > mediana: P(missing) = rate * 1.5
    # Se valor ≤ mediana: P(missing) = rate * 0.5
```

**Hipótese estatística:** P(missing|X) depende do próprio valor de X

### 3.4 Limitações das Funções de Inserção

1. **MAR simplificado:** Usa apenas 1 coluna driver (cenários reais podem ter múltiplas dependências)
2. **MNAR simétrico:** Valores altos sempre têm mais missings (cenário real pode ser inverso)
3. **Sem correlação entre colunas:** Missings em colunas diferentes são independentes

**NOTA:** Estas simplificações são aceitáveis para benchmark inicial, mas devem ser documentadas.

---

## 4. Métricas de Avaliação

### 4.1 Algoritmo de Cálculo (código original)

```python
def calculate_metrics_per_column(original, imputed, missing_mask, col_types):
    for col in columns:
        # IMPORTANTE: Só avaliar onde havia missing
        true_values = original[missing_mask[col]]
        imputed_values = imputed[missing_mask[col]]

        if col_types[col] == 'numeric':
            # R², Pearson, NRMSE
        else:
            # Accuracy

    return média das métricas por coluna
```

**CORRECTO:** Avalia apenas os valores que foram imputados (não todos).

### 4.2 Métricas Numéricas

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| R² | 1 - SS_res/SS_tot | 1=perfeito, 0=média, <0=pior que média |
| Pearson | cov(y,ŷ)/(σ_y·σ_ŷ) | 1=correlação perfeita |
| NRMSE | RMSE/(max-min) | 0=perfeito, normalizado pelo range |

### 4.3 Métricas Categóricas

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| Accuracy | n_correct / n_total | 1=perfeito, 0=nenhum correcto |

### 4.4 Verificação de Implementação

Código original `calculate_metrics_per_column`:
- ✅ Usa missing_mask para filtrar apenas valores imputados
- ✅ Separa numérico vs categórico
- ✅ Trata NaN e valores inválidos
- ✅ Calcula média por coluna no final

**CONCLUSÃO:** Implementação correcta.

---

## 5. Dimensionalidade e Estrutura (v2)

### 5.1 Análise de Dimensionalidade

```python
ratio = n_samples / n_features
is_high_dimensional = ratio < 10 or n_features > n_samples
```

**Critério:** Dataset é high-dimensional se ratio n/p < 10

### 5.2 Detecção de Clusters

```python
# Método do cotovelo para encontrar k óptimo
for k in range(1, max_k):
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# Cotovelo = máximo da segunda derivada
k_optimal = argmax(diff(diff(inertias))) + 1
```

**Limitação:** Funciona mal em datasets sem estrutura clara de clusters.

### 5.3 Densidade Local

```python
# k-NN para calcular distância média aos vizinhos
nn = NearestNeighbors(k=5)
distances = nn.kneighbors(data)
density = 1 / mean(distances)
```

**Uso futuro:** Densidade influencia escolha de k adaptativo.

---

## 6. Acções Necessárias

### 6.1 Detecção de Tipos
- [ ] Copiar `MixedDataHandler` para `iscak_v2/core/`
- [ ] Manter modo interactivo
- [ ] Adicionar testes

### 6.2 Notebook de Benchmark
- [ ] Documentar algoritmos de inserção de missings
- [ ] Documentar métricas
- [ ] Verificar que usa `missing_mask` correctamente
- [ ] Adicionar mais datasets variados

### 6.3 Análise de Dataset
- [ ] Testar em datasets reais
- [ ] Validar detecção de padrões MCAR/MAR

---

## 7. Resumo

| Componente | Status | Notas |
|------------|--------|-------|
| Detecção de tipos (v2) | ⚠️ Incompleto | Falta modo interactivo |
| Taxa de missings | ✅ OK | Implementado |
| Padrão MCAR/MAR | ⚠️ Simplificado | Teste de Little básico |
| Dimensionalidade | ✅ OK | Implementado |
| Estrutura (clusters) | ✅ OK | Implementado |
| Outliers/distribuições | ✅ OK | Implementado |

**PRÓXIMO PASSO:** Antes de avançar para MI, precisamos:
1. Recuperar `MixedDataHandler` completo
2. Criar notebook de benchmark robusto e documentado

# 🌿 Checkpoint - Machine Learning & Modelling | Statistical Computing with R & Python.

Neste projeto, mostramos como a análise estatística de dados se integra ao Machine Learning, usando o **Dataset IRIS** — Um dos conjuntos mais conhecidos nessa área. O trabalho foi feito em Python, com três bibliotecas: 
- Pandas para organizar os dados em tabela
- NumPy para os cálculos numéricos
- Scikit-learn, que fornece tanto o dataset quanto o algoritmo KNN.

Antes de qualquer análise, tratamos a qualidade dos dados: verificamos duplicidades, que podem distorcer o treinamento do modelo, e valores ausentes, que podem comprometer os cálculos estatísticos. Em seguida, usamos o `describe()` para resumir estatisticamente as variáveis — média, desvio padrão, mínimo, máximo e quartis.

Com essas etapas, garantimos que os dados estivessem consistentes e prontos para avançar à aplicação do Machine Learning.

---

## 👥 Integrantes do Grupo:

- **Rafael Medici** — RM: 576249.
- **Gustavo Carreiro** — RM: 5763079.
- **Ludson Rodrigo Marciano Fernandes** — RM: 574997.
- **Lucas Longo Rosette** — RM: 576306.
- **Kayky Cayres Vicente da Silva** — RM: 575091.

---

## 📌 Visão Geral do Projeto.

O projeto contempla a integração entre análise estatística computacional e modelagem preditiva de Machine Learning, dividido em duas etapas principais:
1. **Parte 1: Coleta, Exploração, Tratamento e Validação dos Dados.**
2. **Parte 2: Machine Learning, Modelagem, Avaliação de Desempenho & Otimização de Hiperparâmetros com KNN**.**

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas:

- [x] **Linguagem:** Python 3.
- [x] **Manipulação e Análise Estatística de Dados:** `pandas`, `numpy`.
- [x] **Visualização de Dados:** `matplotlib` (Análise Descritiva, Histogramas, Boxplots e Gráficos de Desempenho).
- [x] **Machine Learning & Modelagem:** `scikit-learn` (`load_iris`, `train_test_split`, `KNeighborsClassifier`, `accuracy_score`, `cross_val_score`, `GridSearchCV`).
- [x] **Métrica de Execução:** Módulo nativo `time` para medição de latência do modelo. 

---

## 📋 Estrutura Detalhada e Etapas do Projeto.

### 🔍 Parte 1: Tratamento, Análise Estatística e Qualidade dos Dados.

1. **Coleta de Dados:**
    - Carregamento do Dataset **IRIS** via biblioteca `scikit-learn` em um DataFrame via biblioteca `pandas`.
    - O Dataset **IRIS** possui **150 amostras**, **3 espécies de flores** (*setosa*, *versicolor*, *virginica*) e **4 variáveis numéricas de entrada** medidas em centímetros: `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)` e `petal width (cm)`.

2. **Exploração Inicial e Diagnóstico:**
   - Verificação das dimensões: `(150, 6)` considerando colunas de **target** e nome da espécie.
   - Tipagem adequada: Colunas numéricas como `float64` e identificadores de classe como `int64`/`category`.
   - **Balanceamento de Classes:** Distribuição perfeitamente equilibrada na origem, contendo exatamente **50 amostras (33.3%)** para cada uma das 3 espécies.

3. **Estatística Descritiva e Padrões de Variáveis:**
    - **Comprimento da Sépala (`sepal length`):** Média de ~5.84 cm, variação entre 4.3 cm e 7.9 cm.
   - **Largura da Sépala (`sepal width`):** Média de ~3.06 cm, variação entre 2.0 cm e 4.4 cm.
   - **Comprimento da Pétala (`petal length`):** Média de ~3.76 cm, variação entre 1.0 cm e 6.9 cm (apresenta maior variabilidade).
   - **Largura da Pétala (`petal width`):** Média de ~1.20 cm, variação entre 0.1 cm e 2.5 cm.
   - **Diferenciação Biológica:** A análise das médias por classe revela que a *setosa* possui pétalas significativamente menores (comprimento médio ~1.46 cm) em relação à *versicolor* (~4.26 cm) e *virginica* (~5.55 cm), tornando-a fácil de separar linearmente dos demais grupos.

4. **Tratamento de Dados Ausentes e Duplicidades:**
   - **Valores Ausentes:** Confirmação de **0 valores nulos/NaN** em todo o Dataset.
   - **Linhas Duplicadas:** Identificação de **1 linha totalmente duplicada** no índice 142 (dados pertencentes à classe *virginica*: `[6.0, 2.9, 4.5, 1.5]`).
   - **Ação:** A linha duplicada foi removida com `drop_duplicates()`, mantendo **149 amostras válidas** no Dataset final.
  
5. **Análise de Inconsistências Físicas e Outliers:**
   - **Consistência Numérica:** Nenhuma medida física inconsistente foi detectada ($\le 0$ cm).
   - **Detecção de Outliers:** Identificados valores atípicos através do método IQR e Boxplots na variável `sepal width (cm)` (valores $< 2.05$ cm e $> 4.05$ cm).
   - **Decisão Técnica de Manutenção:** Os pontos atípicos observados (valores altos para a espécie *setosa* e baixo para a espécie *versicolor*) foram mantidos no modelo por se tratarem de variações biológicas reais e legítimas das espécies, e não de erros de digitação ou medição.

---

### 🤖 Parte 2: Machine Learning, Modelagem e Avaliação com $KNN$.

1. **Divisão dos Dados (Treino e Teste):**
   - Separação na proporção de **70% para treino** e **30% para teste.**
   - Uso de `random_state=42` para reprodutibilidade e `stratify=y` para manter a proporção das classes.

2. **Treinamento e Avaliação Numérica ($K = 1, 3, 5, 7, 9$):**
   - Execução do classificador $KNN$ em loop para avaliar o comportamento do parâmetro de vizinhança $K$.
   - Medição do tempo de processamento por inferência com o módulo `time`.

3. **Validação Cruzada ($5$-Fold Cross Validation):**
   - Aplicação de `cross_val_score` para calcular a média (`media_cv`) e o desvio padrão (`desvio_cv`), avaliando a estabilidade e a escolha do conjunto de teste.
   - **Investigação de Underfitting:** Teste com valores extremos de $K$ ($K=75$ e $K=83$) para demonstrar numericamente como um vizinho excessivamente grande diminui a decisão e degradação do desempenho.

4. **Otimização de Hiperparâmetros (`GridSearchCV`):**
   - Utilização do (*Grid Search*) combinando:
   - $K \in [1, 3, 5, 7, 9, 11, 13, 15, ..., 83]$.
   - Esquemas de ponderação de pesos (`weights`): `'uniform'` vs `'distance'`.
   - Avaliação sobre o impacto das funções de distância quando $K$ atinge valores elevados.

---

## 📊 Resultados Comparativos e Conclusões.

### Tabela de Acurácia Simples (Conjunto de Teste).

| $K$ | Acurácia |
| :---: | :---: |
| 1 | 93.33% |
| 3 | 95.56% |
| **5** | **97.78%** |
| 7 | 95.56% |
| 9 | 95.56% |

---

### 2. Validação Cruzada ($5$-Fold CV no Treino).

| $K$ | Acurácia Média CV (`media_cv`) | Desvio Padrão CV (`desvio_cv`) 
| :---: | :---: | :---: |
| 1 | 94.24% | $\pm$ 3.82% | 
| 3 | 94.24% | $\pm$ 3.82% |
| **5** | **95.24%** | **$\pm$ 3.33%** |
| 7 | 94.24% | $\pm$ 3.82% |
| 9 | 94.24% | $\pm$ 3.82% |
| *75* | *64.43%* | $\pm$ 4.88% |
| *83* | *39.43%* | $\pm$ 3.82% |

---

### 3. Impacto da Ponderação de Pesos (`GridSearchCV`).

- **Pesos Uniformes (`weights='uniform'`):** Com $K$ pequeno ($K=5$), obtém o pico de acurácia (97.78%). No entanto, ao elevar $K$ para valores como $75$ e $83$, a acurácia cai para a faixa de $39\%$ a $64\%$, pois vizinhos distantes passam a ter o mesmo peso que os vizinhos próximos.
- **Pesos por Distância (`weights='distance'`):** Atribui pesos inversamente proporcionais à distância do ponto. Mesmo quando $K=83$, o modelo mantém uma acurácia de **93.33%**, pois os pontos mais próximos exercem maior relevância na decisão, neutralizando o efeito de subajuste por excesso de vizinhos.

---

## 💡 Conclusões do Estudo

1. **Atributos Determinantes:** As medidas de pétala (`petal length` e `petal width`) demonstraram ser os separadores mais fortes para distinguir as espécies.
2. **Modelo Escolhido:** O **$KNN$ com $K = 5$ e peso uniforme** apresentou a melhor performance geral, atingindo **97.78% de acurácia no teste** e a maior taxa na validação cruzada ($95.24\%$).









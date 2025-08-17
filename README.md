Titanic - Machine Learning from Disaster

Este projeto foi desenvolvido a partir do famoso desafio do Titanic disponível no Kaggle. O objetivo é prever a sobrevivência de passageiros utilizando técnicas de ciência de dados e machine learning.

📌 Objetivo do Projeto

Analisar os dados dos passageiros do Titanic.

Tratar valores ausentes e realizar engenharia de atributos.

Criar modelos de machine learning para prever quem teria sobrevivido.

Avaliar os resultados e comparar métricas.

🛠️ Tecnologias Utilizadas

Python

Pandas e NumPy → Manipulação e análise de dados

Matplotlib e Seaborn → Visualização de dados

Scikit-learn → Modelagem e avaliação de modelos

📊 Etapas do Projeto

Análise Exploratória (EDA)

Distribuição de idade, sexo, classe social e taxa de sobrevivência.

Identificação de correlações entre variáveis.

Tratamento de Valores Nulos

A coluna Age apresentava valores ausentes (NaN).

Para efeito estatístico, identifiquei a classe (Pclass) dos passageiros com idade nula.

Preenchi os valores de idade ausentes utilizando a mediana da idade por classe (1ª, 2ª ou 3ª classe).

Pré-processamento dos Dados

Criação de novas features.

Normalização e codificação de variáveis categóricas.

Modelagem

Teste com diferentes algoritmos de classificação, como Regressão Logística e Random Forest.

Cálculo da acurácia no dataset de treino (≈ 69,77%).

Submissão no Kaggle com melhor score de 0.74641.

Avaliação

Uso de métricas como acurácia para medir a performance.

📈 Resultados

Acurácia local (treino/validação): 69,77%

Score no Kaggle (submissão): 0.74641

O modelo final conseguiu prever corretamente cerca de 74,6% dos casos na base de teste, alcançando um desempenho competitivo dentro do desafio do Kaggle.

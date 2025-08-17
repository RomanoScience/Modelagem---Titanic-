''' Escopo do Projeto: Criar um modelo que preveja quais passageiros sobreveveria aos naufrago do titanic
que tipo de pessoa tem maior probabilidade de sobrevivier com base na idade,classe social, nome, sexo etc'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv('C:/Users/ronal/Desktop/Proj_Titanica/Dados_Modelar/train.csv')
print(df.head())
''' Com base nos dados irei analisar os dados Sexo (Sex),Idade (Age) e classe social (Pclass) como valores de entrada e dados de saida irei 
considerar a coluna sobreviveu (Survived)'''
dados_entrada = df[['Sex' ,'Age' ,'Pclass']]
dados_saida  = df['Survived']
''' Agora irei analisar existem valores diferentes do esperado como NaN principalmente no dados de entrada e saida'''
t1 = dados_entrada.isnull().sum() 
t2 = dados_saida.isnull().sum()
''' Visto que apenas a coluna Age nos dados de entrada possuem valores Nan irei calcular a media de idade e complementar esses nan com a media ou mediana'''
media_Age = df['Age'].describe()
''' Percebe-se que os dados de idade tem um desvio padrão alto, portanto usar a media é uma opção ruim pois 
trata-se de uma distrhuição assimetrica portanto a variavel mediana é uma boa opção pois esta na parte central da distruição logo irei considerar
a mediana'''
def normalizacao(data):
 frame = data
 social = []
 social = frame.groupby('Pclass')['Age'].median()
 a = social[1]
 b = social[2]
 c = social[3]
 Age_nova = []
 for i in range(len(frame)):
   idade = frame['Age'][i]
   genero = frame['Sex'][i]
   classe = frame['Pclass'][i]
   if pd.isnull(idade) and classe ==1:
      Age_nova.append(a)
   elif pd.isnull(idade) and classe ==2:
      Age_nova.append(b)
   elif pd.isnull(idade) and classe ==3:
      Age_nova.append(c) 
   else:
      Age_nova.append(idade)
 return Age_nova
age_certa= df[['Sex','Age','Pclass']]
age_certa = normalizacao(age_certa)
''' Agora irei remontar o dada frame com a coluna idade limpa'''
dados_entrada = pd.DataFrame({
    'Sex' : df['Sex'].values,
    'Age' : age_certa,
    'Pclass': df['Pclass'].values
})
dados_entrada ['Sex'] = dados_entrada['Sex'].map({'female': 0, 'male':1})
''' Assim será incluindo na primeira coluna da matriz de dados e o valor do termo independente será  a 1 '''
matriz_entrada = dados_entrada
matriz_entrada.insert(0,'B',1)
matriz_dados = matriz_entrada.values
''' Assim minha matriz de dados de entrada fica [B0 | Sex | Age | Pclass ] n x4 n linha com 4 colunas matriz não quadrada'''
matriz_saida = dados_saida.values
''' Z = M*B assim precisarei usar matriz transposta para transformar a matriz de dados de entrada em uma matriz quadrada '''
transposta = matriz_dados.T
''' Assim multiplicando os dois da igualdade pela matriz T para termos z*T = (M*T)*B'''
entrada = np.dot(transposta,matriz_dados)
Z = np.dot(transposta,matriz_saida)
''' Assim como temos uma matriz quadrada posso isolar a matriz B ou seja aplicar o conceito de matriz inversa da matriz de dados de entrada'''
matriz_inversa = np.linalg.pinv(entrada)
''' Assim temos (Z*T)*(M*T^-1) = B'''
B = np.dot(matriz_inversa,Z)
''' Expressão final: Z= B0 + B1*Sex + B2*Age + B3*Pclass'''
print(' Os coeficientes finais da expressão B0,B1,B2,B3 são: ',B)
modelagem = np.dot(matriz_dados,B)
def sigmode(matriz_final):
    probabilidade = []
    for expoente in matriz_final:
     e = 1/(1+np.exp(-expoente))
     if e > 0 and e < 0.6:
        probabilidade.append(0)
     else:
        probabilidade.append(1)
    return probabilidade
''' Calcularei Acuracia da modelagem  selecionando ao acaso 30% da amostra'''
probabilidade_teste=[]
X = matriz_dados
Y = matriz_saida
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=80)
z_test = np.dot(X_test,B)
probabilidade_teste = sigmode(z_test)
acuracia_treino = accuracy_score(Y_test,probabilidade_teste)
print(f' acurácia dos dados train foi de: {acuracia_treino:.3%}')
' Abaixo o codifo para Submissão do arquivo teste Para plataforma Kaggle'
test = []
test = pd.read_csv('C:/Users/ronal/Desktop/Proj_Titanica/Dados_Testar/test.csv')
dados_treino = test[['Age','Sex','Pclass']]
normalizar = normalizacao(dados_treino)
dados_treino = pd.DataFrame({
    'Sex' : test['Sex'].values,
    'Age' : normalizar,
    'Pclass': test['Pclass'].values
})
dados_treino ['Sex'] = dados_treino['Sex'].map({'female': 0, 'male':1})
dados_treino.insert(0, 'B', 1)  # termo independente
matriz_teste = dados_treino.values
z_prev = np.dot(matriz_teste, B)
prev_final = sigmode(z_prev)

# Cria submissão no formato exigido
submissao = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': prev_final
})

# Salva no CSV
submissao.to_csv('C:/Users/ronal/Desktop/Proj_Titanica/submissao_titanic.csv', index=False)
print("✅ Arquivo 'submissao_titanic.csv' gerado com sucesso!")
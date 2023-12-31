import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

file_path = 'archive02/wine_dataset.csv'
dados = pd.read_csv(file_path)
dados['style'] = dados['style'].replace('red', 1)
dados['style'] = dados['style'].replace('white', 0)
print(dados.head())
print(dados)

y = dados['style']
x = dados.drop('style', axis = 1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

resultado = modelo.score(x_teste, y_teste)
print("Acuria:", resultado)

print(dados.shape, x_treino.shape, x_teste.shape, y_treino.shape, y_teste.shape)

print(y_teste[400:405])

previsao = modelo.predict(x_teste[400:405])
print(previsao)

"""
x = dados[["fixed_acidity","volatile_acidity",
       "citric_acid","residual_sugar","chlorides",
       "free_sulfur_dioxide","total_sulfur_dioxide",
       "density","pH","sulphates","alcohol","quality"]]

y = dados["style"]
print(x.head())
print(y.head())
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

file_path = 'archive02/wine_dataset.csv'
dado = pd.read_csv(file_path)
dado['style'] = dado['style'].replace('red', 1)
dado['style'] = dado['style'].replace('white', 0)
print(dado.head())
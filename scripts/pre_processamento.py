import numpy as np
import pandas as pd
import warnings

from feature_engine.encoding import OneHotEncoder
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 39)

url = 'https://raw.githubusercontent.com/ryanrodr/ml-heart-disease/main/dados/heart_statlog_cleveland_hungary_final.csv'

# Carregar dados
df = pd.read_csv(url)

# Renomear colunas substituindo espaços por underscores
df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# Identificar e converter variáveis categóricas para 'category'
cat_features = ['sex', 'chest_pain_type', 'resting_ecg', 'ST_slope']
df[cat_features] = df[cat_features].astype('category')

# Codificar variáveis categóricas usando OneHotEncoder
onehot = OneHotEncoder(variables=cat_features)
df = onehot.fit_transform(df)

# Selecionar e normalizar variáveis contínuas
colunas_norm = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak']
norm = StandardScaler()
df[colunas_norm] = norm.fit_transform(df[colunas_norm])

# Caminho completo para o arquivo CSV
caminho = '/home/ryanrodr/GitHub/ml-heart-disease/dados/heart_disease_preprocessed.csv'

# Salvar o DataFrame como um arquivo CSV
df.to_csv(caminho, index=False)
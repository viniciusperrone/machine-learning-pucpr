import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings('ignore')

# Importação do Dataset
df = pd.read_excel("csgo_round_snapshots.xlsx")
df.head()

# Análise exploratória
df.info()
df.describe()
df['round_winner'].value_counts().plot(kind='bar')
plt.title('Distribuição do Vencedor do Round')

# Pré-processamento de dados
df.dropna(inplace=True)
df.drop(['match_id', 'round_num'], axis=1, inplace=True)

df = pd.get_dummies(df, columns=['map'], drop_first=True)
df['round_winner'] = df['round_winner'].map({'CT': 0, 'T': 1})

# Seleção de Atributos
X = df.drop('round_winner', axis=1)
y = df['round_winner']

selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features)

# Divisão de treinos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

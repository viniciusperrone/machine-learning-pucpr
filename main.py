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
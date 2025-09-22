#ATIVIDADE FEITA POR: PEDRO GOMES ROBERTE SILVA - 20221si002

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Carregar o dataset completo
cancer = load_breast_cancer()
X_full = cancer.data # Usando todas as 30 features
y_full = cancer.target

print("="*50)
print("VERSÃO B: USANDO 30 FEATURES")
print(f"Features selecionadas: {X_full.shape[1]} (todas)")
print(f"Classes: {cancer.target_names}")
print("="*50)

# 2. Aplicar o pipeline padrão
# Dividir em treino/teste
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.3, random_state=1, stratify=y_full
)

# Normalizar os dados
sc_full = StandardScaler()
sc_full.fit(X_train_full)
X_train_std_full = sc_full.transform(X_train_full)
X_test_std_full = sc_full.transform(X_test_full)

model_30f = Perceptron(eta0=0.01, random_state=1)
model_30f.fit(X_train_std_full, y_train_full)

# 3. Fazer predições e avaliar com as métricas
y_pred_30f = model_30f.predict(X_test_std_full)

# 4. Exibir os resultados
print(f"Acurácia (30 features): {accuracy_score(y_test_full, y_pred_30f) * 100:.2f}%\n")

print("Matriz de Confusão (30 features):")
# Linhas: Real, Colunas: Previsto
#           Previsto Maligno | Previsto Benigno
# Real Maligno      [ TP ]   |      [ FN ]
# Real Benigno      [ FP ]   |      [ TN ]
print(confusion_matrix(y_test_full, y_pred_30f))
print("\nRelatório de Classificação (30 features):")
print(classification_report(y_test_full, y_pred_30f, target_names=cancer.target_names))
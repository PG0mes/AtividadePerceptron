#ATIVIDADE FEITA POR: PEDRO GOMES ROBERTE SILVA - 20221si002

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# 1. Carregar o dataset e selecionar 2 features
cancer = load_breast_cancer()
X = cancer.data[:, :2] 
y = cancer.target

print("="*50)
print("VERSÃO A: USANDO 2 FEATURES")
print(f"Features selecionadas: {cancer.feature_names[:2]}")
print(f"Classes: {cancer.target_names}")
print("="*50)


# 2. Aplicar o pipeline padrão
# Dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# Normalizar os dados
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

model_2f = Perceptron(eta0=0.01, random_state=1)
model_2f.fit(X_train_std, y_train)

# 3. Fazer predições e avaliar com as métricas
y_pred_2f = model_2f.predict(X_test_std)

print(f"Acurácia (2 features): {accuracy_score(y_test, y_pred_2f) * 100:.2f}%\n")

print("Matriz de Confusão (2 features):")
print(confusion_matrix(y_test, y_pred_2f))
print("\nRelatório de Classificação (2 features):")
print(classification_report(y_test, y_pred_2f, target_names=cancer.target_names))


# 4. Função para plotar
def plot_decision_regions(X, y, classifier, test_idx=None):
    markers = ('o', 's')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cancer.target_names[cl],
                    edgecolor='black')

# Plotando o resultado
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt.figure(figsize=(10, 6))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=model_2f)
plt.title('Perceptron no Câncer de Mama (2 Features)')
plt.xlabel(f'{cancer.feature_names[0]} [normalizado]')
plt.ylabel(f'{cancer.feature_names[1]} [normalizado]')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
#ATIVIDADE FEITA POR: PEDRO GOMES ROBERTE SILVA - 20221si002

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# -----------------------------------------------------------
# 1. Carregar e preparar o dataset
# -----------------------------------------------------------
iris = datasets.load_iris()
# Usar apenas as classes 0 (Setosa) e 1 (Versicolor) e as 2 features
X = iris.data[iris.target != 2, :2] 
y = iris.target[iris.target != 2]

# -----------------------------------------------------------
# 2. Passos a seguir
# -----------------------------------------------------------

# Passo 1: Dividir em treino/teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# Passo 2: Normalizar os dados
sc = StandardScaler()
sc.fit(X_train) 
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Passo 3: Treinar o perceptron
model = Perceptron(eta0=0.01, random_state=1)
model.fit(X_train_std, y_train)

# Passo 5: Calcular e reportar a acurácia
y_pred = model.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classes: Setosa vs Versicolor")
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")

# Passo 4: Plotar as regiões de decisão
def plot_decision_regions(X, y, classifier, test_idx=None):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=f'Classe {cl}',
                    edgecolor='black')
    
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Conjunto de teste')


# Combinando os dados normalizados para plotagem
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt.figure(figsize=(10, 6))
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=model, test_idx=range(len(y_train), len(y_combined)))

plt.title('Região de Decisão do Perceptron (Setosa vs Versicolor)')
plt.xlabel('Comprimento da Sépala [normalizado]')
plt.ylabel('Largura da Sépala [normalizado]')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

#Resposta para a Pergunta de Reflexão: "O que acontece se você usar Versicolor vs Virginica (classes 1 e 2)?"
#O Perceptron terá um desempenho muito ruim, com uma acurácia baixa, porque essas duas classes não são linearmente separáveis.
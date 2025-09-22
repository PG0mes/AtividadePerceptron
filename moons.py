#ATIVIDADE FEITA POR: PEDRO GOMES ROBERTE SILVA - 20221si002

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# -----------------------------------------------------------
# 1. Gerar o dataset "Moons"
# -----------------------------------------------------------
X, y = make_moons(
    n_samples=200,
    noise=0.15,
    random_state=42
)

# -----------------------------------------------------------
# 2. Aplicar os mesmos passos do exercício anterior
# -----------------------------------------------------------

# Passo 1: Dividir em treino/teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Passo 2: Normalizar os dados
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Passo 3: Treinar o Perceptron
model = Perceptron(eta0=0.1, random_state=42)
model.fit(X_train_std, y_train)

# Passo 4: Calcular e reportar a acurácia
y_pred = model.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print(f'Análise do Perceptron no Moons Dataset')
print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')

# -----------------------------------------------------------
# 3. Plotar as regiões de decisão
# -----------------------------------------------------------

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
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

# Combinando dados para visualização completa
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt.figure(figsize=(10, 6))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=model)

plt.title('Falha do Perceptron em Dados Não-Lineares (Moons)')
plt.xlabel('Feature 1 [normalizada]')
plt.ylabel('Feature 2 [normalizada]')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

#Resposta para a Pergunta de Reflexão: Como você modificaria o algoritmo para resolver este problema?
#O problema fundamental é que o Perceptron só consegue aprender fronteiras de decisão lineares (retas). Para resolver problemas com dados não-linearmente separáveis como o "Moons", precisamos usar algoritmos mais sofisticados que possam aprender fronteiras não-lineares (curvas).
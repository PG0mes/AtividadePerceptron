import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=50):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = 0.0
        self.errors_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.errors_history = []

        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)
            self.errors_history.append(errors)

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Configurar gerador de cores
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plotar a superfície de decisão
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plotar os pontos de dados
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0], 
            y=X[y == cl, 1],
            alpha=0.8, 
            c=colors[idx],
            marker=markers[idx], 
            label=f'Classe {cl}', 
            edgecolor='black'
        )
    plt.legend(loc='upper left')

# PASSO 1: Gerar o Dataset
print("=" * 50)
print("EXEMPLO: BLOBS SINTÉTICOS")
print("=" * 50)

# make_blobs cria clusters gaussianos
X, y = datasets.make_blobs(
    n_samples=200,        # Total de pontos
    n_features=2,         # Número de features (2 para visualização)
    centers=2,            # Número de clusters (classes)
    cluster_std=1.5,      # Desvio padrão dos clusters
    center_box=(-5, 5),   # Limites para os centros
    random_state=42       # Seed para reprodutibilidade
)

print(f"Dataset gerado:")
print(f"- Amostras: {X.shape[0]}")
print(f"- Features: {X.shape[1]}")
print(f"- Classes: {np.unique(y)}")

# PASSO 2: Dividir em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,      # 30% para teste
    random_state=42,
    stratify=y          # Mantém proporção das classes
)

print(f"\nDivisão treino/teste:")
print(f"- Treino: {len(X_train)} amostras")
print(f"- Teste: {len(X_test)} amostras")

# PASSO 3: Normalização (Importante!)
"""
Por que normalizar?
- Garante que todas features tenham a mesma escala
- Previne que features com valores grandes dominem
- Acelera convergência
- Método: z-score (média=0, desvio=1)
"""
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)  # Fit no treino
X_test_std = scaler.transform(X_test)        # Apenas transform no teste

# PASSO 4: Treinar o Perceptron
ppn = Perceptron(learning_rate=0.01, n_epochs=50)
ppn.fit(X_train_std, y_train)

# PASSO 5: Avaliar o Modelo
y_pred = ppn.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nResultados:")
print(f"- Acurácia: {accuracy:.2%}")
print(f"- Erros finais no treino: {ppn.errors_history[-1]}")

# Verificar convergência
if 0 in ppn.errors_history:
    conv_epoch = ppn.errors_history.index(0)
    print(f"- Convergiu na época: {conv_epoch + 1}")
else:
    print("- Não convergiu completamente")

# PASSO 6: Visualizar Resultados
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Regiões de Decisão
axes[0].set_title('Regiões de Decisão - Blobs')
plot_decision_regions(X_train_std, y_train, classifier=ppn)
axes[0].set_xlabel('Feature 1 (normalizada)')
axes[0].set_ylabel('Feature 2 (normalizada)')

# Subplot 2: Curva de Convergência
axes[1].plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, marker='o')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('Número de erros')
axes[1].set_title('Convergência do Treinamento')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# PASSO 7: Análise dos Pesos Aprendidos
print(f"\nPesos aprendidos:")
print(f"- w1: {ppn.weights[0]:.4f}")
print(f"- w2: {ppn.weights[1]:.4f}")
print(f"- bias: {ppn.bias:.4f}")

# A equação da fronteira de decisão é:
# w1*x1 + w2*x2 + bias = 0
# ou seja: x2 = -(w1/w2)*x1 - (bias/w2)
if ppn.weights[1] != 0:
    slope = -ppn.weights[0] / ppn.weights[1]
    intercept = -ppn.bias / ppn.weights[1]
    print(f"\nEquação da fronteira de decisão:")
    print(f"x2 = {slope:.2f} * x1 + {intercept:.2f}")
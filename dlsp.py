#ATIVIDADE FEITA POR: PEDRO GOMES ROBERTE SILVA - 20221si002

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

#==============================================================================
# PARTE 1: DATASET LINEARMENTE SEPARÁVEL
#==============================================================================
print("="*60)
print("PARTE 1: DATASET LINEARMENTE SEPARÁVEL (CENTROS EM [-2,-2] E [2,2])")
print("="*60)

# 1. Criar dataset customizado bem separado
np.random.seed(42)
class_0_sep = np.random.randn(50, 2) + [-2, -2]
class_1_sep = np.random.randn(50, 2) + [2, 2]
X_sep = np.vstack([class_0_sep, class_1_sep])
y_sep = np.hstack([np.zeros(50), np.ones(50)])

# 2. Treinar o Perceptron
sc_sep = StandardScaler()
X_sep_std = sc_sep.fit_transform(X_sep)
model_sep = Perceptron(eta0=0.01, random_state=42)
model_sep.fit(X_sep_std, y_sep)

# Verificar acurácia
accuracy_sep = model_sep.score(X_sep_std, y_sep)
print(f"Acurácia no dataset separado: {accuracy_sep * 100:.2f}%")

# 3. Análise geométrica: Calcular a reta de decisão
w_sep = model_sep.coef_[0]
b_sep = model_sep.intercept_[0]
x1_min_sep, x1_max_sep = X_sep_std[:, 0].min() - 1, X_sep_std[:, 0].max() + 1
x2_decision_line_sep = -(w_sep[0] * np.array([x1_min_sep, x1_max_sep]) + b_sep) / w_sep[1]

# 4. Plotar o primeiro gráfico
plt.figure(figsize=(10, 6))
plt.scatter(X_sep_std[y_sep == 0, 0], X_sep_std[y_sep == 0, 1],
            color='red', marker='o', label='Classe 0', edgecolor='k')
plt.scatter(X_sep_std[y_sep == 1, 0], X_sep_std[y_sep == 1, 1],
            color='blue', marker='s', label='Classe 1', edgecolor='k')
plt.plot([x1_min_sep, x1_max_sep], x2_decision_line_sep,
         color='green', linestyle='--', linewidth=3, label='Reta de Decisão')
plt.title('Dataset Personalizado e a Geometria da Solução')
plt.xlabel('Feature 1 [normalizada]')
plt.ylabel('Feature 2 [normalizada]')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()


#==============================================================================
# PARTE 2: EXPERIMENTO COM DADOS SOBREPOSTOS
#==============================================================================
print("\n" + "="*60)
print("PARTE 2: EXPERIMENTO COM DADOS SOBREPOSTOS (CENTROS EM [-0.5,-0.5] E [0.5,0.5])")
print("="*60)

# 1. Criar dataset com centros próximos
np.random.seed(42)
class_0_overlap = np.random.randn(50, 2) + [-0.5, -0.5]
class_1_overlap = np.random.randn(50, 2) + [0.5, 0.5]
X_overlap = np.vstack([class_0_overlap, class_1_overlap])
y_overlap = np.hstack([np.zeros(50), np.ones(50)])

# 2. Treinar o Perceptron
sc_overlap = StandardScaler()
X_overlap_std = sc_overlap.fit_transform(X_overlap)
model_overlap = Perceptron(eta0=0.01, random_state=42)
model_overlap.fit(X_overlap_std, y_overlap)

# Verificar acurácia
accuracy_overlap = model_overlap.score(X_overlap_std, y_overlap)
print(f"Acurácia no dataset sobreposto: {accuracy_overlap * 100:.2f}%")
print("A acurácia caiu, pois o Perceptron não consegue mais traçar uma reta perfeita.")


# 3. Análise geométrica: Calcular a reta de decisão
w_overlap = model_overlap.coef_[0]
b_overlap = model_overlap.intercept_[0]
x1_min_overlap, x1_max_overlap = X_overlap_std[:, 0].min() - 1, X_overlap_std[:, 0].max() + 1
x2_decision_line_overlap = -(w_overlap[0] * np.array([x1_min_overlap, x1_max_overlap]) + b_overlap) / w_overlap[1]

# 4. Plotar o segundo gráfico
plt.figure(figsize=(10, 6))
plt.scatter(X_overlap_std[y_overlap == 0, 0], X_overlap_std[y_overlap == 0, 1],
            color='red', marker='o', label='Classe 0', edgecolor='k')
plt.scatter(X_overlap_std[y_overlap == 1, 0], X_overlap_std[y_overlap == 1, 1],
            color='blue', marker='s', label='Classe 1', edgecolor='k')
plt.plot([x1_min_overlap, x1_max_overlap], x2_decision_line_overlap,
         color='green', linestyle='--', linewidth=3, label='Reta de Decisão')
plt.title('Falha do Perceptron com Dados Sobrepostos')
plt.xlabel('Feature 1 [normalizada]')
plt.ylabel('Feature 2 [normalizada]')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
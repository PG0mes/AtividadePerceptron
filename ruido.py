import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from copy import deepcopy

# Parâmetros para o experimento
class_sep_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
flip_y_values = [0, 0.05, 0.1, 0.2]

print("="*60)
print("EXPERIMENTO COMPLETO COM EARLY STOPPING")
print("="*60)

# Loop principal do experimento
for sep in class_sep_values:
    for flip in flip_y_values:
        # 1. Gerar o dataset com os parâmetros da vez
        X, y = make_classification(
            n_samples=500,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=sep,
            flip_y=flip,
            random_state=42
        )

        # 2. Dividir em TREINO, VALIDAÇÃO e TESTE
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )

        # 3. Normalizar os dados
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_val_std = sc.transform(X_val)
        X_test_std = sc.transform(X_test)

        # 4. Treinamento com Early Stopping
        model = Perceptron(eta0=0.01, random_state=42, warm_start=True)
        
        patience = 5
        epochs_no_improve = 0
        best_val_accuracy = 0.0
        best_model = None
        max_epochs = 100

        for epoch in range(max_epochs):
            model.partial_fit(X_train_std, y_train, classes=np.unique(y))
            val_accuracy = model.score(X_val_std, y_val)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
                best_model = deepcopy(model)
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve == patience:
                break
        
        # 5. Avaliar o melhor modelo no conjunto de teste
        if best_model is None:
            best_model = model

        y_pred_final = best_model.predict(X_test_std)
        final_accuracy = accuracy_score(y_test, y_pred_final)
        
        # 6. Imprimir o resultado para esta combinação de parâmetros
        print(f"class_sep: {sep:<4} | flip_y: {flip:<4} | Acurácia (Teste): {final_accuracy * 100:.2f}%")
    print("-" * 60)
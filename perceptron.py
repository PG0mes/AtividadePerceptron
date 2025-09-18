import numpy as np

class Perceptron:
    
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_history = []  # Para acompanhar o progresso

    def activation(self, x):
        """Função de ativação binária (step function)."""
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        
        # PASSO 1: Inicialização dos pesos
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # PASSO 2: Loop de treinamento
        for epoch in range(self.n_epochs):
            errors = 0
            
            # PASSO 3: Para cada exemplo de treinamento
            for idx, x_i in enumerate(X):
                # 3.1: Calcula a saída líquida (net input)
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # 3.2: Aplica função de ativação
                y_predicted = self.activation(linear_output)
                
                # 3.3: Calcula o erro
                error = y[idx] - y_predicted
                
                # 3.4: Atualiza pesos e bias (Regra Delta)
                update = self.learning_rate * error
                self.weights += update * x_i
                self.bias += update
                
                # Conta erros para monitoramento
                errors += int(update != 0.0)
            
            self.errors_history.append(errors)
            
            # Parada antecipada se convergiu
            if errors == 0:
                print(f"Convergiu na época {epoch + 1}")
                break

    def net_input(self, X):
        """Calcula a entrada líquida (weighted sum + bias)."""
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """Faz predições para novos dados."""
        return self.activation(self.net_input(X))

if __name__ == "__main__":
    # Dados de exemplo para teste
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # Operação AND

    # Cria e treina o perceptron
    perceptron = Perceptron(learning_rate=0.1, n_epochs=10)
    perceptron.fit(X, y)

    # Faz predições
    predictions = perceptron.predict(X)
    print("Predições:", predictions)
    print("Pesos finais:", perceptron.weights)
    print("Bias final:", perceptron.bias)
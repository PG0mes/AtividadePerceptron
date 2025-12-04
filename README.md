# Atividade Prática de Inteligência Artificial

## Aluno

- Pedro Gomes R. Silva
- Yuri Pessanha Carneiro


## 1. Iris

- 1. Descrição do Dataset

Número de amostras e features: Foram utilizadas 100 amostras (as duas primeiras classes do dataset Iris) e 2 features selecionadas (comprimento da sépala e comprimento da pétala).

Distribuição das classes: Perfeitamente balanceado, com 50 amostras da classe 0 (Setosa) e 50 da classe 1 (Versicolor).

É linearmente separável? Sim. As duas classes selecionadas com as features escolhidas são classicamente conhecidas por serem perfeitamente separáveis por uma linha reta.

- 2. Resultados

Acurácia no treino e teste: Como o dataset foi usado integralmente para treino e visualização, a acurácia foi de 100%. Em uma divisão treino/teste, o resultado esperado também seria de 100%.

Número de épocas até convergência: O algoritmo convergiu muito rapidamente, em poucas épocas.

Tempo de treinamento: Extremamente baixo.

- 3. Visualizações

Gráfico de convergência: Não foi gerado, mas mostraria a acurácia atingindo 100% rapidamente e se estabilizando.

Regiões de decisão: O gráfico mostrou uma linha reta que separava perfeitamente as duas nuvens de pontos (Setosa e Versicolor), sem nenhum erro de classificação.

Matriz de confusão: Uma matriz perfeita, com zeros nas posições de erro (diagonal secundária) e os totais de cada classe na diagonal principal.

- 4. Análise

O perceptron foi adequado para este problema? Sim, foi perfeitamente adequado. O Perceptron é ideal para problemas binários e linearmente separáveis, que é exatamente o cenário deste exercício.

Que melhorias você sugeriria? Para esta tarefa específica, nenhuma melhoria é necessária. O modelo mais simples resolveu o problema com 100% de eficácia.

Comparação com suas expectativas: O resultado correspondeu exatamente à expectativa de que um classificador linear simples poderia resolver este problema clássico com perfeição.

## 2. Moons

- 1. Descrição do Dataset

Número de amostras e features: 200 amostras e 2 features.

Distribuição das classes: Perfeitamente balanceado, com 100 amostras em cada uma das duas classes.

É linearmente separável? Não. O dataset é projetado para ter um formato de duas luas entrelaçadas, sendo um exemplo clássico de problema não-linear.

- 2. Resultados

Acurácia no treino e teste: A acurácia no conjunto de teste ficou em torno de 85% - 90%. Embora não seja aleatória, está longe de ser perfeita.

Número de épocas até convergência: O algoritmo geralmente converge, mas a fronteira de decisão não é ótima. O número de épocas tende a ser maior do que no caso do Iris.

Tempo de treinamento: Muito baixo.

- 3. Visualizações

Gráfico de convergência: Não foi gerado.

Regiões de decisão: A visualização foi a parte mais importante. Ela mostrou claramente uma linha reta tentando, sem sucesso, separar as duas luas curvas. A linha cortava ambas as luas, ilustrando visualmente por que a acurácia não era 100%.

Matriz de confusão: A matriz mostrou erros de classificação para ambas as classes.

- 4. Análise

O perceptron foi adequado para este problema? Não, foi inadequado. Sua incapacidade de aprender fronteiras não-lineares o tornou a ferramenta errada para este trabalho.

Que melhorias você sugeriria? A substituição do Perceptron por um modelo capaz de aprender fronteiras não-lineares.

Comparação com suas expectativas: O resultado correspondeu à expectativa. O objetivo era demonstrar a limitação do Perceptron, e o experimento fez isso com clareza.

## 3. Breast

- 1. Descrição do Dataset

Número de amostras e features: 569 amostras e 30 features.

Distribuição das classes: Desbalanceado, com 357 amostras da classe "benigno" e 212 da classe "maligno".

É linearmente separável? É considerado um problema quase linearmente separável, pois consegue uma boa separação, mas não perfeita.

- 2. Resultados (para a versão com 30 features)

Acurácia no treino e teste: A acurácia no conjunto de teste foi alta, geralmente na faixa de 94% a 97%.

Número de épocas até convergência: O modelo convergiu rapidamente, em poucas épocas.

Tempo de treinamento: Muito baixo.

- 3. Visualizações

Gráfico de convergência: Não foi gerado.

Regiões de decisão: Não foi possível plotar devido ao alto número de dimensões (30 features).

Matriz de confusão: Esta foi a visualização mais crítica. Ela mostrou que o modelo tinha um bom desempenho geral, mas cometia alguns erros, incluindo os perigosos Falsos Negativos.

- 4. Análise

O perceptron foi adequado para este problema? Em termos de acurácia foi, mas inadequado do ponto de vista médico/prático devido à natureza de seus erros. Ele não oferece controle sobre o tipo de erro que comete.

Que melhorias você sugeriria? A principal melhoria seria usar um modelo que lida melhor com a incerteza e as consequências dos erros.

Comparação com suas expectativas: O resultado mostrou que, embora a acurácia seja uma métrica importante, ela pode ser enganosa em problemas do mundo real, especialmente na área médica, onde o custo de diferentes erros é drasticamente diferente.

## 4. Ruido

- 1. Descrição do Dataset

Número de amostras e features: 500 amostras e 2 features.

Distribuição das classes: Balanceado (250/250).

É linearmente separável? Não. Por design, o dataset tinha sobreposição entre as classes (controlado por class_sep) e ruído nos rótulos (controlado por flip_y).

- 2. Resultados (Resumo do Experimento)

Acurácia no treino e teste: A acurácia no teste variou significativamente, como esperado. Com alta separação (class_sep=3.0) e baixo ruído (flip_y=0), a acurácia foi alta. Com baixa separação (class_sep=0.5) e alto ruído (flip_y=0.2), a acurácia foi muito baixa, próxima de 50-60%.

Número de épocas até convergência: O early stopping foi implementado com sucesso, parando o treino quando a acurácia de validação não melhorava por 5 épocas consecutivas.

Tempo de treinamento: Baixo para cada execução, mas o experimento completo levou mais tempo devido ao loop.

- 3. Visualizações

Gráfico de convergência: Foi possível observar a acurácia de validação oscilando e o treinamento parando quando ela se estabilizava ou piorava, provando a utilidade do early stopping.

Regiões de decisão: Não foram geradas para o relatório final, mas os resultados numéricos substituíram a análise visual.

Matriz de confusão: Não foi o foco principal, mas refletiria a acurácia observada em cada cenário.

- 4. Análise

O perceptron foi adequado para este problema? Foi adequado para o propósito do estudo: analisar a sensibilidade de um modelo linear a dados imperfeitos.

Que melhorias você sugeriria? A principal melhoria testada foi uma técnica de treinamento, não um modelo: o Early Stopping. Ele ajudou a encontrar um bom ponto de parada e evitou que o modelo continuasse treinando em dados ruidosos sem necessidade, prevenindo o sobreajuste.

Comparação com suas expectativas: O experimento confirmou as expectativas de forma clara: a performance de modelos de machine learning é altamente dependente da qualidade dos dados de entrada.

## 5. DLSP

- 1. Descrição do Dataset

Cenário A: 100 amostras (50/50), 2 features. Sim, perfeitamente separável (centros em [-2,-2] e [2,2]).

Cenário B: 100 amostras (50/50), 2 features. Não, não separável (centros em [-0.5,-0.5] e [0.5,0.5]).

- 2. Resultados

Acurácia no treino e teste:

Cenário A: 100%.

Cenário B: < 100% (geralmente em torno de 90%, dependendo da sobreposição).

Número de épocas até convergência: Convergência rápida no Cenário A; mais lenta ou instável no Cenário B.

Tempo de treinamento: Muito baixo em ambos os cenários.

- 3. Visualizações
   
Gráfico de convergência: Não foi gerado.

Regiões de decisão: Foi a visualização central.

No Cenário A, a reta de decisão calculada a partir dos pesos do modelo separou os dois grupos de forma limpa e perfeita.

No Cenário B, a reta de decisão foi mostrada cortando através dos grupos sobrepostos, ilustrando os pontos que seriam classificados incorretamente.

Matriz de confusão: Refletiria as acurácias de 100% (Cenário A) e <100% (Cenário B).

- 4. Análise

O perceptron foi adequado para este problema? Perfeitamente adequado para o Cenário A; inadequado para o Cenário B.

Que melhorias você sugeriria? Nenhuma para o Cenário A. Para o Cenário B, os mesmos modelos não-lineares do Exercício 2 seriam necessários para desenhar uma fronteira curva em volta dos grupos.

Comparação com suas expectativas: Este exercício forneceu a prova visual e geométrica para a teoria por trás de todos os outros exercícios. A capacidade de extrair os pesos e plotar a reta de decisão conectou diretamente o resultado matemático do treinamento com uma representação visual clara, atendendo perfeitamente às expectativas.

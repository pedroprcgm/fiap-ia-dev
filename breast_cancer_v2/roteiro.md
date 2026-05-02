
# Abertura - 1 min (tela inicial)
Explicar o problema
Explicar a solução

# Bibliotecas e base dados - 1 min (mostrar seções #1 e #2)
Pincelar as bibliotecas, mencionar as mais relevantes
Falar sobre o dataset, destacar as colunas e o tamanho da base
Destacar os métodos que utilizamos para visualizar os dados e analisar a integridade da base

# Pré processamento - 2min (seção #3)
Destacar a conversão de M e B para 1 e 0 e como isso facilita para o modelo
Manter o código comentado de remoção de colunas e destacar como testamos remover colunas que achamos desnecessárias, porém o resultado não foi satisfatório
Destacar colunas ID e Unnamed removidas
Destacar a matriz de correlação

# Treinamento e resultado dos modelos - 2min (seção #4 e #5)
Destacar os dois modelos selecionados - SVC linear, Regressão linear
Destacar o peso do erro na análise (1 pra 5)
Mostrar o resultado da comparação entre os modelos
SVC linear com melhor resposta, menos erros (Recall melhor) - importante para medicina e ainda mais nesse caso especial

# Visualização da tomada de decisão dos modelos - 2 min (seção #6)
Mostrar feature importance de ambos 
Destacar como os modelos não possuem feature importance diretamente e o cálculo é feito de forma diferente para ambos
Destacar o SHAP para visualização mais rica do peso de cada feature no modelo
(Se persistir com erro no SHAP do SVC destacar que a base de dados gera algum erro de cálculo e o gráfico gerado fica desproporcional)

# Encerramento - 2 min
Destacar o aprendizado, os testes feitos, os desafios devido a talvez correlação alta entre features (talvez isso seja o problema de overflow em alguns calculos)
Destacar que o modelo SVC Linear seria o mais indicado nesse estudo devido ao recall melhor, devido ao peso do erro. Salientar que mesmo assim a análise humana e profissional segue sendo necessária
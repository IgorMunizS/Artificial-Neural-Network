import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Definindo função de ativação
# Função degrau bipolar
def step(x):
    if x >= 0:
        return 1
    else:
        return 0


# Dados de treinamento
bias = 1

training_data = [(np.array([0, 0, bias]), 0),
                 (np.array([0, 1, bias]), 1),
                 (np.array([1, 0, bias]), 1),
                 (np.array([1, 1, bias]), 1)]

# vetor pesos
w1 = 0.3
w2 = 0.4
b = -0.5

vetor_pesos = [w1, w2, b]

#Inicialização de paramêtros
max_epocas = 10
epoca = 0
learning_rate = 0.1
erro = 1

# Fase de treinamento

while epoca < max_epocas and erro >0:

    epoca += 1
    vetor_error = []

    #plot
    w1, w2, b = vetor_pesos
    m = -b/w1
    n = -b/w2

    d = n
    c = -n/m
    line_x = np.array([0, m])
    line_y  = c*line_x + d
    plt.plot(line_x, line_y)
    scatter_x = [x[0][0] for x in training_data]
    scatter_y = [x[0][1] for x in training_data]
    color_target = [x[1] for x in training_data]
    plt.scatter(scatter_x, scatter_y, s=75, c=color_target) #c=[x[1] for x in training_data]
    plt.title('Epoca: %s'%(epoca))
    plt.show()


    for n in range(len(training_data)):

        x, y_expected = training_data[n]
        entrada = np.dot(vetor_pesos, x)
        y  = step(entrada)
        erro = (y_expected - y)
        vetor_error.append(abs(erro))
        for i in range(3):
            vetor_pesos[i] += learning_rate*erro*x[i]





    erro = np.sum(vetor_error)
    print(erro)
    print(vetor_error)


# Fase de teste
acertos = 0
it = 0
while it < 100:

    for x, y_expected in training_data:
        entrada = np.dot(vetor_pesos, x)
        y = step(entrada)
        #print("{} -> {}".format(entrada, y))
        if y == y_expected:
            acertos += 1
        else:
            erro += 1

    it += 1

print("Acurácia: ", acertos/4, "%" )
print(vetor_pesos)
print(epoca)
print("Erros: ", erro/4, "%")


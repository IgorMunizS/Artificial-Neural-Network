import numpy as np
import random
import matplotlib.pyplot as plt
import math
plt.style.use('ggplot')

# Definindo função de ativação
# Função degrau bipolar
def step(x):
    if x >= 0:
        return 1
    else:
        return -1

def sigmoide(u):
        return 1 / (1 + math.exp(-u))

def linear(u):
    return u

def valor_final(y):
    return 1 if y > 0.5 else 0


# Dados de treinamento
bias = 1

training_data = [(np.array([-1, -1, bias]), -1),
                 (np.array([-1, 1, bias]), 1),
                 (np.array([1, -1, bias]), 1),
                 (np.array([1, 1, bias]), 1)]

# vetor pesos

w1 = random.random()
w2 = random.random()
b = random.random()

vetor_pesos = [w1, w2, b]

#Inicialização de paramêtros
max_epocas = 10000
epoca = 0
learning_rate = 0.1
threshold = 0.0000001
erro = 0
erroAtual = 2
erroAnterior = 1

# Fase de treinamento

while epoca < max_epocas and abs(erroAtual - erroAnterior) >threshold:

    erroAnterior = erroAtual
    erroAtual = 0
    epoca += 1

    #plot
    # w1, w2, b = vetor_pesos
    # m = -b/w1
    # n = -b/w2
    #
    # d = n
    # c = -n/m
    # line_x = np.array([0, m])
    # line_y  = c*line_x + d
    # plt.plot(line_x, line_y)
    # scatter_x = [x[0][0] for x in training_data]
    # scatter_y = [x[0][1] for x in training_data]
    # color_target = [x[1] for x in training_data]
    # plt.scatter(scatter_x, scatter_y, s=75, c=color_target) #c=[x[1] for x in training_data]
    # plt.title('Epoca: %s'%(epoca))
    # plt.show()


    for n in range(len(training_data)):

        x, y_expected = training_data[n]

        net_input = np.dot(vetor_pesos, x)
        erro = (y_expected - net_input)
        erroAtual += erro*erro

        for i in range(3):

            vetor_pesos[i] += learning_rate*erro*x[i]





    erroAtual = erroAtual/len(training_data)
    print(erroAtual)



# Fase de teste
acertos = 0
it = 0
erro =0
while it < 100:

    for x, y_expected in training_data:
        entrada = np.dot(vetor_pesos, x)
        y = step(entrada)
        #y = valor_final(y)
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


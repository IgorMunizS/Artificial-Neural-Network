# Rede Neural Artificial Multilayer Perceptron com algoritmo backpropagation
# Autor: Igor Muniz Soares 12/09/2017

import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import random




#função de ativação do neurônio
def sigmoide(u):
    return 1 / (1 + math.exp(-u)) #Valor de a escolhido como 1.7 a partir de estudos heurísticos para dar uma maior margem a convergência da sigmoide


class Neuronio(object):

    def __init__(self, neuron_id):

        self.id = neuron_id
        self.output = 0
        self.input = []
        self.weights = []

    def set_inputs(self, neuron_input, weights = 0):
        inputDataNew = neuron_input[:]

        self.input = inputDataNew

        self.input.append(1.0)

        if weights == 0:
            self.weights = []
            for i in range(len(self.input)):
                self.weights.append(np.random.random())

        else:
            self.weights = weights


    def activateNeuron(self):
        #print("Camada")
        #print(self.input, self.weights)
        u = np.dot(self.input, self.weights)
        self.output = sigmoide(u)




class NetworkMlp(object):

    def __init__(self, inputData, outputData, nHiddenLayer, nNeuron, weights = 0, learningRate = 0.1):

        self.hiddenLayer = []
        self.outLayer = []
        self.nNeuron = nNeuron
        self.inputData = inputData
        self.outputData = outputData
        self.nHiddenLayer = nHiddenLayer
        self.learningRate = learningRate
        self.inputDataNow = []
        self.outputDataNow =[]

        if weights == 0:
            weights = []
            for m in range(nHiddenLayer +1):
                weights.append([])
            for m in range(len(nNeuron)):
                for n in range(nNeuron[m]):
                    weights[m].append([])
            for m in range(len(outputData[0])):
                weights[nHiddenLayer].append([])
                    # for _ in range(len(inputData[0]) +1):
                    #     weights[m][n].append(np.random.random())

            for x in range(len(weights)):
                for y in range(len(weights[x])):
                    if x == 0:
                        for _ in range(len(inputData[0]) +1):
                            weights[x][y].append(np.random.random())
                    else:
                        for _ in range(nNeuron[x-1] +1):
                            weights[x][y].append(np.random.uniform())


        #print(weights)
        self.weights = weights

        self.nextData = []

        for i in range(self.nHiddenLayer):
            self.hiddenLayer.append([])

        # criação das camadas escondidas
        for i,j in enumerate(self.nNeuron):

            for n in range(j):
                    self.hiddenLayer[i].append(Neuronio('H' + str(i)+str(n)))


        # criação da camada de saída

        for n in range(len(outputData[0])):
            self.outLayer.append(Neuronio('S' + str(n)))



    def propagate(self, inputDataNow, outputDataNow):

        self.inputDataNow = inputDataNow
        self.outputDataNow = outputDataNow
        self.nextData = []
        for i in range(self.nHiddenLayer):
            #self.hiddenLayer.append([])
            self.nextData.append([])

        camada = 0
        for i,j in enumerate(self.nNeuron):

            if i == 0:
                for n in range(j):
                    self.hiddenLayer[i][n].set_inputs(self.inputDataNow, self.weights[i][n])
                    self.hiddenLayer[i][n].activateNeuron()
                    self.nextData[i].append(self.hiddenLayer[i][n].output)

            else:
                for n in range(j):
                    self.hiddenLayer[i][n].set_inputs(self.nextData[i-1], self.weights[i][n])
                    self.hiddenLayer[i][n].activateNeuron()
                    self.nextData[i].append(self.hiddenLayer[i][n].output)
            camada += 1

        for n in range(len(self.outputDataNow)):
            self.outLayer[n].set_inputs(self.nextData[camada-1], self.weights[camada][n])
            self.outLayer[n].activateNeuron()
            #print(self.outLayer[n].output)

    def backpropagation(self):
        # Erro entre valor desejado e a saída
        erro = self.calculaErro(self.outputDataNow)
        self.newWeights =[]
        self.newWeights = copy.deepcopy(self.weights)
        #print(erro)

        # Vetor gradiente local em relação a todos neurônios da camada de saída
        gradOut = self.gradienteLocal(erro)
        #print(gradOut)
        #print(self.newWeights)

        # Atualiza valor dos pesos da camada de saída

        l = self.nHiddenLayer # posição da última camada - Camada de Saída
        for n in range(len(self.outLayer)):
            for m in range(len(self.weights[l][0])):

                self.newWeights[l][n][m] += self.learningRate*gradOut[n]*self.outLayer[n].input[m]

        # Atualiza os valores dos pesos das camadas escondidas

        while l > 0:

            gradOutH = self.gradienteLocalH(gradOut, l)
            l -= 1

            for n in range(len(self.weights[l])):
                for m in range(len(self.weights[l][0])):
                    self.newWeights[l][n][m] += self.learningRate * gradOutH[n] * self.hiddenLayer[l][n].input[m]

        #Nova geração de pesos
        #print(self.newWeights, self.weights)
        self.weights = copy.deepcopy(self.newWeights)



    def calculaErro(self, vetorDesejado):
        erro = []
        for n in range(len(vetorDesejado)):
            erro.append(vetorDesejado[n] - self.outLayer[n].output)
        return erro

    def gradienteLocal(self, erro):
        vetorGradiente = []
        for n in range(len(self.outLayer)):
            derivada = self.outLayer[n].output*(1 - self.outLayer[n].output)
            gradiente = erro[n]*derivada
            vetorGradiente.append(gradiente)
        return vetorGradiente

    def gradienteLocalH(self, grad, nHidden):
        vetorGradienteH = []

        for n in range(len(self.hiddenLayer[nHidden-1])):
            vetorPesos = []
            derivada = self.hiddenLayer[nHidden-1][n].output*(1 - self.hiddenLayer[nHidden-1][n].output)
            for i in range(len(self.outLayer)):
                vetorPesos.append(self.weights[nHidden][i][n])

            combinacao = np.dot(vetorPesos, grad)
            vetorGradienteH.append(combinacao*derivada)
        return vetorGradienteH

    def treinarEpocas(self, maxEpoch = 1000):
        erroQuad = 1


        erroPlot = []
        epoca = 0
        while epoca < maxEpoch:

            erro = []
            for n in range(len(self.inputData)):
                self.propagate(self.inputData[n], self.outputData[n])
                self.backpropagation()
                erro.append(self.erroQuadratico(self.outputData[n]))

            erroQuad = np.mean(erro)
            erroPlot.append(erroQuad)
            epoca += 1

        self.printFinalData(epoca, erroQuad)

    def treinarConvergir(self, threshold=0.00000001):
        erroQuad = 1
        erroQuadOld = 0

        erroPlot = []
        epoca = 0




        while abs(erroQuadOld - erroQuad) > threshold:

            # Shuffle valores de entrada para uma melhor apresentação a rede
            s = list(zip(self.inputData, self.outputData))
            random.shuffle(s)
            self.inputData, self.outputData = zip(*s)


            erroQuadOld = erroQuad
            erro = []
            for n in range(len(self.inputData)):
                self.propagate(self.inputData[n], self.outputData[n])
                self.backpropagation()
                erro.append(self.erroQuadratico(self.outputData[n]))

            erroQuad = np.mean(erro)
            erroPlot.append(erroQuad)
            epoca += 1

        self.printFinalData(epoca, erroQuad)
        self.plotErro(epoca, erroPlot)


    def erroQuadratico(self, vetorDesejado):

        erro = self.calculaErro(vetorDesejado)
        erroQuad = np.sum(np.array(erro)**2)/2
        return erroQuad

    def printFinalData(self, epoca, erro):
        print("Total de épocas para treinamento: " , epoca)
        print("Erro quadrático médio final: ", erro)
        print("Configuração de pesos finais: ", self.weights)

    def plotErro(self, epoca, vetorErro):

        coord_x = np.arange(0,epoca)
        plt.title("Erro quadrático médio")
        plt.xlabel("Época")
        plt.ylabel("Erro")
        plt.plot(coord_x, vetorErro, c= 'b')
        plt.show()


    def testNetwork(self, testData, resultData):

        acertoPorcent = []

        for n in range(len(testData)):
            self.propagate(testData[n], resultData[n])
            erro = self.calculaErro(resultData[n])

            acertoPorcent.append((1 - np.mean([abs(x) for x in erro]))*100)

            print("{}->{}".format(testData[n], [x.output for x in self.outLayer]))

        acertoTest = np.mean(acertoPorcent)
        print("Acerto do Teste: " + str(acertoTest) + "%")



# Configuração do DataSet de treinamento
# Uma matriz contendo os valores de entrada e outra contendo os valores de saída

inputData = [[0.05, 0.1]]
outputData = [[0.01, 0.99]]
weights = [[[0.15, 0.2, 0.35], [0.25, 0.3, 0.35]], [[0.4, 0.45, 0.6], [0.5, 0.55, 0.6]]]

# inputData = [[0,0],[0,1],[1,0],[1,1]]
# outputData = [[0],[1],[1],[0]]
# weights = [[[0.15, 0.2, 0.35], [0.25, 0.3, 0.35]], [[0.4, 0.45, 0.6], [0.5, 0.55, 0.6]]]


mlp = NetworkMlp(inputData=inputData, outputData=outputData, nHiddenLayer=1, nNeuron=[2], weights=weights, learningRate= 0.5)
#mlp.treinarConvergir()
mlp.treinarEpocas(2)
mlp.testNetwork(inputData,outputData)



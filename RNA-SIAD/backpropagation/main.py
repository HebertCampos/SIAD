from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

x1 = [72 , 75 , 71 , 60 , 85 , 95 , 62 , 55 , 331 , 498 , 73 , 83 , 83 , 67 , 90 , 60 , 64 , 80 , 56 , 72 , 55 , 75 , 85 , 75 , 60 , 80 , 70 , 84 , 72 , 83 , 62 , 80 , 55]
x2 = [600 , 400 , 440 , 320 , 400 , 640 , 350 , 525 , 2300 , 2100 , 440 , 450 , 468 , 350 , 600 , 350 , 270 , 420 , 460 , 600 , 525 , 500 , 420 , 450 , 320 , 468 , 290 , 400 , 600 , 500 , 500 , 638 , 460]
v_esperado = [225000 , 315000 , 490000 , 420000 , 320000 , 680000 , 380000 , 460000 , 3200000 , 3200000 , 520000 , 624300 , 568000 , 330000 , 330000 , 380000 , 380000 , 638100 , 275000 , 225000 , 460000 , 629300 , 280000 , 450000 , 431000 , 638100 , 625000 , 330000 , 230000 , 624000 , 265000 , 638100 , 460000]
# epocas = 1
epocas = 210
tx_aprend = 0.2

"""
Uma rede neural é organizada em camadas

inicializa_rede: aceita 3 parâmetros =>
    - n_entradas - o número de entradas
    - n_oculto - número de neurônios na camada oculta
    - n_saidas - número de saídas
"""

#pesos: números aleatórios no intervalo de 0 a 1
def inicializa_rede(n_entradas, n_oculto, n_saidas):
	rede = list()
    #A camada oculta tem um neurônio com 2 pesos de entrada.
    # pesos_oculta = n_entradas + 1;
	camada_oculta = [{'pesos':[random() for i in range(n_entradas + 1)]} for i in range(n_oculto)]
	rede.append(camada_oculta)
    
    #A camada de saída tem 2 neurônios, cada um com 1 peso mais o bias.
	camada_saida = [{'pesos':[random() for i in range(n_oculto + 1)]} for i in range(n_saidas)]
	rede.append(camada_saida)
	return rede
"""
Cálculo da soma ponderada das entradas > semelhante a regressão linear
ativacao = sum(peso_i * entrada_i) + bias
"""
def ativacao(pesos, entradas):
    ativacao = pesos[-1]
    for i in range(len(pesos)-1):
        ativacao += pesos[i] * entradas[i]
    return ativacao
"""
Ativação da função logística sigmóide em forma de S
para retropropagação - recebe valor de entrada e pode reproduzir
um número entre 0 e 1 em uma curva S

Transferindo uma função de ativação sigmóide
saída =  1 / (1 + e^(-activation))
""" 

# Transferir neurônio - implementa a sigmóide
def transferir(ativacao):
    return 1.0 / (1.0 + exp(-ativacao))

# Método que retorna as saídas da última camada
def propagar(rede, linha):
	entradas = linha
	for camada in rede:
		novas_entradas = []
		for neuronio in camada:
			activation = ativacao(neuronio['pesos'], entradas)
			neuronio['saida'] = transferir(activation)
			novas_entradas.append(neuronio['saida'])
		entradas = novas_entradas
	return entradas

"""
drivada sigmóde = saída * (1.0 - saída)
"""

def transferencia_derivada(saida):
	return saida * (1.0 - saida)

"""
Retropropagação: calcular o erro para cada neurônio de saída
"""
def propagar_error(rede, esperado):
	for i in reversed(range(len(rede))):
		camada = rede[i]
		errors = list()
		if i != len(rede)-1:
			for j in range(len(camada)):
				error = 0.0
				for neuronio in rede[i + 1]:
					error += (neuronio['pesos'][j] * neuronio['delta'])
				errors.append(error)
		else:
			for j in range(len(camada)):
				neuronio = camada[j]
				errors.append(neuronio['saida'] - esperado[j])
		for j in range(len(camada)):
			neuronio = camada[j]
			neuronio['delta'] = errors[j] * transferencia_derivada(neuronio['saida'])

def atualizar_pesos(rede, linha, l_rate):
	for i in range(len(rede)):
		entradas = linha[:-1]
		if i != 0:
			entradas = [neuronio['saida'] for neuronio in rede[i - 1]]
		for neuronio in rede[i]:
			for j in range(len(entradas)):
				neuronio['pesos'][j] -= l_rate * neuronio['delta'] * entradas[j]
			neuronio['pesos'][-1] -= l_rate * neuronio['delta']
   
# Treinamento da rede por um número de épocas
def treinamento_rede(rede, train, l_rate, n_epoca, n_saidas):
	for epoca in range(n_epoca):
		sum_error = 0
		for row in train:
			saidas = propagar(rede, row)
			esperado = [0 for i in range(n_saidas)]
			esperado[row[-1]] = 1
			sum_error += sum([(esperado[i]-saidas[i])**2 for i in range(len(esperado))])
			propagar_error(rede, esperado)
			atualizar_pesos(rede, row, l_rate)
		print('>epoca=%d, lrate=%.3f, error=%.3f' % (epoca, l_rate, sum_error))

def prever(rede, linha):
	saidas = propagar(rede, linha)
	return saidas.index(max(saidas))

def acuracea(actual, predicted):
	correto = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correto += 1
	return correto / float(len(actual)) * 100.0

def normalizar(dataset, minmax):
	norm_x1 = [((x-min(dataset))/(max(dataset)-min(dataset))) for x in dataset]
   
def dataset_minmax(dataset):
	minmax = list()
	stats = [min(dataset), max(dataset)]
	return stats

def denormalize(y, x):
    y_max_normalize = max(x)
    y_min_normalize = min(x)
    
    final = (y)*(y_max_normalize - y_min_normalize) + y_min_normalize 
    
    return final

# Test training backprop algorithm
# Test Backprop on Seeds dataset
seed(1)
dataset = x1
init = inicializa_rede(x1[0], 2, 1)

print(init)

minmax = dataset_minmax(dataset)
normalizar(dataset, minmax)

l_rate = 0.2
n_epoch = 500




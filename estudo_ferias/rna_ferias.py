from func import biasFunc, pesosFunc, saidaNeuroFunc

# entradas
x1 = [1,1]
x2 = [0,1]
x3 = [1,1]
entradas = [x1, x2, x3]
v_esperdo = [0,1]


# quantidade de neuronios camada oculta 01
neuronios_ocultos_1 = 2
neuronios_saida = 1

# pesos iniciais para camada oculta 01
pesos_entradas_para_oculto = pesosFunc(neuronios_ocultos_1,len(entradas))

# bias iniciais
bias_neuronios = biasFunc(neuronios_ocultos_1+neuronios_saida)

# entrada que esta sendo treinda
interacao = 0

# saida camada oculta
neuronios_ocultos = saidaNeuroFunc(neuronios_ocultos_1,entradas, pesos_entradas_para_oculto, bias_neuronios[:-1],  interacao)


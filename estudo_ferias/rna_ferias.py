from func import biasFunc, pesosFunc, saidaNeuroFunc

# entradas
x1 = [1,1]
x2 = [0,1]
x3 = [1,1]
entradas = [x1, x2, x3]
valor_esperado = [0,1]


# quantidade de neuronios camada oculta 01
neuronios_ocultos_1 = 2
neuronios_saida = 1

# pesos iniciais para camada oculta 01
pesos_entradas_para_oculto = pesosFunc(neuronios_ocultos_1,len(entradas))
# pesos iniciais para o neuronio de saida
pesos_entradas_para_saida = pesosFunc(neuronios_saida, neuronios_ocultos_1)

# bias iniciais
bias_neuronios = biasFunc(neuronios_ocultos_1+neuronios_saida)

# entrada que esta sendo treinda
interacao = 0

# saida camada oculta
saida_neuronios_ocultos = saidaNeuroFunc(neuronios_ocultos_1,entradas, pesos_entradas_para_oculto, bias_neuronios[:-1],  interacao)

saida_neuronio_saida = saidaNeuroFunc(neuronios_saida, [saida_neuronios_ocultos], pesos_entradas_para_saida,[bias_neuronios[-1]], 0)
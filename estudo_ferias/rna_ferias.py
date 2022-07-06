from func import biasFunc, pesosFunc

# entradas
x1 = []
x2 = []
x3 = []
entradas = [x1, x2, x3]
v_esperdo = []


# quantidade de neuronios camada oculta 01
neuronios_ocultos_1 = 2
neuronios_saida = 1

# pesos
pesos_entradas_para_oculto = pesosFunc(neuronios_ocultos_1,len(entradas))
bias_neuronios = biasFunc(neuronios_ocultos_1+neuronios_saida)
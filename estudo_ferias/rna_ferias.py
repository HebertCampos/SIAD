from func import *

# entradas
entrada_1 = [11, 11, 10]
entrada_2 = [10, 11, 10]
entrada_3 = [11, 11, 10]
esperado = [10, 11, 10]

entradas = [entrada_1, entrada_2, entrada_3]

valor_esperado = [esperado]

normalizado_entradas = normalizacaoFunc(entradas)
normalizado_valor_esperado = normalizacaoFunc(valor_esperado)


# quantidade de neuronios camada oculta 01
neuronios_ocultos_1 = 2
neuronios_saida = 1

# pesos iniciais para camada oculta 01
pesos_entradas_para_oculto = pesosFunc(neuronios_ocultos_1,len(normalizado_entradas))
# pesos iniciais para o neuronio de saida
pesos_entradas_para_saida = pesosFunc(neuronios_saida, neuronios_ocultos_1)

# bias iniciais
bias_neuronios = biasFunc(neuronios_ocultos_1+neuronios_saida)

# taxa de aprendizado
tx_aprendizado = 0.2

# entrada que esta sendo treinda
interacao = 0

# saida camada oculta
saida_neuronios_ocultos = saidaNeuroFunc(neuronios_ocultos_1,normalizado_entradas, pesos_entradas_para_oculto, bias_neuronios[:-1],  interacao)

saida_neuronio_saida = saidaNeuroFunc(neuronios_saida, [saida_neuronios_ocultos], pesos_entradas_para_saida,[bias_neuronios[-1]], 0)

if (saida_neuronio_saida[0] >= normalizado_valor_esperado[0][interacao]-0.01) and (saida_neuronio_saida[0] <= normalizado_valor_esperado[0][interacao]+0.01):
    interacao += 1

else:
    erro = normalizado_valor_esperado[0][interacao] - saida_neuronio_saida[0]
    delta_erro_saida = saida_neuronio_saida[0]*(1-saida_neuronio_saida[0])*erro
    delta_erro_oculto = deltaCamadaOcultaFunc(saida_neuronio_saida[0], pesos_entradas_para_oculto, delta_erro_saida)
    pesos_entradas_para_oculto = ajustePesosOcultoFunc(pesos_entradas_para_oculto, tx_aprendizado, normalizado_entradas, delta_erro_oculto, interacao)
    pesos_entradas_para_saida = ajustePesosSaidaFunc(pesos_entradas_para_saida, tx_aprendizado, [saida_neuronios_ocultos], delta_erro_saida, 0)
    delta_bias_oculto = delta_erro_oculto * tx_aprendizado
    delta_bias_saida = delta_erro_saida * tx_aprendizado
    bias_neuronios = [delta_bias_oculto, delta_bias_oculto, delta_bias_saida]
    
print([pesos_entradas_para_oculto, pesos_entradas_para_saida], bias_neuronios, saida_neuronio_saida)
    
from func import ajuste_pesos, biasFunc, delta_camada_oculta, normalizacaoFunk, pesosFunc, saidaNeuroFunc

# entradas
entrada_1 = [11, 11, 10]
entrada_2 = [10, 11, 10]
entrada_3 = [11, 11, 10]
esperado = [10, 11, 10]

entradas = [entrada_1, entrada_2, entrada_3]

valor_esperado = [esperado]

normalizado_entradas = normalizacaoFunk(entradas)
normalizado_valor_esperado = normalizacaoFunk(valor_esperado)


# quantidade de neuronios camada oculta 01
neuronios_ocultos_1 = 2
neuronios_saida = 1

# pesos iniciais para camada oculta 01
pesos_entradas_para_oculto = pesosFunc(neuronios_ocultos_1,len(normalizado_entradas))
# pesos iniciais para o neuronio de saida
pesos_entradas_para_saida = pesosFunc(neuronios_saida, neuronios_ocultos_1)

# bias iniciais
bias_neuronios = biasFunc(neuronios_ocultos_1+neuronios_saida)

# entrada que esta sendo treinda
interacao = 0

# saida camada oculta
saida_neuronios_ocultos = saidaNeuroFunc(neuronios_ocultos_1,entradas, pesos_entradas_para_oculto, bias_neuronios[:-1],  interacao)

saida_neuronio_saida = saidaNeuroFunc(neuronios_saida, [saida_neuronios_ocultos], pesos_entradas_para_saida,[bias_neuronios[-1]], 0)

if (saida_neuronio_saida[0] >= normalizado_valor_esperado[0][interacao]-0.01) and (saida_neuronio_saida[0] <= normalizado_valor_esperado[0][interacao]+0.01):
    interacao += 1

else:
    erro = normalizado_valor_esperado[0][interacao] - saida_neuronio_saida[0]
    delta_erro_saida = saida_neuronio_saida[0]*(1-saida_neuronio_saida[0])*erro
    delta_erro_oculto = delta_camada_oculta(saida_neuronio_saida[0], pesos_entradas_para_oculto, delta_erro_saida)

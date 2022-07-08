from func import *

# entradas
# entrada_1 = [11, 11, 10]
# entrada_2 = [10, 11, 10]
# entrada_3 = [11, 11, 10]
# esperado = [10, 11, 10]

entrada_1 = [72 , 75 , 71 , 60 , 85 , 95 , 62 , 55 , 331 , 498 , 73 , 83 , 83 , 67 , 90 , 60 , 64 , 80 , 56 , 72 , 55 , 75 , 85 , 75 , 60 , 80 , 70 , 84 , 72 , 83 , 62 , 80 , 55]
entrada_2 = [600 , 400 , 440 , 320 , 400 , 640 , 350 , 525 , 2300 , 2100 , 440 , 450 , 468 , 350 , 600 , 350 , 270 , 420 , 460 , 600 , 525 , 500 , 420 , 450 , 320 , 468 , 290 , 400 , 600 , 500 , 500 , 638 , 460]
esperado = [225000 , 315000 , 490000 , 420000 , 320000 , 680000 , 380000 , 460000 , 3200000 , 3200000 , 520000 , 624300 , 568000 , 330000 , 330000 , 380000 , 380000 , 638100 , 275000 , 225000 , 460000 , 629300 , 280000 , 450000 , 431000 , 638100 , 625000 , 330000 , 230000 , 624000 , 265000 , 638100 , 460000]


entradas = [entrada_1, entrada_2]

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
tx_aprendizado = 0.8

# entrada que esta sendo treinda
interacao = 0
epoca = 10000

saidas = []

while True:
    # if len(saidas) > len(esperado):
    #     saidas = []
    
    if epoca > 0:
        # saida camada oculta
        saida_neuronios_ocultos = saidaNeuroFunc(neuronios_ocultos_1,normalizado_entradas, pesos_entradas_para_oculto, bias_neuronios[:-1],  interacao)

        saida_neuronio_saida = saidaNeuroSaidaFunc(neuronios_saida, [saida_neuronios_ocultos], pesos_entradas_para_saida,[bias_neuronios[-1]], 0)

        # if (saida_neuronio_saida[0] >= normalizado_valor_esperado[0][interacao]-0.01) and (saida_neuronio_saida[0] <= normalizado_valor_esperado[0][interacao]+0.01):
        # interacao += 1
        saidas.append(denormalize(saida_neuronio_saida[0] ,valor_esperado[0]))
            # print(f'i {interacao}')

        # else:
        erro = normalizado_valor_esperado[0][interacao] - saida_neuronio_saida[0]
        delta_erro_saida = saida_neuronio_saida[0]*(1-saida_neuronio_saida[0])*erro
        delta_erro_oculto = deltaCamadaOcultaFunc(saida_neuronios_ocultos, pesos_entradas_para_oculto, delta_erro_saida)
        pesos_entradas_para_oculto = ajustePesosOcultoFunc(pesos_entradas_para_oculto, tx_aprendizado, normalizado_entradas, delta_erro_oculto, interacao)
        pesos_entradas_para_saida = ajustePesosSaidaFunc(pesos_entradas_para_saida, tx_aprendizado, [saida_neuronios_ocultos], delta_erro_saida, 0)
        delta_bias_oculto = deltaOcultoFunc(delta_erro_oculto, tx_aprendizado)
        delta_bias_saida = delta_erro_saida * tx_aprendizado
        bias_neuronios = [dbo for dbo in delta_bias_oculto] #[delta_bias_oculto, delta_bias_oculto, delta_bias_saida]
        bias_neuronios.append(delta_erro_saida)
        interacao += 1    
        if interacao > len(entradas[0])-1:
            interacao = 0
            epoca -= 1
            print(f'e {epoca}')
            # print([pesos_entradas_para_oculto, pesos_entradas_para_saida], bias_neuronios, saida_neuronio_saida)

            
    else: break
    
print([pesos_entradas_para_oculto, pesos_entradas_para_saida], bias_neuronios, saida_neuronio_saida)
saidasPrint = [print(f'{x:.0f}') for x in saidas[-len(esperado):]]
print('\n\n')
entradas_espe = [print(f'{x:.0f}') for x in esperado]
# print(saidas[-len(esperado):])
    
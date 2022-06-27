# caso queira testar 
# comentar linhas 12, 22, 23, 58
# retirar comentarios das linhas 13, 26, 27, 60
# 
# --------------------------------------------------------------------------------------
from defs import *
import os

os.system('cls' if os.name == 'nt' else 'clear')

x1 = [72 , 75 , 71 , 60 , 85 , 95 , 62 , 55 , 331 , 498 , 73 , 83 , 83 , 67 , 90 , 60 , 64 , 80 , 56 , 72 , 55 , 75 , 85 , 75 , 60 , 80 , 70 , 84 , 72 , 83 , 62 , 80 , 55]
x2 = [600 , 400 , 440 , 320 , 400 , 640 , 350000 , 525 , 2300 , 2100 , 440 , 450 , 468 , 350 , 600 , 350 , 270 , 420 , 460 , 600 , 525 , 500 , 420 , 450 , 320 , 468 , 290 , 400 , 600 , 500 , 500 , 638100 , 460000]
v_esperado = [225000 , 315000 , 490000 , 420000 , 320000 , 680000 , 380000 , 460000 , 3200000 , 3200000 , 520000 , 624300 , 568000 , 330000 , 330000 , 380000 , 380000 , 638100 , 275000 , 225000 , 460000 , 629300 , 280000 , 450000 , 431000 , 638100 , 625000 , 330000 , 230000 , 624000 , 265000 , 638100 , 460000]
#epocas = 1
epocas = 10000
tx_aprend = 0.2

# normalizando as entradas entre 0-1
'''
Xatual = (X-Xmin) / (Xmax - Xmin)
'''
print()

norm_x1 = [((x-min(x1))/(max(x1)-min(x1))+1) for x in x1]
norm_x2 = [((x-min(x2))/(max(x2)-min(x2))+1) for x in x2]
norm_v_esperado = [((x-min(v_esperado))/(max(v_esperado)-min(v_esperado))+1) for x in v_esperado]

# melhores valores do treinamento - acuracea - >90%
#pesos = [[9.384723216892949, 11.384723216892471], [9.384723216892949, 11.384723216892471], [4.513571691899537, 4.513571691899537]]
#bias = [[11.384723216892471, 11.384723216892471], [3.5135716918996227]]

# gerando aleatoriamento os primeiros pesos e bias
pesos = peso()  # peso = [[x1n1.x2n1],[x1n2,x2n2],[n1n3,n2n3]]
bias = biass()  # bias = [[n1,n2],[n3]]

# x = z(max - min) + min
def denormalize(y, x):
    y_max_normalize = max(x)
    y_min_normalize = min(x)
    
    final = (y-1)*(y_max_normalize - y_min_normalize) + y_min_normalize 
    
    return final

# entrando no loop com quantidade de epocas
for q in range(epocas):
    saida_n3_list = []
    for it in range(len(x1)):
        # calcula o somatorio = (x1*peso[0][0]+x2*peso[0][1]+...+xn*peso[0][n])
        som_N1 = (norm_x1[it] * pesos[0][0])+(norm_x2[it] *pesos[0][1])
        som_N2 = (norm_x1[it] * pesos[1][0])+(norm_x2[it] *pesos[1][1])
        saida_N1 = 1/(1+math.exp(-(som_N1 + bias[0][0])))
        saida_N2 = 1/(1+math.exp(-(som_N2 + bias[0][1])))
        
        som_N3 = (saida_N1*pesos[2][0])+(saida_N2*pesos[2][1])
        saida_N3 = 1/(1+math.exp(-(som_N3 + bias[1][0])))
        
        saida_n3_list.append(saida_N3)
        
        # print('saída N3: ', saida_N3)
        # print('Desnormalizando a saída N3  ')
        # desnormalize_saidaN3 = denormalize(saida_N3)
        # print(desnormalize_saidaN3)
        
        erro  = abs(norm_v_esperado[it] - saida_N3)
        
        erro_peso_N1 = pesos[2][0] * erro #ErroP8
        erro_peso_N2 = pesos[2][1] * erro #ErroP9
        
        for n in range(2):
            pesos[n][0] = pesos[n][0] + tx_aprend * erro_peso_N1 * (saida_N3*(1-saida_N3)) * saida_N1
        for n in range(2):
            pesos[n][1] = pesos[n][1] + tx_aprend * erro_peso_N1 * (saida_N3*(1-saida_N3)) * saida_N2
        
        pesos[2][0] = pesos[2][0] + tx_aprend * erro * (saida_N3*(1-saida_N3)) * saida_N3 
        pesos[2][1] = pesos[2][1] + tx_aprend * erro * (saida_N3*(1-saida_N3)) * saida_N3
        
        bias[0][0] = bias[0][0] + tx_aprend * erro_peso_N1 * (saida_N3*(1-saida_N3)) * saida_N1
        bias[0][1] = bias[0][1] + tx_aprend * erro_peso_N2 * (saida_N3*(1-saida_N3)) * saida_N2
        bias[1][0] = bias[1][0] + tx_aprend * erro * (saida_N3*(1-saida_N3)) * saida_N3
        porcent = erro/saida_N3*100

        #print(f'{100-porcent:.2f}% item {it}\nsaida N3 = {saida_N3:.2f}\nvalor esperado = {norm_v_esperado[it]:.2f}\nerro = {abs(erro):.2f}\n')
        
print(f'{100-porcent:.2f}% item {it}\nsaida N3 = {saida_N3:.2f}\nvalor esperado = {norm_v_esperado[it]:.2f}\nerro = {abs(erro):.2f}\n')

print(pesos, bias)

print('')
print('len(saida_n3_list): ', len(saida_n3_list))
print('len(v_esperado): ', len(v_esperado))

print('')
print('lista de saidas N3: ', saida_n3_list)

print('')
print('v_esperado: \n')
print(v_esperado)
print('----x-----x----')

list_v_desnormalizado_n3 = []
percentualList_saidaN3 = []
list_diferenca_valor_esperado_desnormalizado = []

for index in range(len(saida_n3_list)):
    desnormalize_saidaN3 = denormalize(v_esperado[index], saida_n3_list)
    list_v_desnormalizado_n3.append(desnormalize_saidaN3)
    
print('Desnormalizando a saída N3\n')

for index in range(len(list_v_desnormalizado_n3)):
    diferenca =  (v_esperado[index]-list_v_desnormalizado_n3[index])
    list_v_desnormalizado_n3[index] = list_v_desnormalizado_n3[index] + diferenca
    
    porcentEsperadoSaidaN3 = (v_esperado[index]/list_v_desnormalizado_n3[index])*100
    percentualList_saidaN3.append(porcentEsperadoSaidaN3)
    
print(list_v_desnormalizado_n3)

print('')
print('Percentual da lista de saidaN3:\n')
print(percentualList_saidaN3)
for index in range(len(list_v_desnormalizado_n3)):
    list_diferenca_valor_esperado_desnormalizado.append(v_esperado[index]-list_v_desnormalizado_n3[index])

print('')
print('Lista da Diferença de Valores => Valor Esperado - Valor Desnormalizado:')
print('')
print(list_diferenca_valor_esperado_desnormalizado)

# #X1
# print('X1 normalizado: \n')
# print(norm_x1)
# print('\n')

# print('X1: \n')
# print(x1)
# print('\n')

# listaX1_desnormalized = []

#for index in range(len(norm_x1)):
    #valor = denormalize(norm_x1[index], x1)
    #valor = [(y*(max(x1) - min(x1)) + min(x1) ) for y in norm_x1]
    #listaX1_desnormalized.append(valor)

# norm_t = [((x-min(x1))/(max(x1)-min(x1))) for x in x1]
# valores = [y*(max(x1) - min(x1)) + min(x1) for y in norm_t]
# print('\n')
# print('\nnorm_t: ', norm_t)


#desnormalizar norm_x1
# valores = [(y-1)*(max(x1) - min(x1))-1 + min(x1) for y in norm_x1]

# for index in range(len(valores)):
#     diferenca =  (x1[index]-valores[index])
#     valores[index] = valores[index] + diferenca

# print('valores: ', valores)

# print('lista desnormalizada - x1: \n')
# print(listaX1_desnormalized)

# norm_v_esperado = [((x-min(v_esperado))/(max(v_esperado)-min(v_esperado))) for x in v_esperado]
# d = [x*(max(v_esperado)-min(v_esperado))+min(v_esperado) for x in norm_v_esperado]
# print(d)

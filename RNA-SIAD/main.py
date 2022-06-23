from defs import *

# x1 = [140,130,90,75,100]
# x2 = [500,600,400,450,800]
# v_esperado = [700000,650000,450000,330000,675000]
# x1 = [85,97,280,282,73,78]
# x2 = [550,1300,1800,1800,550,550]
# v_esperado = [545244,450000,5200000,1550000,395000,300000]

x1 = [72 , 75 , 71 , 60 , 85 , 95 , 62 , 55 , 331 , 498 , 73 , 83 , 83 , 67 , 90 , 60 , 64 , 80 , 56 , 72 , 55 , 75 , 85 , 75 , 60 , 80 , 70 , 84 , 72 , 83 , 62 , 80 , 55]
x2 = [600 , 400 , 440 , 320 , 400 , 640 , 350000 , 525 , 2300 , 2100 , 440 , 450 , 468 , 350 , 600 , 350 , 270 , 420 , 460 , 600 , 525 , 500 , 420 , 450 , 320 , 468 , 290 , 400 , 600 , 500 , 500 , 638100 , 460000]
v_esperado = [225000 , 315000 , 490000 , 420000 , 320000 , 680000 , 380000 , 460000 , 3200000 , 3200000 , 520000 , 624300 , 568000 , 330000 , 330000 , 380000 , 380000 , 638100 , 275000 , 225000 , 460000 , 629300 , 280000 , 450000 , 431000 , 638100 , 625000 , 330000 , 230000 , 624000 , 265000 , 638100 , 460000]
epocas = 40000
tx_aprend = 0.2
print(f'{len(x1)}  {len(x2)}  {len(v_esperado)}')

# normalizando as entradas entre 0-1
norm_x1 = [(x-min(x1))/(max(x1)-min(x1)) for x in x1]
norm_x2 = [(x-min(x2))/(max(x2)-min(x2)) for x in x2]
norm_v_esperado = [(x-min(v_esperado))/(max(v_esperado)-min(v_esperado)) for x in v_esperado]

# gerando aleatoriamento os primeiros pesos e bias
pesos = peso()  # peso = [[x1n1.x2n1],[x1n2,x2n2],[n1n3,n2n3]]
bias = biass()  # bias = [[n1,n2],[n3]]

# entrando no loop com quantidade de epocas
for q in range(epocas):
    for it in range(len(x1)):
        # calcula o somatorio = (x1*peso[0][0]+x2*peso[0][1]+...+xn*peso[0][n])
        som_N1 = (norm_x1[it] * pesos[0][0])+(norm_x2[it] *pesos[0][1])
        som_N2 = (norm_x1[it] * pesos[1][0])+(norm_x2[it] *pesos[1][1])
        saida_N1 = 1/(1+math.exp(-(som_N1 + bias[0][0])))
        saida_N2 = 1/(1+math.exp(-(som_N2 + bias[0][1])))
        
        som_N3 = (saida_N1*pesos[2][0])+(saida_N2*pesos[2][1])
        saida_N3 = 1/(1+math.exp(-(som_N3 + bias[1][0])))
        
        erro  = norm_v_esperado[it] - saida_N3
        
        erro_peso_N1 = pesos[2][0] * erro
        erro_peso_N2 = pesos[2][1] * erro
        
        for n in range(2):
            pesos[n][0] = pesos[n][0] + tx_aprend * erro_peso_N1 * (saida_N3*(1-saida_N3)) * saida_N1
        for n in range(2):
            pesos[n][1] = pesos[n][1] + tx_aprend * erro_peso_N1 * (saida_N3*(1-saida_N3)) * saida_N2
        pesos[2][0] = pesos[2][0] + tx_aprend * erro * (saida_N3*(1-saida_N3)) * saida_N3 
        pesos[2][1] = pesos[2][1] + tx_aprend * erro * (saida_N3*(1-saida_N3)) * saida_N3
        # Calcula o erro da v_esperado calculada
        
    print(f'epoca = {q}\nsaida N3 = {saida_N3}\nvalor esperado = {norm_v_esperado[-1]}\nerro = {erro}')


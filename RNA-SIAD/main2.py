from defs import *

# x1 = [140,130,90,75,100]
# x2 = [500,600,400,450,800]
# v_esperado = [700000,650000,450000,330000,675000]
x1 = [78,85,97,280,282,73]
x2 = [550,550,1300,1800,1800,550]
v_esperado = [300000, 545244,45000,5200000,1550000,395000]

epocas = 10000
tx_aprend = 0.005

# normalizando as entradas entre 0-1
n_x1 = [(x-min(x1))/(max(x1)-min(x1)) for x in x1]
n_x2 = [(x-min(x2))/(max(x2)-min(x2)) for x in x2]
norm_v_esperado = [(x-min(v_esperado))/(max(v_esperado)-min(v_esperado)) for x in v_esperado]

# gerando aleatoriamento os primeiros pesos e bias
pesos = peso()  # peso = [[x1n1.x1n2],[x2n1,x2n2],[n1n3,n2n3]]
bias = biass()  # bias = [[n1,n2],[n3]]

# entrando no loop com quantidade de epocas
for q in range(epocas):
    norm_x1 = sum([abs(x) for x in n_x1])
    norm_x2 = sum([abs(x) for x in n_x1])
    # calcula o somatorio = (x1*peso[0][0]+x2*peso[0][1]+...+xn*peso[0][n])
    soma0 = (norm_x1*pesos[0][0])+(norm_x2*pesos[0][1])
    soma1 = (norm_x1*pesos[1][0])+(norm_x2*pesos[1][1])
    # somatoria = [somatori0(norm_x1, norm_x2, pesos[0][j]) for j in range(2)]   # esta gerando matriz [[p/n1],[p/n2]]
    # soma0=sum([abs(x) for x in somatoria[0]])
    # soma1=sum([abs(x) for x in somatoria[1]])
    saida_N1_N2 = [1/(1+math.exp(-(soma0 + bias[0][0])))+bias[0][0],1/(1+math.exp(-(soma1 + bias[0][1])))+bias[0][1]]    # juntando a matriz
    
    
    
    # v_esperado do Neuronio 1 e 2 - usando ativação segmoid
    # N1 =1/1+e**(somatoria_N1_N2 + bias[0][0])
    # N2 =1/1+e**(somatoria_N1_N2 + bias[0][1])
    # saida_N1_N2 = saidaN(somatoria_N1_N2, bias[0])

    #Calculando somatorio neuronio 3 = (N1*peso[1][0]+N2*peso[1][1])
    somatoria_N3 = (saida_N1_N2[0]*pesos[2][0])+(saida_N1_N2[1]*pesos[2][1])     # esta gerando matriz [[p8],[p9]]
    # somatorio_N3 = list(zip(somatoria_N3[0], somatoria_N3[1]))      # juntando a matriz
    
    # v_esperado no Neuronio 3 - usando ativação segmoid = 1/1+e**(somatoria_N1_N2 + bias[1][0])
    n3 = 1/(1+math.exp(-(somatoria_N3+bias[1][0])))+bias[1][0]
    print('#############################')
    print(f'v_esperado N3{n3}\n')

    # Calcula o erro da v_esperado calculada
    # neuronio3 = sum([abs(x) for x in n3[0]])
    v_esp = sum([abs(x) for x in norm_v_esperado])
    #erro = [norm_v_esperado[x]-n3[x][0] for x in range(len(v_esperado))]
    erro = v_esp - n3
    print(f'erro = {erro}\n')
    # print(f'v_esperado esperada{norm_v_esperado}\n\n')
    print('#############################')

    # Tentar usar  =  (SUM erros)**2

    erroP8 = [saida_N1_N2[x][0]*erro[x] for x in range(len(erro))]
    erroP9 = [saida_N1_N2[x][1]*erro[x] for x in range(len(erro))]

    # novos pesos
    for j in range(2):
        for i in range(len(x1)):
            pesos[j][0] = pesos[j][0] + tx_aprend * erroP8[i] * norm_x1[i] * (n3[i][0]*(1-n3[i][0]))

    for j in range(2):
        for i in range(len(x1)):
            pesos[j][1] = pesos[j][1] + tx_aprend * erroP9[i] * norm_x2[i] * (n3[i][0]*(1-n3[i][0]))
            



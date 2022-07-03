import random, math

def peso():
    peso = []
    for i in range(3):
        p =[]
        for j in range(2):
            p.append(random.randint(-1,1))
        peso.append(p)
    return peso

def biass():
    bias = []
    b = []
    for i in range(2):
        b.append(random.randint(-1,1))
    bias.append(b)
    b = [random.randint(-1,1)]
    bias.append(b)
    return bias

# somatorio = m2[i]*peso[0][j]+cond[i]*peso[1][j]
def somatori0(norm_m2, norm_cond, peso):
    som = []
    for i in range(len(norm_m2)):
        s = 0
        s += norm_m2[i]*peso[0]+norm_cond[i]*peso[1]
        som.append(s)
    return som

def saidaN(somatorio, bias):
    #ativação 1/(1+math.exp(somatorio[j]+bias[i]))
    atv =[]
    for j in range(len(somatorio)):
        ns = []
        for i in range(len(bias)):
            n = 1/(1+math.exp(-(somatorio[j][i]+bias[i])))
            ns.append(n)
        atv.append(ns)
    return atv

def denormalize(y, x):
    y_max_normalize = max(x)
    y_min_normalize = min(x)
    
    final = (y)*(y_max_normalize - y_min_normalize) + y_min_normalize 
    
    return final

def derivadaSigmoid(value):
    return value*(1-value)
import math
import random


def pesosFunc(x,y):
    pesos = []
    for a in range(x):
        p = []
        for b in range(y):
            p.append(random.randint(-2,2))
        pesos.append(p)
    return pesos

def biasFunc(n):
    b = []
    for x in range(n):
        b.append(random.randint(-2,2))
    return b

def saidaNeuroFunc(n, e, p, b, index):
    saida = []
    s = 0
    for n in range(n):
        for i in range(len(e)):
            s += e[i][index] * p[n][i]
        saida.append(1/(1+math.exp(-(s + b[n]))))
    return saida

def saidaNeuroSaidaFunc(n, e, p, b, index):
    saida = []
    s = 0
    for n in range(n):
        for i in range(len(e)):
            s += e[i][index] * p[index][i]
        saida.append(1/(1+math.exp(-(s + b[n]))))
    return saida

def normalizacaoFunc(x):
    entrada = []
    for e in range(len(x)):
        resultado = []
        for i in x[e]:
            norma = (i - min(x[e])) / (max(x[e]) - min(x[e]))
            resultado.append(norma)
        entrada.append(resultado)
    return entrada

def deltaCamadaOcultaFunc(sa, w, des):
    soma = 0
    do = []
    for s in sa:
        for i in range(len(w[0])):
            soma += i*des
        do.append(s*(1-s)*s)
    return do

def ajustePesosOcultoFunc(w, a, e, d, index):
    peso = []
    for x in range(len(w)):
        p = []
        for y in range(len(e)):
            p.append(w[x][y] + a * e[y][index] * d[x])
        peso.append(p)
    return peso

def ajustePesosSaidaFunc(w, a, e, d, index):
    peso = []
    for x in range(len(w)):
        p = []
        for y in range(len(e[index])):
            p.append(w[x][y] + a * e[index][y] * d)
        peso.append(p)
    return peso

def denormalize(y, x):
    y_max_normalize = max(x)
    y_min_normalize = min(x)
    final = (y)*(y_max_normalize - y_min_normalize) + y_min_normalize 
    return final

def deltaOcultoFunc(de,a):
    dt = [d*a for d in de]
    return dt
        
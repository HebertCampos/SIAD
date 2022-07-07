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

def deltaCamadaOcultaFunc(s, w, des):
    soma = 0
    for s in range(len(w)):
        for i in range(len(w[0])):
            soma += i*des
    return s*(1-s)*s

def ajustePesosOcultoFunc(w, a, e, d, index):
    peso = []
    for x in range(len(w)):
        p = []
        for y in range(len(e)):
            p.append(w[x][y] + a * e[y][index] * d)
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
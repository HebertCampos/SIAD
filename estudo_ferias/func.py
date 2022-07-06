import random


def pesosFunc(x,y):
    pesos = []
    for a in range(x):
        p = []
        for b in range(y):
            p.append(random.randint(-2,2))
        pesos.append(p)
    return pesos

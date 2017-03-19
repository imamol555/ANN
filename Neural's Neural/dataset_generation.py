import os
import numpy as np
import random
from random import randint

flat_array = []
def graph_input():
    i = 0
    while i < len(flat_array):
        del flat_array[i]
        i += 1
    adj_matrix = []

    # vertex numbering starts from 0
    for i in range(0, 32):

        temp = []
        for j in range(0, 32):
            temp.append(0)
        adj_matrix.append(temp)

    for u in range(0, 32):
        for v in range(0, 32):
            if (u == v):
                adj_matrix[u][v] = adj_matrix[v][u] = 0

            else:
                tmp_no = random.randint(1,6)
                if (v % tmp_no == 0):
                    adj_matrix[u][v] = adj_matrix[v][u] = 0
                if (u % tmp_no == 0  ):
                    adj_matrix[u][v] = adj_matrix[v][u] = 0
                else:
                    adj_matrix[u][v] = adj_matrix[v][u] = randint(0, 9)

            #print("u = ", u)
            #print("v = ", v)
            #print(adj_matrix[u][v])
            if (adj_matrix[u][v] > 0):
                flat_array.append(adj_matrix[u][v])
            else:
                flat_array.append(0)


    #print(flat_array)
    #print(len(flat_array))



    return flat_array
#graph_input()
def labels():
    no_zero = 0
    i = 0
    while i < len(flat_array):
        if flat_array[i] == 0:
            no_zero += 1
        i += 1
    #print("no zero", no_zero)
    if no_zero < 512:
        result = [1,0]

    else:
        result = [0,1]
    return result

#labels()

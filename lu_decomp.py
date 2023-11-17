import numpy as np
import itertools
import math


def matrix_creation():
    N = 6
    a = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            a[i, j] = 3. / (0.6 * i * j + 1)
    return a

def matrix_minor_check(a, i=1):
    chck = 1
    if i != a.shape[0]:
        if np.linalg.det(a[:i, :i]) == 0:
            chck = 0
            return chck
        i += 1
        chck = matrix_minor_check(a, i)
    return bool(chck)

def permutation_matrix(a):
    a1 = a.copy()
    permutation_list = list(itertools.permutations(range(a.shape[0])))
    for i in range(math.factorial(a.shape[0])):
        for j in range(a.shape[0]):
            a1[:, [j]] = a[:, [permutation_list[i][j]]]
        if matrix_minor_check(a1):
            permutation = permutation_list[i][::]
            return a1, permutation
        a1 = a.copy()
    return a

def diy_lu(a):
    N = a.shape[0]
    u = a.copy()
    L = np.eye(N)
    for j in range(N - 1):
        lam = np.eye(N)
        gamma = u[j + 1:, j] / u[j, j]
        lam[j + 1:, j] = -gamma
        u = lam @ u
        lam[j + 1:, j] = gamma
        L = L @ lam
    return L, u, L @ u

def matrix_restoration(a_permutated, permutation):
    a_restored = a_permutated.copy()
    for i in range(a_permutated.shape[0]):
        a_restored[:, [permutation[i]]] = a_permutated[:, [i]]
    return a_restored


np.set_printoptions(precision=3)
a = matrix_creation()
permutation = list()
a[1, 1] = 3
print(a)
check = matrix_minor_check(a)
if not check:
    a, permutation = permutation_matrix(a)
print(a)
L, u, a_counted = diy_lu(a)
if not check:
    a_counted_restored = matrix_restoration(a_counted, permutation)
else:
    a_counted_restored = a_counted
print(a_counted_restored)

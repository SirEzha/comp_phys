import numpy as np
import matplotlib.pyplot as plt

def original_matrix_creation():
    a = np . zeros ((15 , 28))
    a [2: -2 ,1] = 1; a [2 ,2:6] = 1
    a [2:7 ,6] = 1; a [7: -2 ,7] = 1
    a [7 ,2:7] = 1; a [ -3 ,2:7] = 1
    a [2: -2 , 10] = 1; a [2: -2 , 14] = 1
    a [2: -2 , 18] = 1; a [-3 ,10:19] = 1
    a[2:13, 26] = 1
    a[12, 21:26] = 1; a[7, 21:26] = 1; a[2, 21:26] = 1
    return a, a.shape

def svd_decomp(a):
    u, s, v = np.linalg.svd(a)
    rank = 0
    for i in range(s.shape[0]):
        if s[i] != 0:
            rank += 1
        else:
            return rank, u, s, v
    return rank, u, s, v

def best_approximation(rank, u, s, v, dims):
    for i in range(1, rank + 1):
        s_new = np.zeros(dims)
        for j in range(0, i):
            s_new[j, j] = s[j]
        a_new = u @ s_new @ v
        image = plt.imshow(a_new)
        plt.title(i)
        plt.show()


original_matrix, original_matrix_dimensions = original_matrix_creation()
original_matrix_rank, svd_u, svd_s, svd_v = svd_decomp(original_matrix)
print(svd_s)
print('Original matrix rank =', original_matrix_rank)
best_approximation(original_matrix_rank, svd_u, svd_s, svd_v, original_matrix_dimensions)
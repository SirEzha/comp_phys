import numpy as np

def matrix_construct(length):
    a = np.ones((2, length))
    for i in range(length):
        a[0, i] = length - i - 1/2
    return a

def reverse_singular_construct(a):
    u, s, v = np.linalg.svd(a)
    s_new = np.zeros((a.shape[1], a.shape[0]))
    for i in range(len(s)):
        if s[i] != 0:
            s_new[i, i] = 1 / (s[i])
        else:
            for j in range(i, len(s)):
                s_new[j, j] = 0
            return s_new, u, v
    return s_new, u, v

def f_min(s_new, u, v, a):
    f = v.T @ s_new @ u.T @ a
    return f


a = np.array((1, 0))
matrix = matrix_construct(10)
s_reversed, u, v = reverse_singular_construct(matrix)
f_minimal = f_min(s_reversed, u, v, a)
print(f_minimal)

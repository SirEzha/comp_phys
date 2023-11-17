import numpy as np

def design_matrix(x, m):
    """Construct the design matrix with monomials x**k for k=0..m-1"""
    length = len(x)
    matrix = np.zeros((length, m))
    for i in range(length):
        for j in range(m):
            matrix[i][j] = x[i] ** j
    return matrix


def lsq_poly(x, y, m):
    """Construct the LSQ polynomial of degree `m-1`.

    Parameters
    ----------
    x : array_like
        Sample points
    y : array_like
        Measured values
    m : int
        The number of coefficients of the LSQ polynomial
        (i.e. the degree of the polynomial is `m-1`)

    Returns
    -------
    p : callable
        np.polynomial.Polynomial instance, representing the LSQ polynomial

    Examples
    --------
    >>> p = lsq_poly([1, 2, 3], [4, 5, 6], m=2)
    >>> p(np.array([1.5, 2.5]))
    array([4.5, 5.5])

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Expect paired data.")
    if x.shape[0] <= m:
        raise ValueError("Cannot fit a degree-%s polynomial through %s points" % (m, x.shape[0]))

    matrix = design_matrix(x, m)
    matrix_mp_inv = np.linalg.pinv(matrix)
    solution_matrix = matrix_mp_inv @ y
    polynomial = np.polynomial.Polynomial(solution_matrix)
    return polynomial

def sigma(x, y, m):
    r"""Compute $\sigma_m$."""
    n = len(x)
    summ = 0
    for i in range(n):
        summ_part = lsq_poly(x, y, 1)
        summ += (summ_part(x[i]) - y[i]) ** 2
    sigma_male_prev = summ / (n - 1)
    summ = 0
    for i in range(n):
        summ_part = lsq_poly(x, y, 2)
        summ += (summ_part(x[i]) - y[i]) ** 2
    sigma_male_prev_prev = summ / (n - 2)
    summ = 0
    for j in range(3, m):
        for i in range(n):
            summ_part = lsq_poly(x, y, m)
            summ += (summ_part(x[i]) - y[i]) ** 2
        sigma_male = summ / (n - m)
        summ = 0
        if (sigma_male - sigma_male_prev) <= 1e-16 and (sigma_male_prev - sigma_male_prev_prev) <= 1e-16:
          return j
        sigma_male_prev_prev = sigma_male_prev
        sigma_male_prev = sigma_male
    return m


def lsq_qr(x, y, m):
    """Solve the LSQ problem via the QR decomp of the design matrix.

    Parameters
    ----------
    x : array_like
        Sample points
    y : array_like
        Measured values
    m : int
        The degree of the LSQ polynomial

    Returns
    -------
    p : callable
        np.polynomial.Polynomial instance, representing the LSQ polynomial

    """
    a = design_matrix(x, m)
    q, r = np.linalg.qr(a)
    r_1 = r[:m, :]
    f_matrix = q.T @ y
    f = f_matrix[:m]
    solution = np.linalg.inv(r_1) @ f
    return np.polynomial.Polynomial(solution)




x = np.asarray([-1, -0.7, -0.43, -0.14, 0.14, 0.43, 0.71, 1, 1.29, 1.57, 1.86, 2.14, 2.43, 2.71, 3])
y = np.asarray([-2.25, -0.77, 0.21, 0.44, 0.64, 0.03, -0.22, -0.84, -1.2, -1.03, -0.37, 0.61, 2.67, 5.04, 8.90])

solution = lsq_qr(x, y, 5)
solution2 = lsq_poly(x, y, 5)
print(solution, solution2, sep='\n')
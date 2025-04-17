"""
Gaussian elimination with scaled partial pivoting.
"""

import numpy as np

def compute_scales(A):
    """
    Compute scale factors for each row of matrix A.
    :param A: numpy.ndarray
    :return: numpy.ndarray of scale factors
    """
    # Maximum absolute value in each row as scale factors
    return np.max(np.abs(A), axis=1)

def select_pivot(A, scales, k):
    """
    Select pivot row for column k using scaled partial pivoting.
    :param A: numpy.ndarray
    :param scales: numpy.ndarray of scale factors
    :param k: current pivot index
    :return: index of pivot row
    """
    # Compute scaled ratios and select pivot
    ratios = np.abs(A[k:, k]) / scales[k:]
    pivot_rel = np.argmax(ratios)
    return k + pivot_rel

def eliminate_step(A, b, pivot_row, k):
    """
    Perform elimination for step k, updating A and b in-place.
    :param A: numpy.ndarray
    :param b: numpy.ndarray
    :param pivot_row: row index to swap with k
    :param k: current pivot index
    """
    n = A.shape[0]
    for i in range(k+1, n):
        if A[k, k] == 0:
            raise ZeroDivisionError(f"Zero pivot encountered at step {k}")
        factor = A[i, k] / A[k, k]
        # Eliminate entry
        A[i, k:] -= factor * A[k, k:]
        b[i] -= factor * b[k]

def back_substitution(U, f):
    """
    Perform back-substitution on upper-triangular matrix U.
    :param U: numpy.ndarray
    :param f: numpy.ndarray
    :return: solution vector x
    """
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    # Solve from bottom row up
    for i in range(n-1, -1, -1):
        if U[i, i] == 0:
            raise ZeroDivisionError(f"Zero diagonal at back substitution index {i}")
        x[i] = (f[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def gauss_scaled(A, b):
    """
    Perform Gaussian elimination with scaled partial pivoting.
    Collect intermediate matrices and return final solution.
    :param A: numpy.ndarray
    :param b: numpy.ndarray
    :return: (x, intermediates)
    """
    n = A.shape[0]
    if n != b.shape[0]:
        raise ValueError("Matrix A and vector b must have compatible dimensions")
    intermediates = []
    # Record original matrix
    intermediates.append(A.copy())
    # Compute scale factors once on original A
    scales = compute_scales(A)
    # Forward elimination
    for k in range(n-1):
        pivot = select_pivot(A, scales, k)
        # Swap pivot row with current row
        A[[k, pivot]] = A[[pivot, k]]
        b[[k, pivot]] = b[[pivot, k]]
        eliminate_step(A, b, pivot, k)
        intermediates.append(A.copy())
    # Back substitution for final solution
    x = back_substitution(A, b)
    return x, intermediates

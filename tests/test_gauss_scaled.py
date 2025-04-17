import numpy as np
import pytest
from Modules.gauss_scaled import compute_scales, select_pivot, eliminate_step, back_substitution, gauss_scaled

def test_compute_scales_simple():
    A = np.array([[2, 1], [3, 4]], float)
    scales = compute_scales(A)
    assert np.allclose(scales, np.array([2, 4]))

def test_select_pivot_scaled():
    A = np.array([[2, 1], [3, 4]], float)
    scales = np.array([2, 4])
    pivot = select_pivot(A, scales, 0)
    assert pivot == 1

def test_eliminate_and_back_sub():
    A = np.array([[2., 1.], [3., 4.]], float)
    b = np.array([5., 6.])
    scales = compute_scales(A.copy())
    pivot = select_pivot(A.copy(), scales, 0)
    eliminate_step(A, b, pivot, 0)
    x = back_substitution(A, b)
    expected = np.linalg.solve(np.array([[2,1],[3,4]]), np.array([5,6]))
    assert np.allclose(x, expected)

def test_gauss_scaled_solution():
    A = np.array([[3, -13, 9, 3], [-6, 4, 1, -18], [6, -2, 2, 4], [12, -8, 6, 10]], float)
    b = np.array([-19, -34, 16, 26], float)
    x, intermediates = gauss_scaled(A.copy(), b.copy())
    expected = np.linalg.solve(np.array([[3, -13, 9, 3], [-6, 4, 1, -18], [6, -2, 2, 4], [12, -8, 6, 10]], float), np.array([-19, -34, 16, 26], float))
    assert np.allclose(x, expected)
    assert len(intermediates) == A.shape[0]

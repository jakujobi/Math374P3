import numpy as np
import pytest
from src.gauss_sp import scaled_partial_pivot_gauss


def test_3x3_solution():
    # Example 3×3 system requiring pivoting
    A = np.array([[3.0, 1.0, 2.0],
                  [1.0, 2.0, 0.0],
                  [0.0, 1.0, 1.0]])
    b = np.array([10.0, 8.0, 3.0])
    # Solve with steps
    x, steps = scaled_partial_pivot_gauss(A.copy(), b.copy(), return_steps=True)
    # Compare to NumPy
    x_np = np.linalg.solve(A, b)
    assert np.allclose(x, x_np, atol=1e-12)
    # Check residual norm
    res = np.linalg.norm(A.dot(x) - b)
    assert res < 1e-11


def test_4x4_solution():
    # Example 4×4 system
    A = np.array([[2.0, 1.0, 1.0, 0.0],
                  [4.0, 3.0, 3.0, 1.0],
                  [8.0, 7.0, 9.0, 5.0],
                  [6.0, 7.0, 9.0, 8.0]])
    b = np.array([4.0, 10.0, 28.0, 29.0])
    x = scaled_partial_pivot_gauss(A, b)
    # Compare to NumPy
    x_np = np.linalg.solve(A, b)
    assert np.allclose(x, x_np, atol=1e-12)
    # Check residual norm
    res = np.linalg.norm(A.dot(x) - b)
    assert res < 1e-11


def test_return_steps_structure():
    # Simple 2×2 system to check logging structure
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([5.0, 6.0])
    x, steps = scaled_partial_pivot_gauss(A.copy(), b.copy(), return_steps=True)
    # Steps should be a list of dicts
    assert isinstance(steps, list)
    assert all(isinstance(step, dict) for step in steps)
    # Each log entry must include keys:
    for step in steps:
        assert 'step' in step
        assert 'A' in step and 'b' in step
    # The last step must be back-substitution on i=0
    last = steps[-1]
    assert last['step'] == 'back_substitution'
    assert last.get('i') == 0 
import numpy as np

def scaled_partial_pivot_gauss(A, b, return_steps=False, tol=1e-10):
    """
    Solves Ax = b using Gaussian elimination with scaled partial pivoting.
    Returns the solution vector x, and optionally step-by-step logs.
    
    Parameters:
    -----------
    A : array-like
        Coefficient matrix
    b : array-like
        Right-hand side vector
    return_steps : bool, optional
        If True, return detailed steps of the algorithm
    tol : float, optional
        Tolerance for detecting near-singular matrices
        
    Returns:
    --------
    x : ndarray
        Solution vector
    steps : list, optional
        Detailed steps of the algorithm (if return_steps=True)
    """
    # Convert inputs to numpy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]
    # Validate dimensions
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if b.size != n:
        raise ValueError("Vector b length must equal A dimension.")

    steps = []
    # Compute scaling factors for each row
    s = np.max(np.abs(A), axis=1)
    
    # Check for zero scaling factors
    if np.any(s == 0):
        raise ValueError("Matrix contains a row of zeros.")

    # Forward elimination with scaled partial pivoting
    for k in range(n - 1):
        # Determine pivot row based on scaled ratios
        ratios = np.abs(A[k:, k]) / s[k:]
        idx_max = np.argmax(ratios)
        p = k + idx_max
        ratio = float(ratios[idx_max])  # scaled ratio for pivot
        
        # Check for near-singular matrix
        if abs(A[p, k]) < tol:
            raise ValueError("Matrix is singular or nearly singular.")
            
        # Swap rows if necessary, logging ratio
        if p != k:
            A[[k, p], :] = A[[p, k], :]
            b[k], b[p] = b[p], b[k]
            steps.append({
                "step": "swap",
                "k": k,
                "pivot_row": p,
                "ratio": ratio,
                "A": A.copy(),
                "b": b.copy()
            })
        else:
            steps.append({
                "step": "pivot",
                "k": k,
                "pivot_row": p,
                "ratio": ratio,
                "A": A.copy(),
                "b": b.copy()
            })
        # Eliminate entries below pivot
        for i in range(k + 1, n):
            # Compute multiplier with fraction components
            num = A[i, k]
            den = A[k, k]
            m = num / den
            # Perform elimination row update
            A[i, k:] = A[i, k:] - m * A[k, k:]
            b[i] = b[i] - m * b[k]
            steps.append({
                "step": "elimination",
                "k": k,
                "i": i,
                "multiplier": m,
                "mult_num": num,
                "mult_den": den,
                "A": A.copy(),
                "b": b.copy()
            })

    # Back substitution to solve for x
    x = np.zeros(n, dtype=float)
    for i in reversed(range(n)):
        if abs(A[i, i]) < tol:
            raise ValueError("Matrix is singular or nearly singular.")
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        steps.append({
            "step": "back_substitution",
            "i": i,
            "value": x[i],
            "A": A.copy(),
            "b": b.copy()
        })

    if return_steps:
        return x, steps
    return x

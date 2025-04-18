import numpy as np
import pandas as pd
from typing import Tuple

def compute_givens_rotation_coefficients(a: float, b: float) -> Tuple[float, float]:
    """
    Compute the Givens rotation matrix elements.
    
    Args:
        a: Diagonal element
        b: Element to be zeroed
        
    Returns:
        Tuple[float, float]: Cosine and sine of the rotation angle (c, s)
    """
    r = np.sqrt(a**2 + b**2)
    c = a / r
    s = -b / r
    return c, s

def solve_linear_system_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve a system of linear equations using QR decomposition with Givens rotations.
    
    The method solves the equation Ax = b by first decomposing A into QR,
    where Q is orthogonal and R is upper triangular, then solving Rx = Q^T b.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n x 1)
        
    Returns:
        np.ndarray: Solution vector x
        
    Note:
        This implementation uses Givens rotations to compute the QR decomposition,
        which is numerically stable for solving linear systems.
    """
    n = A.shape[0]
    
    # Initialize R as a copy of A and Q as identity matrix
    R = A.copy()
    Q = np.identity(n)
    
    # Compute QR decomposition using Givens rotations
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Compute rotation parameters
            c, s = compute_givens_rotation_coefficients(R[i,i], R[j,i])
            
            # Apply rotation to R matrix (only affected rows change)
            R[i], R[j] = (R[i] * c + R[j] * (-s)), (R[i] * s + R[j] * c)
            
            # Update Q matrix for final solution
            Q[i], Q[j] = (Q[i] * c + Q[j] * (-s)), (Q[i] * s + Q[j] * c)
    
    # Solve Rx = Q^T b using back substitution
    y = np.dot(Q, b)  # Compute Q^T b
    x = np.zeros(n)
    
    # Back substitution
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i,i]
        
    return x

def format_solution(x: np.ndarray) -> pd.DataFrame:
    """
    Format the solution vector as a pandas DataFrame.
    
    Args:
        x: Solution vector
        
    Returns:
        pd.DataFrame: Formatted solution with variable names
    """
    return pd.DataFrame(
        {f"x{i+1}": [val] for i, val in enumerate(x)},
        dtype=float
    )

# Example usage
if __name__ == "__main__":
    # Example problem setup
    A = np.array([
        [10, 8, 12, 0, 78, 12, 35],
        [8, 12, 34.1, 0, 0, 0, 0],
        [0, 34, 12, 6, 0, 5, 0],
        [0, 5.5, 6, 6, 10, 0, 1.1],
        [0, 4.3, 0, 10, 9, 12, 0],
        [0, 0, 2, 12, 122, 0, 11],
        [0, 1, 0, 14, 0, 11, 4]
    ])
    
    b = np.array([30.1, 54.01, 52.01, 22, 31, 44, 29])
    
    # Solve system
    solution = solve_linear_system_qr(A, b)
    
    # Format and display results
    result_df = format_solution(solution)
    print(f'solution: {result_df}')
    print(f'vector b: {np.dot(A, solution)}')
import numpy as np
import pandas as pd
from typing import Optional
import diagonally_dominant_matrix as dm

def successive_over_relaxation(
    A: np.ndarray,
    b: np.ndarray,
    omega: float,
    eps: float,
    max_iterations: int,
    x0: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Solve a system of linear equations using the Successive Over-Relaxation (SOR) method.
    
    The method is an iterative method that solves the equation Ax = b using a relaxation
    parameter omega to accelerate convergence.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n x 1)
        omega: Relaxation parameter (1 < omega < 2)
        eps: Convergence criterion (tolerance)
        max_iterations: Maximum number of iterations
        x0: Initial guess for the solution (optional)
        
    Returns:
        pd.DataFrame: DataFrame containing iteration history if method converges
        None: If matrix does not satisfy convergence conditions
        
    Note:
        The matrix A must be either diagonally dominant or symmetric positive definite
        for the method to converge.
    """
    size = A.shape[0]
    
    # Extract diagonal and non-diagonal elements
    diagonal_matrix = np.diag(np.diag(A))
    matrix_without_diagonal = A - diagonal_matrix

    all_eigenvalues_positive = np.all(np.linalg.eigvals(A) > 0)
    matrix_symmetric = np.array_equal(A, A.T)
    
    # Check convergence conditions
    has_unique_solution = (
        dm.is_diagonally_dominant(A) or
        (matrix_symmetric and all_eigenvalues_positive)
    )
    
    if not has_unique_solution:
        return None
    
    # Initialize solution vectors
    if x0 is None:
        current_solution = np.zeros(size)
    else:
        current_solution = np.copy(x0)
    
    previous_solution = np.copy(current_solution)
    
    # Initialize results tracking
    results = {
        "Step": [0],
        "Error": [0],
        **{f"X{i+1}": [current_solution[i]] for i in range(size)}
    }
    
    # Main iteration loop
    for iteration in range(max_iterations):
        # Update each component
        for i in range(size):
            # Calculate new value using SOR formula
            row_sum = np.dot(matrix_without_diagonal[i,:], current_solution)
            gauss_seidel_step = (b[i] - row_sum) / diagonal_matrix[i,i]
            current_solution[i] = (1 - omega) * previous_solution[i] + omega * gauss_seidel_step
        
        # Calculate error
        error = np.abs(current_solution - previous_solution)
        max_error = np.max(error)
        
        # Update results
        results["Step"].append(iteration + 1)
        results["Error"].append(max_error)
        for i in range(size):
            results[f"X{i+1}"].append(current_solution[i])
        
        # Check convergence
        if max_error < eps:
            return pd.DataFrame(results)
            
        previous_solution = np.copy(current_solution)
    
    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Configure pandas display format for better precision
    pd.set_option('display.float_format', '{:.8f}'.format)

    # Example problem setup
    A = np.array([
        [10, 2, 1],
        [1, 10, 2],
        [1, 1, 10]
    ])
    b = np.array([13, 13, 12])
    
    # Initial guess (optional)
    x0 = np.zeros(3)
    
    # Solve system
    result = successive_over_relaxation(
        A=A,
        b=b,
        omega=1.1,
        eps=1e-4,
        max_iterations=30,
        x0=x0
    )
    
    print(result)
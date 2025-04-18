import numpy as np
import pandas as pd
from typing import Optional
import diagonally_dominant_matrix as dm

def jacobi_method(
    A: np.ndarray,
    b: np.ndarray,
    eps: float,
    max_iterations: int,
    x0: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Solve a system of linear equations using the Jacobi iterative method.
    
    The method solves the equation Ax = b iteratively, where A is a coefficient matrix,
    b is the right-hand side vector, and x is the solution vector.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n x 1)
        eps: Convergence criterion (tolerance)
        max_iterations: Maximum number of iterations
        x0: Initial guess for the solution (optional)
        
    Returns:
        pd.DataFrame: DataFrame containing iteration history
    """
    size = A.shape[0]
    
    # Extract diagonal and non-diagonal elements
    diagonal_matrix = np.diag(np.diag(A))
    matrix_without_diagonal = A - diagonal_matrix
    
    # Check convergence condition
    if not dm.is_diagonally_dominant(A):
        return None
    
    # Initialize solution vectors
    if x0 is None:
        current_solution = np.zeros(size)
    else:
        current_solution = np.copy(x0)
    
    next_solution = np.copy(current_solution)
    
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
            row_sum = np.dot(matrix_without_diagonal[i,:], current_solution)
            next_solution[i] = (b[i] - row_sum) / diagonal_matrix[i,i]
        
        # Calculate error
        error = np.abs(next_solution - current_solution)
        max_error = np.max(error)
        
        # Update results
        results["Step"].append(iteration + 1)
        results["Error"].append(max_error)
        for i in range(size):
            results[f"X{i+1}"].append(next_solution[i])
        
        # Check convergence
        if max_error < eps:
            return pd.DataFrame(results)
            
        # Update solution for next iteration
        current_solution = np.copy(next_solution)
    
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
    result = jacobi_method(
        A=A,
        b=b,
        eps=1e-6,
        max_iterations=1000,
        x0=x0
    )
    
    print(result)
   
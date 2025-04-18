import numpy as np
import pandas as pd
import diagonally_dominant_matrix as dm

def gauss_seidel_method(
    A: np.ndarray,
    b: np.ndarray,
    eps: float,
    max_iterations: int
) -> pd.DataFrame:
    """
    Solve a system of linear equations using the Gauss-Seidel method.
    
    The method iteratively solves the equation Ax = b, where A is a coefficient matrix,
    b is the right-hand side vector, and x is the solution vector.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n x 1)
        eps: Convergence criterion (tolerance)
        max_iterations: Maximum number of iterations
        
    Returns:
        pd.DataFrame: DataFrame containing iteration history if solution converges
        None: If no unique solution exists
        
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
    
    # Check existence of unique solution
    has_unique_solution = (
        dm.is_diagonally_dominant(A) or
        (matrix_symmetric and all_eigenvalues_positive)
    )
    
    if not has_unique_solution:
        return None
    
    # Initialize solution vectors
    current_solution = np.copy(b)
    previous_solution = np.copy(b)
    
    # Initialize results tracking
    results = {
        "Step": [0],
        "Error": [0],
        **{f"X{i+1}": [current_solution[i]] for i in range(size)}
    }
    
    # Main iteration loop
    for iteration in range(max_iterations):
        for i in range(size):
            # Calculate new value for x[i]
            row_sum = np.dot(matrix_without_diagonal[i,:], current_solution)
            current_solution[i] = (b[i] - row_sum) / diagonal_matrix[i,i]
        
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
    
    # Solve system
    result = gauss_seidel_method(
        A=A,
        b=b,
        eps=1e-6,
        max_iterations=1000
    )
    
    if result is not None:
        print(result)
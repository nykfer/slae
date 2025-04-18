import numpy as np

def is_diagonally_dominant(matrix: np.ndarray) -> bool:
    """
    Check if a given matrix is diagonally dominant.
    
    A matrix is diagonally dominant if the absolute value of each diagonal element
    is greater than the sum of absolute values of all other elements in its row.
    
    Args:
        matrix: Input matrix as numpy array or nested list
        
    Returns:
        bool: True if the matrix is diagonally dominant, False otherwise
        
    Example:
        >>> matrix = np.array([[5, -2, 1], 
                             [-1, 4, 2], 
                             [1, -1, 3]])
        >>> is_diagonally_dominant(matrix)
        True
    """

    matrix_size = matrix.shape[0]
    
    for row_idx in range(matrix_size):
        diagonal_element = abs(matrix[row_idx][row_idx])
        sum_of_other_elements = sum(abs(matrix[row_idx][col_idx]) 
                                  for col_idx in range(matrix_size) 
                                  if row_idx != col_idx)
        
        if diagonal_element <= sum_of_other_elements:
            return False
            
    return True



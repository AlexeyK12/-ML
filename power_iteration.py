import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    n = data.shape[0]
    
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(num_steps):
        Av = np.dot(data, v)
        eigenvalue = np.dot(Av, v)
        v = Av / np.linalg.norm(Av)
    
    return eigenvalue, v
# SLAE (Systems of Linear Algebraic Equations) Solver

This project implements various numerical methods for solving Systems of Linear Algebraic Equations. It provides a comprehensive suite of iterative and direct methods, each with its own advantages and trade-offs.

## Implemented Algorithms

### 1. Gauss-Seidel Method
**Mathematical Formula:**
For a system Ax = b, where A is an n × n matrix, the Gauss-Seidel iteration is:

```
x[i]^(k+1) = (1/a[ii]) * (b[i] - Σ(j=1 to i-1) a[ij]*x[j]^(k+1) - Σ(j=i+1 to n) a[ij]*x[j]^(k))
```

where:
- x[i]^(k) is the i-th component of x at iteration k
- a[ij] are the elements of matrix A
- b[i] is the i-th component of vector b

**Advantages:**
- Generally faster convergence than the Jacobi method
- Memory efficient (updates values in-place)
- Simple implementation
- Works well for diagonally dominant matrices

**Limitations:**
- Requires the matrix to be either diagonally dominant or symmetric positive definite
- Sequential nature makes parallelization difficult
- Convergence rate depends on the matrix properties

### 2. Jacobi Method
**Mathematical Formula:**
The Jacobi iteration formula is:

```
x[i]^(k+1) = (1/a[ii]) * (b[i] - Σ(j≠i) a[ij]*x[j]^(k))
```

where:
- All variables are updated simultaneously using values from previous iteration
- Convergence requires |a[ii]| > Σ(j≠i) |a[ij]| (diagonal dominance)

**Advantages:**
- Highly parallelizable due to independent calculations
- Simple to understand and implement
- Numerically stable for diagonally dominant matrices
- Good for sparse matrices

**Limitations:**
- Slower convergence compared to Gauss-Seidel
- Requires additional memory for storing the previous iteration
- Convergence only guaranteed for diagonally dominant matrices

### 3. Successive Over-Relaxation (SOR)
**Mathematical Formula:**
SOR extends the Gauss-Seidel method with a relaxation parameter ω:

```
x[i]^(k+1) = (1-ω)*x[i]^(k) + (ω/a[ii]) * (b[i] - Σ(j=1 to i-1) a[ij]*x[j]^(k+1) - Σ(j=i+1 to n) a[ij]*x[j]^(k))
```

where:
- ω is the relaxation parameter (1 < ω < 2)
- Optimal ω depends on matrix properties
- ω = 1 reduces to Gauss-Seidel method

**Advantages:**
- Faster convergence than both Gauss-Seidel and Jacobi methods
- Adjustable relaxation parameter for optimization
- Can significantly reduce the number of iterations needed

**Limitations:**
- Requires optimal relaxation parameter (ω) selection
- More complex implementation
- Same convergence requirements as Gauss-Seidel
- Can diverge if relaxation parameter is not chosen properly

### 4. Givens Rotation (QR Decomposition)
**Mathematical Formula:**
Givens rotation coefficients are computed as:

```
c = cos(θ) = a/sqrt(a² + b²)
s = sin(θ) = -b/sqrt(a² + b²)
```

The Givens rotation matrix G(i,j,θ) has the form:
```
[1    ...   0   ...   0    ...   0]
[.    ...   .   ...   .    ...   .]
[0    ...   c   ...  -s    ...   0]
[.    ...   .   ...   .    ...   .]
[0    ...   s   ...   c    ...   0]
[.    ...   .   ...   .    ...   .]
[0    ...   0   ...   0    ...   1]
```

The solution process:
```
Ax = b → QRx = b → Rx = Q'b
```

**Advantages:**
- Direct method (non-iterative)
- Numerically stable
- Always produces a solution if one exists
- Good for dense matrices

**Limitations:**
- Higher computational complexity O(n³)
- Requires more memory than iterative methods
- Less efficient for large sparse matrices

## Code Structure and Implementation

### Core Components
1. **Individual Method Implementations**:
   - `Gauss_Seidel.py`: Implementation of the Gauss-Seidel method
   - `Jacobi_Method.py`: Implementation of the Jacobi iterative method
   - `relaxation_methods.py`: Implementation of the SOR method
   - `givens_rotation.py`: Implementation of QR decomposition using Givens rotations

2. **Utility Modules**:
   - `diagonally_dominant_matrix.py`: Helper functions for matrix property checking


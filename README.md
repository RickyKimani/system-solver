
# Gaussian Elimination Solver

This Python script was created to address a real challenge I encountered while studying for my chemical mass balance class. I needed to solve a 4x4 matrix representing a system of equations, and I wanted to implement a solution that demonstrated each step of the Gaussian elimination process. This tool allows anyone to solve a system of linear equations through Gaussian elimination while seeing every operation and transformation applied to the augmented matrix.

## Features

- **Step-by-Step Output**: Shows each row swap, normalization, and elimination during the forward elimination phase.
- **Back Substitution**: Completes the solution with a clear back-substitution process.
- **Detailed Solution Output**: Prints the solution in a readable format with calculation time.

## Requirements

- Python 3.x
- NumPy for matrix operations (install with `pip install numpy`)

## Usage

1. **Terminal Input**:
   - Run the script and input the coefficient matrix and answer vector as space-separated values.
   - The script will display each step of the Gaussian elimination, showing how the solution is derived.

## How to Run

```bash
python gaussian_elimination_solver.py
```

Follow the prompts to enter your matrix values, and see the full solution with all steps laid out.

## Example

Input:
- Coefficient matrix A:
  ```
  2 1 -1 2
  -3 -1 2 -11
  -2 1 2 -3
  1 2 -1 -1
  ```
- Answer vector b:
  ```
  8 -15 -3 4
  ```

The output will display each row transformation and the final solution vector for the unknowns.

## Why Gaussian Elimination?

Gaussian elimination is a foundational tool in linear algebra and essential for solving linear systems, especially in chemical engineering, where mass balance problems often lead to systems of equations. This script provides a clear and educational walkthrough of the Gaussian elimination process, reinforcing understanding for users who need both the solution and the methodology.

---

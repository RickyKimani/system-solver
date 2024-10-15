import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    # Augment the matrix
    aug_matrix = np.hstack([A, b.reshape(-1, 1)])
    
    print("Initial augmented matrix:")
    print(aug_matrix, "\n")
    
    # Forward elimination
    for i in range(n):
        # Pivot for maximum element in column i
        max_row = np.argmax(np.abs(aug_matrix[i:, i])) + i
        if i != max_row:
            aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]
            print(f"Swapped row {i+1} with row {max_row+1}:")
            print(aug_matrix, "\n")

        # Make pivot element 1 and eliminate below
        pivot = aug_matrix[i, i]
        if pivot != 0:
            aug_matrix[i] = aug_matrix[i] / pivot
            print(f"Made pivot of row {i+1} equal to 1:")
            print(aug_matrix, "\n")
        else:
            print("No unique solution.")
            return None

        # Eliminate the entries below the pivot
        for j in range(i+1, n):
            factor = aug_matrix[j, i]
            aug_matrix[j] = aug_matrix[j] - factor * aug_matrix[i]
            print(f"Eliminated element in row {j+1}, column {i+1}:")
            print(aug_matrix, "\n")

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = aug_matrix[i, -1] - np.dot(aug_matrix[i, i+1:n], x[i+1:n])
        print(f"Back substitution for x[{i+1}]: {x[i]}")

    return x

def input_matrix(prompt):
    print(prompt)
    matrix = []
    n = int(input("Enter the number of equations (and unknowns): "))
    for i in range(n):
        row = list(map(float, input(f"Enter row {i+1} (space-separated coefficients): ").split()))
        matrix.append(row)
    return np.array(matrix)

def input_vector(prompt):
    print(prompt)
    vector = list(map(float, input("Enter the answer vector (space-separated values): ").split()))
    return np.array(vector)

def unknowns_vector(prompt):
    print(prompt)
    vector = list(map(str, input("Enter the unknowns vector (space separated values): ").split()))
    return np.array(vector)

# Input the coefficient matrix (A) and answer matrix (b)
A = input_matrix("Enter the coefficient matrix:")
b = input_vector("Enter the answer matrix:")
c = unknowns_vector("Enter the unknowns vector:")

# Solve the system
solution = gaussian_elimination(A, b)
print("\nSolution:")
print(f"{c} = {solution}")

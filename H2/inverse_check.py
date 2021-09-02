from numpy.linalg import matrix_power
import numpy as np

matrix = np.array([[4,7],[2,6]])

print("Matrix: ", '\n', matrix, '\n')

inverse = matrix_power(matrix, -1)

print("Inverse: ", '\n', inverse)

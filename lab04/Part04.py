import numpy as np


def normalize_matrix(matrix):
    normalized_matrix = np.round(matrix / np.sum(matrix), 2)

    return normalized_matrix


def sum_rows(matrix):
    row_sums = np.sum(matrix, axis=1)
    return row_sums


def multiply_matrix(matrix, vector):
    result_vector = np.dot(matrix, vector)
    return result_vector


def normalize_matrix_another(matrix):
    size = matrix.shape[0]
    result_matrices = []

    for i in range(size):
        column_i = matrix[:, i]
        new_matrix = np.zeros_like(matrix, dtype=float)

        for j in range(size):
            new_matrix[:, j] = column_i[j] / column_i

        result_matrices.append(new_matrix.T)

    return result_matrices


matrix_small = np.array([8, 3, 7, 6])
matrix_big = np.array([[7, 8, 2, 2],
                       [6, 6, 3, 5],
                       [1, 5, 6, 4],
                       [3, 2, 7, 7]])

print("\nМатрица приоритетов критериев:")
print(matrix_small)

normalized_matrix = normalize_matrix(matrix_small)
print("\nНормированная матрица приоритетов критериев:")
print(normalized_matrix)

print("\nМатрица материалов и критериев:")
print(matrix_big)

np.set_printoptions(formatter={'float': lambda x: "{:0.2f}".format(x)})

result_new_matrix = normalize_matrix_another(matrix_big)

column1 = normalize_matrix(sum_rows(result_new_matrix[0]))
column2 = normalize_matrix(sum_rows(result_new_matrix[1]))
column3 = normalize_matrix(sum_rows(result_new_matrix[2]))
column4 = normalize_matrix(sum_rows(result_new_matrix[3]))
new_normalized_matrix = np.column_stack((column1, column2, column3, column4))

result_matrix = np.zeros((len(matrix_small), len(matrix_small)))

for i in range(len(matrix_small)):
    result_matrix[i, :] = matrix_small[i] / matrix_small

result_multiply_vector_another = multiply_matrix(new_normalized_matrix, normalize_matrix(sum_rows(result_matrix)))

print("\nЧетыре промежуточные матрицы нормирования:")
for i, new_matrix in enumerate(result_new_matrix):
    print(f"\nМатрица {i + 1}:")
    print(new_matrix)
    print("\nСогласованность:")
    print((np.sum(normalized_matrix*(np.sum(new_matrix, axis=0))) - 4)/2.7)
    print("\nСумма строк этой матрицы")
    print(sum_rows(new_matrix))
    print("\nНормальная матрица суммы:")
    print(normalize_matrix(sum_rows(new_matrix)))

print("\nНормированная матрица другим способом:")
print(new_normalized_matrix)

print("\nПромежуточная матрица нормирования:")
print(result_matrix)

print("\nСогласованность:")
print((np.sum(normalized_matrix*(np.sum(result_matrix, axis=0))) - 4)/2.7)

print("\nНормированная матрица другим способом:")
print(normalize_matrix(sum_rows(result_matrix)))

np.set_printoptions(formatter={'float': lambda x: "{:0.4f}".format(x)})

print("\nРезультат перемножения двух матриц:")
print(result_multiply_vector_another)
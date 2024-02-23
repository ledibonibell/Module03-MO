import numpy as np


def normalize_matrix(matrix):
    normalized_matrix = np.round(matrix / np.sum(matrix), 2)

    return normalized_matrix


def normalize_matrix_columns(matrix):
    normalized_matrix = np.round(matrix / np.sum(matrix, axis=0), 2)

    return normalized_matrix


def pairwise_comparison(matrix):
    size = matrix.shape[0]
    result_matrix = np.zeros_like(matrix, dtype=float)

    for i in range(size):
        for j in range(i + 1, size):
            comparison_result = float(input(f"Является ли элемент [{i + 1}][{j + 1}] главным? Введите 1, 0 или 0.5: "))

            if comparison_result == 1:
                result_matrix[i, j] = 1
            elif comparison_result == 0:
                result_matrix[j, i] = 1
            elif comparison_result == 0.5:
                result_matrix[i, j] = 0.5
                result_matrix[j, i] = 0.5
            else:
                print("Некорректный ввод. Используйте 1, 0 или 0.5!")
                return None

    return result_matrix


def sum_rows(matrix):
    row_sums = np.sum(matrix, axis=1)
    return row_sums


def multiply_matrix(matrix, vector):
    result_vector = np.dot(matrix, vector)
    return result_vector


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

normalized_matrix_big = normalize_matrix_columns(matrix_big)

matrix_pairwise_comparison = pairwise_comparison(matrix_big)
result_row_sums = sum_rows(matrix_pairwise_comparison)
normalized_row_sums = normalize_matrix(result_row_sums)

result_multiply_vector = multiply_matrix(normalized_matrix_big, normalized_row_sums)

print("\nНормированная матрица материалов и критериев:")
print(normalized_matrix_big)

if matrix_pairwise_comparison is not None:
    print("\nМатрица попарного сравнения:")
    print(matrix_pairwise_comparison)

print("\nМатрица суммы строк попарного сравнения:")
print(result_row_sums)

print("\nНормальная матрица суммы строк попарного сравнения:")
print(normalized_row_sums)

np.set_printoptions(formatter={'float': lambda x: "{:0.4f}".format(x)})

print("\nРезультат перемножения двух матриц:")
print(result_multiply_vector)

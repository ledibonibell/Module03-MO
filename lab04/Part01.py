import numpy as np


def normalize_matrix(matrix):
    normalized_matrix = np.round(matrix / np.sum(matrix), 2)

    return normalized_matrix


def find_max_min_in_columns(matrix, col_indices):
    if max(col_indices) >= matrix.shape[1]:
        raise ValueError("Некорректные индексы столбцов")

    max_min_dict = {}

    for i in col_indices:
        max_element = np.max(matrix[:, i])
        min_element = np.min(matrix[:, i])

        max_min_dict[i] = {'max': max_element, 'min': min_element}

    return max_min_dict


def multiply_max_numbers(matrix, col_indices, multiplier_array):
    max_values = [np.max(matrix[:, i]) for i in col_indices]
    result_array = np.array([max_val * multiplier_array[i] for i, max_val in enumerate(max_values)])

    return result_array


def compute_expression(matrix, col_indices, max_min_dict, skipped_column):
    result_matrix = np.zeros_like(matrix, dtype=float)

    for i in range(matrix.shape[1]):
        if i == skipped_column:
            result_matrix[:, i] = matrix[:, i]
        elif i in col_indices:
            max_val = max_min_dict[i]['max']
            min_val = max_min_dict[i]['min']

            result_matrix[:, i] = np.round((matrix[:, i] - min_val) / (max_val - min_val), 2)

    return result_matrix


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

skipped_column_input = int(input("\nВведите номер главного критерия (от 1 до 4): ")) - 1

if skipped_column_input < 0 or skipped_column_input >= matrix_big.shape[1]:
    raise ValueError("Некорректный номер столбца")

col_indices = [i for i in range(matrix_big.shape[1]) if i != skipped_column_input]

multiplier_array = np.array([float(input(f"Введите коэффициент для неглавного параметра №{i + 1}: ")) for i in range(3)])

result_multiply_array = multiply_max_numbers(matrix_big, col_indices, multiplier_array)

max_min_dict = find_max_min_in_columns(matrix_big, col_indices)

print("\nМаксимальные и минимальные элементы для каждого не главного критерия:")
for col_index, values in max_min_dict.items():
    print(f"Столбец {col_index + 1}: Максимальный элемент = {values['max']}, Минимальный элемент = {values['min']}")

print(f"\nМинимально допустимые уровни для не главных критериев:")
print(result_multiply_array)

result_matrix = compute_expression(matrix_big, col_indices, max_min_dict, skipped_column_input)
print(f"\nРезультат вычислений выражения для каждого элемента выбранных столбцов матрицы 4x4:")
print(result_matrix)
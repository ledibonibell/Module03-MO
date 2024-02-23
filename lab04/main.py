import numpy as np
import matplotlib.pyplot as plt


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


"""########
Данные нам условия
########"""

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

"""########
Поиск данных из части 1
########"""

print("\nЧАСТЬ 1")

skipped_column_input = int(input("\nВведите номер главного критерия (от 1 до 4): ")) - 1

if skipped_column_input < 0 or skipped_column_input >= matrix_big.shape[1]:
    raise ValueError("Некорректный номер столбца")

col_indices = [i for i in range(matrix_big.shape[1]) if i != skipped_column_input]

multiplier_array = np.array([float(input(f"Введите коэффициент для неглавного параметра №{i + 1}: ")) for i in range(3)])

result_multiply_array = multiply_max_numbers(matrix_big, col_indices, multiplier_array)

max_min_dict = find_max_min_in_columns(matrix_big, col_indices)

"""########
Вывод данных из части 1
########"""

print("\nМаксимальные и минимальные элементы для каждого не главного критерия:")
for col_index, values in max_min_dict.items():
    print(f"Столбец {col_index + 1}: Максимальный элемент = {values['max']}, Минимальный элемент = {values['min']}")

print(f"\nМинимально допустимые уровни для не главных критериев:")
print(result_multiply_array)

result_matrix4x4 = compute_expression(matrix_big, col_indices, max_min_dict, skipped_column_input)
print(f"\nРезультат вычислений выражения для каждого элемента выбранных столбцов матрицы 4x4:")
print(result_matrix4x4)

"""########
Поиск данных из части 2
########"""

print("\nЧАСТЬ 2")

try:
    selected_columns = [int(input('Введите номер первого главного криетрия: ')) - 1,
                        int(input('Введите номер второго главного криетрия: ')) - 1]
except ValueError:
    print("Некорректный ввод. Пожалуйста, введите целые числа")
    exit()

if any(col < 0 or col >= matrix_big.shape[1] for col in selected_columns):
    print("Некорректные номера столбцов. Пожалуйста, введите корректные номера")
    exit()

print("\nВ папке проекта лежит график lab02.png")

x = matrix_big[:, selected_columns[0]]
y = matrix_big[:, selected_columns[1]]

max_x_coord = 10
max_y_coord = 10

colors = ['red', 'green', 'blue', 'purple']
materials = ['Береза', 'Сосна', 'Дуб', 'Лиственница']

"""########
Вывод данных из части 2
########"""

for i in range(len(x)):
    plt.scatter(x[i], y[i], label=materials[i], color=colors[i])
    plt.text(x[i], y[i], '', fontsize=8, ha='right', va='bottom')

plt.scatter(max_x_coord, max_y_coord, color='black', marker='o', label='Точка утопии')

plt.grid(True)

plt.xlabel("Обработка")
plt.ylabel("Долговечность")
plt.title('Критерий Парето')
plt.legend()

plt.savefig("lab02.png")

"""########
Поиск данных из части 3
########"""

print("\nЧАСТЬ 3")

normalized_matrix_big = normalize_matrix_columns(matrix_big)

matrix_pairwise_comparison = pairwise_comparison(matrix_big)
result_row_sums = sum_rows(matrix_pairwise_comparison)
normalized_row_sums = normalize_matrix(result_row_sums)

result_multiply_vector = multiply_matrix(normalized_matrix_big, normalized_row_sums)

"""########
Вывод данных из части 3
########"""

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

np.set_printoptions(formatter={'float': lambda x: "{:0.2f}".format(x)})

"""########
Поиск данных из части 4
########"""

print("\nЧАСТЬ 4")

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

"""########
Вывод данных из части 4
########"""

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


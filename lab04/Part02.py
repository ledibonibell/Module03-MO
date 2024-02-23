import numpy as np
import matplotlib.pyplot as plt

matrix_big = np.array([[7, 8, 2, 2],
                       [6, 6, 3, 5],
                       [1, 5, 6, 4],
                       [3, 2, 7, 7]])

try:
    selected_columns = [int(input('Введите номер первого главного криетрия: ')) - 1,
                        int(input('Введите номер второго главного криетрия: ')) - 1]
except ValueError:
    print("Некорректный ввод. Пожалуйста, введите целые числа")
    exit()

if any(col < 0 or col >= matrix_big.shape[1] for col in selected_columns):
    print("Некорректные номера столбцов. Пожалуйста, введите корректные номера")
    exit()

x = matrix_big[:, selected_columns[0]]
y = matrix_big[:, selected_columns[1]]

max_x_coord = x[np.argmax(matrix_big[:, selected_columns[0]])]
max_y_coord = y[np.argmax(matrix_big[:, selected_columns[1]])]

colors = ['red', 'green', 'blue', 'purple']
materials = ['Береза', 'Сосна', 'Дуб', 'Лиственница']

for i in range(len(x)):
    plt.scatter(x[i], y[i], label=materials[i], color=colors[i])
    plt.text(x[i], y[i], '', fontsize=8, ha='right', va='bottom')

plt.scatter(max_x_coord, max_y_coord, color='black', marker='o', label='Точка утопии')

plt.grid(True)

plt.xlabel(f'Критерий №{selected_columns[0]}')
plt.ylabel(f'Критерий №{selected_columns[1]}')
plt.title('Критерий Парето')
plt.legend()

plt.savefig("lab02.png")

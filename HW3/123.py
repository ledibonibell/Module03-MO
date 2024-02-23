def calculate_late_start(duration, precedency, delaying):
    ls = {}
    for i in precedency.keys():
        if not precedency[i]:
            ls[i] = delaying[i]
        else:
            ls[i] = max([duration[i] + ls[i] for i in precedency[i]])
    return ls


def calculate_early_end(duration, precedency):
    ee = {}
    for i in precedency.keys():
        if not precedency[i]:
            ee[i] = duration[i]
        else:
            ee[i] = max([ee[i] + duration[i] for i in precedency[i]])
    return ee


def calculate_early(way, early_ij):
    early_i = {}
    for i in way.keys():
        if not way[i]:
            early_i[i] = 0
        else:
            early_i[i] = max(early_ij[i] for i in way[i])
    return early_i


def calculate_late(way, duration, late_ij):
    late_i = {}
    for i in way.keys():
        if not way[i]:
            late_i[i] = 0
        else:
            late_i[i] = max(late_ij[i] + duration[i] for i in way[i])
    return late_i


def calculate_full_reserve(coordinate, duration, late_j, early_i):
    reserve = {}
    for i in coordinate.keys():
        reserve[i] = late_j[coordinate[i][1]] - early_i[coordinate[i][0]] - duration[i]
    return reserve


def calculate_free_reserve(coordinate, duration, early_ij):
    reserve = {}
    for i in coordinate.keys():
        reserve[i] = early_ij[coordinate[i][1]] - early_ij[coordinate[i][0]] - duration[i]
    return reserve


# def find_critical_path(coordinates, durations):
#     critical_path = []
#     current_node = [key for key in coordinates.keys() if coordinates[key][0] not in coordinates.values()][0]
#
#     while current_node in coordinates:
#         critical_path.append(current_node)
#         successor_node = coordinates[current_node][1]
#
#         if durations[current_node] + late_start[current_node] == early_end[current_node]:
#             current_node = successor_node
#         else:
#             break
#
#     return critical_path


durations = {'a': 3, 'b': 5, 'c': 2, 'd': 4, 'e': 3, 'f': 1, 'g': 4, 'h': 3, 'i': 3, 'j': 2, 'k': 5}
precedence = {'a': [], 'b': [], 'c': ['b'], 'd': ['b'], 'e': [], 'f': ['a'], 'g': ['e', 'd'], 'h': ['c', 'f', 'g'],
              'i': ['c', 'f', 'g'], 'j': ['h'], 'k': ['i']}

delay = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0, 'k': 0}
ways = {'1': [], '2': ['a'], '3': ['b'], '4': ['e', 'd'], '5': ['c', 'f', 'g'], '6': ['h'], '7': ['i'], '8': ['j', 'k']}
coordinates = {'a': ['1', '2'], 'b': ['1', '3'], 'c': ['3', '5'], 'd': ['3', '4'], 'e': ['1', '4'], 'f': ['2', '5'],
               'g': ['4', '5'], 'h': ['5', '6'], 'i': ['5', '7'], 'j': ['6', '8'], 'k': ['7', '8']}

late_start = calculate_late_start(durations, precedence, delay)
early_end = calculate_early_end(durations, precedence)

early = calculate_early(ways, early_end)
late = calculate_late(ways, durations, late_start)

full_reserve = calculate_full_reserve(coordinates, durations, late, early)
free_reserve = calculate_free_reserve(coordinates, durations, early)

print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format("-Ways-", "-OR-", "-LS-", "-EE-", "-FullR-", "-FreeR-"))
for key in durations.keys():
    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(key, durations[key], late_start[key], early_end[key],
                                                             full_reserve[key], free_reserve[key]))

print("\n{:<10} {:<10} {:<10} {:<10}".format("-Peak-", "-tp-", "-tn-", "-R-"))
for key in ways.keys():
    print("{:<10} {:<10} {:<10} {:<10}".format(key, early[key], late[key], late[key] - early[key]))


n = 11
# start_nodes_values = [1, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7]
# end_nodes_values = [2, 3, 4, 5, 4, 5, 5, 6, 7, 8, 8]
# durations_values = [3, 5, 3, 1, 4, 2, 4, 3, 3, 2, 5]

start_nodes_values = [1, 2, 7, 1, 3, 4, 5, 8, 1, 6, 9]
end_nodes_values = [2, 7, 8, 3, 4, 5, 8, 9, 6, 9, 10]
durations_values = [1, 14, 1, 2, 3, 1, 8, 2, 10, 1, 2]

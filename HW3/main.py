def critical_path(amount, start_params, end_params, duration):
    earliest_start_times = [0] * amount
    latest_start_times = [float('inf')] * amount

    for k in range(amount):
        max_start = earliest_start_times[start_params[k]] + duration[k]
        if earliest_start_times[end_params[k]] < max_start:
            earliest_start_times[end_params[k]] = max_start

    latest_start_times[end_params[amount - 1]] = earliest_start_times[end_params[amount - 1]]

    for k in range(amount - 1, -1, -1):
        min_finish = latest_start_times[end_params[k]] - duration[k]
        if latest_start_times[start_params[k]] > min_finish:
            latest_start_times[start_params[k]] = min_finish

    earliest_start = [earliest_start_times[start_params[k]] for k in range(amount)]
    earliest_finish = [earliest_start[k] + duration[k] for k in range(amount)]
    latest_finish = [latest_start_times[end_params[k]] for k in range(amount)]
    latest_start = [latest_finish[k] - duration[k] for k in range(amount)]
    total_float = [latest_finish[k] - earliest_finish[k] for k in range(amount)]
    free_float = [earliest_start_times[end_params[k]] - earliest_finish[k] for k in range(amount)]
    critical_path_tasks = [1] + [end_params[k] for k in range(amount) if total_float[k] == 0]

    return earliest_start, latest_start, earliest_finish, latest_finish, total_float, free_float, critical_path_tasks


# def input_tasks_data(amount):
#     works = []
#     start_work = []
#     end_work = []
#     durations = []
#
#     for i in range(amount):
#         work_name = input(f"Enter name for task {i + 1}: ")
#         start_time = int(input(f"Enter start time for task {work_name}: "))
#         end_time = int(input(f"Enter end time for task {work_name}: "))
#         duration_time = int(input(f"Enter duration for task {work_name}: "))
#
#         works.append(work_name)
#         start_work.append(start_time)
#         end_work.append(end_time)
#         durations.append(duration_time)
#
#     return works, start_work, end_work, durations
#
# n = int(input("Enter the number of tasks: "))
# result = critical_path(*input_tasks_data(n))

n = 11

works = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]

start_work = [1, 1, 3, 3, 1, 2, 4, 5, 5, 6, 7]
end_work = [2, 3, 5, 4, 4, 5, 5, 6, 7, 8, 8]
durations = [3, 5, 3, 4, 3, 1, 4, 3, 3, 2, 5]

# start_work = [1, 2, 7, 1, 3, 4, 5, 8, 1, 6, 9]
# end_work = [2, 7, 8, 3, 4, 5, 8, 9, 6, 9, 10]
# durations = [1, 14, 1, 2, 3, 1, 8, 2, 10, 1, 2]

result = critical_path(n, start_work, end_work, durations)

print("\n  JOB\t\tDUR\t\tРН\t\tПН\t\tРО\t\tПО\t\tПР\t\tСР")
for i in range(n):
    print(f"{works[i]}\t{start_work[i]}-{end_work[i]}\t\t{durations[i]}\t\t{result[0][i]}\t\t{result[1][i]}\t\t{result[2][i]}\t\t{result[3][i]}\t\t{result[4][i]}\t\t{result[5][i]}")

print("\nКритический путь:")
for j in range(len(result[6])-1):  # Iterate up to the second-to-last element
    for i in range(n):
        if result[6][j] == start_work[i] and result[6][j + 1] == end_work[i]:
            print(f"{works[i]}\t", end='')

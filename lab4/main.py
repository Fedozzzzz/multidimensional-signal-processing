import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

N = 8


def compute_bpa(line):
    line = np.array([line[0], line[4], line[2], line[6], line[1], line[5], line[3], line[7]])
    result = np.zeros(N)

    for i in range(4):
        result[i] = line[i] + line[i + 4]
    for i in range(4, 8):
        result[i] = -line[i] + line[i - 4]

    line = result.copy()

    for i in range(2):
        result[i] = line[i] + line[i + 2]
    for i in range(2, 4):
        result[i] = -line[i] + line[i - 2]
    for i in range(4, 6):
        result[i] = line[i] - line[i + 2]
    for i in range(6, 8):
        result[i] = line[i] + line[i - 2]

    line = result.copy()

    result[0] = line[0] + line[1]
    result[1] = line[0] - line[1]
    result[2] = line[2] - line[3]
    result[3] = line[2] + line[3]
    result[4] = line[4] + line[5]
    result[5] = line[4] - line[5]
    result[6] = line[6] - line[7]
    result[7] = line[6] + line[7]

    return result


def compute_reverse_bpa_2n(matrix):
    matrix = np.array(matrix.copy())

    for i in range(len(matrix)):
        matrix[i] = compute_reverse_bpa(matrix[i])

    for i in range(len(matrix[0])):
        result = compute_reverse_bpa(
            [matrix[0][i], matrix[1][i], matrix[2][i], matrix[3][i], matrix[4][i], matrix[5][i], matrix[6][i],
             matrix[7][i]])
        for j in range(len(matrix)):
            matrix[j][i] = result[j]

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = matrix[i][j]

    return matrix


def compute_bpa_2n(matrix):
    matrix = np.array(matrix.copy())
    for i in range(len(matrix)):
        matrix[i] = compute_bpa(matrix[i])

    for i in range(len(matrix[0])):
        result = compute_bpa(
            [matrix[0][i], matrix[1][i], matrix[2][i], matrix[3][i], matrix[4][i], matrix[5][i], matrix[6][i],
             matrix[7][i]])
        for j in range(len(matrix)):
            matrix[j][i] = result[j]

    return matrix


def compute_reverse_bpa(line):
    line = np.array(line.copy())
    result = np.zeros(N)

    for i in range(4):
        result[i] = line[i] + line[i + 4]
    for i in range(4, 8):
        result[i] = -line[i] + line[i - 4]

    line = result.copy()

    for i in range(2):
        result[i] = line[i] + line[i + 2]
    for i in range(2, 4):
        result[i] = -line[i] + line[i - 2]
    for i in range(4, 6):
        result[i] = line[i] - line[i + 2]
    for i in range(6, 8):
        result[i] = line[i] + line[i - 2]

    line = result.copy()

    result[0] = line[0] + line[1]
    result[1] = line[0] - line[1]
    result[2] = line[2] - line[3]
    result[3] = line[2] + line[3]
    result[4] = line[4] + line[5]
    result[5] = line[4] - line[5]
    result[6] = line[6] - line[7]
    result[7] = line[6] + line[7]

    return [result[0], result[7], result[4], result[3], result[2], result[5], result[6], result[1]]


# def compute_filter(signal, noisy_signal):
#     lu_decomposition = lu(noisy_signal)
#     result = np.linalg.solve(lu_decomposition, signal)
#     return result
#

def compute_filter(signal, noisy_signal):
    result = np.linalg.solve(noisy_signal, signal)
    return result


def compute_normalized_matrix(matrix):
    matrix = np.array(matrix.copy())
    matrix /= N * N
    return matrix


def compute_multiplication(lval, rval):
    result = np.dot(lval, rval)
    return result


def print_matrix(matrix, places):
    for i in range(N):
        for j in range(N):
            print(f"{matrix[i][j]:.{places}f}", end="")
            if j != N - 1:
                print("\t", end="")
        print("\n")
    print("\n")


def print_debug_matrix(matrix, places):
    print("<<DEBUG>>")
    for i in range(N):
        for j in range(N):
            print(f"{matrix[i][j]:.{places}f}", end="")
            if j != N - 1:
                print("\n", end="")
        print("\n")
    print("<<DEBUG>>")


def read_matrix(path_to_file):
    with open(path_to_file, 'r') as file:
        return np.reshape(np.array([line.split() for line in file]).astype(float),
                          (N, N),
                          order='C')


def plot_matrix(matrix, name):
    plot.figure()
    plot.imshow(matrix, cmap='coolwarm')
    plot.colorbar()
    plot.title(name)
    plot.show()


# 3D график
def plot_matrix_3d(matrix, name):
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(len(matrix[0]))
    Y = np.arange(len(matrix))
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, np.array(matrix), cmap='coolwarm')
    ax.set_title(name)
    fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10, orientation='vertical')
    plot.show()


def save_as_excell_table(data, file_name="output"):
    df = pd.DataFrame(data)
    df.to_excel("{}.xlsx".format(file_name), index=False, header=False)


def main():
    signal = read_matrix('source_matrix.txt').tolist()
    noisy_signal_1 = read_matrix('noise_matrix_1.txt').tolist()
    noisy_signal_2 = read_matrix('noise_matrix_2.txt').tolist()

    print("Исходный сигнал:")
    print_matrix(signal, 0)
    # print_debug_matrix(signal, 0)
    plot_matrix(signal, "Исходный сигнал")
    plot_matrix_3d(signal, "Исходный сигнал 3д")
    save_as_excell_table(signal, "signal")

    print("Первый сигнал с единичной помехой:")
    print_matrix(noisy_signal_1, 0)
    # print_debug_matrix(noisy_signal_1, 0)
    plot_matrix(noisy_signal_1, "Первый сигнал с единичной помехой")
    plot_matrix_3d(noisy_signal_1, "Первый сигнал с единичной помехой 3д")
    save_as_excell_table(noisy_signal_1, "noisy_signal_1")

    print("Второй сигнал с единичной помехой (в другом месте):")
    print_matrix(noisy_signal_2, 0)
    # print_debug_matrix(noisy_signal_2, 0)
    plot_matrix(noisy_signal_2, "Второй сигнал с единичной помехой (в другом месте)")
    plot_matrix_3d(noisy_signal_2, "Второй сигнал с единичной помехой (в другом месте) 3д")
    save_as_excell_table(noisy_signal_2, "noisy_signal_2")

    specter_of_signal = compute_bpa_2n(signal)
    specter_of_signal = compute_normalized_matrix(specter_of_signal)

    print("Спектр сигнала без помех:")
    print_matrix(specter_of_signal, 3)
    # print_debug_matrix(specter_of_signal, 3)
    plot_matrix(specter_of_signal, "Спектр сигнала без помех")
    plot_matrix_3d(specter_of_signal, "Спектр сигнала без помех")
    save_as_excell_table(specter_of_signal, "specter_of_signal")

    specter_of_noisy_signal_1 = compute_bpa_2n(noisy_signal_1)
    specter_of_noisy_signal_1 = compute_normalized_matrix(specter_of_noisy_signal_1)

    print("Спектр первого сигнала с единичной помехой:")
    print_matrix(specter_of_noisy_signal_1, 3)
    # print_debug_matrix(specter_of_noisy_signal_1, 3)
    plot_matrix(specter_of_noisy_signal_1, "Спектр первого сигнала с единичной помехой")
    plot_matrix_3d(specter_of_noisy_signal_1, "Спектр первого сигнала с единичной помехой 3д")
    save_as_excell_table(specter_of_noisy_signal_1, "specter_of_noisy_signal_1")

    filter_matrix = compute_filter(specter_of_signal, specter_of_noisy_signal_1)

    print("Матрица фильтра:")
    print_matrix(filter_matrix, 3)
    # print_debug_matrix(filter_matrix, 3)

    plot_matrix(filter_matrix, "Матрица фильтра")
    plot_matrix_3d(filter_matrix, "Матрица фильтра")
    save_as_excell_table(filter_matrix, "filter_matrix")

    print("<<<<<Apply filter>>>>>")

    specter_of_noisy_signal_2 = compute_bpa_2n(noisy_signal_2)
    specter_of_noisy_signal_2 = compute_normalized_matrix(specter_of_noisy_signal_2)

    print("Спектр второго сигнала с единичной помехой:")
    print_matrix(specter_of_noisy_signal_2, 3)
    # print_debug_matrix(specter_of_noisy_signal_2, 3)
    plot_matrix(specter_of_noisy_signal_2, "Спектр второго сигнала с единичной помехой")
    plot_matrix_3d(specter_of_noisy_signal_2, "Спектр второго сигнала с единичной помехой")
    save_as_excell_table(specter_of_noisy_signal_2, "specter_of_noisy_signal_2")

    filtered_noisy_signal_1 = compute_multiplication(specter_of_noisy_signal_1, filter_matrix)
    filtered_noisy_signal_1 = compute_reverse_bpa_2n(filtered_noisy_signal_1)

    print("Первый сигнал с единичной помехой после применения фильтра Адамара:")
    print_matrix(filtered_noisy_signal_1, 1)
    # print_debug_matrix(filtered_noisy_signal_1, 1)
    plot_matrix(filtered_noisy_signal_1, "Первый сигнал с единичной помехой после применения фильтра Адамара")
    plot_matrix_3d(filtered_noisy_signal_1, "Первый сигнал с единичной помехой после применения фильтра Адамара")
    save_as_excell_table(filtered_noisy_signal_1, "filtered_noisy_signal_1")

    filtered_noisy_signal_2 = compute_multiplication(specter_of_noisy_signal_2, filter_matrix)
    filtered_noisy_signal_2 = compute_reverse_bpa_2n(filtered_noisy_signal_2)

    print("Второй сигнал с единичной помехой после применения фильтра Адамара:")
    print_matrix(filtered_noisy_signal_2, 1)

    plot_matrix(filtered_noisy_signal_2, "Второй сигнал с единичной помехой после применения фильтра Адамара:")
    plot_matrix_3d(filtered_noisy_signal_2, "Второй сигнал с единичной помехой после применения фильтра Адамара:")
    save_as_excell_table(filtered_noisy_signal_2, "filtered_noisy_signal_2")


if __name__ == "__main__":
    main()

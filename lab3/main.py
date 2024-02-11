import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from math import sqrt


def read_matrix(path_to_file):
    with open(path_to_file, 'r') as file:
        return np.reshape(np.array([line.split() for line in file]).astype(int),
                          (15, 15),
                          order='C')


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def print_data(signal, signal_noised, delta, title="title"):
    print("------------{}------------------".format(title))
    print("Signal-noise:")
    print_matrix(signal_noised)
    print("///////////////////////////////////")
    print("Signal:")
    print_matrix(signal)
    print("///////////////////////////////////")
    print("Delta:")
    print("///////////////////////////////////")
    print_matrix(delta)
    print("---------------------------------------------")


def format_matrix_to_string(M, name=None):
    s = ""
    if name is not None:
        nameStr = str(name)  # ' ' +str(name)+' '
        filler = ' \t'
        while len(s) < len(M) * len(filler):
            s += filler
        half = int((len(s) - len(nameStr)) / 2)
        s = s[:half] + nameStr + s[half + len(nameStr) - 1:]
        s += '\n'
    for i in range(len(M)):
        V = M[i]
        for j in range(len(V)):
            s += str(V[j]) + '\t'
            s = s[:-1] + '\n'
    return s


def get_matrix_surface(M):
    X, Y = np.meshgrid(np.arange(len(M[0])), np.arange(len(M)))
    return X, Y, np.array(M)


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


def apply_anisotropic_filter(signal, signal_noise, k, window_size, anisotropic_filter, force_int=True):
    row = len(signal)
    col = len(signal[0])
    amount = 1.0 / k
    delta = []
    signal_result = []

    if force_int:
        amount = int(amount)

    for i in range(row):
        d = []
        s = []
        for j in range(col):
            d.append(0)
            s.append(0)
        delta.append(d)
        signal_result.append(s)

    for i in range(row):
        for j in range(col):
            if i == 0 or i == row - 1 or j == 0 or j == col - 1:
                signal_result[i][j] = signal_noise[i][j]
            else:
                sg = 0
                for k1 in range(window_size):
                    for k2 in range(window_size):
                        sg += anisotropic_filter[k1][k2] * signal_noise[i - 1 + k1][j - 1 + k2]
                signal_result[i][j] = sg / amount

    if force_int:
        for i in range(row):
            for j in range(col):
                signal_result[i][j] = int(signal_result[i][j])
                signal_noise[i][j] = int(signal_noise[i][j])
                delta[i][j] = int(delta[i][j])

    for i in range(row):
        for j in range(col):
            delta[i][j] = signal_noise[i][j] - signal_result[i][j]

    return signal_result, signal_noise, delta


def apply_statistic_filter(signal, signal_noise, window_size, m, force_int=True):
    row = len(signal)
    col = len(signal[0])
    delta = []
    signal_result = []
    for i in range(row):
        d = []
        s = []
        for j in range(col):
            d.append(0)
            s.append(0)
        delta.append(d)
        signal_result.append(s)

    for i in range(row):
        for j in range(col):
            sum1 = 0
            sum2 = 0.0
            for k1 in range(window_size):
                for k2 in range(window_size):
                    try:
                        sum1 += signal_noise[i - 1 + k1][j - 1 + k2]
                    except:
                        pass

            G = (1.0 * sum1) / pow(window_size, 2)
            for k1 in range(window_size):
                for k2 in range(window_size):
                    try:
                        sum2 += pow(signal_noise[i - 1 + k1][j - 1 + k2] - G, 2)
                    except:
                        pass
            D = sum2 / (pow(window_size, 2) - 1)
            nu = m * sqrt(D)
            if (signal_noise[i][j] - G) < nu:
                signal_result[i][j] = signal_noise[i][j]
            else:
                signal_result[i][j] = G

    for i in range(row):
        for j in range(col):
            delta[i][j] = signal_noise[i][j] - signal_result[i][j]
            if force_int:
                signal_result[i][j] = int(signal_result[i][j])
                signal_noise[i][j] = int(signal_noise[i][j])
                delta[i][j] = int(delta[i][j])

    return signal_result, signal_noise, delta


def save_as_excell_table(data, file_name="output"):
    df = pd.DataFrame(data)
    df.to_excel("{}.xlsx".format(file_name), index=False, header=False)


def main():
    # Статистический фильтр
    a = 1.26
    window_size = 3

    # Анизотропный фильтр
    anisotropic_filter = [[1, 2, 1],
                          [2, 4, 3],
                          [1, 2, 2]]
    k = 1.0 / 18.0

    source_matrix = read_matrix('source_matrix.txt').tolist()
    noise_matrix = read_matrix('noise_matrix.txt').tolist()

    (signal, signal_noise, delta) = apply_anisotropic_filter(source_matrix, noise_matrix, k, window_size,
                                                             anisotropic_filter)

    save_as_excell_table(signal, 'signal_anisotropic')
    save_as_excell_table(delta, 'delta_anisotropic')

    print_data(signal, signal_noise, delta, title='Anisotropic filter')

    plot_matrix_3d(signal_noise, "Зашумленный сигнал 3д")
    plot_matrix(signal_noise, "Зашумленный сигнал")
    plot_matrix_3d(signal, "Анизотропная фильтрация сигнал 3д")
    plot_matrix(signal, "Анизотропная фильтрация сигнал")
    plot_matrix_3d(delta, "Анизотропная фильтрация разница 3д")
    plot_matrix(delta, "Анизотропная фильтрация разница")

    (signal, signal_noise, delta) = apply_statistic_filter(source_matrix, noise_matrix, window_size, a)

    save_as_excell_table(signal, 'signal_statistic')
    save_as_excell_table(delta, 'delta_statistic')

    print_data(signal, signal_noise, delta, title='Statistic filter')

    plot_matrix_3d(signal, "Статистическая фильтрация сигнал 3д")
    plot_matrix(signal, "Статистическая фильтрация сигнал")
    plot_matrix_3d(delta, "Статистическая фильтрация разница 3д")
    plot_matrix(delta, "Статистическая фильтрация разница")


if __name__ == '__main__':
    main()

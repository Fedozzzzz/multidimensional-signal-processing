import matplotlib.pyplot as plot
import numpy as np
from math import sqrt

from matplotlib import cbook, cm
from matplotlib.colors import LightSource

# Статистический фильтр
a = 1.26
# Анизотропный фильтр (k x [filter])
anisotropic_filter = [[1, 2, 1],
                      [2, 4, 3],
                      [1, 2, 2]]
k = 1.0 / 18.0


def read_matrix(path_to_file):
    with open(path_to_file, 'r') as file:
        return np.reshape(np.array([line.split() for line in file]).astype(int),
                          (15, 15),
                          order='C')


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


def plot_matrix(M, name='unknown matrix'):
    fig = plot.figure(figsize=(6, 3.2))
    # fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.set_title(str(name))
    plot.imshow(M)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    # fig.canvas.set_window_title(name)
    plot.colorbar(orientation='vertical')
    plot.show()

    # plot.imshow(M, cmap='hot', interpolation='nearest')
    # plot.show()


def get_matrix_surface(M):
    X = np.arange(0, len(M[0]))
    Y = np.arange(0, len(M))
    X, Y = np.meshgrid(X, Y)
    return X, Y, M


def plot_matrix_3D(M, name='unknown matrix'):
    # fig = plot.figure(figsize=(6, 3.2))
    X, Y, Z = get_matrix_surface(M)
    # Z = np.array(Z)
    # ax = fig.add_subplot(projection='3d')
    # ax.set_title(str(name))
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Spectral_r', linewidth=0,
    #                        antialiased=True)
    # # fig.canvas.set_window_title(name)
    # fig.set_t
    # fig.show()

    # Set up plot
    fig, ax = plot.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(np.array(Z), cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(np.array(X), np.array(Y), np.array(Z), rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)

    plot.show()


def apply_anisotropic_filter(signal, signal_noise, k, anisotropic_filter, force_int=True):
    ROW = len(signal)
    COL = len(signal[0])
    amount = 1.0 / k
    WINDOW_SIZE = len(anisotropic_filter)
    delta = []
    signal = []

    if force_int:
        amount = int(amount)

    for i in range(ROW):
        d = []
        s = []
        for j in range(COL):
            d.append(0)
            s.append(0)
            delta.append(d)
            signal.append(s)

    for i in range(ROW):
        for j in range(COL):
            if i == 0 or i == ROW - 1 or j == 0 or j == COL - 1:
                signal[i][j] = signal_noise[i][j]
            else:
                sg = 0
                for k1 in range(WINDOW_SIZE):
                    for k2 in range(WINDOW_SIZE):
                        sg += anisotropic_filter[k1][k2] * signal_noise[i - 1 + k1][j - 1 + k2]
                signal[i][j] = sg / amount

    if force_int:
        for i in range(ROW):
            for j in range(COL):
                signal[i][j] = int(signal[i][j])
                signal_noise[i][j] = int(signal_noise[i][j])
                delta[i][j] = int(delta[i][j])

    for i in range(ROW):
        for j in range(COL):
            delta[i][j] = signal_noise[i][j] - signal[i][j]

    # print("Anisotropic filter:")
    # print(format_matrix_to_string(signal_noise, "Signal-noise"))
    # print(format_matrix_to_string(signal, "Signal"))
    # print(format_matrix_to_string(delta, "Delta (Signal-noise - Signal)"))
    # print("\n=======================================================\n")

    return signal, signal_noise, delta


def apply_statistic_filter(signal, signal_noise, window_size, m, force_int=True):
    ROW = len(signal)
    COL = len(signal[0])
    WINDOW_SIZE = window_size
    delta = []
    signal = []
    for i in range(ROW):
        d = []
        s = []
        for j in range(COL):
            d.append(0)
            s.append(0)
        delta.append(d)
        signal.append(s)

    for i in range(ROW):
        for j in range(COL):
            sum1 = 0
            sum2 = 0.0
            for k1 in range(WINDOW_SIZE):
                for k2 in range(WINDOW_SIZE):
                    try:
                        sum1 += signal_noise[i - 1 + k1][j - 1 + k2]
                    except: pass

            G = (1.0 * sum1) / pow(WINDOW_SIZE, 2)
            for k1 in range(WINDOW_SIZE):
                for k2 in range(WINDOW_SIZE):
                    try:
                        sum2 += pow(signal_noise[i - 1 + k1][j - 1 + k2] - G, 2)
                    except: pass
            D = sum2 / (pow(WINDOW_SIZE, 2) - 1)
            nu = m * sqrt(D)
            if (signal_noise[i][j] - G) < nu:
                signal[i][j] = signal_noise[i][j]
            else:
                signal[i][j] = G

    for i in range(ROW):
        for j in range(COL):
            delta[i][j] = signal_noise[i][j] - signal[i][j]
            if force_int:
                signal[i][j] = int(signal[i][j])
                signal_noise[i][j] = int(signal_noise[i][j])
                delta[i][j] = int(delta[i][j])
    # print("Statistic filter:")
    # print(format_matrix_to_string(signal_noise, "Signal-noise"))
    # print(format_matrix_to_string(signal, "Signal"))
    # print(format_matrix_to_string(delta, "Delta (Signal-noise - Signal)"))
    return signal, signal_noise, delta

source_matrix = read_matrix('source_matrix.txt').tolist()
noise_matrix = read_matrix('noise_matrix.txt').tolist()

print(source_matrix)

(signal, signal_noise, delta) = apply_anisotropic_filter(source_matrix, noise_matrix, k, anisotropic_filter)
print(signal)
print(delta)

plot_matrix_3D(delta, "Анизотропная фильтрация 3д")
plot_matrix(delta, "Анизотропная фильтрация")
plot_matrix_3D(signal_noise, "Зашумленный сигнал 3д")
plot_matrix(signal_noise, "Зашумленный сигнал")
plot_matrix_3D(signal, "Анизотропная фильтрация сигнал 3д")
plot_matrix(signal, "Анизотропная фильтрация сигнал")

(signal, signal_noise, delta) = apply_statistic_filter(source_matrix, noise_matrix, 3, a)
plot_matrix_3D(delta, "Статистическая фильтрация разница 3D")
plot_matrix(delta, "Статистическая фильтрация разница")
plot_matrix_3D(signal, "Статистическая фильтрация сигнал 3д")
plot_matrix(signal, "Статистическая фильтрация сигнал")
plot.show()

import numpy as np
from PIL import Image
import os
import csv
import multiprocessing as mp
import time
from matplotlib import pyplot as plt
from sklearn import datasets


def plot_graph_time(cores, times, proc_count):
    ideal_time = np.array([times[0] / x for x in range(1, proc_count + 1)])
    #plt.plot(cores, ideal_time, linestyle='-', marker='o', label='Ideal')
    plt.plot(cores, times, linestyle='-', marker='o', label='fidelity')
    plt.xlabel('threads, n')
    plt.ylabel('fidelity')
    plt.grid(visible=True)
    plt.legend()
    plt.savefig(f'./graph_{time.time()}.png')
    plt.show()


def get_file_names(directory):
    files = os.listdir(directory)
    return files


def get_random_weights(shape):
    w = np.random.random(shape) - 0.5
    return w


def get_weights(fname, shape):
    try:
        with open(fname, 'r', newline='') as file:
            reader = csv.reader(file, delimiter=';')
            w = []
            for row in reader:
                w.append(row)
            w = np.array(w, dtype=np.float64)
    except:
        with open(fname, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            w = get_random_weights(shape)
            for row in w:
                writer.writerow(row)
    return w


def write_weights_to_csv(fname, arr_min_weights):
    with open(fname, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for w in arr_min_weights:
            writer.writerow(w)

def get_image():  # возвращает матрицу 0 и 1
    digits = datasets.load_digits()
    return digits


def back_propagation(w, w_1, x, x_1_in, x_1_out, x_2_in, x_2_out, x_id):
    nu = 0.005

    de_dx_o2 = 2*(x_2_out - x_id)
    dx_o2_dx_i2 = np.exp(-x_2_in) / (1 + np.exp(-x_2_in))**2
    dx_i2_dw2 = x_1_out.reshape([x_1_out.shape[0], 1])
    de_dw2 = np.multiply(dx_i2_dw2, de_dx_o2 * dx_o2_dx_i2)  # градиент для весов: скрытый слой - выходной
    #print(de_dw2.shape)

    dx_i2_dx_o1 = w_1
    de_dx_o1 = np.sum(np.multiply(dx_i2_dx_o1, de_dx_o2 * dx_o2_dx_i2), axis=1)
    dx_o1_dx_i1 = np.exp(-x_1_in) / np.power((1 + np.exp(-x_1_in)), 2)
    dx_i1_dw1 = x.reshape([x.shape[0], 1])
    de_dw1 = np.multiply(dx_i1_dw1, de_dx_o1 * dx_o1_dx_i1)

    w_new = w.copy() - de_dw1 * nu
    w_1_new = w_1.copy() - de_dw2 * nu

    #print(de_dw1.shape, de_dw2.shape)
    return w_new, w_1_new



def main(id, p, digits):
    count_images = len(digits.images) - 200
    queue = np.arange(count_images)
    answers = []
    w = get_weights('weights.csv', [64, 16])
    w_1 = get_weights('weights_1.csv', [16, 10])
    for k in range(25):
        np.random.shuffle(queue)
        for j in range(count_images):
            i = queue[j]
            image = digits.images[i]
            name = int(digits.target[i])
            x = np.asarray(image).reshape(image.shape[0] * image.shape[1])
            x_out = 1 / (1 + np.exp(-x))

            x_1_in = np.matmul(x_out, w)
            x_1_out = 1/(1+np.exp(-x_1_in))

            x_2_in = np.matmul(x_1_out, w_1)
            x_2_out = 1/(1 + np.exp(-x_2_in))

            b = np.argsort(x_2_out)[-1]  # индекс нейрона с наибольшим выходным сигналом (ответ сети)
            answer = 1 if b == name else 0
            answers.append(answer)
            x_id_out = np.zeros(10)
            x_id_out[name] = 1

            new_weights, new_weights_1 = back_propagation(w, w_1, x, x_1_in, x_1_out, x_2_in, x_2_out, x_id_out)
            w = new_weights.copy()
            w_1 = new_weights_1.copy()
        fidelity = np.sum(answers) / len(answers)
        #print(f'fidelity on training = {fidelity}')
    return w, w_1, fidelity


def testing(s_w, s_w_1, digits):
    answers = []
    w = s_w.copy()
    w_1 = s_w_1.copy()
    for i in range(len(digits.images) - 200, len(digits.images)):
        image = digits.images[i]
        name = int(digits.target[i])
        x = np.asarray(image).reshape(image.shape[0] * image.shape[1])
        x_out = 1/(1+np.exp(-x))

        x_1_in = np.matmul(x_out, w)
        x_1_out = 1/(1+np.exp(-x_1_in))

        x_2_in = np.matmul(x_1_out, w_1)
        x_2_out = 1/(1 + np.exp(-x_2_in))

        b = np.argsort(x_2_out)[-1]  # индекс нейрона с наибольшим выходным сигналом (ответ сети)
        answer = 1 if b == name else 0
        answers.append(answer)
        x_id_out = np.zeros(10)
        x_id_out[name] = 1
    fidelity = np.sum(answers) / len(answers)
    #print(f'fidelity on testing = {fidelity}')
    return fidelity


if __name__ == '__main__':
    threads = 8
    id_dict = {'а': 0, 'б': 1, 'в': 2, 'г': 3, 'е': 4}
    folder_data = '.\data'
    file_names = get_file_names(folder_data)

    cores = np.arange(1, threads + 1, 1)
    end = 10
    times_arr = np.empty(0)
    imges = get_image()
    queue = np.arange(len(imges.images) - 200)
    arr_fid_per_cores = []
    for core in cores:

        args = [(x, core, imges) for x in range(core)]
        res = mp.Pool().starmap(main, args)

        new_weights = []
        new_weights_1 = []
        arr_fid = []
        for res_per_thread in res:
            new_weights.append(res_per_thread[0])
            new_weights_1.append(res_per_thread[1])
            arr_fid.append(res_per_thread[2])

        best_fid_index = np.argsort(arr_fid)[-1]

        w = new_weights[best_fid_index].copy()
        w_1 = new_weights_1[best_fid_index].copy()
        best_fid = arr_fid[best_fid_index]
        print(f'best fidelity on training data: {best_fid}')

        fid_test = testing(w, w_1, imges)
        arr_fid_per_cores.append(fid_test)
        print(core, fid_test)
    #print(times_arr)
    plot_graph_time(cores, arr_fid_per_cores, threads)

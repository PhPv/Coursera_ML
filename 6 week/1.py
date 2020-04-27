from skimage.io import imread
import skimage
import numpy as np
from sklearn.cluster import KMeans

import pandas as pd
from scipy import log10
import sklearn


import matplotlib.pyplot as plt
from matplotlib import pyplot
import pylab


image = imread('parrots.jpg')

R,G,B = [], [], []
i = []


# 473*713
# загружаем картинку
image_float = skimage.img_as_float(image)

# преобразуем данные
R = skimage.img_as_float(image)[:, :, 0].ravel()
G = skimage.img_as_float(image)[:, :, 1].ravel()
B = skimage.img_as_float(image)[:, :, 2].ravel()
RGB = np.column_stack((R, G, B))

for n_clusters in range(1, 21):
    #обучаем классификатор
    clf = KMeans(random_state=241, init='k-means++', n_clusters=n_clusters).fit(RGB)

    #вытаскиваем центра кластеров и номера класстеров для каждого пикселя
    cl_cen = clf.cluster_centers_
    lab = clf.labels_

    #создаем матрицу для каждого пикселя с номером кластера
    RGB_cl = np.reshape(lab, (-1, 713))

    #создаем матрицу со средними цветами
    RGB_av = np.copy(image_float)
    for i in range(len(cl_cen)):
        RGB_av[RGB_cl == i] = cl_cen[i]

    #ищем медиану для центров
    RGB_med = np.copy(image_float)
    for i in range(len(cl_cen)):
        med_r = np.median(RGB_med[:, :, 0][RGB_cl == i])
        med_g = np.median(RGB_med[:, :, 1][RGB_cl == i])
        med_b = np.median(RGB_med[:, :, 2][RGB_cl == i])
        RGB_med[RGB_cl == i] = [med_r, med_g, med_b]


    def PSNR(img1, img2):
        MSE = np.mean((img1 - img2) ** 2)
        PSNR = 10 * log10(np.max(img1) / MSE)
        return PSNR

    print(n_clusters, 'average: ' + str(PSNR(image_float, RGB_av)), 'median ' + str(PSNR(image_float, RGB_med)))


    plt.imshow(RGB_av)
    plt.show()
    plt.imshow(RGB_med)
    plt.show()
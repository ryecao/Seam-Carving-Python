# Author: ryecao
# ryecao@gmail.com

import cv2
import numpy as np
import operator
from multiprocessing import Pool
from matplotlib import pyplot as plt
import profile
import copy

def find_a_seam(m_energy):
    row_num, col_num = m_energy.shape
    print row_num,"px X",col_num,"px"
    m_energy = np.absolute(m_energy).astype(int)

    cost = np.zeros((row_num, col_num)).astype(int)

    for i in xrange(0, row_num):
        for j in xrange(0, col_num):
            if i == 0:
                cost[i][j] = m_energy[i][j]
            else:
                if j and j < (col_num - 1):
                    cost[i][j] = min(
                        cost[i - 1][j - 1], cost[i - 1][j], cost[i - 1][j + 1]) + m_energy[i][j]
                elif j == (col_num - 1):
                    cost[i][j] = min(
                        cost[i - 1][j - 1], cost[i - 1][j]) + m_energy[i][j]
                elif j == 0:
                    cost[i][j] = min(
                        cost[i - 1][j], cost[i - 1][j + 1]) + m_energy[i][j]
                else:
                    cost[i][j] = cost[i - 1][j] + m_energy[i][j]

    cor = np.zeros(row_num).astype(int)

    for i, e in reversed(list(enumerate(cost))):
        if i == (row_num - 1):
            cor_cur = cost[i].tolist().index(np.min(cost[i]))
        else:
            if cor_cur and cor_cur < (col_num - 1):
                if (cost[i][cor_cur - 1] + m_energy[i + 1][cor_cur]) == cost[i + 1][cor_cur]:
                    cor_cur = cor_cur - 1
                elif (cost[i][cor_cur + 1] + m_energy[i + 1][cor_cur]) == cost[i + 1][cor_cur]:
                    cor_cur = cor_cur + 1
            elif cor_cur == 0:
                if (cost[i][cor_cur + 1] + m_energy[i + 1][cor_cur]) == cost[i + 1][cor_cur]:
                    cor_cur = cor_cur + 1
            elif cor_cur == col_num - 1:
                if (cost[i][cor_cur - 1] + m_energy[i + 1][cor_cur]) == cost[i + 1][cor_cur]:
                    cor_cur = cor_cur - 1
        cor[i] = cor_cur
    return cor


def sobel_rgb(img, orient):
    img = img.astype(np.uint8)
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

    if orient:
        # sobel_y
        sobel = np.add(np.add(cv2.Sobel(img_r, cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_REPLICATE), cv2.Sobel(
            img_g, cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_REPLICATE)), cv2.Sobel(img_b, cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_REPLICATE))
        sobel = np.transpose(sobel)
    else:
        # sobel_x
        sobel = np.add(np.add(cv2.Sobel(img_r, cv2.CV_64F, 1, 0, ksize=5, borderType=cv2.BORDER_REPLICATE), cv2.Sobel(
            img_g, cv2.CV_64F, 1, 0, ksize=5, borderType=cv2.BORDER_REPLICATE)), cv2.Sobel(img_b, cv2.CV_64F, 1, 0, ksize=5, borderType=cv2.BORDER_REPLICATE))
    return sobel


def resize(img, amount, orient, mode):
    # amount: how many pixels you want to change.
    # orient: 0 for horizontal, sobel_x ; 1 for vertical r, sobel_y_t.
    # mode: 0 for shrink, 1 for englarge
    m_rgb = np.transpose(img) if orient else img
    row_num, col_num, rgb = m_rgb.shape
    m_res = m_rgb.tolist()
    for i in xrange(0, amount):
        sobel = sobel_rgb(np.array(m_res), orient)
        seam = find_a_seam(sobel)
        if mode:
            for row in xrange(0, row_num):
                interpolation_pixel = [(m_res[row][seams[i][row] - 1][0] + m_res[row][seams[i][row] + 1][0]) / 2, (m_res[row][seams[i][row] - 1][
                    1] + m_res[row][seams[i][row] + 1][1]) / 2, (m_res[row][seams[i][row] - 1][2] + m_res[row][seams[i][row] + 1][2]) / 2]
                m_res[row].insert(seams[i][row], interpolation_pixel)
        else:
            for row in xrange(0, row_num):
                m_res[row].pop(seam[row])

    m_res = np.array(m_res)
    if orient:
        m_res = np.transpose(m_res)
    return m_res

img = cv2.imread('sample.jpg')  # input image
img2 = resize(img, 30, 0, 0) # 30 pixels
cv2.imwrite('output.jpg', img2)  # output image

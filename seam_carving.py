import cv2
import numpy as np
import operator
from multiprocessing import Pool
from matplotlib import pyplot as plt


def find_all_seams(m_energy):
    row_num, col_num = m_energy.shape
    cor = {}
    energy = {}
    for nth in xrange(0, col_num):
        cor_cur = nth
        energy_of_seam = abs(int(m_energy[0][cor_cur]))
        cor[nth] = [cor_cur]
        for i in xrange(1, row_num):
            eng = min(m_energy[i][cor_cur - 1] if cor_cur else 0, m_energy[i][
                      cor_cur], m_energy[i][cor_cur + 1] if cor_cur != (col_num - 1) else col_num)
            if cor_cur and eng == m_energy[i][cor_cur - 1]:
                cor_cur = cor_cur - 1
            elif cor_cur != (col_num - 1) and eng == m_energy[i][cor_cur + 1]:
                cor_cur = cor_cur + 1

            cor[nth].append(cor_cur)
            energy_of_seam += abs(int(eng))
        energy[nth] = energy_of_seam

    sorted_energy = sorted(energy.iteritems(), key=operator.itemgetter(1))
    res = []
    for item in sorted_energy:
        key_e, value_e = item
        res.append(cor[key_e])
    return res


def find_a_seam(m_energy):
    row_num, col_num = m_energy.shape
    cost = []
    max_cost = 0
    for i in xrange(0, row_num):
        cost_row = []
        for j in xrange(0, col_num):
            if i == 0:
                cost_row.append(abs(int(m_energy[i][j])))
            else:
                if j and j < (col_num - 1):
                    cost_row.append(
                        abs(int(min(cost[i - 1][j - 1], cost[i - 1][j], cost[i - 1][j + 1]))) + abs(int(m_energy[i][j])))
                elif j == (col_num - 1):
                    cost_row.append(
                        abs(int(min(cost[i - 1][j - 1], cost[i - 1][j]))) + abs(int(m_energy[i][j])))
                elif j == 0:
                    cost_row.append(
                        abs(int(min(cost[i - 1][j], cost[i - 1][j + 1]))) + abs(int(m_energy[i][j])))
                else:
                    cost_row.append(
                        abs(int(cost[i - 1][j])) + abs(int(m_energy[i][j])))
        # print cost_row
        cost.append(cost_row)

    # file1 = open("newfile.txt", "w")
    # for item in m_energy:
    #     file1.write("%s\n" % item)
    # file1.write(
    #     "=============================================================")
    # for item in cost:
    #     file1.write("%s\n" % item)
    # file1.close()
    # for i in xrange(0, row_num):
    #     for j in xrange(0, col_num):
    #         if cost[i][j] > max_cost:
    #             max_cost = cost[i][j]

    # energy_fig=[]
    # for i in xrange(0, row_num):
    #     energy_fig_row = []
    #     for j in xrange(0, col_num):
    #         energy_fig_row.append([int(float(cost[i][j])/max_cost*255),int(float(cost[i][j])/max_cost*255),int(float(cost[i][j])/max_cost*255)])
    #     energy_fig.append(energy_fig_row)

    # plt.subplot(211), plt.imshow(cost, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # cv2.imwrite('h8.jpg', np.array(energy_fig))
    cor = []
    for i, e in reversed(list(enumerate(cost))):
        if i == (row_num - 1):
            # print min(cost[i]), i, cost[i].index(min(cost[i]))
            cor_cur = cost[i].index(min(cost[i]))
        else:
            if cor_cur and cor_cur < (col_num - 1):
                if (cost[i][cor_cur - 1] + abs(int(m_energy[i + 1][cor_cur]))) == cost[i + 1][cor_cur]:
                    cor_cur = cor_cur - 1
                elif (cost[i][cor_cur + 1] + abs(int(m_energy[i + 1][cor_cur]))) == cost[i + 1][cor_cur]:
                    cor_cur = cor_cur + 1
            elif cor_cur == 0:
                if (cost[i][cor_cur + 1] + abs(int(m_energy[i + 1][cor_cur]))) == cost[i + 1][cor_cur]:
                    cor_cur = cor_cur + 1
            elif cor_cur == col_num - 1:
                if (cost[i][cor_cur - 1] + abs(int(m_energy[i + 1][cor_cur]))) == cost[i + 1][cor_cur]:
                    cor_cur = cor_cur - 1
        cor.append(cor_cur)
    cor.reverse()
    return cor


def sobel_rgb(img):
    row_num, col_num, rgb = img.shape
    img = img.tolist()
    img_r = np.zeros((row_num, col_num)).tolist()
    img_g = np.zeros((row_num, col_num)).tolist()
    img_b = np.zeros((row_num, col_num)).tolist()
    for i in xrange(0, row_num):
        for j in xrange(0, col_num):
            img_r[i][j] = np.uint8(img[i][j][0])
            img_g[i][j] = np.uint8(img[i][j][1])
            img_b[i][j] = np.uint8(img[i][j][2])

    img_r = np.array(img_r)
    img_g = np.array(img_g)
    img_b = np.array(img_b)
    sobelx = np.add(np.add(cv2.Sobel(img_r, cv2.CV_64F, 1, 0, ksize=5, borderType=cv2.BORDER_REPLICATE), cv2.Sobel(
        img_g, cv2.CV_64F, 1, 0, ksize=5, borderType=cv2.BORDER_REPLICATE)), cv2.Sobel(img_b, cv2.CV_64F, 1, 0, ksize=5, borderType=cv2.BORDER_REPLICATE))
    sobely = np.add(np.add(cv2.Sobel(img_r, cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_REPLICATE), cv2.Sobel(
        img_g, cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_REPLICATE)), cv2.Sobel(img_b, cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_REPLICATE))
    return sobelx, sobely


def resize(img, amount, orient, mode):
    # orient: 0 for horizontal resize; sobel_x , 1 for vertical resize; sobel_y_t
    # mode: 0 for shrink, 1 for englarge
    m_rgb = np.transpose(img) if orient else img
    row_num, col_num, rgb = m_rgb.shape
    m_res = m_rgb.tolist()
    for i in xrange(0, amount):
        # m_sobel = np.array(m_sobel)
        sobel_x, sobel_y = sobel_rgb(np.array(m_res))
        sobel_y_t = np.transpose(sobel_y)
        m_sobel = sobel_y_t if orient else sobel_x
        seam = find_a_seam(m_sobel)
        m_sobel = m_sobel.tolist()
        if mode:
            for row in xrange(0, row_num):
                interpolation_pixel = [(m_res[row][seams[i][row] - 1][0] + m_res[row][seams[i][row] + 1][0]) / 2, (m_res[row][seams[i][row] - 1][
                    1] + m_res[row][seams[i][row] + 1][1]) / 2, (m_res[row][seams[i][row] - 1][2] + m_res[row][seams[i][row] + 1][2]) / 2]
                m_res[row].insert(seams[i][row], interpolation_pixel)
                # m_res[row][seam[row]]=[255,0,0]
        else:
            for row in xrange(0, row_num):
                m_res[row].pop(seam[row])
                # m_sobel[row].pop(seam[row])
                # m_res[row][seams[row]]= [int(float(i)/255*255),int(float(i)/255*255),int(float(i)/255*255)]
        # sobel_x,sobel_y = sobel_rgb(np.array(m_res))
    # for i in xrange(0,row_num):
    # 	m_res[i] = [v for v in m_res[i] if v!=[-1,-1,-1]]

    # return np.transpose(np.array(m_res)) if orient else np.array(m_res)
    return np.array(m_res)


img = cv2.imread('hikari-loveletter.jpg')
# img = cv2.imread('01-1372466146-2319814.jpg')
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# lap = laplacian_rgb(laplacian)
# img = img.tolist()
# sobelx_t = np.transpose(sobelx)
# img2 = resize(img, 20, 0, 0, laplacian, laplacian)
img2 = resize(img, 270, 0, 0)
# print img2.shape
cv2.imwrite('h6.jpg', img2)
# plt.subplot(221),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(222),plt.imshow(img2,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(223),plt.imshow(sobely,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

import cv2
import numpy as np
import re
from scipy.spatial import Delaunay


def getdata(f, rate):
    Data = []
    datasize = int(f.readline())
    for i in range(datasize):
        txt = f.readline()
        x = re.split("\s", txt)
        Data.append((int(int(x[0]) / rate), int(int(x[1]) / rate)))
    Data = np.array(Data)
    return Data


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2)


# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside(x, y, p1, p2, p3):
    # Calculate area of triangle ABC
    A = area(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])

    # Calculate area of triangle PBC
    A1 = area(x, y, p2[0], p2[1], p3[0], p3[1])

    # Calculate area of triangle PAC
    A2 = area(p1[0], p1[1], x, y, p3[0], p3[1])

    # Calculate area of triangle PAB
    A3 = area(p1[0], p1[1], p2[0], p2[1], x, y)

    # Check if sum of A1, A2 and A3
    # is same as A
    if A == A1 + A2 + A3:
        return True
    else:
        return False


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def q1(image1, image2, tri1, tri2, rate):
    image1 = resize(image1, 1 / rate)
    image2 = resize(image2, 1 / rate)


    warp_mats1 = []
    warp_mats2 = []

    for t in range(tri1.shape[0]):
        warp_mats1.append(cv2.getAffineTransform(tri1[t].astype(np.float32), tri2[t].astype(np.float32)))

    for t in range(tri1.shape[0]):
        warp_mats2.append(cv2.getAffineTransform(tri2[t].astype(np.float32), tri1[t].astype(np.float32)))

    h = image1.shape[0]
    w = image1.shape[1]
    map1 = np.array([[[0, 0]] * w] * h, dtype=np.float32)
    map2 = np.array([[[0, 0]] * w] * h, dtype=np.float32)

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            for t in range(tri1.shape[0]):
                if isInside(j, i, tri1[t, 0], tri1[t, 1], tri1[t, 2]):
                    map1[i, j] = np.sum(np.matmul(warp_mats1[t], np.array([[j, i, 1]]).transpose()), axis=1)
                    break
            for t in range(tri2.shape[0]):
                if isInside(j, i, tri2[t, 0], tri2[t, 1], tri2[t, 2]):
                    map2[i, j] = np.sum(np.matmul(warp_mats2[t], np.array([[j, i, 1]]).transpose()), axis=1)
                    break
        print(i, '/', image1.shape[0])

    video = cv2.VideoWriter('morph.mp4', -1, 30, (w, h))

    coefs1 = np.array([[[[0] * 2] * w] * h] * 45, dtype='float16')
    coefs2 = np.array([[[[0] * 2] * w] * h] * 45, dtype='float16')
    poses = np.array([[[[0] * 2] * w] * h])

    for i in range(h):
        for j in range(w):
            poses[:, i, j] = [j, i]

    for it in range(45):
        fr = it / 44
        coefs1[it, :, :] = np.add(np.multiply(poses, fr), np.multiply(map1, (1 - fr)))
        coefs2[it, :, :] = np.add(np.multiply(poses, (1 - fr)), np.multiply(map2, fr))

    res_stack = []
    for it in range(45):
        fr2 = it / 44
        fr1 = 1 - fr2
        res2 = image2.copy()
        res1 = image1.copy()
        for i in range(coefs1.shape[1]):
            for j in range(coefs1.shape[2]):
                ch = coefs1[it, i, j, 0]
                cw = coefs1[it, i, j, 1]
                w1 = np.floor(ch)
                w2 = np.ceil(ch)
                h1 = np.floor(cw)
                h2 = np.ceil(cw)
                if w1 == w2:
                    w2 += 1
                if h1 == h2:
                    h2 += 1
                s1 = (ch - w1) * (cw - h1)
                s2 = (w2 - ch) * (cw - h1)
                s3 = (ch - w1) * (h2 - cw)
                s4 = (w2 - ch) * (h2 - cw)
                res2[i, j] = (np.multiply(image2[min(h - 1, int(h1)), min(w - 1, int(w1))], s4) +
                              np.multiply(image2[min(h - 1, int(h2)), min(w - 1, int(w2))], s1) +
                              np.multiply(image2[min(h - 1, int(h2)), min(w - 1, int(w1))], s2) +
                              np.multiply(image2[min(h - 1, int(h1)), min(w - 1, int(w2))], s3)).astype('int')

                ch = coefs2[it, i, j, 0]
                cw = coefs2[it, i, j, 1]
                w1 = np.floor(ch)
                w2 = np.ceil(ch)
                h1 = np.floor(cw)
                h2 = np.ceil(cw)
                if w1 == w2:
                    w2 += 1
                if h1 == h2:
                    h2 += 1
                s1 = (ch - w1) * (cw - h1)
                s2 = (w2 - ch) * (cw - h1)
                s3 = (ch - w1) * (h2 - cw)
                s4 = (w2 - ch) * (h2 - cw)
                res1[i, j] = (np.multiply(image1[min(h - 1, int(h1)), min(w - 1, int(w1))], s4) +
                              np.multiply(image1[min(h - 1, int(h2)), min(w - 1, int(w2))], s1) +
                              np.multiply(image1[min(h - 1, int(h2)), min(w - 1, int(w1))], s2) +
                              np.multiply(image1[min(h - 1, int(h1)), min(w - 1, int(w2))], s3)).astype('int')

        res1 = np.multiply(res1, fr1)
        res2 = np.multiply(res2, fr2)
        res = np.add(res1, res2).astype('int')
        res[res > 255] = 255
        res[res < 0] = 0
        if it == 14:
            cv2.imwrite('res03.jpg', res.astype('uint8'))
        if it == 29:
            cv2.imwrite('res04.jpg', res.astype('uint8'))
        res_stack.append(res.astype('uint8'))
        print(it, '/ 44')

    for i in range(45):
        video.write(res_stack[i])
    for i in range(45):
        video.write(res_stack.pop(44-i))

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    rate = 1
    f = open("landmarks1.dat", "r")
    shape1 = getdata(f, rate)
    f = open("landmarks2.dat", "r")
    shape2 = getdata(f, rate)

    tri2 = Delaunay(shape1)

    tries = tri2.simplices
    tri1 = shape1[tries]
    tri2 = shape2[tries]

    image1 = cv2.imread("res01.jpg")
    image2 = cv2.imread("res02.jpg")

    q1(image1, image2, tri1, tri2, rate)

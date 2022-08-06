import cv2
import numpy as np
import tensorflow as tf
from scipy import sparse
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as linalg


def normalize(img):
    # linear normalizing img between 0 to 255
    return ((img - img.min()) * (255 / (img.max() - img.min()))).astype('uint8')


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def q2(rate, x, y, background, mask, fore_raw, sigma=1):
    mask = cv2.GaussianBlur(mask, (55, 55), sigma)

    mask = resize(mask, rate)
    mask_foreground = np.zeros((background.shape[0], background.shape[1]), dtype='int')
    mask_foreground[y:y + mask.shape[0], x:x + mask.shape[1]] = mask

    mask_background = np.subtract(255, mask_foreground)

    back_mask = np.divide(mask_background, 255)
    fore_mask = np.divide(mask_foreground, 255)

    fore_raw = resize(fore_raw, rate)

    bb, gb, rb = cv2.split(background[y:y + mask.shape[0], x:x + mask.shape[1]])
    bf, gf, rf = cv2.split(fore_raw)

    rank = mask.shape[0] * mask.shape[1]
    mat0 = []
    mat1 = []

    h = mask.shape[0]
    w = mask.shape[1]





    bb = np.array(bb, dtype='int')
    gb = np.array(gb, dtype='int')
    rb = np.array(rb, dtype='int')

    bf = np.array(bf, dtype='int')
    gf = np.array(gf, dtype='int')
    rf = np.array(rf, dtype='int')

    print('Matrix created')
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    for j in range(w):
        if np.max(mask[:, j]) == 0: continue
        for i in range(h):
            if mask[i, j] != 0:
                try:
                    bb[i, j] = np.sum(np.multiply(bf[i - 1:i + 2, j - 1:j + 2], laplacian))
                    gb[i, j] = np.sum(np.multiply(gf[i - 1:i + 2, j - 1:j + 2], laplacian))
                    rb[i, j] = np.sum(np.multiply(rf[i - 1:i + 2, j - 1:j + 2], laplacian))
                except:
                    continue



    print('Vector created')

    for j in range(w):
        for i in range(h):
            if mask[i, j] == 0:
                mat0.append(1)
                mat1.append(0)
            else:
                mat0.append(4)
                mat1.append(-1)

    mat2 = mat1.copy()
    mat3 = mat1.copy()
    mat4 = mat1.copy()

    mat1.pop(0)
    mat1.append(0)
    mat2.pop(-1)
    mat2.insert(0, 0)

    for i in range(h):
        mat3.pop(0)
        mat3.append(0)
        mat4.pop(-1)
        mat4.insert(0, 0)

    print('Diags created')

    diags = np.array([-h, -1, 0, 1, h])
    data = np.stack((mat3, mat1, mat0, mat2, mat4), axis=0)
    mat = sparse.spdiags(data, diags, rank, rank)

    mat = csr_matrix(mat)
    bb = csr_matrix(bb)
    gb = csr_matrix(gb)
    rb = csr_matrix(rb)

    bb = bb.reshape((rank, 1), order='F')
    gb = gb.reshape((rank, 1), order='F')
    rb = rb.reshape((rank, 1), order='F')

    bb = np.divide(bb, 1)
    gb = np.divide(gb, 1)
    rb = np.divide(rb, 1)

    print('Matrices created')

    bf = linalg.spsolve(mat, bb).reshape((h, w), order='F')
    print('Blue completed')
    gf = linalg.spsolve(mat, gb).reshape((h, w), order='F')
    print('Green completed')
    rf = linalg.spsolve(mat, rb).reshape((h, w), order='F')
    print('Red completed')

    bf[bf < 0] = 0
    gf[gf < 0] = 0
    rf[rf < 0] = 0

    bf[bf > 255] = 255
    gf[gf > 255] = 255
    rf[rf > 255] = 255

    foreground = np.zeros((background.shape[0], background.shape[1], 3), dtype='int')
    foreground[y:y + mask.shape[0], x:x + mask.shape[1]] = cv2.merge((bf, gf, rf))

    fore_mask = cv2.merge((fore_mask, fore_mask, fore_mask))
    back_mask = cv2.merge((back_mask, back_mask, back_mask))

    res_fore = np.multiply(fore_mask, foreground)
    res_back = np.multiply(back_mask, background)

    res = res_fore + res_back

    return res


if __name__ == '__main__':
    background = cv2.imread('res06.jpg', 1)
    mask = cv2.imread('mask.jpg', 0)[:420, :820]
    foreground = cv2.imread('res05.jpg', 1)[:420, :820]

    res = q2(0.85, 20, 290, background, mask, foreground, 2)

    background = cv2.imread('res06.jpg', 1)
    mask = cv2.imread('mask.jpg', 0)[421:738, :718]
    foreground = cv2.imread('res05.jpg', 1)[421:738, :718]

    res = q2(1.4, 900, 600, res, mask, foreground, 1)

    cv2.imwrite('res07.jpg', res)
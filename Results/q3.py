import cv2
import numpy as np
from scipy import sparse, ndimage



def q3(im1, im2, mask, levels, a, b,c):
    laplacian_stack = []

    im1 = np.array(im1, dtype='int')
    im2 = np.array(im2, dtype='int')

    optimized_im1 = im1.copy()
    optimized_im2 = im2.copy()

    optimized_mask1 = ndimage.gaussian_filter(mask, sigma=c)
    optimized_mask2 = np.subtract(255, optimized_mask1)

    optimized_mask1 = cv2.merge((optimized_mask1, optimized_mask1, optimized_mask1))
    optimized_mask2 = cv2.merge((optimized_mask2, optimized_mask2, optimized_mask2))

    binary_omask1 = np.divide(optimized_mask1, 255)
    binary_omask2 = np.divide(optimized_mask2, 255)

    last_level1 = im1.copy()
    last_level2 = im2.copy()

    for layer in range(levels):
        gauss = int((layer + 1) * a)

        optimized_im1 = cv2.GaussianBlur(im1.astype('uint8'), (215, 215), gauss).astype('int')
        optimized_im2 = cv2.GaussianBlur(im2.astype('uint8'), (215, 215), gauss).astype('int')

        l1 = np.subtract(last_level1, optimized_im1)
        l2 = np.subtract(last_level2, optimized_im2)

        l1 = np.multiply(l1, binary_omask1)
        l2 = np.multiply(l2, binary_omask2)

        laplacian_stack.insert(0, l1 + l2)
        print(layer, '/', levels - 1)

        optimized_im1 = cv2.GaussianBlur(im1.astype('uint8'), (215, 215), gauss).astype('int')
        optimized_im2 = cv2.GaussianBlur(im2.astype('uint8'), (215, 215), gauss).astype('int')

        optimized_mask1 = ndimage.gaussian_filter(mask, sigma=int(gauss * b))
        optimized_mask2 = np.subtract(255, optimized_mask1)

        optimized_mask1 = cv2.merge((optimized_mask1, optimized_mask1, optimized_mask1))
        optimized_mask2 = cv2.merge((optimized_mask2, optimized_mask2, optimized_mask2))

        binary_omask1 = np.divide(optimized_mask1, 255)
        binary_omask2 = np.divide(optimized_mask2, 255)

        last_level1 = optimized_im1.copy()
        last_level2 = optimized_im2.copy()

    optimized_im1 = np.multiply(optimized_im1, binary_omask1)
    optimized_im2 = np.multiply(optimized_im2, binary_omask2)

    res = optimized_im1 + optimized_im2

    for layer in range(levels):
        res += laplacian_stack.pop(0)

    res[res < 0] = 0
    res[res > 255] = 255
    return np.array(res, dtype='uint8')


if __name__ == '__main__':
    im1 = cv2.imread("res08.jpg", 1)
    im2 = cv2.imread('res09.jpg', 1)
    mask = cv2.imread('mask.jpg', 0)
    res = q3(im1, im2, mask, 20, 3, 4,3)
    cv2.imwrite('res10.jpg', res)

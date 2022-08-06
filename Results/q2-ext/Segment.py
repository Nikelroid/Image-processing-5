import sys
import numpy as np
import cv2
import scipy as sp



def q4(mask):
    print('working')

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.array(mask, dtype='int')
    mask[(mask > 0)] = 1
    mask[mask == -1] = 2
    mask = np.array(mask, dtype='uint8')
    mask, bgdModel, fgdModel = cv2.grabCut(org_image, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    mask_main = cv2.resize(mask, (0, 0), fx=4, fy=4)
    cv2.imwrite('mask.jpg', mask_main)
    cv2.imwrite('for_optimized.jpg', org_image[:mask_main.shape[0],:mask_main.shape[1]])


def operate(event, x, y, arg1, arg2):
    global mask_label
    global mouseX, mouseY

    if event == cv2.EVENT_RBUTTONDOWN:
        if mask_label == 2:
            mask_label = 1
        elif mask_label == 1:
            mask_label = 3
        elif mask_label == 3:
            mask_label = 0
        else:
            mask_label = 2

    if event == cv2.EVENT_MOUSEMOVE and mask_label == 0:
        try:
            image[y - r:y + r + 1, x - r:x + r + 1] = radial31
            mask[y - r:y + r + 1, x - r:x + r + 1] = \
                np.add(mask[y - r:y + r + 1, x - r:x + r + 1], radial2)
        except:
            print('Out of range')

    if event == cv2.EVENT_MOUSEMOVE and mask_label == 1:
        try:
            image[y - r0:y + r0 + 1, x - r0:x + r0 + 1] = radial30
            image[image < 0] = 0
            mask[y - r0:y + r0 + 1, x - r0:x + r0 + 1] = \
                np.multiply(mask[y - r0:y + r0 + 1, x - r0:x + r0 + 1], radial4)
        except:
            print('Out of range')

    image[image > 255] = 255
    cv2.imshow('Segment', image)
    cv2.waitKey(1)
    # saves clicked points in Xcoords and Ycoords
    if event == cv2.EVENT_LBUTTONDBLCLK:
        q4(mask)
        sys.exit()
    mouseX, mouseY = x, y


Xcoords = []
Ycoords = []
mask_label = 2
r = 5
r0 = 15
radial30 = np.full((2 * r0 + 1, 2 * r0 + 1, 3), (0, 0, 255))
radial31 = np.full((2 * r + 1, 2 * r + 1, 3), (255, 0, 0))

radial4 = np.full((2 * r0 + 1, 2 * r0 + 1), 0)
radial2 = np.full((2 * r + 1, 2 * r + 1), 2)

# get image  from files
org_image = cv2.imread("res05.jpg", 1)

mask = np.zeros(org_image.shape[:2], dtype='int') - 1
main_image = org_image.copy()
image = org_image.copy()
# set a window for show image
cv2.namedWindow('Segment')

cv2.setMouseCallback('Segment', operate)
# make a true loop for get update image which is showing in window


while (1):
    cv2.imshow('Segment', image)
    k = cv2.waitKey(20) & 0xFF
    # if 'esc' clicked, loop will be break and program closes
    if k == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

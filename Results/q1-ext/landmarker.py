# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


def operate1(event, x, y, arg1, arg2):
    global mouseX, mouseY
    global shape1
    # saves clicked points in Xcoords and Ycoords
    if event == cv2.EVENT_LBUTTONDOWN:
        shape1 = np.append(shape1, [[x, y]], axis=0)
        cv2.circle(image1, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow('im1', image1)
        mouseX, mouseY = x, y


def operate2(event, x, y, arg1, arg2):
    global mouseX, mouseY
    global shape2
    # saves clicked points in Xcoords and Ycoords
    if event == cv2.EVENT_LBUTTONDOWN:
        shape2 = np.append(shape2, [[x, y]], axis=0)
        cv2.circle(image2, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow('im2', image2)
        mouseX, mouseY = x, y


# construct the argument parser and parse the arguments


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

# load the input image, resize it, and convert it to grayscale
image1 = cv2.imread("res01.jpg")
# image1 = imutils.resize(image1, width=500)
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape1 = predictor(gray, rect)
    shape1 = face_utils.shape_to_np(shape1)
    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape1:
        cv2.circle(image1, (x, y), 1, (0, 0, 255), -1)

# show the output image with the face detections + facial landmarks

image2 = cv2.imread("res02.jpg")
# image2 = imutils.resize(image2, width=500)
gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape2 = predictor(gray, rect)
    shape2 = face_utils.shape_to_np(shape2)
    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape2:
        cv2.circle(image2, (x, y), 1, (0, 0, 255), -1)

# show the output image with the face detections + facial landmarks

# set a window for show image
cv2.namedWindow('im1')
cv2.namedWindow('im2')

# set callback for created image window, as draw_circle function
cv2.setMouseCallback('im1', operate1)
cv2.setMouseCallback('im2', operate2)

# make a true loop for get update image which is showing in window
while (1):
    cv2.imshow('im1', image1)
    cv2.imshow('im2', image2)
    k = cv2.waitKey(20) & 0xFF
    # if 'esc' clicked, loop will be break and program closes
    if k == 13:
        break

cv2.destroyWindow("im1")
cv2.destroyWindow("im2")

with open('landmarks1.dat', 'w') as f:
    f.write(str(shape1.shape[0] + 4) + '\n')
    f.write('0 0\n')
    f.write('0 ' + str(image1.shape[0]) + '\n')
    f.write(str(image1.shape[1]) + ' 0\n')
    f.write(str(image1.shape[1]) + ' ' + str(image1.shape[0]) + '\n')
    for q in shape1:
        f.write(str(q[0]))
        f.write(' ')
        f.write(str(q[1]))
        f.write('\n')

with open('landmarks2.dat', 'w') as f:
    f.write(str(shape2.shape[0] + 4) + '\n')
    f.write('0 0\n')
    f.write('0 ' + str(image1.shape[0]) + '\n')
    f.write(str(image2.shape[1]) + ' 0\n')
    f.write(str(image2.shape[1]) + ' ' + str(image1.shape[0]) + '\n')
    for q in shape2:
        f.write(str(q[0]))
        f.write(' ')
        f.write(str(q[1]))
        f.write('\n')

cv2.waitKey(0)

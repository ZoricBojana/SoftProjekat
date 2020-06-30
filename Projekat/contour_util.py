import numpy as np
import cv2
from image_util import *

def contours_difference(prev_img, current_img):
    prev_gray = image_gray(prev_img)
    current_gray = image_gray(current_img)

    difference = cv2.absdiff(prev_gray, current_gray)

    #display_image(difference)

    bin = cv2.adaptiveThreshold(difference, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 4)

    bin = dilate(bin)
    bin = erode(bin)
    #display_image(bin)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_people = []
    for contour in contours:
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        if width > 8 and width < 70 and height > 10 and height < 70:
            contours_people.append(contour)

    imgCopy = current_img.copy()
    cv2.drawContours(imgCopy, contours_people, -1, (255, 0, 0), 1)
    #display_image(imgCopy)
    return imgCopy

def detect_contours(img):
    #detekcija ljudi na slici

    img_gray = image_gray(img)

    # prema histogramu, plato se nalazi negde na 18
    #ret, image_bin = cv2.threshold(img_gray, 17, 255, cv2.THRESH_BINARY_INV) # ret je vrednost praga, image_bin je binarna slika
    image_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 5)
    #image_bin = image_util.dilate(image_bin)
    #image_bin = image_util.erode(image_bin)
    #display_image(image_bin)

    image_bin = dilate(image_bin)
    image_bin = erode(image_bin)

    contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_people = []
    for contour in contours:
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        if width > 10 and width < 50 and height > 10 and height <50:
            contours_people.append(contour)

    imgCopy = img.copy()
    cv2.drawContours(imgCopy, contours_people, -1, (255, 0, 0), 1)

    return imgCopy

def detect_contoursBrownColor(frame):
    # odvoji smedju pozadinu
    img_gray = image_gray(frame)
    bin_img = cv2.inRange(img_gray, 5, 20)
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = frame.copy()
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    #display_image(img)
    contours_people = []
    for contour in contours:
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        if width > 10 and width < 35 and height > 10 and height < 35:
            contours_people.append(contour)

    imgCopy = frame.copy()
    cv2.drawContours(imgCopy, contours_people, -1, (255, 0, 0), 1)
    #display_image(imgCopy)
    return imgCopy, contours_people

def detect_contoursHSV(frame):
    BROWN_MIN = np.array([10, 100, 20], np.uint8)
    BROWN_MAX = np.array([30, 255, 200], np.uint8)

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #display_image(hsv_img)
    frame_threshed = cv2.inRange(hsv_img, BROWN_MIN, BROWN_MAX)
    #cv2.imwrite('output2.jpg', frame_threshed)

def detect_areaCanny(img):
    #detekcija ivica platoa
    T1 = 50 #ispod T1 nije ivica
    T2 = 100 # iznad T2 jeste ivica
    #izmedju T1 i T2 slaba ivica
    imgCopy = img.copy()
    THETA = np.pi / 180
    #50, 100
    #img = invert(img)
    img_gs = image_gray(imgCopy)
    #histogram(img_gs)
    #display_image(img_gs)
    edges_img = cv2.Canny(img_gs, T1, T2, apertureSize=3)
    #plt.imshow(edges_img, "gray")

    # minimalna duzina linije
    min_line_length = 250

    # Hough transformacija
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=THETA, threshold=100, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)

    final_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 < 50 or y2 < 50 or x1 < 155 or x2 > 550:
        #if y1 < 50 or y2 > 200 or x1 < 155 or x2 > 550:
            continue
        print ((x1, y1), (x2, y2))
        cv2.line(imgCopy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        final_line = line[0]

    #display_image(imgCopy)
    return final_line


def find_contour_centeroid(cnt):
    # vraca centeroid konture: cnt => kontura
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

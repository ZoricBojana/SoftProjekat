import numpy as np
import cv2
from sklearn import datasets
import matplotlib.pyplot as plt


sum_of_nums = 0
k = 0
n = 0

# ucitavanje videa
frame_num = 0
cap = cv2.VideoCapture("video/video9.mp4")
cap.set(1, frame_num)  # indeksiranje frejmova


# analiza videa frejm po frejm

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 150, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def histogram(img_gray):
    hist_full = cv2.calcHist([img_gray], [0], None, [255], [0, 255])
    plt.plot(hist_full)
    plt.show()

def detect_areaCanny(img):
    T1 = 50 #ispod T1 nije ivica
    T2 = 100 # iznad T2 jeste ivica
    #izmedju T1 i T2 slaba ivica

    THETA = np.pi / 180
    #50, 100
    #img = invert(img)
    img_gs = image_gray(img)
    #histogram(img_gs)
    #display_image(img_gs)
    edges_img = cv2.Canny(img_gs, T1, T2, apertureSize=3)
    #plt.imshow(edges_img, "gray")

    # minimalna duzina linije
    min_line_length = 250

    # Hough transformacija
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=THETA, threshold=100, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 < 50 or y2 < 50 or x1 < 155 or x2 > 550:
            continue
        print ((x1, y1), (x2, y2))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('image', img)
    display_image(img)


def detect_contours(img):
    img_gray = image_gray(img)
    image_ada_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    display_image(image_ada_bin)
    return

while frame_num < 5:

    frame_num += 1
    ret_val, frame = cap.read()

    # ako frejm nije zahvacen
    if not ret_val:
        break

    #detect_contours(frame)

    if(frame_num == 1):
        detect_areaCanny(frame)





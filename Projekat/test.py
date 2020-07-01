import numpy as np
import cv2
from sklearn import datasets
import matplotlib.pyplot as plt
from image_util import *
from contour_util import *
from model.ObjectTracker import ObjectTracker
from sklearn.metrics import mean_absolute_error
import pandas as pd

def get_video_title(order_number):
    if (order_number < 1 or order_number > 10):
        return None
    return "video/video" + str(order_number) + ".mp4"

def cross_line(x,y):
    if x > 180 and x < 480 and y < 250 and y > 200:
        return True
    return False

def print_MAE(my_results):
    raw_data = pd.read_csv('res.txt', header=0, names=['file', 'count'])
    res = raw_data['count'].tolist()
    # print(res)
    print('MAE: ', mean_absolute_error(res, my_results))


if __name__ == '__main__':

    my_results = []

    for i in range(1,11):
        # ucitavanje videa
        frame_num = 0 # frejm, tj broj frejma koji se obradjuje
        counter = 0 # brojac pesaka
        tracker = ObjectTracker()
        cap = cv2.VideoCapture(get_video_title(i))
        cap.set(1, frame_num)  # indeksiranje frejmova

        #izlazni video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('output'+ str(i) + '.avi', fourcc, 25, (640,480))

        first_img = None
        fift_img = None
        while True:
            frame_num += 1
            ret_val, frame = cap.read()
            # if not frame_num%3 == 0:
            #     continue
            retVal = frame
            if not ret_val:
                break
            prev = frame
            if(frame_num == 1):
                first_img = frame
                # linija: y: 150, x1: 180, x2: 480
                cv2.line(frame, (180, 225), (480, 225), (0, 255, 0), 2)
                #display_image(frame)
                #detected = detect_areaCanny(frame)
                #print(detected)

            with_contoures, contoures = detect_contoursBrownColor(frame)
            centers = []
            for cont in contoures:
                center = find_contour_centeroid(cont)
                centers.append(list(center))
                if (cross_line(center[0], center[1])):
                    counter += 1
                    with_contoures = cv2.circle(with_contoures, (center[0], center[1]), radius=2, color=(0, 0, 255), thickness=2)
            #print(centers)
            tracker.update(tuple(centers))
            cv2.line(with_contoures, (180, 225), (480, 225), (0, 255, 0), 2)
            video.write(with_contoures)

            ret_val, frame = cap.read()


            curr = frame
        my_results.append(len(tracker.objects))

        print(counter, ' vs ' , len(tracker.objects))
        #print(counter)

        cv2.destroyAllWindows()
        video.release()


    print_MAE(my_results)


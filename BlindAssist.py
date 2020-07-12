import sys
import numpy as np
import cv2
from os import system
import io
import time
from os.path import isfile, join
import re
import pygame
import os

sourceFileDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(sourceFileDir)

pygame.mixer.init()


def Sound_play(WAVFILE):
    pygame.mixer.music.load(WAVFILE)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue


def Label_play(object_id):
    if object_id == 2:
        Sound_play("bicycle.wav")
    elif object_id == 3:
        Sound_play("bird.wav")
    elif object_id == 4:
        Sound_play("boat.wav")
    elif object_id == 5:
        Sound_play("bottle.wav")
    elif object_id == 6:
        Sound_play("bus.wav")
    elif object_id == 7:
        Sound_play("car.wav")
    elif object_id == 8:
        Sound_play("cat.wav")
    elif object_id == 9:
        Sound_play("chair.wav")
    elif object_id == 10:
        Sound_play("cow.wav")
    elif object_id == 11:
        Sound_play("table.wav")
    elif object_id == 12:
        Sound_play("dog.wav")
    elif object_id == 13:
        Sound_play("horse.wav")
    elif object_id == 14:
        Sound_play("motorbike.wav")
    elif object_id == 15:
        Sound_play("person.wav")
    elif object_id == 16:
        Sound_play("pottedplant.wav")
    elif object_id == 17:
        Sound_play("sheep.wav")
    elif object_id == 18:
        Sound_play("sofa.wav")
    elif object_id == 19:
        Sound_play("train.wav")
    elif object_id == 20:
        Sound_play("monitor.wav")


fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
sound_count = 0


LABELS = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'monitor')

camera_width = 320
camera_height = 240

# small x = on left    small y = on top

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

net = cv2.dnn.readNet('lrmodel/MobileNetSSD/MobileNetSSD_deploy.xml',
                      'lrmodel/MobileNetSSD/MobileNetSSD_deploy.bin')
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

try:

    while True:
        t1 = time.perf_counter()

        ret, color_image = cap.read()
        if not ret:
            break

        height = color_image.shape[0]
        width = color_image.shape[1]

        blob = cv2.dnn.blobFromImage(color_image, 0.007843, size=(
            300, 300), mean=(127.5, 127.5, 127.5), swapRB=False, crop=False)
        net.setInput(blob)
        out = net.forward()
        out = out.flatten()

        TL_inside = (int(width*0.218), int(height*0.125))
        BR_inside = (int(width*0.781), int(height*0.875))

        # draw inside box
        cv2.rectangle(color_image, TL_inside, BR_inside, (20, 20, 255), 3)

        for box_index in range(100):
            if out[box_index + 1] == 0.0:
                break
            base_index = box_index * 7
            if (not np.isfinite(out[base_index]) or
                not np.isfinite(out[base_index + 1]) or
                not np.isfinite(out[base_index + 2]) or
                not np.isfinite(out[base_index + 3]) or
                not np.isfinite(out[base_index + 4]) or
                not np.isfinite(out[base_index + 5]) or
                    not np.isfinite(out[base_index + 6])):
                continue

            if box_index == 0:
                detectframecount += 1

            # tranfer object information to object_info_overlay
            object_info_overlay = out[base_index:base_index + 7]

            # minimum scores for object detection is 65
            min_score_percent = 65
            source_image_width = width
            source_image_height = height

            # set value of class_id(float) and percentage(int)
            base_index = 0
            class_id = object_info_overlay[base_index + 1]
            percentage = int(object_info_overlay[base_index + 2] * 100)

            # filter objects by scores
            if (percentage <= min_score_percent):
                continue

            # filter objects by class_id
            if (class_id == 20.0 or class_id == 0.0 or class_id == 8.0 or class_id == 10.0 or class_id == 12.0 or class_id == 13.0 or class_id == 17.0 or class_id == 19.0):
                continue

            # set coordinates of a box
            x1 = max(0, int(out[base_index + 3] * height))
            y1 = max(0, int(out[base_index + 4] * width))
            x2 = min(height, int(out[base_index + 5] * height))
            y2 = min(width, int(out[base_index + 6] * width))

            # Real coordinates for each object
            box_left = int(object_info_overlay[base_index + 3] * source_image_width)
            box_top = int(object_info_overlay[base_index + 4] * source_image_height)
            box_right = int(object_info_overlay[base_index + 5] * source_image_width)
            box_bottom = int(object_info_overlay[base_index + 6] * source_image_height)

            label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"

            box_color = (255, 128, 0)
            box_thickness = 3

            # set center coordinates of each object
            x_center = int(
                ((object_info_overlay[base_index + 3] + object_info_overlay[base_index + 5])/2) * source_image_width)
            y_center = int(
                ((object_info_overlay[base_index + 4] + object_info_overlay[base_index + 6])/2) * source_image_height)

            # filter objects by center position
            if ((x_center > TL_inside[0]) and (x_center < BR_inside[0]) and (y_center > TL_inside[1]) and (y_center < BR_inside[1])):

                # draw center point of each object
                cv2.circle(color_image, (x_center, y_center), 5, box_color, -1)

                # draw box of each object
                cv2.rectangle(color_image, (box_left, box_top),
                              (box_right, box_bottom), box_color, box_thickness)

                label_background_color = (125, 175, 75)
                label_text_color = (255, 255, 255)

                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]

                print(label_text)
                print("")

                # draw label of each object
                cv2.rectangle(color_image, (label_left - 1, label_top - 1),
                              (label_right + 1, label_bottom + 1), label_background_color, -1)
                cv2.putText(color_image, label_text, (label_left, label_bottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

                # Find the area of each objects
                object_width = (box_right - box_left)
                object_height = (box_bottom - box_top)
                object_area = (object_width*object_height)

                objectnumber = int(class_id)

                if sound_count == 3:
                    Label_play(objectnumber)
                    sound_count = 0

                else:
                    sound_count += 1

        cv2.putText(color_image, fps,       (width-170, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(color_image, detectfps, (width-170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)

        cv2.namedWindow('USB Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('USB Camera', cv2.resize(color_image, (width, height)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        if framecount >= 15:
            fps = "(Playback) {:.1f} FPS".format(time1/15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime


except:
    import traceback
    traceback.print_exc()

finally:

    print("\n\nFinished\n\n")

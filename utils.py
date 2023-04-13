import statistics
from scipy.ndimage import histogram
from scipy.stats import linregress
import cv2
import numpy as np
import time
import signal 
import sys
import threading
import copy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model


GLOBAL_ROI_ARRAY = []
gaussian_kernel = 101
median_kernel = 5
RECT_ROWS = 180
RECT_COLS = 120
KERNEL = 151
ROWS = 200
COLS = 200

# seems to work way better at 2 than anything else.. not sure why
DIFFARRAY_THRESH = 2

LINEAR_BEST_FIT_NUM_POINTS = 10

MAX_COOLDOWN = 15
cooldown_counter = MAX_COOLDOWN

N = 0.45
M = 3
P_lo = 10
P_hi = 6000

blur_kernel = np.ones((KERNEL,KERNEL),np.float32)/(KERNEL*KERNEL)

def calculate_entropy_tsallis(histogram, q):
    entropy = (1/(q-1))*(1-np.sum(histogram**q))
    return entropy

def calculate_entropy(histogram):
    entropy = -1*np.sum(np.log2(histogram+0.0000001))
    return entropy

def find_all_locations(list_2D):
    total_list = []
    for idx, i in enumerate(list_2D):
        for jdx, j in enumerate(i):
            if list_2D[idx][jdx] == 255:
                total_list.append((idx, jdx))
    return total_list

def find_geographic_center(list_of_points):
    x_vals = [n[0] for n in list_of_points]
    y_vals = [n[1] for n in list_of_points]
    x_avg = sum(x_vals) / len(x_vals)
    y_avg = sum(y_vals) / len(y_vals)
    return (int(x_avg), int(y_avg))


def calculate_linear_best_fit(list_of_points):
    x_array = np.linspace(1, len(list_of_points), len(list_of_points), dtype=np.intc)
    res = linregress(x_array, np.array(list_of_points))
    return int((res.intercept + int(len(list_of_points)/2)*res.slope))


def calculate_ROI(image1, previous_location):
    global GLOBAL_ROI_ARRAY
    if len(GLOBAL_ROI_ARRAY) > 10:
        GLOBAL_ROI_ARRAY = GLOBAL_ROI_ARRAY[1:]
    diff_array = np.zeros((ROWS,COLS), dtype=np.uint8)
    if len(GLOBAL_ROI_ARRAY) > 5:
        diff_array = (abs(np.array(image1).astype(np.intc) - np.array(GLOBAL_ROI_ARRAY[-1]).astype(np.intc))).astype(np.uint8)
        diff_array = cv2.medianBlur(diff_array, median_kernel)
        diff_array = cv2.GaussianBlur(diff_array, (gaussian_kernel, gaussian_kernel), 0)
        diff_array[diff_array < DIFFARRAY_THRESH] = 0
        diff_array[diff_array > DIFFARRAY_THRESH] = 255
        try:
            max_val_location = find_geographic_center(find_all_locations(diff_array))
        except:
            max_val_location = previous_location
        #max_val_location = np.argmax(diff_array)
        #max_val_location = (int(max_val_location / ROWS), int(max_val_location % COLUMNS))
        GLOBAL_ROI_ARRAY.append(image1)
        return max_val_location, diff_array
    else:
        GLOBAL_ROI_ARRAY.append(image1)
        return None, None
    

def is_key_frame(movement_list, cooldown):
    ## check if the amount of movement in the frame is less than N percent of the average of the past M frames 
    ## if so, and if the total movement in the frame is greater than P_lo and smaller than P_hi, it is a key frame.

    mvmt_avg = 0
    for i in range(M):
        mvmt_avg += movement_list[-2 - i]
    mvmt_avg /= 4

    if movement_list[-1] < mvmt_avg * N:
        if (movement_list[-1] > P_lo) and (movement_list[-1] < P_hi):
            return True
    else:
        return False
import statistics
from scipy.ndimage import histogram
import cv2
import numpy as np

GLOBAL_ROI_ARRAY = []
gaussian_kernel = 101
median_kernel = 5
KERNEL = 151
ORIG_ROWS = 720
ORIG_COLS = 1280
ROWS = 200
COLUMNS = 200

COOL_DOWN_VAL = 100
#q = 10

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

def calculate_ROI(image1, previous_location):
    global GLOBAL_ROI_ARRAY
    if len(GLOBAL_ROI_ARRAY) > 10:
        GLOBAL_ROI_ARRAY = GLOBAL_ROI_ARRAY[1:]
    diff_array = np.zeros((ROWS,COLUMNS), dtype=np.uint8)
    if len(GLOBAL_ROI_ARRAY) > 5:
        diff_array = (abs(np.array(image1).astype(np.intc) - np.array(GLOBAL_ROI_ARRAY[-1]).astype(np.intc))).astype(np.uint8)
        diff_array = cv2.medianBlur(diff_array, median_kernel)
        diff_array = cv2.GaussianBlur(diff_array, (gaussian_kernel, gaussian_kernel), 0)
        diff_array[diff_array<2] = 0
        diff_array[diff_array>2] = 255
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
    
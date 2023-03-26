import matplotlib.pyplot as plt
import copy
from utils import *
from memory_profiler import profile
from time import sleep

@profile
def main():
    images = []
    keyframe_images = []
    entropy_vals = []
    background_image = np.zeros((ROWS, COLUMNS))


    keyframe = False
    cooldown = True
    cool_counter = 0

    max_val_input_tracker = [0,0,0,0,0,0,0,0,0]
    max_val_output_tracker = [0,0,0,0,0,0,0,0,0]
    max_val_location, saliency_map, max_val_location_temp = None, None, None
    count = 0
    total_count = 0
    vid = cv2.VideoCapture(0)

    previous_max_val_location = (0,0)

    while(True):
        if (len(images) > 50000):
            images.pop(0)
            entropy_vals.pop(0)

        ret, frame = vid.read()
        frame = cv2.resize(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY), (ROWS,COLUMNS), interpolation=cv2.INTER_AREA)

        #if total_count < 50:
        #    background_image =((background_image + frame)/2)
        #else:
        #    frame = abs(frame.astype(np.intc) - background_image.astype(np.intc)).astype(np.uint8)
        #    frame = cv2.medianBlur(frame, median_kernel)
        #    frame = cv2.GaussianBlur(frame, (gaussian_kernel, gaussian_kernel), 0)
        #    frame[frame<50] = 0
        #    frame[frame>49] = 255

        if keyframe == False:
            text = "NORMAL"
            color = (0,255,255)
        elif keyframe == True:
            text = "KEY FRAME"
            color = (255, 0, 0)

        if cooldown==True:
            cool_counter += 1
            if cool_counter >= COOL_DOWN_VAL:
                    cooldown = False
                    cool_counter = 0
        if (count % 20 == 0):
            images.append(frame)
            hist = np.array(histogram(np.array(images), 0, 255, 256))
            hist = hist/sum(hist)
            entropy = calculate_entropy(hist)
            if len(entropy_vals) > 2:
                if cooldown == False:
                    if ((entropy_vals[-1] > entropy_vals[-2]) and (entropy_vals[-1] > entropy)) or\
                        ((entropy_vals[-1] < entropy_vals[-2] and entropy_vals[-1] < entropy)):
                        keyframe = True
                        cooldown = True
                    else:
                        keyframe = False

            entropy_vals.append(entropy)

        max_val_location_temp, saliency_map =calculate_ROI(frame, previous_max_val_location)

        if max_val_location_temp is not None:
            mvltnumpy = np.array(max_val_location_temp)
            max_val_input_tracker.append(mvltnumpy)
            max_val_location = (9.2673e-03)*mvltnumpy + (7.4138e-02)*max_val_input_tracker[-1]\
                                +(2.5948e-01)*max_val_input_tracker[-2] + (5.1897e-01)*max_val_input_tracker[-3]\
                                +(6.4871e-01)*max_val_input_tracker[-4] + (5.1897e-01)*max_val_input_tracker[-5]\
                                +(2.5948e-01)*max_val_input_tracker[-6] + (7.4138e-02)*max_val_input_tracker[-7]\
                                +(9.2673e-03)*max_val_input_tracker[-8]\
                                -(-6.1930e-16)*max_val_output_tracker[-1] - (1.0609e+00)*max_val_output_tracker[-2]\
                                -(-4.4087e-16)*max_val_output_tracker[-3] - (2.9089e-01)*max_val_output_tracker[-4]\
                                -(-6.8339e-17)*max_val_output_tracker[-5] - (2.0430e-02)*max_val_output_tracker[-6]\
                                -(-2.0086e-18)*max_val_output_tracker[-7] - (1.7177e-04)*max_val_output_tracker[-8]
            max_val_output_tracker.append(max_val_location)
            max_val_output_tracker.pop(0)
            max_val_input_tracker.pop(0)
            previous_max_val_location = max_val_location
            try:
                start_point_row = (int(max_val_location[0]) - int(KERNEL/2))
            except:
                start_point_row = 0
            try:
                start_point_column = int(max_val_location[1]) - int(KERNEL/2)
            except:
                start_point_column = 0
            try:
                end_point_row = int(max_val_location[0]) + int(KERNEL/2)
            except:
                end_point_row = ROWS-1
            try:
                end_point_column = int(max_val_location[1]) + int(KERNEL/2)
            except:
                end_point_column = COLUMNS-1

            frame = cv2.rectangle(copy.deepcopy(frame),
                (start_point_column, start_point_row), (end_point_column, end_point_row), (0,255,0), 2)
        frame = cv2.resize(frame, (ORIG_COLS, ORIG_ROWS), interpolation=cv2.INTER_AREA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cooldown_text = "cooldown" if cooldown==True else "active"
        cooldown_color = (255,0,0) if cooldown==True else (0,255,255)
        cv2.putText(frame, 
                    text, 
                    (50, 50), 
                    font, 1, 
                    color, 
                    2, 
                    cv2.LINE_4)
        cv2.putText(frame, 
                    cooldown_text, 
                    (50, 100), 
                    font, 1, 
                    color, 
                    2, 
                    cv2.LINE_4)
        if saliency_map is not None:
            cv2.imshow('saliency map', saliency_map + 50)
        cv2.imshow('frame', frame)
        cv2.waitKey(10) & 0xFF
        count = (count + 1)%100
        total_count += 1

    vid.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    main()
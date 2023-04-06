from utils import *
from memory_profiler import profile

## global variables accessed by all threads
kill_threads = False
className = None

## set up gesture recognition model
##
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
# Initialize mediapipe model
hands = mpHands.Hands(max_num_hands = 1, min_detection_confidence = 0.7)
# Load gesture recognizer model
model = load_model('mp_hand_gesture')
# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
##
##

## start by emptying ./tmp folder
saved_ims = [n for n in os.listdir('./tmp/') if n[0] != '.']
if len(saved_ims) > 0:
    for image in saved_ims:
        os.remove(f"./tmp/{image}")
del saved_ims

def signal_handler(signal, frame):
    global kill_threads
    print('Interrupted!')
    vid.release()
    cv2.destroyAllWindows()
    kill_threads = True
    sys.exit(0)

def gest_rec(cropped_image):
    global className
    # make classname global so that GUI thread can access it
    x, y, c = cropped_image.shape
    # Get hand landmark prediction
    result = hands.process(cropped_image)
    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx,lmy])
        # Predict gesture in Hand Gesture Recognition project
        prediction = model.predict([landmarks], verbose=0)
        classID = np.argmax(prediction)
        className = classNames[classID]
    else:
        # hopefully, if GR does not find a gesture, it will print None
        className = None


def image_save(image, image_count, row_start, row_end, column_start, column_end):
    # flip image and expand ROI boundaries because the ROI was calculated using 200x200 instead of 720x1280
    rect_image = cv2.flip(image[int(row_start*3.7):int(row_end*3.7), int(column_start*6.4):int(column_end*6.4)], 1)
    try:
        gest_rec_thread = threading.Thread(target=gest_rec, args=[rect_image])
        gest_rec_thread.start()
        cv2.imwrite(f"./tmp/image_{image_count}.jpg", rect_image)
    # sometimes the image write fails, maybe because the rectangle boundaries are off
    except Exception as e:
        print("could not save or process image")
        print(e)


#@profile
def data_processing():
    global text
    global color
    global cooldown
    global keyframe
    global max_val_location_temp
    global max_val_location
    global saliency_map

    ## this thread writes to these variables
    images = []
    entropy_vals = []
    movement_list = [0,0,0,0,0]
    keyframe = False
    cooldown = True
    cool_counter = 0
    #
    #
    # initialize ROI inputs and outputs for filtering purposes
    max_val_input_tracker = [np.array([n,n]) for n in [0]*20]
    max_val_output_tracker = [np.array([n,n]) for n in [0]*20]
    max_val_location, saliency_map, max_val_location_temp = None, None, None
    # previous location is passed to ROI function to stay in the same place if no movement
    previous_max_val_location = (0,0)

    # sleep for 2 seconds to wait for camera to warm up
    time.sleep(2)

    while(True):
        # check if program has been terminated
        if kill_threads == True:
            break

        max_val_location_temp, saliency_map =calculate_ROI(frame, previous_max_val_location)

        if max_val_location_temp is not None:
            movement_list.append(np.sum(saliency_map/255))
            keyframe = is_key_frame(movement_list)
            mvltnumpy = np.array(max_val_location_temp)
            max_val_input_tracker.append(mvltnumpy)

            max_val_location_x = calculate_linear_best_fit([int(n[0]) for n in max_val_input_tracker][-15:])
            max_val_location_y = calculate_linear_best_fit([int(n[1]) for n in max_val_input_tracker][-15:])
        
            max_val_location = (max_val_location_x, max_val_location_y)

            max_val_output_tracker.append(max_val_location)
            max_val_output_tracker.pop(0)
            max_val_input_tracker.pop(0)
            movement_list.pop(0)
            previous_max_val_location = max_val_location


#@profile
def main():
    global frame
    global entropy
    global vid
    global proc_thread

    # keep track of how many frames have been read
    image_count = 0
    # open video capture device (0 for computer cam, 1 for usb cam)
    vid = cv2.VideoCapture(0)

    # start signal processing thread
    proc_thread = threading.Thread(target=data_processing)
    proc_thread.start()

    # tie function signal_handler to interrupt signal to safely handle a crash
    signal.signal(signal.SIGINT, signal_handler)

    while(True):
        # check if there has been termination
        if kill_threads == True:
            break
        # read video frame from capture device
        ret, frame_temp = vid.read()
        # increment frame counter for keeping track
        image_count += 1 
        # flip image, turn to gray, and resize down to 200x200
        frame = cv2.resize(cv2.cvtColor(cv2.flip(frame_temp, 1), cv2.COLOR_BGR2GRAY), (ROWS,COLUMNS), interpolation=cv2.INTER_AREA)

        # read max_val_location from signal processing thread
        if max_val_location is not None:
            # use max_val_location to find ROI in rectangle
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
            # create frame final by putting a rectangle on the small image
            frame_final = cv2.rectangle(copy.deepcopy(frame),
                (start_point_column, start_point_row), (end_point_column, end_point_row), (0,255,0), 2)

        try:
            frame_final = cv2.resize(frame_final, (ORIG_COLS, ORIG_ROWS), interpolation=cv2.INTER_AREA)
        except:
            frame_final = cv2.resize(frame, (ORIG_COLS, ORIG_ROWS), interpolation=cv2.INTER_AREA)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if keyframe == False:
            text = "NORMAL"
            color = (0,255,255)
        elif keyframe == True:
            # we have a key frame to save and process
            text = "KEY FRAME"
            color = (255, 0, 0)
            ## crop image by current rectangle value
            save_thread = threading.Thread(target=image_save, args=[frame_temp, 
                                                                    image_count, 
                                                                    start_point_row, 
                                                                    end_point_row, 
                                                                    start_point_column, 
                                                                    end_point_column])
            save_thread.start()
        cooldown_text = "cooldown" if cooldown==True else "active"
        cooldown_color = (255,0,0) if cooldown==True else (0,255,255)
        cv2.putText(frame_final, 
                    text, 
                    (50, 50), 
                    font, 1, 
                    color, 
                    2, 
                    cv2.LINE_4)
        cv2.putText(frame_final,
                    str(className),
                    (50, 100),
                    font, 1,
                    color,
                    2,
                    cv2.LINE_4)
        if saliency_map is not None:
            cv2.imshow('saliency map', saliency_map + 50)
        cv2.imshow('frame', frame_final)
        cv2.waitKey(1) & 0xFF

    
    vid.release()
    cv2.destroyAllWindows()
    

if __name__=="__main__":
    main()
import matplotlib.pyplot as plt
from utils import *
## utils imports most necessary libraries

NUM = 30

def main():
    for i in range(2,10,1):
        key_frames(i)

def key_frames(q):
    images = []
    vidcap = cv2.VideoCapture('movie.mov')
    success,image = vidcap.read()
    count = 0
    while success:
        if count % NUM == 0:
            #print(f"frame {count}")
            images.append(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (100,100), interpolation=cv2.INTER_AREA))
        success, image = vidcap.read()
        count += 1

    the_big_image = np.array([])

    total_entropy = [0,0]
    total_entropy2 = [0,0]

    extremas = []

    for idx, i in enumerate(images):
        hist = np.array(histogram(i, 0, 255, 256))
        hist = hist/sum(hist)
        entropy = calculate_entropy_tsallis(hist, q) 
        if ((total_entropy[-1] > total_entropy[-2]) and (total_entropy[-1] > entropy)) or\
                ((total_entropy[-1] < total_entropy[-2] and total_entropy[-1] < entropy)):
            extremas.append({"frame":(idx-1)*NUM,"image":images[idx-1],"entropy":total_entropy[-1]})
        total_entropy.append(entropy)

        the_big_image = np.append(the_big_image, i)
        hist2 = np.array(histogram(the_big_image, 0, 255, 256))
        hist2 = hist2/sum(hist2)
        entropy2 = calculate_entropy_tsallis(hist2, q)
        total_entropy2.append(entropy2)

    total_entropy3 = []
    for idx, i in enumerate(total_entropy2):
        if (idx > 0):
            total_entropy3.append(total_entropy2[idx] - total_entropy2[idx-1])

    fig, axs = plt.subplots(2)
    axs[0].plot(total_entropy[2:])
    for i in extremas:
        axs[0].plot(i["frame"]/NUM, i['entropy'], '.r')
    axs[1].plot(total_entropy3[2:])

    plt.show()

################################################################
################################################################

if __name__=="__main__":
    main()
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as newfft
import cv2
from VideoGet import VideoGet

filtersize = 7
threshhold = 15
fps = 30


deltas = []
threshs= []

deltas1 = []
deltas2 = []
deltas3 = []

def threadVideoGet(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    video_getter = VideoGet(source).start()
    start_time = time.time()
    lastFrame = None
    currFrame = 1

    rolling_frames = np.zeros((30, video_getter.height, video_getter.width), dtype=np.uint8)

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        # wait until its time
        if ((time.time() - start_time) < (currFrame * 1 / fps)):
            continue


        if (time.time() - start_time) > 20:
            break

        print(currFrame/(time.time() - start_time))
        currFrame += 1
        frame = video_getter.frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (filtersize, filtersize), 0)

        # if the first frame is None, initialize it
        if lastFrame is None:
            lastFrame = gray
            continue
        
        # use rolling frames here

        np.roll(rolling_frames, 1, axis=0)  # toooooooooo slow
        rolling_frames[0,:,:] = gray
        lastFrame= gray

        tmp = rolling_frames[1,:,:]
        curr_averaged_frame = np.mean(rolling_frames[(0,1,2),:,:], axis=0).astype(np.uint8)
        delta1 = cv2.absdiff(lastFrame, gray)
        delta2 = cv2.absdiff(np.mean(rolling_frames[(14,15,16),:,:], axis=0).astype(np.uint8), curr_averaged_frame)
        delta3 = cv2.absdiff(rolling_frames[29,:,:], curr_averaged_frame)

        cv2.imshow("delta1", delta1)
        cv2.imshow("delta2", delta2)
        cv2.imshow("delta3", delta3)

        deltas1.append(np.sum(delta1))
        deltas2.append(np.sum(delta2))
        deltas3.append(np.sum(delta3))
        

        # thresh = cv2.threshold(frameDelta, threshhold, 255, cv2.THRESH_BINARY)[1]
        # thresh = cv2.dilate(thresh, None, iterations=2)

        # threshs.append(np.sum(thresh))
        # deltas.append(np.sum(frameDelta))
        
        


    video_getter.stream.release()
    cv2.destroyAllWindows()


threadVideoGet()

# fftfreq = newfft.fftfreq(len(deltas),1/29)
# fft = newfft.fft(threshs)

# plt.plot(fftfreq[1:len(deltas)//2], fft[1:len(deltas)//2])
# plt.show()


plt.plot(deltas1)
plt.plot(deltas2)
plt.plot(deltas3)
# plt.plot(threshs)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as newfft
import cv2

filtersize = 7
threshhold = 15

vs = cv2.VideoCapture(0)
frame = None
firstFrame = None
lastFrame = None

deltas = []
threshs= []


while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    frame = vs.read()[1]

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break
    # resize the frame, convert it to grayscale, and blur it

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (filtersize, filtersize), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        lastFrame = gray
        continue
    
    frameDelta = cv2.absdiff(lastFrame, gray)
    lastFrame=gray
    thresh = cv2.threshold(frameDelta, threshhold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)

    threshs.append(np.sum(thresh))
    deltas.append(np.sum(frameDelta))

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()

#fft

fftfreq = newfft.fftfreq(len(deltas),1/29)
fft = newfft.fft(threshs)

plt.plot(fftfreq[1:len(deltas)//2], fft[1:len(deltas)//2])
plt.show()


plt.plot(deltas)
plt.plot(threshs)
plt.show()
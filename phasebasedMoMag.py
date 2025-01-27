from __future__ import print_function
from builtins import next
from builtins import range
from perceptual.filterbank import *
import cProfile

import cv2

# determine what OpenCV version we are using
try:
    import cv2.cv as cv
    USE_CV2 = True
except ImportError:
    # OpenCV 3.x does not have cv2.cv submodule
    USE_CV2 = False
    
import sys
import numpy as np

from pyr2arr import Pyramid2arr
from temporal_filters import IdealFilterWindowed, ButterBandpassFilter


def phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq):

    # initialize the steerable complex pyramid
    steer = Steerable(5)
    pyArr = Pyramid2arr(steer)

    print("Reading:", vidFname, end=' ')

    # get vid properties
    vidReader = cv2.VideoCapture(vidFname)
    if USE_CV2:
        # OpenCV 2.x interface
        vidFrames = int(vidReader.get(cv.CV_CAP_PROP_FRAME_COUNT))    
        width = int(vidReader.get(cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(vidReader.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        fps = int(vidReader.get(cv.CV_CAP_PROP_FPS))
        func_fourcc = cv.CV_FOURCC
    else:
        # OpenCV 3.x interface
        vidFrames = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))    
        width = int(vidReader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vidReader.get(cv2.CAP_PROP_FPS))
        func_fourcc = cv2.VideoWriter_fourcc

    if np.isnan(fps):
        fps = 30

    print(' %d frames' % vidFrames, end=' ')
    print(' (%d x %d)' % (width, height), end=' ')
    print(' FPS:%d' % fps)

    # video Writer
    fourcc = func_fourcc('M', 'J', 'P', 'G')
    vidWriter = cv2.VideoWriter(vidFnameOut, fourcc, int(fps), (width,height), 1)
    print('Writing:', vidFnameOut)

    # how many frames
    nrFrames = min(vidFrames, maxFrames)

    # read video
    #print steer.height, steer.nbands

    # setup temporal filter
    filter = IdealFilterWindowed(windowSize, lowFreq, highFreq, fps=fpsForBandPass, outfun=lambda x: x[0])
    #filter = ButterBandpassFilter(1, lowFreq, highFreq, fps=fpsForBandPass)

    print('FrameNr:', end=' ') 
    for frameNr in range( nrFrames + windowSize ):
        print(frameNr, end=' ')
        sys.stdout.flush() 

        if frameNr < nrFrames:
            # read frame
            _, im = vidReader.read()
               
            if im is None:
                # if unexpected, quit
                break
			
            # convert to gray image
            if len(im.shape) > 2:
                grayIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            else:
                # already a grayscale image?
                grayIm = im

            # get coeffs for pyramid
            coeff = steer.buildSCFpyr(grayIm)

            # add image pyramid to video array
            # NOTE: on first frame, this will init rotating array to store the pyramid coeffs                 
            arr = pyArr.p2a(coeff)


            phases = np.angle(arr)
            #print(coeff.shape, coeff, arr.shape, arr, phases.shape, phases)

            # add to temporal filter
            # filter.update([phases])
            filter.update([grayIm.flatten()])

            # try to get filtered output to continue            
            try:
                filteredPhases = next(filter)
            except StopIteration:
                continue

            print('*', end=' ')
            
            # motion magnification
            magnifiedPhases = (phases - filteredPhases) + filteredPhases*factor
            
            # create new array
            newArr = np.abs(arr) * np.exp(magnifiedPhases * 1j)

            # create pyramid coeffs     
            newCoeff = pyArr.a2p(newArr)
            
            # reconstruct pyramid
            out = steer.reconSCFpyr(newCoeff)

            # clip values out of range
            out[out>255] = 255
            out[out<0] = 0
            
            # make a RGB image
            rgbIm = np.empty( (out.shape[0], out.shape[1], 3 ) )
            rgbIm[:,:,0] = out
            rgbIm[:,:,1] = out
            rgbIm[:,:,2] = out
            
            #write to disk
            res = cv2.convertScaleAbs(rgbIm)
            vidWriter.write(res)

    # free the video reader/writer
    vidReader.release()
    vidWriter.release()   


################# main script

vidFname = 'media/test.avi'
#vidFname = 'media/WIN_20151208_17_11_27_Pro.mp4.normalized.avi'
#vidFname = 'media/embryos01_30s.mp4'
#vidFname = 'media/guitar.mp4'

# maximum nr of frames to process
maxFrames = 60000
# the size of the sliding window
windowSize = 30
# the magnifaction factor
factor = 20
# the fps used for the bandpass
fpsForBandPass = -1 # use -1 for input video fps
# low ideal filter
lowFreq = 0.2
# high ideal filter
highFreq = 2
# output video filename
vidFnameOut = vidFname + '-Mag%dIdeal-lo%d-hi%d.avi' % (factor, lowFreq, highFreq)

phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq)


#cProfile.run('phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq)')

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 13:01:08 2018
@author: Jason Ioffe
"""

import numpy as np
import cv2

FRAME_CAPTION = 'OpenCV - Webcam Capture'

# This is all it takes to start capturing
# from the primary camera with OpenCV
print('Starting camera capture...')
cap = cv2.VideoCapture(0)
print('Camera capture started!')

while(True):
    # For each frame, capture the image contents
    # ret contains a boolean: true means the capture read was successful
    # frame will contain the actual pixel data as a numpy array in BGR format
    ret, frame = cap.read()
    cv2.imshow(FRAME_CAPTION, frame)
    
    #passing 1 means that this will be non-blocking, so this loop continues
    #27 is the ESC key
    if cv2.waitKey(1) & 0xff == 27:
        break

# When everything done, release the capture
cap.release()
print('Camera capture released')
cv2.destroyAllWindows()
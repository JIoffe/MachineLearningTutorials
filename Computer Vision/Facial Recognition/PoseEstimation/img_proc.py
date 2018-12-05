# -*- coding: utf-8 -*-
"""
Utilities for face pose estimation using CV2 and dlib
Created on Wed Oct 17 17:26:30 2018

@author: Jason Ioffe
"""

from threading import Thread
import numpy as np
import cv2
import dlib

FRAME_CAPTION = 'OpenCV and dlib - Filtered Face Landmarks'
font = cv2.FONT_HERSHEY_SIMPLEX

class FaceDetectionThread(Thread):
    def __init__(self, caption, show_cam=True):
        Thread.__init__(self)
        self.continue_processing = True
        self.caption = caption
        self.show_cam = show_cam
    
    def run(self):
        detector = dlib.get_frontal_face_detector()

        print('Starting camera capture...')
        cap = cv2.VideoCapture(0)
        print('Camera capture started!')

        while(self.continue_processing):
            ret, frame = cap.read()
        
            # detect faces in the grayscale image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)
        
            if self.show_cam:
                cv2.imshow(FRAME_CAPTION, frame)
        
            key = cv2.waitKey(1) & 0xff
            if(key == 27):
                self.continue_processing = False
    
        cap.release()
        print('Camera capture released')
        
        cv2.destroyAllWindows()
         
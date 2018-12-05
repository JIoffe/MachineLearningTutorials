# -*- coding: utf-8 -*-
"""
Bare-bones example of displaying detected
face bounds and landmarks using dlib.
@author: Jason Ioffe
"""

import numpy as np
import cv2
import dlib

FRAME_CAPTION = 'OpenCV and dlib - Face Feature Detection'
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

FACE_BOUNDS_COLOR = (255,255,0)
FACE_LANDMARK_COLOR = (255,255,255)

font = cv2.FONT_HERSHEY_SIMPLEX

# Use Dlib for face detection and landmark prediction
# this is greatly simplified compared to OpenCV
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

print('Starting camera capture...')
cap = cv2.VideoCapture(0)
print('Camera capture started!')

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    faces = detector(gray, 0)
    for (i, rect) in enumerate(faces):
        min_x = rect.left()
        min_y = rect.top()
        max_x = rect.right()
        max_y = rect.bottom()
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), FACE_BOUNDS_COLOR, 2)
        
        shape = predictor(gray, rect)
                
        # We can assume this range because we're using a 68-point training set
        for j in range(0,68):
            part = shape.part(j)
            coords = (part.x, part.y)
            cv2.circle(frame, coords, 1, FACE_LANDMARK_COLOR, -1)
            
        cv2.putText(frame, 'Face {0}'.format(i+1), (min_x, min_y - 3), font, 0.5, FACE_BOUNDS_COLOR, 1, cv2.LINE_AA)
    
    cv2.imshow(FRAME_CAPTION, frame)
    
    key = cv2.waitKey(1) & 0xff
    if(key == 27):
        break
    
cap.release()
print('Camera capture released')

cv2.destroyAllWindows()
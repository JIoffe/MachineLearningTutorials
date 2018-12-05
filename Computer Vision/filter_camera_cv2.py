# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:47:12 2018

@author: Jason Ioffe
"""

import numpy as np
import cv2

FRAME_CAPTION = 'OpenCV - Camera with Filters'

# For Menu
font = cv2.FONT_HERSHEY_SIMPLEX
selected_color = (255,0,0)
non_selected_color = (255,255,255)

available_filters = ['None', 'B&W', 'Blur', 'Threshold', 'Edges']
filter_mode = 1

# Func definitions
def draw_menu(img):
    x = 10
    y = 20
    for i, value in enumerate(available_filters, 1):
        if(i == filter_mode):
            color = selected_color
        else:
            color = non_selected_color
            
        # Draw text with shadow first
        cv2.putText(img, '{0}) {1}'.format(i, value), (x + 2, y * i + 2), font, 0.5, (0,0,0), 1, cv2.LINE_AA)            
        cv2.putText(img, '{0}) {1}'.format(i, value), (x, y * i), font, 0.5, color, 1, cv2.LINE_AA)

def blur(frame):
    downsample_factor = 4
    kernal_size = 15
    
    original_size = img.shape[:2][::-1]
    downscaled_size = tuple((int)(max(ti/downsample_factor,1)) for ti in original_size)
    
    blurred_img = cv2.resize(frame, downscaled_size, interpolation = cv2.INTER_NEAREST)
    blurred_img = cv2.GaussianBlur(blurred_img,(15, 15),0)
    blurred_img = cv2.resize(blurred_img, original_size, interpolation = cv2.INTER_CUBIC)
    return blurred_img

def apply_filter(frame):
    if(filter_mode == 2):
        # Convert to grey and then RGB so we keep grayscale values but can still draw
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    elif(filter_mode == 3):
        return blur(frame)
    elif(filter_mode == 4):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY_INV)[1]
    elif(filter_mode == 5):
        return cv2.Canny(frame,100,255)
    else:
        return frame

print('Starting camera capture...')
cap = cv2.VideoCapture(0)
print('Camera capture started!')

while(True):
    ret, frame = cap.read()
    
    img = apply_filter(frame)
    draw_menu(img)
    cv2.imshow(FRAME_CAPTION, img)
    
    key = cv2.waitKey(1) & 0xff
    if(key == 27):
        break
    elif(key >= 49 and key <= 53):
        filter_mode = key - 48

# When everything done, release the capture
cap.release()
print('Camera capture released')
cv2.destroyAllWindows()
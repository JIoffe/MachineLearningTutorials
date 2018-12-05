# -*- coding: utf-8 -*-
"""
Simple example of using haar cascades to detect faces with OpenCV


@author: Jason Ioffe
"""
import numpy as np
import cv2

FRAME_CAPTION = 'OpenCV - Haar Cascades for Face Detection'

font = cv2.FONT_HERSHEY_SIMPLEX
selected_color = (255,0,0)
non_selected_color = (255,255,255)
face_rect_color = (255,200,200)
eye_rect_color = (200,255,200)
ui_shadow = (0,0,0)

# Haar cascades can be trained against just about anything
# OpenCV comes with a number of pre-built cascades for faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

available_filters = ['None', 'Blur', 'Cute Eyes', 'Face Swap']
filter_mode = 1

# Alpha channel images behave a little oddly...
def load_alpha_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    B, G, R, A = cv2.split(img)
    B = cv2.bitwise_and(B, B, mask=A)
    G = cv2.bitwise_and(G, G, mask=A)
    R = cv2.bitwise_and(R, R, mask=A)
    return cv2.merge([B, G, R])

cute_eye = load_alpha_image('eye_cute.png')

#def bitblt(src, dest, x, y):
#    rows,cols,channels = src.shape
#    roi = dest[x:x+rows, y:y+cols ]
def render_menu(img):
    x = 10
    y = 20
    for i, value in enumerate(available_filters, 1):
        if(i == filter_mode):
            color = selected_color
        else:
            color = non_selected_color
            
        # Draw text with shadow first
        cv2.putText(img, '{0}) {1}'.format(i, value), (x + 1, y * i + 1), font, 0.5, (0,0,0), 1, cv2.LINE_AA)            
        cv2.putText(img, '{0}) {1}'.format(i, value), (x, y * i), font, 0.5, color, 1, cv2.LINE_AA)
    
def label_faces(img, img_grey, faces):
    output = img.copy()
    
    for i, (x,y,w,h) in enumerate(faces, 1):
        caption = 'Face {}'.format(i)
        cv2.rectangle(output,(x,y),(x+w,y+h),face_rect_color,2)
        cv2.putText(output, caption, (x + 1, y - 9), font, 0.5, ui_shadow, 1, cv2.LINE_AA)
        cv2.putText(output, caption, (x, y - 10), font, 0.5, face_rect_color, 1, cv2.LINE_AA)
        
        # Pull the region of interest to narrow down the search for eyes
        roi_gray = img_grey[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.circle(output, (int(x+ex+ew/2), int(y+ey+eh/2)), int((ew+eh)/3), eye_rect_color, 1)
          
    return output

def blur_faces(img, img_grey, faces):
    output = img.copy()
    for (x,y,w,h) in faces:
        roi = img[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(roi,(23,23),45)
        output[y:y+blurred_face.shape[0], x:x+blurred_face.shape[1]] = blurred_face
        
    return output

def draw_cute_eyes(img, img_grey, faces):
    output = img.copy()
    
    for (x,y,w,h) in faces:
        # Pull the region of interest to narrow down the search for eyes
        roi_gray = img_grey[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            if ew < 1 or eh < 1:
                break

            destX = ex+x
            destY = ey+y
            
            roi = output[destY:destY+eh, destX:destX+ew]
            
            # Resize the cute_eye to more-or-less
            # match the detected eye size
            resized_eye = cv2.resize(cute_eye,(ew, eh), interpolation = cv2.INTER_CUBIC)
            
            roi = cv2.add(roi, resized_eye)
            output[destY:destY+eh, destX:destX+ew] = roi[0:eh, 0:ew]  
            
    return output

def face_swap(img, img_grey, faces):
    output = img.copy()
    
    if len(faces) == 2:
        (x1,y1,w1,h1) = faces[0]
        (x2,y2,w2,h2) = faces[1]
        
        roi1 = output[y1:y1+h1, x1:x1+w1]
        roi2 = output[y2:y2+h2, x2:x2+w2]
        
        roi1 = cv2.resize(roi1,(w2, h2), interpolation = cv2.INTER_CUBIC)
        roi2 = cv2.resize(roi2,(w1, h1), interpolation = cv2.INTER_CUBIC)
        
        output[y1:y1+h1, x1:x1+w1] = roi2[0:h1, 0:w1]
        output[y2:y2+h2, x2:x2+w2] = roi1[0:h2, 0:w2]
    else:
        txt = 'Face swap requires exactly 2 faces!'
        cv2.putText(output, txt, (20, 200), font, 0.5, ui_shadow, 1, cv2.LINE_AA)
        cv2.putText(output, txt, (21, 201), font, 0.5, face_rect_color, 1, cv2.LINE_AA)
        
    return output

filter_funcs = [label_faces, blur_faces, draw_cute_eyes, face_swap]

print('Starting camera capture...')
cap = cv2.VideoCapture(0)
print('Camera capture started!')

while(True):
    ret, frame = cap.read()
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    faces = face_cascade.detectMultiScale(frame_grey, 1.3, 5)
    output = filter_funcs[filter_mode-1](frame, frame_grey, faces)
    
    render_menu(output)
    
    cv2.imshow(FRAME_CAPTION, output)
    
    key = cv2.waitKey(1) & 0xff
    if(key == 27):
        break
    elif(key >= 49 and key <= 53):
        filter_mode = key - 48
        print('Setting filter mode to {}: {}'.format(filter_mode, available_filters[filter_mode-1]))
    
# When everything done, release the capture
cap.release()
print('Camera capture released')
cv2.destroyAllWindows()
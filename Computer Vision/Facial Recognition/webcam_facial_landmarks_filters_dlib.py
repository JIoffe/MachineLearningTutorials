# -*- coding: utf-8 -*-
"""
Simple toy examples of filters that can be done
with facial landmark detection using opencv and dlib
@author: Jason Ioffe
"""

import numpy as np
import cv2
import dlib

FRAME_CAPTION = 'OpenCV and dlib - Filtered Face Landmarks'
# This 68 point training set is nearly 100MB!
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

# Text options, eg. for menu
font = cv2.FONT_HERSHEY_SIMPLEX
selected_color = (255,0,0)
non_selected_color = (255,255,255)

filter_names = ['None', 'Blur', 'Unblur', 'Anime', 'Wizard', 'Greek Theatre']
filter_mode = 1

# Images to help with some of the filters
anime_eye_right = cv2.imread('eye_right.png')
anime_eye_right_mask = cv2.imread('eye_right_mask.png', cv2.IMREAD_GRAYSCALE)
stage_curtains = cv2.imread('stage-curtains-backdrop.jpg')

# The 68-pt training set for this face landmark shape predictor will always
# label specific landmarks at the same indices
def extract_range(shape, start, end, offset_x = 0, offset_y = 0):
    return np.matrix([[p.x + offset_x, p.y + offset_y] for p in shape.parts()[start:end]])

def extract_right_eye(shape, offset_x = 0, offset_y = 0):
    return extract_range(shape, 36, 42, offset_x, offset_y)

def extract_left_eye(shape, offset_x = 0, offset_y = 0):
    return extract_range(shape, 42, 48, offset_x, offset_y)
    
def extract_points(shape, offset_x = 0, offset_y = 0):
    # Apply an optional offset to account for ROI shifts
    return np.matrix([[p.x + offset_x, p.y + offset_y] for p in shape.parts()])

def get_convex_hull_mask(convex_hull, roi_shape):
    mask = np.zeros(roi_shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, convex_hull, (255,255,255))
    mask_inv = cv2.bitwise_not(mask)
    
    return (mask, mask_inv)
 
def get_shape_mask(shape, roi_shape, offset_x = 0, offset_y = 0):
    convex_hull = cv2.convexHull(extract_points(shape, offset_x, offset_y))    
    return get_convex_hull_mask(convex_hull, roi_shape)

def render_menu(img):
    x = 10
    y = 20
    for i, value in enumerate(filter_names, 1):
        if(i == filter_mode):
            color = selected_color
        else:
            color = non_selected_color
            
        # Draw text with shadow first
        cv2.putText(img, '{0}) {1}'.format(i, value), (x + 1, y * i + 1), font, 0.5, (0,0,0), 1, cv2.LINE_AA)            
        cv2.putText(img, '{0}) {1}'.format(i, value), (x, y * i), font, 0.5, color, 1, cv2.LINE_AA)

def no_op(img, gray, faces):
    return img

def blur_impl(img, downsample_factor=8, kernal_size=15):
    # Want to go crazy? Numpy uses (height, width) but cv2 expects (width, height)!
    original_size = img.shape[:2][::-1]
    blur = img.copy()
    
    # Downscale image before blur and upsample back to get a strong blur
    downscaled_size = tuple((int)(max(ti/downsample_factor,1)) for ti in original_size)
    
    blur = cv2.resize(blur, downscaled_size, interpolation = cv2.INTER_NEAREST)
    blur = cv2.GaussianBlur(blur,(kernal_size, kernal_size),0)
    blur = cv2.resize(blur, original_size, interpolation = cv2.INTER_CUBIC)
    return blur
    
def blur_face(img, gray, faces):
    output = img.copy()
    
    for rect in faces:
        min_x = rect.left()
        min_y = rect.top()
        max_x = rect.right()
        max_y = rect.bottom()
        
        # Blur the entire ROI around the face
        roi = output[min_y:max_y, min_x:max_x]
        blur = blur_impl(roi)
            
        # Create a mask around the facial landmarks
        # the convex hull has to be translated to account for the roi's location
        shape = predictor(gray, rect)
        (mask, mask_inv) = get_shape_mask(shape, roi.shape[:2], -min_x, -min_y)
        
        # Use bitwise ops to place the blurred image on top
        blur = cv2.bitwise_and(blur,blur,mask = mask)
        roi = cv2.bitwise_and(roi,roi,mask = mask_inv)
        roi = cv2.add(roi,blur)
#
        output[min_y:max_y, min_x:max_x] = roi
        
    return output

def unblur_face(img, gray, faces):
    output = blur_impl(img);
    
    for rect in faces:
        min_x = rect.left()
        min_y = rect.top()
        max_x = rect.right()
        max_y = rect.bottom()
        
        # Here it's reverse - take an roi of the original
        # and add to the blurred
        roi = img[min_y:max_y, min_x:max_x]
        bg = output[min_y:max_y, min_x:max_x]
            
        # Create a mask around the facial landmarks
        # the convex hull has to be translated to account for the roi's location
        shape = predictor(gray, rect)
        (mask, mask_inv) = get_shape_mask(shape, roi.shape[:2], -min_x, -min_y)
        
        # Use bitwise ops to place the in-focus face on top
        fg = cv2.bitwise_and(roi,roi,mask = mask)    
        bg = cv2.bitwise_and(bg,bg,mask = mask_inv)
        bg = cv2.add(bg,fg)

        output[min_y:max_y, min_x:max_x] = bg
        
    return output

# Because why not make red glowing eyes?
# Maybe 2.0 will have lens flares :)
def wizard(img, gray, faces):
    eye_color = (255, 255, 255)
    
    output = img.copy()
    glow = np.zeros(img.shape, np.uint8)
    for rect in faces:
        shape = predictor(gray, rect)
        
        left_eye = cv2.convexHull(extract_left_eye(shape))        
        cv2.fillConvexPoly(glow, left_eye, eye_color)
        
        right_eye = cv2.convexHull(extract_right_eye(shape))        
        cv2.fillConvexPoly(glow, right_eye, eye_color)
        
    # Mask the full glow and overlay to the output
    glow_gray = cv2.cvtColor(glow, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(glow_gray,0,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    opaque_glow = cv2.bitwise_and(glow,glow,mask = mask)
    output = cv2.bitwise_and(output,output,mask = mask_inv)
    output = cv2.add(output, opaque_glow)
        
    # But of course we need bloom
    bloom = blur_impl(glow, downsample_factor=2, kernal_size=15) * 2
    output = cv2.add(output, bloom)
    
    return output

def apply_transformed_overlay(src, dst, mask, cols, rows, pts_src, pts_dst):
    M = cv2.getAffineTransform(pts_src,pts_dst)
    overlay = cv2.warpAffine(src,M,(cols,rows))
    mask = cv2.warpAffine(mask,M,(cols,rows)) 
    mask_inv = cv2.bitwise_not(mask)
    
    overlay = cv2.bitwise_and(overlay,overlay,mask = mask)    
    dst = cv2.bitwise_and(dst,dst,mask = mask_inv)
    dst = cv2.add(dst, overlay)
    return dst

def apply_eye_overlay(eye, mask, output, cols, rows, pts_src, ref_a, ref_b, ref_c, ref_d):
    dst_a = [ref_b[0], ref_c[1]]
    dst_b = [ref_d[0], ref_c[1]]
    dst_c = [ref_b[0], ref_a[1]]
            
    pts_dst = np.float32([dst_a,dst_b,dst_c])
    
    roi = output[0:eye.shape[0], 0:eye.shape[1]]
    roi = apply_transformed_overlay(eye, roi, mask, cols, rows, pts_src, pts_dst)
    output[0:roi.shape[0], 0:roi.shape[1]] = roi
    
def approximate_face_color(img, rect, multiplier=1.5):
    # Crude, but not terrible
    # Get the mean of all the pixels on the vertical
    # center line
    min_x = rect.left()
    min_y = rect.top()
    max_x = rect.right()
    max_y = rect.bottom()
    mid_y = int((min_y + max_y) / 2)
    # Average the central pixels... probably silly
    color = np.sum(img[mid_y, min_x:max_x], axis=0) / (max_x - min_x) * multiplier
    return color.tolist()
    
# Probably the most elaborate one here!
# Has to even out the face and draw those cute anime eyes
# The easiest part of this is probably the nose
def anime_face(img, gray, faces):
    mouth_color = (200,200,0)
    outline_color = (20,20,20)
    
    output = img.copy()
    rows,cols,ch = img.shape
    
    for rect in faces:     
        skin_color = approximate_face_color(img, rect)
        
        shape = predictor(gray, rect)
        shape_pts = extract_points(shape)

        convex_hull = cv2.convexHull(shape_pts)
        cv2.fillConvexPoly(output, convex_hull, skin_color)
        cv2.polylines(output, [convex_hull], True, outline_color, thickness=2, lineType=cv2.LINE_AA)
        
        # Compute shadow color
        shadow_color = tuple((int)(max(ti / 2,1)) for ti in skin_color)
        
        # Draw nose :)
        nose_shadow = cv2.convexHull(extract_range(shape, 27, 32))
        cv2.fillConvexPoly(output, nose_shadow, shadow_color)
        
        # Draw mouth - outline last
        mouth = cv2.convexHull(extract_range(shape, 60, 68))
        cv2.fillConvexPoly(output, mouth, mouth_color)
        cv2.polylines(output, [mouth], True, outline_color, thickness=2, lineType=cv2.LINE_AA)
        
        # Open up a few windows to the soul using affine transformations
        # For large "anime" eyes we're going to stretch the eye midway down the cheek
        eye = cv2.resize(anime_eye_right, (cols,rows), interpolation = cv2.INTER_NEAREST)
        mask = cv2.resize(anime_eye_right_mask, (cols,rows), interpolation = cv2.INTER_NEAREST)   
        pts_src = np.float32([[0,0],[rows, 0],[0,cols]])    
        
        # R Eye - Left of screen
        ref_a = shape_pts[2].getA1()
        ref_b = shape_pts[17].getA1()
        ref_c = shape_pts[19].getA1()
        ref_d = shape_pts[21].getA1()
        apply_eye_overlay(eye, mask, output, cols, rows, pts_src, ref_a, ref_b, ref_c, ref_d)
        
        # L Eye - Left of screen
        ref_a = shape_pts[14].getA1()
        ref_b = shape_pts[26].getA1()
        ref_c = shape_pts[24].getA1()
        ref_d = shape_pts[22].getA1()
        apply_eye_overlay(eye, mask, output, cols, rows, pts_src, ref_a, ref_b, ref_c, ref_d)
        
    return output

def theatre_masks(img, gray, faces):
    mask_fill = (255,255,255)
    mask_gap = (0,0,0)
    
    rows,cols,ch = img.shape
    output = cv2.resize(stage_curtains, (cols,rows), interpolation = cv2.INTER_NEAREST)
    overlay = np.zeros(img.shape, np.uint8)
    
    for rect in faces:
        shape = predictor(gray, rect)
        shape_pts = extract_points(shape)
        convex_hull = cv2.convexHull(shape_pts)
        cv2.fillConvexPoly(overlay, convex_hull, mask_fill)
        
        left_eye = cv2.convexHull(extract_left_eye(shape))        
        cv2.fillConvexPoly(overlay, left_eye, mask_gap)
        
        right_eye = cv2.convexHull(extract_right_eye(shape)) 
        cv2.fillConvexPoly(overlay, right_eye, mask_gap)
        
        mouth = cv2.convexHull(extract_range(shape, 60, 68))
        cv2.fillConvexPoly(overlay, mouth, mask_gap)
        
    # The colors are pure white so just add without a mask
    output = cv2.add(output, overlay)
        
    return output
    

filter_funcs = [no_op, blur_face, unblur_face, anime_face, wizard, theatre_masks]


# Use Dlib for face detection and landmark prediction
# this is greatly simplified compared to OpenCV
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

print('Starting camera capture...')
cap = cv2.VideoCapture(0)
print('Camera capture started!')

while(True):
    ret, frame = cap.read()
    
    # detect faces in the grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    img = filter_funcs[filter_mode-1](frame, gray, faces)
    render_menu(img)
    cv2.imshow(FRAME_CAPTION, img)
    
    key = cv2.waitKey(1) & 0xff
    if(key == 27):
        break
    elif(key >= 49 and key <= 54):
        filter_mode = key - 48
        print('Setting filter mode to {}: {}'.format(filter_mode, filter_names[filter_mode-1]))
    
cap.release()
print('Camera capture released')

cv2.destroyAllWindows()


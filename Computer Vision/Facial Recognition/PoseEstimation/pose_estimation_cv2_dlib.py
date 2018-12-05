# -*- coding: utf-8 -*-
"""
Basic example of facial pose estimation using OpenCV and dlib

@author: Jason Ioffe
"""

import numpy as np
import cv2
import dlib

FRAME_CAPTION = 'OpenCV - Webcam Capture'
SHAPE_PREDICTOR_PATH = '../shape_predictor_68_face_landmarks.dat'

font = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_COLOR = (200,200,255)

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

BB_SIZE = 10
bounding_box = np.float32([[-BB_SIZE, BB_SIZE, BB_SIZE],
                           [-BB_SIZE, BB_SIZE, -BB_SIZE],
                           [BB_SIZE, BB_SIZE, -BB_SIZE],
                           [BB_SIZE, BB_SIZE, BB_SIZE],
                           [-BB_SIZE, -BB_SIZE, BB_SIZE],
                           [-BB_SIZE, -BB_SIZE, -BB_SIZE],
                           [BB_SIZE, -BB_SIZE, -BB_SIZE],
                           [BB_SIZE, -BB_SIZE, BB_SIZE]])

bb_lines = [[0,1],[1,2],[2,3],[0,3],
              [4,5],[5,6],[6,7],[4,7],
              [0,4],[1,5],[2,6],[3,7]]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

def get_euler_angles(rotation_vec):
    # The rotation vec provided by SolvePnP is in Rodrigues form
    # and must be converted into Euler angles
    rotation_mat, jacobian = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    
    return euler_angles
    
#def predict_pose(object_points, image_points, cam_matrix, dist_coeffs):
#    ret, rotation_vec, translation_vec = cv2.solvePnP(object_points, image_points, cam_matrix, dist_coeffs)
    
    
print('Starting camera capture...')
cap = cv2.VideoCapture(0)
print('Camera capture started!')

while(True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    for rect in faces:
        shape = predictor(frame, rect)
        shape = np.float32([[p.x, p.y] for p in shape.parts()])
        
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                        shape[39], shape[42], shape[45], shape[31], shape[35],
                        shape[48], shape[54], shape[57], shape[8]])

        ret, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
        projected_bb, jacobian = cv2.projectPoints(bounding_box, rotation_vec, translation_vec, 
                                                   cam_matrix, dist_coeffs)

        t = projected_bb
        projected_bb = projected_bb.reshape(8, 2)
        
        for i1, i2 in bb_lines:
            cv2.line(frame, tuple(projected_bb[i1]), tuple(projected_bb[i2]), OVERLAY_COLOR)
        
        # From the rotation vector, we can approximate the euler angles of the rotation
        euler_angles = get_euler_angles(rotation_vec)
        euler_angles = euler_angles.flatten()
        
        x = rect.left()
        y = rect.top() - 5
        cv2.putText(frame, 'Yaw: {0:0.1f} Pitch: {1:0.1f} Roll: {2:0.1f}'.format(euler_angles[1], euler_angles[0], euler_angles[2]), (x, y), font, 0.5, OVERLAY_COLOR, 1, cv2.LINE_AA)    

    cv2.imshow(FRAME_CAPTION, frame)
    
    #passing 1 means that this will be non-blocking, so this loop continues
    #27 is the ESC key
    if cv2.waitKey(1) & 0xff == 27:
        break

# When everything done, release the capture
cap.release()
print('Camera capture released')
cv2.destroyAllWindows()
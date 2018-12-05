# -*- coding: utf-8 -*-
"""
Encapsulates the face detection and webcam rendering side of this app

@author: Jason Ioffe
"""

import numpy as np
import cv2
import dlib

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

face_rect_color = (255,200,200)

class WebcamFaceDetector():
    def __init__(self, caption, predictor, show_cam=True,highlight_faces=True,mirror_x=False,obscure_feed=False):
        self.caption = caption
        self.show_cam = show_cam
        self.highlight_faces = highlight_faces
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = predictor
        self.mirror_x = mirror_x
        self.obscure_feed = obscure_feed
        self.highlight_landmarks = True

    def begin_capture(self):
        print('Starting camera capture...')
        self.cap = cv2.VideoCapture(0)
        print('Camera capture started!')
        
    def release_capture(self):
        self.cap.release()
        print('Camera capture released')    
        cv2.destroyAllWindows()
        
    def capture_faces(self):
        ret, frame = self.cap.read()
        
        if self.mirror_x:
            frame = cv2.flip(frame, 1)
        
        # detect faces in the grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        # Sort from left to right
        faces = list(faces)
        faces.sort(key=lambda rect: rect.left())
        
        capture_results = []
        
        if self.obscure_feed:
            # Blur and remove RG channels
            original_size = frame.shape[:2][::-1]
            downscaled_size = tuple((int)(max(ti/12,1)) for ti in original_size)

            frame = cv2.resize(frame, downscaled_size, interpolation = cv2.INTER_NEAREST)
            frame[:, :, 1:3] = 0
            frame = cv2.GaussianBlur(frame,(15, 15),0)
            frame = cv2.resize(frame, original_size, interpolation = cv2.INTER_CUBIC)
        
        for rect in faces:
            min_x = rect.left()
            min_y = rect.top()
            max_x = rect.right()
            max_y = rect.bottom()
            
            if self.highlight_faces:
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), face_rect_color, 2)
            
            shape = self.predictor(gray, rect)
            shape = np.float32([[p.x, p.y] for p in shape.parts()])
            
            if self.highlight_landmarks:
                for pt in shape:
                    cv2.circle(frame, tuple(pt), 1, (255,200,200), -1)
        
            image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

            ret, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
            
            # This is almost magical, but we need a 4x4 matrix
            # That we can pass through OpenGL to a shader
            # Also I notice that the Yaw, roll and x translation each need to be reversed for OpenGL
            rotation_vec[1] = -rotation_vec[1]
            rotation_vec[2] = -rotation_vec[2]
            R, jacobian = cv2.Rodrigues(rotation_vec)
          
            view_matrix = np.float32([[R[0][0], R[0][1], R[0][2], -translation_vec[0]],
                        [R[1][0], R[1][1], R[1][2], translation_vec[1]],
                        [R[2][0], R[2][1], R[2][2], translation_vec[2]],
                        [0.0, 0.0, 0.0, 1.0]])
    
            # Note that there is no row for scale. The object will scale onscreen as a
            # function of the projection matrix (which should match the camera!)


            capture_results.append([view_matrix, shape])
        
        if self.show_cam:
            cv2.imshow(self.caption, frame)
            
        return capture_results, frame.shape[:-1]
        
    
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:02:39 2018

@author: Jason Ioffe
"""
from threading import Thread
import time
import sys
import numpy as np
import cv2
import dlib

# Local to this project 
from rendering.pumpkin_face_pose_renderer import PumpkinFacePoseRenderer
from img_proc import FaceDetectionThread

SHAPE_PREDICTOR_PATH = '../shape_predictor_68_face_landmarks.dat'
CAMERA_FRAME_CAPTION = 'Face Pose Estimation - Camera Feed'
GL_FRAME_CAPTION = 'Face Pose Estimation with 3D Mesh'
GL_FRAME_WIDTH, GL_FRAME_HEIGHT = 1024, 768

predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
renderer = PumpkinFacePoseRenderer(GL_FRAME_WIDTH, GL_FRAME_HEIGHT)

#class RenderingThread(Thread):
#    def __init__(self):
#        Thread.__init__(self)
#
#    def run(self):
#        models = {
#            'PKN': './models/pumpkin.obj',
#            'CUT': './models/pumpkin_carving.obj'
#        }
#        
#        images = {
#            'BG': ['./images/moonlit-forest.png', True],
#            'PKN': ['./images/pumpkin.png', False],
#            'PKN-N': ['./images/pumpkin_noise.png', False],
#        }
#        
#        shader_paths = {
#            'BG': ['./shaders/fsq.vshader', './shaders/fsq.fshader'],
#            'PKN': ['./shaders/pumpkin.vshader', './shaders/pumpkin.fshader'],
#            'CUT': ['./shaders/pumpkin_cutaway.vshader', './shaders/pumpkin_cutaway.fshader']
#        }
#        
#        renderer.init(GL_FRAME_CAPTION, models, images, shader_paths)
#        print('Starting rendering thread...')
#        renderer.start()
      
def main():
    models = {
        'PKN': './models/pumpkin_facemap2.obj',
        'PIN': './models/pineapple.obj'
    }
    
    images = {
        'BG': ['./images/moonlit-forest.png', True],
        'PKN': ['./images/pumpkin_facemap.png', False],
        'PKN-N': ['./images/pumpkin_facemap_displace.png', False],
        'PIN': ['./images/pineapple.png', False],
        'PIN-N': ['./images/pineapple_bump.png', False],
    }
    
#    models = {
#        'PKN': './models/pineapple.obj',
#        'CUT': './models/pumpkin_carving3.obj'
#    }
#    
#    images = {
#        'BG': ['./images/moonlit-forest.png', True],
#        'PKN': ['./images/pineapple.png', False],
#        'PKN-N': ['./images/pineapple_bump.png', False],
#    }
    
    shader_paths = {
        'BG': ['./shaders/fsq.vshader', './shaders/fsq.fshader'],
        'PKN': ['./shaders/pumpkin.vshader', './shaders/pumpkin.fshader'],
        'CUT': ['./shaders/pumpkin_cutaway.vshader', './shaders/pumpkin_cutaway.fshader']
    }
    
    renderer.init(GL_FRAME_CAPTION, predictor, models, images, shader_paths)
    print('Starting rendering thread...')
    renderer.start()
#    rendering_thread = RenderingThread()
#    img_proc_thread = FaceDetectionThread(CAMERA_FRAME_CAPTION, show_cam=False)
#    
#    img_proc_thread.start()
#    rendering_thread.start()
    
main()
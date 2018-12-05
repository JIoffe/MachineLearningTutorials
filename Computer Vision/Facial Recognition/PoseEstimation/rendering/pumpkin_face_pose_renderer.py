# -*- coding: utf-8 -*-
"""
Rendering system for the Face to pumpkin mesh example
Not crazy about how GLUT handles things and might refactor in the future
@author: Jason Ioffe
"""

import math
import time
import numpy as np

import cv2

from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pyrr
from pyrr import Matrix44, Vector4, Vector3, Quaternion

# From local project
from .obj_loader import ObjLoader
from .textures import TextureLoader
from .geometry import BatchedGeometry, FullScreenQuad
from .webcam_capture import WebcamFaceDetector

class PumpkinFacePoseRenderer():
    def __init__(self, w, h):
        self.frame_width = w
        self.frame_height = h
        self.last_time = time.time()
        self.rot = 0
        return
    
    def __compile_shader_program(self, vshader, fshader):
        VERTEX_SHADER = shaders.compileShader(vshader, GL_VERTEX_SHADER)
        FRAGMENT_SHADER = shaders.compileShader(fshader, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

    def __compile_shader_from_disk(self, vshader_path, fshader_path):
        with open(vshader_path, "r") as vert_file:
            vshader = vert_file.read()
        with open(fshader_path, "r") as frag_file:
            fshader = frag_file.read()
        return self.__compile_shader_program(vshader, fshader)

    def __build_shaders_from_map(self, shader_paths):
        self.shaders = {}
        self.uniform_locations = {}
        for key, value in shader_paths.items():
            shader = self.__compile_shader_from_disk(value[0], value[1])
            self.shaders[key] = shader
            self.uniform_locations[key] = {
                'mvp': glGetUniformLocation(shader, 'mvp'),
                'mv': glGetUniformLocation(shader, 'mv'),
                'normal_matrix': glGetUniformLocation(shader, 'normal_matrix'),
                'diffuse': glGetUniformLocation(shader, 'diffuse'),
                'bump': glGetUniformLocation(shader, 'bump')
            }

    def __build_geometry_from_map(self, models):
        self.geometries = {}
        obj_loader = ObjLoader()
        
        for key, value in models.items():
            vertices, tex_coords, normals, indices = obj_loader.load(value, flip_faces=False)
            self.geometries[key] = BatchedGeometry(vertices, tex_coords, normals, indices)
            
    def __build_images_from_map(self, images):
        self.textures = {}
        tex_loader = TextureLoader()
        
        for key, value in images.items():
            tex_id = tex_loader.load(value[0], flipY=value[1])
            self.textures[key] = tex_id
        
            
    def __create_perspective_matrix(self, fov, ratio, near, far):
        mat = pyrr.matrix44.create_perspective_projection_matrix(fov, ratio, near, far)
        return np.transpose(mat)
            
    def init(self, rendering_caption, predictor, models={}, images={}, shader_paths={}):
        self.face_detector = WebcamFaceDetector(rendering_caption, predictor, show_cam=True, highlight_faces=True, mirror_x=True,obscure_feed=True)
        self.face_detector.begin_capture()
        
        # Starts the GL context - must happen first!
        glutInit()
        glutInitDisplayMode(GLUT_RGBA)
        glutInitWindowSize(self.frame_width, self.frame_height)
        glutInitWindowPosition(200, 200)
        self.window = glutCreateWindow(rendering_caption)
        
        self.proj_matrix = self.__create_perspective_matrix(45.0, self.frame_width/self.frame_height, 0.1, 500.0)
        
        self.__build_shaders_from_map(shader_paths)
        self.__build_images_from_map(images)
        self.__build_geometry_from_map(models)
        self.fsq = FullScreenQuad()
        
        # Always enable the vertex attributes in this example
        glEnableVertexAttribArray(0)
        
        # Cull backfaces by default
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        
    def on_draw(self):
        capture_results, capture_shape = self.face_detector.capture_faces()
        
        current_time = time.time()
        since_last_frame = current_time - self.last_time
        self.last_time = current_time

        # Draw background each frame - no need to clear color
        glClear(GL_DEPTH_BUFFER_BIT)
        
        glDepthMask(GL_FALSE)
        shaders.glUseProgram(self.shaders['BG'])
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.textures['BG'])
        glUniform1i(self.uniform_locations['BG']['diffuse'], 0); 
        self.fsq.draw()
              
        glDepthMask(GL_TRUE)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)  

        shaders.glUseProgram(self.shaders['PKN'])
        uniform_locations = self.uniform_locations['PKN']
        i = 0
        
        for view_matrix, shape in capture_results:
            mvp_matrix = np.transpose(np.matmul(self.proj_matrix, view_matrix))
            normal_matrix = np.linalg.inv(view_matrix)[0:3,0:3]
            
            if i % 2 == 1:
                target = 'PIN'
                bump_target = 'PIN-N'
            else:
                target = 'PKN'
                bump_target = 'PKN-N'
                
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.textures[target])
            glUniform1i(uniform_locations['diffuse'], 0);
            
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.textures[bump_target])
            glUniform1i(uniform_locations['bump'], 1);
            
            glUniformMatrix4fv(uniform_locations['mvp'], 1, GL_FALSE, mvp_matrix)
            glUniformMatrix4fv(uniform_locations['mv'], 1, GL_FALSE, view_matrix)
            glUniformMatrix3fv(uniform_locations['normal_matrix'], 1, GL_FALSE, normal_matrix)
            self.geometries[target].draw()
            i += 1
 
        glDisableVertexAttribArray(2)
        glDisableVertexAttribArray(1)
        
        glutSwapBuffers()
        
    def start(self):
        glutDisplayFunc(self.on_draw)
        glutIdleFunc(self.on_draw)
        glutMainLoop()
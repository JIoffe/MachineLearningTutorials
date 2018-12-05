# -*- coding: utf-8 -*-
"""
Quick utility to map out the vertex numbers of an OBJ file
from front view. Used to help map the mesh against a face landmark dataset
Created on Thu Oct 18 10:57:06 2018

@author: Jason Ioffe
"""

import numpy as np
import cv2
import pyrr
from pyrr import Matrix44, Vector4, Vector3, Quaternion

# From local project
from rendering.obj_loader import ObjLoader

FRAME_CAPTION = 'Obj Visualizer'
FRAME_WIDTH = 1024
FRAME_HEIGHT = 768

font = cv2.FONT_HERSHEY_SIMPLEX
label_color = (255,255,255)
vertex_color = (200,255,255)

model_path = './models/pumpkin_carving.obj'

def get_screen_coords(coords, frame_width, frame_height):
    return int((coords[0] + 1.0) / 2 * frame_width), int( (1.0 - ((coords[1] + 1.0) / 2)) * frame_height)

        
proj_matrix = mat = pyrr.matrix44.create_perspective_projection_matrix(45.0, FRAME_WIDTH/FRAME_HEIGHT, 0.1, 1000.0)
view_matrix = pyrr.matrix44.create_look_at(np.array([0,40,900], dtype=np.float32),
                                            np.array([0,0,0], dtype=np.float32),
                                            np.array([0, 1, 0], dtype=np.float32))

mvp_matrix = np.matmul(proj_matrix, view_matrix)

obj_loader = ObjLoader()
vertices, tex_coords, normals, indices = obj_loader.load(model_path, flip_faces=False)

# Add an extra "1" to allow 4x4 matrix multiplaction
transformed_vertices = np.zeros((vertices.shape[0], 4), np.float32)
transformed_vertices[:,:-1] = vertices
transformed_vertices[:,3] = 1.0

transformed_vertices = np.transpose(mvp_matrix.dot(np.transpose(transformed_vertices)))

index = 0

while True:
    img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    
    for v in vertices:
        x, y = get_screen_coords(v, FRAME_WIDTH, FRAME_HEIGHT)
        cv2.circle(img, (x,y), 2, vertex_color, thickness=-1)
    
    x, y = get_screen_coords(vertices[index], FRAME_WIDTH, FRAME_HEIGHT)
    cv2.putText(img, '{0}'.format(index), (x + 1, y + 1), font, 0.5, label_color, 1, cv2.LINE_AA) 
    
    cv2.imshow(FRAME_CAPTION, img)
    
    key = cv2.waitKey(0) & 0xff
    if(key == 27):
        break 
    elif key == 32:
        index += 1
        
    if index >= len(vertices):
        index = 0
        
cv2.destroyAllWindows()
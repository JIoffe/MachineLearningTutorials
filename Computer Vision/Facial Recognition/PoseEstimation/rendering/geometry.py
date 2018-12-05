# -*- coding: utf-8 -*-
"""
Simple utilities for rendering geometry
@author: Jason Ioffe
"""

import numpy as np
from OpenGL.GL import *

def setup_array_buffer(buffer, data):
    byte_size = ArrayDatatype.arrayByteCount(data)
    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    glBufferData(GL_ARRAY_BUFFER, byte_size, data, GL_STATIC_DRAW)
    
def setup_element_array_buffer(buffer, data):
    byte_size = ArrayDatatype.arrayByteCount(data)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, byte_size, data, GL_STATIC_DRAW)
    
class BatchedGeometry():
    def __init__(self, vdata, tex_coords, normals, indices):
        # It would be better to interleave the arrays
        self.buffers = glGenBuffers(4)
        self.offset_0 = ctypes.c_void_p(0)
        
        setup_array_buffer(self.buffers[0], vdata);
        setup_array_buffer(self.buffers[1], tex_coords);
        setup_array_buffer(self.buffers[2], normals);
        
        self.count = len(indices)
        setup_element_array_buffer(self.buffers[3], indices)
    
    def update_vertex_data(self, data, byte_size):
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glBufferData(GL_ARRAY_BUFFER, byte_size, data, GL_STATIC_DRAW)
        
    def draw(self, bind_tex=True, bind_normals=True):
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, self.offset_0)
        
        if bind_tex:
            glBindBuffer(GL_ARRAY_BUFFER, self.buffers[1])
            glVertexAttribPointer(1, 2, GL_FLOAT, False, 0, self.offset_0)
        
        if bind_normals:
            glBindBuffer(GL_ARRAY_BUFFER, self.buffers[2])
            glVertexAttribPointer(2, 3, GL_FLOAT, False, 0, self.offset_0)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[3])
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_SHORT, self.offset_0)
        
class FullScreenQuad():
    def __init__(self):
        vertices = np.array([
                [ -1, 1 ],
                [ 1, 1 ],
                [ 1,-1 ],
                [ -1,-1 ]
            ], np.float32)
    
        indices = np.array([2,1,0,0,3,2], np.uint8)
        
        self.buffers = glGenBuffers(2)
        self.offset_0 = ctypes.c_void_p(0)
        
        setup_array_buffer(self.buffers[0], vertices)
        setup_element_array_buffer(self.buffers[1], indices)
        
    def draw(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 0, self.offset_0)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[1])
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, self.offset_0)
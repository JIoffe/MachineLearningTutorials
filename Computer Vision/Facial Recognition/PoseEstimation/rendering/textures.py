# -*- coding: utf-8 -*-
"""
Utilities for texture ops

@author: Jason Ioffe
"""

# Give an alias to not conflict with python's open()
from PIL import Image
from OpenGL.GL import *

class TextureLoader():
    def __init__(self):
        return
    
    def __load_image_data(self, path, flipY=True):
        im = Image.open(path)
        
        # Flip Y for OpenGL
        if flipY:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
        
        try:
            image_data = im.tobytes("raw", "RGBA", 0, -1)
        except (ValueError, SystemError):
            image_data = im.tobytes("raw", "RGBX", 0, -1)
            
        return im.size[0], im.size[1], image_data
    
    def load(self, path, flipY=True):       
        tex_id = glGenTextures(1)
        w, h, image_data = self.__load_image_data(path, flipY)
        
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glTexImage2D(GL_TEXTURE_2D, 0, 3, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        
        return tex_id
# -*- coding: utf-8 -*-
"""
Simple OBJ loader that only supports triangles and a single "group" of faces
Reads tex, normal and group information into numpy arrays
@author: Jason Ioffe
"""

import numpy as np
    
class ObjLoader():
    def __init__(self):
        return
    
    def load(self, path, flip_faces=False):       
        vertices = []
        normals = []
        tex_coords = []
        faces = []

        
        with open(path, "r") as obj_file:
            for line in obj_file:
                if line.startswith('#'):
                   continue
               
                values = line.split()
                if not values:
                    continue
                
                if values[0] == 'v':
                    v = [float(v) for v in values[1:4]]
                    vertices.append(v)
                elif values[0] == 'vn':
                    n = [float(n) for n in values[1:4]]
                    normals.append(n)
                elif values[0] == 'vt':
                    vt = [float(vt) for vt in values[1:3]]
                    tex_coords.append(vt)
                elif values[0] == 'f':
                    for f in values[1:4]:
                        faces.append(f)
        
        print('Read {}: {} vertices'.format(path, len(vertices)))
        
        if flip_faces:
            faces = faces[::-1]
            
        # Now for the fun part. OBJ format has separate indices for verts, norms, and texcoords.
        # So we need to explicitly duplicate vert positions to make up for smoothing and UV groups
            
        n = len(vertices)
                
        output_tex_coords = np.zeros((n, 2), dtype=np.float32)
        output_normals = np.zeros((n, 3), dtype=np.float32)
        indices = []
        
        for face in faces:
            face_indices = [int(i) - 1 for i in face.split('/')]
            vi = face_indices[0]
            n_indices = len(face_indices)
            
            # OBJ file format should store each index as a trio - V/VT/VN
            # But sometimes it will have a single index - assume that single index is for everything
            if(n_indices == 1):
                ti = vi
                ni = vi
            else:
                ti = face_indices[1]
                ni = face_indices[2]
            
            if vi in indices:
                # Double up
                indices.append(len(vertices))
                vertices.append(vertices[vi])
                output_tex_coords = np.append(output_tex_coords, [tex_coords[ti]], axis=0)
                output_normals = np.append(output_normals, [normals[ni]], axis=0)
            else:
                if len(tex_coords) > ti:
                    output_tex_coords[vi] = tex_coords[ti]
                    
                if len(normals) > ni:
                    output_normals[vi] = normals[ni]
                    
                indices.append(vi)

        return np.array(vertices, dtype=np.float32), output_tex_coords.astype(np.float32), output_normals.astype(np.float32), np.array(indices, dtype=np.uint16)
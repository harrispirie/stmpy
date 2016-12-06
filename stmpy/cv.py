import numpy as np
import cv2

'''
This module interfaces with the opencv computer vision library to provide tools
for advanced image processing.  An additional dependency has been added that is
not part of the standard stmpy installation. As such this module is not
imported when the user calls: import stmpy.

Usage: import stmpy.cv as scv

Requirements: cv2 (from opencv library - try "brew install opencv" on mac)
'''

def bilateralFilter(F, d=10, si=1.0, sd=1.0):
    '''
    A wraper for the openCV bilateral filter.  Applies a non-linear,
    edge-preserving filter to a 2D image.  If input is a 3D dataset, this will
    apply the filter to each layer by iterating over the first index.

    Inputs:
        d - Diameter of pixel neighborhood used in filtering.
        si - Width of gaussian in pixel intensity space.
        sd - Width of gaussian in real space.

    Usage: output = bilateralFilter(image, d=10, si=1.0, sd=1.0)

    Issues: If data dominated by one number (e.g. center pixel in the FT)
    then filter does not work. 
    '''
    def filter2d(img, d, si, sd):
        norm = float(np.max(img))
        data = np.float32(img/norm)
        out = cv2.bilateralFilter(data, d, si, sd)
        return norm * np.float64(out)
    if len(F.shape) is 2:
        return filter2d(F, d, si, sd)
    if len(F.shape) is 3:
        out = np.zeros_like(F)
        for ix, layer in enumerate(F):
            out[ix] = filter2d(layer, d, si, sd)
        return out
    else:
        print('ERR: Input must be 2D or 3D numpy array.')


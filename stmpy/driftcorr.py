from __future__ import print_function
try:
    import cv2
except ModuleNotFoundError:
    print("Please install opencv-python module using following command:\npip3 install opencv-python")
import stmpy
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.ndimage as snd
from scipy.interpolate import interp1d, interp2d
from skimage import transform as tf
from skimage.feature import peak_local_max
from pprint import pprint
import types

'''

REFERENCES:
[1] MH Hamidian, et al. "Picometer registration of zinc impurity states in Bi2Sr2CaCu2O8+d for phase determination in intra-unit-cell Fourier transform STM", New J. Phys. 14, 053017 (2012).
[2] JA Slezak, PhD thesis (Ch. 3), http://davisgroup.lassp.cornell.edu/theses/Thesis_JamesSlezak.pdf

History:
    2017-04-28      CREATED BY JIANFENG GE
    04/29/2019      RL : Add documents for all functions. Add another method to calculate phasemap.
                            Add inverse FFT method to apply the drift field.
    03/25/2021      RL : Change the whole drift corr library to function based library
'''

##################################################################################
######################### Wrapped functions for easy use #########################
##################################################################################      
        
def find_drift_parameter(A, r=None, w=None, mask3=None, cut1=None, cut2=None, bp_angle=None, orient=None, bp_c=None,\
                sigma=10, method='lockin', even_out=False, show=True, **kwargs):
    '''
    This method find drift parameters from a 2D map automatically.

    Input:
        A           - Required : 2D array of topo or LIY in real space.
        r           - Optional : width of the gaussian mask, ratio to the full map size, to remove low-q noise, =r*width
                                    Set r=None will disable this mask.
        w           - Optional : width of the mask that filters out noise along qx=0 and qy=0 lines.
                                    Set w=None will disable this mask.
        mask3       - Optional : Tuple for custom-defined mask. mask3 = [n, offset, width], where n is order of symmetry, 
                                    offset is initial angle, width is the width of the mask. e.g., mask3 = [4, np.pi/4, 5], 
                                    or mask3 = [6, 0, 10].
                                    Set mask3=None will disable this mask.
        even_out    - Optional : Boolean, if True then Bragg peaks will be rounded to the make sure there are even number of lattice
        cut1        - Optional : List of length 1 or length 4, specifying how much bad area or area with too large drift to be cut
        cut2        - Optional : List of length 1 or length 4, specifying after local drift correction how much to crop on the edge
        angle_offset- Optional : The min offset angle of the Bragg peak to the x-axis, in unit of rad
        bp_angle    - Optional : The angle between neighboring Bragg peaks, if not given, it will be computed based on all Bragg peaks
        orient      - Optional : The orientation of the Bragg peaks with respect to the x-axis
        bp_c        - Optional : The correct Bragg peak position that user wants after the drift correction  
        sigma       - Optional : Floating number specifying the size of mask to be used in phasemap()
        method      - Optional : Specifying which method to apply the drift correction
                                    "lockin": Interpolate A and then apply it to a new set of coordinates, (x-ux, y-uy)
                                    "convolution": Used inversion fft to apply the drift fields
        show        - Optional : Boolean, if True then A and Bragg peaks will be plotted out.
        **kwargs    - Optional : key word arguments for findBraggs function

    Returns:
        p           : A dict of parameters that can be directly applied to drift correct orther 2D or 3D datasets

    Usage:
        p = find_drift(z, sigma=4, cut1=None, cut2=[0,7,0,7], show=True)

    History:
        06/23/2020  - RL : Initial commit.
    '''
    p = {}
    if cut1 is not None:
        A = cropedge(A, n=cut1)

    # find the Bragg peak before the drift correction 
    bp1 = findBraggs(A, r=r, w=w, mask3=mask3, show=show, **kwargs)
    bp1 = sortBraggs(bp1, s=np.shape(A))

    if bp_c is None:
        # Find the angle between each Bragg peaks
        if bp_angle is None:
            N = len(bp1)
            Q = bp_to_q(bp1, A)
            angles = []
            for i in range(N-1):
                angles.append(np.arctan2(*Q[i+1]) - np.arctan2(*Q[i]))
            # Here is the commonly used angles in the real world
            angle_list = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
            offset = np.absolute(np.mean(angles) - angle_list)
            index = np.argmin(offset)
            bp_angle = angle_list[index]
            if orient is None:
                orient = np.absolute(np.arctan2(*Q[0]))
        # Calculate the correction position of each Bragg peak
        bp_c = generate_bp(A, bp1, angle=bp_angle, orient= orient, even_out=even_out)
    
    # Find the phasemap 
    thetax, thetay, Q1, Q2 = phasemap(A, bp=bp_c, method=method, sigma=sigma)

    phix = fixphaseslip(thetax, method='unwrap')
    phiy = fixphaseslip(thetay, method='unwrap')
    ux, uy = driftmap(phix, phiy, Q1, Q2, method=method)
    z_temp = driftcorr(A, ux, uy, method=method, interpolation='cubic')
    
    # This part interpolates the drift corrected maps
    if cut2 is None:
        z_c = z_temp
    else:
        bp3 = findBraggs(z_temp, r=r, w=w, mask3=mask3, **kwargs)
        z_c = cropedge(z_temp, n=cut2, bp=bp3, force_commen=True)
        p['bp3'] = bp3
    
    # This part displays the intermediate maps in the process of drift correction
    if show is True:
        fig, ax = plt.subplots(1, 2, figsize=[8, 4])
        c = np.mean(phix)
        s = np.std(phix)
        fig.suptitle('Phasemaps after fixing phase slips:')
        ax[0].imshow(phix, origin='lower', clim=[c-5*s, c+5*s])
        ax[1].imshow(phiy, origin='lower', clim=[c-5*s, c+5*s])
        
        A_fft = stmpy.tools.fft(A, zeroDC=True)
        B_fft = stmpy.tools.fft(z_c, zeroDC=True)
        c1 = np.mean(A_fft)
        s1 = np.std(A_fft)
        
        c2 = np.mean(A)
        s2 = np.std(A)

        fig, ax = plt.subplots(2, 2, figsize=[8, 8])
        fig.suptitle('Maps before and after drift correction:')
        ax[0,0].imshow(A, cmap=stmpy.cm.blue2, origin='lower', clim=[c2-5*s2, c2+5*s2])
        ax[0,1].imshow(A_fft, cmap=stmpy.cm.gray_r, origin='lower', clim=[0, c1+5*s1])
        ax[1,0].imshow(z_c, cmap=stmpy.cm.blue2, origin='lower', clim=[c2-5*s2, c2+5*s2])
        ax[1,1].imshow(B_fft, cmap=stmpy.cm.gray_r, origin='lower', clim=[0, c1+5*s1])
    
    p['cut1'] = cut1
    p['cut2'] = cut2
    p['r'] = r
    p['w'] = w
    p['mask3'] = mask3
    p['sigma'] = sigma
    p['method'] = method
    p['even_out'] = even_out
    p['bp_c'] = bp_c
    p['bp_angle'] = bp_angle
    p['orient'] = orient
    p['bp1'] = bp1
    p['phix'] = phix
    p['phiy'] = phiy
    p['ux'] = ux
    p['uy'] = uy

    return z_c, p

def apply_drift_parameter(A, p, **kwargs):
    '''
    Apply the drifr correction parameters p to the 2D or 3D map A.

    Input:
        A           - Required : 2D or 3D map to be drift corrected
        p           - Required : A collection of parameters to be used in drift correction.
                    Use parameters (those parameters should be generated by find_drift_parameter automatically)
                        cut1   : 
                        cut2   :
                        ux     :
                        uy     :
                        method :
                        bp3    :
        **kwargs    - Optional : 

    Returns:
        A_c          : 2D or 3D map with drift removed.

    Usage:
        A_c = apply_drift_parameter(A, p)

    History:
        06/23/2020  - RL : Initial commit.
    '''
    data_c = np.copy(A)

    if p['cut1'] is None:
        data_c = data_c
    else:
        data_c = cropedge(data_c, n=p['cut1'])
    data_corr = driftcorr(data_c, ux=p['ux'], uy=p['uy'], method=p['method'], interpolation='cubic')
    if p['cut2'] is None:
        data_out = data_corr
    else:
        data_out = cropedge(data_corr, bp=p['bp3'], n=p['cut2'], force_commen=True)
    return data_out


##################################################################################
###################### Basic building blocks for OOD use #########################
##################################################################################

def get_para(A, a0=None, size=None, angle=np.pi/2, orient=np.pi/4, 
            pixels=None, even_out=False, use_a0=False):
    '''
    Get parameters that are useful for the drift correction

    Input:
        A           - Required : Spy object of topo (2D) or map (3D).
        a0          - Optional : Lattice constant in the unit of nm.
        size        - Optional : Size of the map in the unit of nm. If not offered, it'll be created
                                    automatically from header file.
        angle       - Optional : Angle of the lattice. If the lattice is n-fold symmetry, then angle = 2*pi/n
        orient      - Optional : Angle of the 1st Bragg peak. It's actually the orientation of the scan frame 
                                    with respect to the Lattice.
        pixels      - Optional : Number of pixels of the topo/map. If not offered, it'll be created
                                    automatically from header file.
        even_out    - Optional : Boolean, if True then Bragg peaks will be rounded to the make sure there are even number of lattice

    Returns:
        p           - Dict, contains necessary information for the 

    Usage:
        import stmpy.driftcorr as dfc
        dfc.getAttrs(topo, a0=a0)
    '''
    if size is None:
        try:
            size = A.header['scan_range'][-2:]
        except KeyError:
            try:
                #size = float(A.header['Grid settings'].split(";")[-2])
                size = [float(k)
                        for k in A.header['Grid settings'].split(";")[-2:]]
            except:
                print(
                    "Error: Cannot find map size from header. Please input it manually.")
    if pixels is None:
        try:
            #pixels = int(A.header['scan_pixels'][-1])
            pixels = [int(k) for k in A.header['scan_pixels'][-2:]]
        except KeyError:
            try:
                pixels = int(A.header['Grid dim'].split()[-1][:-1])
            except:
                print(
                    "Error: Cannot find number of pixels from header. Please input it manually.")
    if not isinstance(size, list):
        sizex, sizey = size, size
    else:
        sizex, sizey = size

    if not isinstance(pixels, list):
        pixelx, pixely = pixels, pixels
    else:
        pixelx, pixely = pixels

    if a0 is None:
        use_a0 = False
        a0 = 1

    # parameters related to the map itself
    A.dfc_para = {
        'a0': a0,
        'size': np.array([sizex, sizey]),
        'pixels': np.array([pixelx, pixely]),
        'qmag': np.array([sizex, sizey]) / a0,
        'qscale': np.array([pixelx, pixely]) / (2*np.array([sizex, sizey]) / a0),
        'angle': angle,
        'orient': orient,
        'use_a0': use_a0,
        'even_out': even_out,
    }

def find_drift(self, A, r=None, w=None, mask3=None, cut1=None, cut2=None, \
                sigma=10, method='convolution', even_out=False, show=True, **kwargs):
    '''
    This method find drift field from a 2D map automatically.

    Input:
        A           - Required : 2D array of topo or LIY in real space.
        r           - Optional : width of the gaussian mask to remove low-q noise, =r*width
                                    Set r=None will disable this mask.
        w           - Optional : width of the mask that filters out noise along qx=0 and qy=0 lines.
                                    Set w=None will disable this mask.
        mask3       - Optional : Tuple for custom-defined mask. mask3 = [n, offset, width], where n is order of symmetry, offset is initial angle, width is 
                                    the width of the mask. e.g., mask3 = [4, np.pi/4, 5], or mask3 = [6, 0, 10]
                                    Set mask3=None will disable this mask.
        even_out    - Optional : Boolean, if True then Bragg peaks will be rounded to the make sure there are even number of lattice
        cut1        - Optional : List of length 1 or length 4, specifying after global shear correction how much to crop on the edge
        cut2        - Optional : List of length 1 or length 4, specifying after local drift correction how much to crop on the edge
        sigma       - Optional : Floating number specifying the size of mask to be used in phasemap()
        method      - Optional : Specifying which method to apply the drift correction
                                    "lockin": Interpolate A and then apply it to a new set of coordinates,
                                                    (x-ux, y-uy)
                                    "convolution": Used inversion fft to apply the drift fields
        show        - Optional : Boolean, if True then A and Bragg peaks will be plotted out.
        **kwargs    - Optional : key word arguments for findBraggs function

    Returns:
        coords      -  (4x2) array contains Bragg peaks in the format of [[x1,y1],[x2,y2],...,[x4,y4]]

    Usage:
        t.find_drift(t.z, sigma=4, cut1=None, cut2=[0,7,0,7], show=True)

    History:
        06/09/2020  - RL : Initial commit. 
    '''

    if not hasattr(self, 'parameters'):
        self = getAttrs(self, a0=None, size=None, angle=np.pi/2, orient=np.pi/4, pixels=np.shape(A)[::-1], \
                            even_out=even_out, use_a0=None)
    # Find Bragg peaks that will be used in the drift correction part
    self.dfcPara = {
        'cut1': cut1,
        'cut2': cut2,
        'method': method,
        'sigma': sigma,
    }
    if cut1 is not None:
        A = cropedge(A, n=cut1)
    if not hasattr(self, 'bp_parameters'):
        self.bp1 = findBraggs(A, r=r, w=w, mask3=mask3, update_obj=True, obj=self,  \
                                show=show, even_out=even_out, **kwargs)
    else:
        self.bp1 = findBraggs(A, r=r, w=w, mask3=mask3, update_obj=True, obj=self,  \
                                show=show, even_out=even_out, **kwargs)
        # self.bp1 = findBraggs(A, obj=self, show=show)

    self.bp1 = sortBraggs(self.bp1, s=np.shape(A))
    if self.parameters['angle'] is None:
        N = len(self.bp1)
        Q = bp_to_q(self.bp1, A)
        angles = []
        for i in range(N-1):
            angles.append(np.arctan2(*Q[i+1]) - np.arctan2(*Q[i]))
        # Here are the commonly used angles in the real world
        angle_list = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
        offset = np.absolute(np.mean(angles) - angle_list)
        index = np.argmin(offset)
        self.parameters['angle'] = angle_list[index]
        
        if self.parameters['orient'] is None:
            orient = np.absolute(np.arctan2(*Q[0]))
            self.parameters['orient'] = orient
        
    
    # This is the correct value for the Bragg peak
    self.bp2 = generate_bp(A, self.bp1, angle=self.parameters['angle'], orient= self.parameters['orient'], 
                            even_out=self.parameters['even_out'], obj=self)
    
    # This part corrects for the drift 
    thetax, thetay, Q1, Q2 = phasemap(A, bp=self.bp2, method=method, sigma=sigma)

    self.phix = fixphaseslip(thetax, method='unwrap')
    self.phiy = fixphaseslip(thetay, method='unwrap')
    self.ux, self.uy = driftmap(self.phix, self.phiy, Q1, Q2, method=method)
    ztemp = driftcorr(A, self.ux, self.uy, method=method, interpolation='cubic')
    
    # This part interpolates the drift corrected maps
    self.bp3 = findBraggs(ztemp, obj=self)
    if cut2 is None:
        cut2 = 0
        force_commen = False
    else:
        force_commen = True
    self.zc = cropedge(ztemp, n=cut2, bp=self.bp3, force_commen=force_commen)
    
    
    # This part displays the intermediate maps in the process of drift correction
    if show is True:
        fig, ax = plt.subplots(1, 2, figsize=[8, 4])
        c = np.mean(self.phix)
        s = np.std(self.phix)
        fig.suptitle('Phasemaps after fixing phase slips:')
        ax[0].imshow(self.phix, origin='lower', clim=[c-5*s, c+5*s])
        ax[1].imshow(self.phiy, origin='lower', clim=[c-5*s, c+5*s])
        
        A_fft = stmpy.tools.fft(A, zeroDC=True)
        B_fft = stmpy.tools.fft(self.zc, zeroDC=True)
        c1 = np.mean(A_fft)
        s1 = np.std(A_fft)
        
        c2 = np.mean(A)
        s2 = np.std(A)

        fig, ax = plt.subplots(2, 2, figsize=[8, 8])
        fig.suptitle('Maps before and after drift correction:')
        ax[0,0].imshow(A, cmap=stmpy.cm.blue2, origin='lower', clim=[c2-5*s2, c2+5*s2])
        ax[0,1].imshow(A_fft, cmap=stmpy.cm.gray_r, origin='lower', clim=[0, c1+5*s1])
        ax[1,0].imshow(self.zc, cmap=stmpy.cm.blue2, origin='lower', clim=[c2-5*s2, c2+5*s2])
        ax[1,1].imshow(B_fft, cmap=stmpy.cm.gray_r, origin='lower', clim=[0, c1+5*s1])
        
    self.bp = findBraggs(self.zc, obj=self)


def correct(self, use):
    '''
    Use attributes of object "self" to correc the 3D map use.

    Input:
        self        - Required : Spy object of topo (2D) or map (3D).
        use         - Required : 3D map to be corrected with attributes of the object.
        
    Returns:
        N/A

    Usage:
        d.correct(d.LIY)

    History:
        06/09/2020  - RL : Initial commit. 
    '''    
    data_c = np.copy(use)
    if self.dfcPara['cut1'] is None:
        data_c = data_c
    else:
        data_c = cropedge(data_c, n=self.dfcPara['cut1'])
    data_corr = driftcorr(data_c, ux=self.ux, uy=self.uy,
                          method=self.dfcPara['method'], interpolation='cubic')
    if self.dfcPara['cut2'] is None:
        data_out = cropedge(data_corr, bp=self.bp3, n=0, force_commen=False)
    else:
        data_out = cropedge(data_corr, bp=self.bp3, n=self.dfcPara['cut2'], force_commen=True)
    self.liy_c = data_out

def __update_parameters(obj, a0=None, bp=None, pixels=None, size=None, use_a0=True):

    if use_a0 is True:
        center = (np.array(pixels)-1) // 2
        Q = bp - center
        q1, q2, q3, q4, *_ = Q
        delta_qx = (np.absolute(q1[0]-q3[0])+np.absolute(q2[0]-q4[0])) / 2
        delta_qy = (np.absolute(q1[1]-q3[1])+np.absolute(q2[1]-q4[1])) / 2
        sizex = np.absolute(
            delta_qx / (4 * a0 * np.cos(obj.parameters['angle'])))
        sizey = np.absolute(
            delta_qy / (4 * a0 * np.cos(obj.parameters['angle'])))

        bp_x = np.min(bp[:, 0])
        ext_x = pixels[0] / (pixels[0] - 2*bp_x)
        bp_y = np.min(bp[:, 1])
        ext_y = pixels[1] / (pixels[1] - 2*bp_y)

        obj.parameters['size'] = np.array([sizex, sizey])
        obj.parameters['pixels'] = np.array(pixels)
        obj.parameters['qscale'] = np.array([ext_x, ext_y])

        obj.qx = bp[0] - center
        obj.qy = bp[1] - center

    else:
        center = (np.array(pixels)-1) // 2
        bp_x = np.min(bp[:, 0])
        ext_x = pixels[0] / (pixels[0] - 2*bp_x)
        bp_y = np.min(bp[:, 1])
        ext_y = pixels[1] / (pixels[1] - 2*bp_y)

        obj.parameters['size'] = np.array(
            pixels) / obj.parameters['pixels'] * obj.parameters['size']
        obj.parameters['pixels'] = np.array(pixels)
        obj.parameters['qscale'] = np.array([ext_x, ext_y])
        obj.qx = bp[0] - center
        obj.qy = bp[1] - center

##################################################################################
################## Basic building blocks for drift correction ####################
##################################################################################

#1 - findBraggs
def findBraggs(A, rspace=True, min_dist=5, thres=0.25, r=None,
                 w=None, mask3=None, even_out=False, precise=False, 
                 width=10, p0=None, show=False):
    '''
    Find Bragg peaks in the unit of pixels of topo or FT pattern A using peak_local_max. If obj is offered,
    an attribute of bp will be created for obj.

    Input:
        A           - Required : 2D array of topo in real space, or FFT in q space.
        min_dist    - Optional : Minimum distance (in pixels) between peaks. Default: 5
        thres       - Optional : Minimum intensity of Bragg peaks relative to max value. Default: 0.25
        rspace      - Optional : Boolean indicating if A is real or Fourier space image. Default: True
        r           - Optional : width of the gaussian mask to remove low-q noise, =r*width
                                    Set r=None will disable this mask.
        w           - Optional : width of the mask that filters out noise along qx=0 and qy=0 lines.
                                    Set w=None will disable this mask.
        mask3       - Optional : Tuple for custom-defined mask. mask3 = [n, offset, width], where n is order of symmetry, offset is initial angle, width is 
                                    the width of the mask. e.g., mask3 = [4, np.pi/4, 5], or mask3 = [6, 0, 10]
                                    Set mask3=None will disable this mask.
        even_out    - Optional : Boolean, if True then Bragg peaks will be rounded to the make sure there are even number of lattice
        precise     - Optional : Boolean, if True then a 2D Gaussian fit will be used to find the precise location of Bragg peaks
        width       - Optional : Integer, defines how large the 2D Gaussian fit will be performed around each Bragg peaks
        p0          - Optional : List of initial parameters for fitting. Default: p0 = [amplitude,x0,y0,sigmaX,sigmaY,offset]=[1, width, width, 1, 1, 0]
        show        - Optional : Boolean, if True then data A and Bragg peaks will be plotted.

    Returns:
        coords      -  (4x2) array contains Bragg peaks in the format of [[x1,y1],[x2,y2],...,[x4,y4]]

    Usage:
        import stmpy.driftcorr as dfc
        bp = dfc.findBraggs(A, min_dist=10, thres=0.2, rspace=True, show=True)

    History:
        04/28/2017      JG : Initial commit.
        04/29/2019      RL : Add maskon option, add outAll option, and add documents.

    '''

    if rspace is True:
        F = stmpy.tools.fft(A, zeroDC=True)
    else:
        F = np.copy(A)
    # Remove low-q high intensity data with multiple masks
    *_, Y, X = np.shape(A)
    if r is not None:
        Lx = X * r
        Ly = Y * r
        x = np.arange(X)
        y = np.arange(Y)
        p0 = [int(X/2), int(Y/2), Lx, Ly, 1, np.pi/2]
        G = 1-stmpy.tools.gauss2d(x, y, p=p0)
    else:
        G = 1
    if w is not None:
        mask2 = np.ones([Y, X])
        mask2[Y//2-int(Y*w):Y//2+int(Y*w), :] = 0
        mask2[:, X//2-int(X*w):X//2+int(X*w)] = 0
    else:
        mask2 = 1
    if mask3 is None:
        mask3 = 1
    else:
        mask3 = mask_bp(A, p=mask3)
    F *= G * mask2 * mask3

    coords = peak_local_max(F, min_distance=min_dist, threshold_rel=thres)
    coords = np.fliplr(coords)

    # This part is to make sure the Bragg peaks are located at even number of pixels
    if even_out is not False:
        coords = __even_bp(coords, s=np.shape(A))

    if precise is not False:
        coords = np.asarray(coords, dtype='float32')
        if p0 is None:
            p0 = [1, width, width, 1, 1, 0]
        for i in range(len(coords)):
            area = stmpy.tools.crop(F/np.sum(F), cen=[int(k) for k in coords[i]], width=width)
            popt, g = fitGaussian2d(area, p0=p0)
            coords[i][0] += popt[1] - width
            coords[i][1] += popt[2] - width

    # This part shows the Bragg peak positions
    if show is not False:
        plt.figure(figsize=[4, 4])
        c = np.mean(F)
        s = np.std(F)
        plt.imshow(F, cmap=plt.cm.gray_r, interpolation='None',
                   origin='lower', clim=[0, c+5*s], aspect=1)
        plt.plot(coords[:, 0], coords[:, 1], 'r.')
        plt.gca().set_aspect(1)
        plt.axis('tight')

        center = (np.array(np.shape(A)[::-1])-1) // 2
        print('The coordinates of the Bragg peaks are:')
        pprint(coords)
        print()
        print('The coordinates of the Q vectors are:')
        pprint(coords-center)

    return coords

# help function: fitting 2D gaussian peaks around Bragg peaks
def fitGaussian2d(data, p0):
    ''' Fit a 2D gaussian to the data with initial parameters p0. '''
    data = np.array(data)
    def gauss(xy,amplitude,x0,y0,sigmaX,sigmaY,offset):
        x,y = xy
        theta = 90
        x0=float(x0);y0=float(y0)
        a =  0.5*(np.cos(theta)/sigmaX)**2 + 0.5*(np.sin(theta)/sigmaY)**2 
        b = -np.sin(2*theta)/(2*sigmaX)**2 + np.sin(2*theta)/(2*sigmaY)**2
        c =  0.5*(np.sin(theta)/sigmaX)**2 + 0.5*(np.cos(theta)/sigmaY)**2
        g = offset+amplitude*np.exp(-( a*(x-x0)**2 -2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ))
        return g.ravel()
    x = np.arange(data.shape[0]);  y = np.arange(data.shape[1])
    X,Y = np.meshgrid(x,y)
    popt, pcov = opt.curve_fit(gauss, (X,Y), data.ravel(), p0=p0)
    return popt, gauss((X,Y),*popt).reshape(data.shape)

# help function: custom mask to remove unwanted Bragg peaks
def mask_bp(A, p):

    n, offset, thres, *_ = p
    s = np.shape(A)[-1]
    t = np.arange(s)
    x, y = np.meshgrid(t, t)
    center = (np.array([s, s])-1) // 2
    mask = np.ones_like(x)
    theta = 2 * np.pi / n
    for i in range(n):
        angle = theta * i + offset
        index = np.where(np.absolute(np.cos(angle)*(y-center[1]) - \
                                     np.sin(angle)*(x-center[0])) < thres)
        mask[index] = 0
    return mask

# help function: make sure the Bragg peaks are located on even pixels
def __even_bp(bp, s):
    '''
    This internal function rounds the Bragg peaks to their nearest even number of Q vectors.
    '''
    *_, s2, s1 = s
    center = (np.array([s1, s2])-1) // 2
    bp_temp = bp - center
    for i, ix in enumerate(bp_temp):
        for j, num in enumerate(ix):
            if (num % 2) != 0:
                if num > 0:
                    bp_temp[i, j] = num + 1
                elif num <0:
                    bp_temp[i, j] = num - 1
            else:
                pass
    bp_even = bp_temp + center
    return bp_even


#2 - cropedge
def cropedge(A, n, bp=None, c1=2, c2=2,
             a1=None, a2=None, force_commen=False):
    """
    Crop out bad pixels or highly drifted regions from topo/dos map.

    Inputs:
        A           - Required : 2D or 3D array of image to be cropped.
        n           - Required : List of integers specifying how many bad pixels to crop on each side.
                                    Order: [left, right, down, up].
        force_commen- Optional : Boolean determining if the atomic lattice is commensurate with
                                    the output image.

    Returns:
        A_crop  - 2D or 3D array of image after cropping.

    Usage:
        import stmpy.driftcorr as dfc
        A_crop = dfc.cropedge(A, n=5)

    History:
        06/04/2019      RL : Initial commit.
        11/30/2019      RL : Add support for non-square dataset
    """

    if not isinstance(n, list):
        n = [n]
    if force_commen is False:
        B = _rough_cut(A, n=n)
        print('Shape before crop:', end=' ')
        print(A.shape)
        print('Shape after crop:', end=' ')
        print(B.shape)
        return B
    else:
        if n != 0:
            B = _rough_cut(A, n)
        else:
            B = np.copy(A)
        *_, L2, L1 = np.shape(A)
        if bp is None:
            bp = findBraggs(A, show=False)
        bp = sortBraggs(bp, s=np.shape(A))
        bp_new = bp - (np.array([L1, L2])-1) // 2
        N1 = compute_dist(bp_new[0], bp_new[1])
        N2 = compute_dist(bp_new[0], bp_new[-1])

        if a1 is None:
            a1 = c1 * L1 / N1
        if a2 is None:
            a2 = a1
            #a2 = c2 * L2 / N2
        *_, L2, L1 = np.shape(B)
        L_new1 = a1 * ((L1)//(a1))
        L_new2 = a2 * ((L2)//(a2))
        t1 = np.arange(L1)
        t2 = np.arange(L2)
        if len(np.shape(A)) == 2:
            f = interp2d(t1, t2, B, kind='cubic')
            t_new1 = np.linspace(0, L_new1, num=L1+1)
            t_new2 = np.linspace(0, L_new2, num=L2+1)
            z_new = f(t_new1[:-1], t_new2[:-1])
        elif len(np.shape(A)) == 3:
            z_new = np.zeros([np.shape(A)[0], L2, L1])
            for i in range(len(A)):
                f = interp2d(t1, t2, B[i], kind='cubic')
                t_new1 = np.linspace(0, L_new1, num=L1+1)
                t_new2 = np.linspace(0, L_new2, num=L2+1)
                z_new[i] = f(t_new1[:-1], t_new2[:-1])
        else:
            print('ERR: Input must be 2D or 3D numpy array!')
        return z_new

# help function: crop edge without any interpolation
def _rough_cut(A, n):
    B = np.copy(A)
    if len(n) == 1:
        n1 = n2 = n3 = n4 = n[0]
    else:
        n1, n2, n3, n4, *_ = n
    if len(B.shape) is 2:
        if n2 == 0:
            n2 = -B.shape[1]
        if n4 == 0:
            n4 = -B.shape[0]
        return B[n3:-n4, n1:-n2]
    elif len(B.shape) is 3:
        if n2 == 0:
            n2 = -B.shape[2]
        if n4 == 0:
            n4 = -B.shape[1]
        return B[:, n3:-n4, n1:-n2]    

# 4. phasemap
def phasemap(A, bp, sigma=10, method="lockin"):
    '''
    Calculate local phase and phase shift maps. Two methods are available now: spatial lockin or Gaussian mask convolution

    Input:
        A           - Required : 2D arrays after global shear correction with bad pixels cropped on the edge
        bp          - Required : Coords of Bragg peaks of FT(A), can be computed by findBraggs(A)
        sigma       - Optional : width of DC filter in lockin method or len(A)/s
        method      - Optional : Specify which method to use to calculate phase map.
                                "lockin": Spatial lock-in method to find phase map
                                "convolution": Gaussian mask convolution method to find phase map

    Returns:
        thetax      -       2D array, Phase shift map in x direction, relative to perfectly generated cos lattice
        thetay      -       2D array, Phase shift map in y direction, relative to perfectly generated cos lattice
        Q1          -       Coordinates of 1st Bragg peak
        Q2          -       Coordinates of 2nd Bragg peak

    Usage:
        import stmpy.driftcorr as dfc
        thetax, thetay, Q1, Q2 = dfc.phasemap(A, bp, sigma=10, method='lockin')

    History:
        04/28/2017      JG : Initial commit.
        04/29/2019      RL : Add "convolution" method, and add documents.
        11/30/2019      RL : Add support for non-square dataset
    '''

    *_, s2, s1 = A.shape
    if not isinstance(sigma, list):
        sigma = [sigma]
    if len(sigma) == 1:
        sigmax = sigmay = sigma[0]
    else:
        sigmax, sigmay, *_ = sigma
    s = np.minimum(s1, s2)
    bp = sortBraggs(bp, s=np.shape(A))
    t1 = np.arange(s1, dtype='float')
    t2 = np.arange(s2, dtype='float')
    x, y = np.meshgrid(t1, t2)
    Q1 = 2*np.pi*np.array([(bp[0][0]-int((s1-1)/2))/s1,
                           (bp[0][1]-int((s2-1)/2))/s2])
    Q2 = 2*np.pi*np.array([(bp[1][0]-int((s1-1)/2))/s1,
                           (bp[1][1]-int((s2-1)/2))/s2])
    if method is "lockin":
        Axx = A * np.sin(Q1[0]*x+Q1[1]*y)
        Axy = A * np.cos(Q1[0]*x+Q1[1]*y)
        Ayx = A * np.sin(Q2[0]*x+Q2[1]*y)
        Ayy = A * np.cos(Q2[0]*x+Q2[1]*y)
        Axxf = FTDCfilter(Axx, sigmax, sigmay)
        Axyf = FTDCfilter(Axy, sigmax, sigmay)
        Ayxf = FTDCfilter(Ayx, sigmax, sigmay)
        Ayyf = FTDCfilter(Ayy, sigmax, sigmay)
        thetax = np.arctan2(Axxf, Axyf)
        thetay = np.arctan2(Ayxf, Ayyf)
        return thetax, thetay, Q1, Q2
    elif method is "convolution":
        t_x = np.arange(s1)
        t_y = np.arange(s2)
        xcoords, ycoords = np.meshgrid(t_x, t_y)
        # (2.* np.pi/s)*(Q1[0] * xcoords + Q1[1] * ycoords)
        exponent_x = (Q1[0] * xcoords + Q1[1] * ycoords)
        # (2.* np.pi/s)*(Q2[0] * xcoords + Q2[1] * ycoords)
        exponent_y = (Q2[0] * xcoords + Q2[1] * ycoords)
        A_x = A * np.exp(np.complex(0, -1)*exponent_x)
        A_y = A * np.exp(np.complex(0, -1)*exponent_y)
        # sx = sigma
        # sy = sigma * s1 / s2
        sx = sigmax
        sy = sigmay
        Amp = 1/(4*np.pi*sx*sy)
        p0 = [int((s-1)/2), int((s-1)/2), sx, sy, Amp, np.pi/2]
        G = stmpy.tools.gauss2d(t_x, t_y, p=p0, symmetric=True)
        T_x = sp.signal.fftconvolve(A_x, G, mode='same',)
        T_y = sp.signal.fftconvolve(A_y, G, mode='same',)
        R_x = np.abs(T_x)
        R_y = np.abs(T_y)
        phi_y = np.angle(T_y)
        phi_x = np.angle(T_x)
        return phi_x, phi_y, Q1, Q2
    else:
        print('Only two methods are available now:\n1. lockin\n2. convolution')

#5 - fixphaseslip
def fixphaseslip(A, thres=None, maxval=None, method='unwrap', orient=0):
    '''
    Fix phase slip by adding 2*pi at phase jump lines.

    Inputs:
        A       - Required : 2D arrays of phase shift map, potentially containing phase slips
        thres   - Optional : Float number, specifying threshold for finding phase jumps in diff(A). Default: None
        method  - Optional : Specifying which method to fix phase slips.
                                "unwrap": fix phase jumps line by line in x direction and y direction, respectively
                                "spiral": fix phase slip in phase shift maps by flattening A into a 1D array in a spiral way
        orient  - Optional : Used in "spiral" phase fixing method. 0 for clockwise and 1 for counter-clockwise

    Returns:

        phase_corr      -       2D arrays of phase shift map with phase slips corrected

    Usage:
        import stmpy.driftcorr as dfc
        thetaxf = dfc.fixphaseslip(thetax, method='unwrap')

    History:
        04/28/2017      JG : Initial commit.
        04/29/2019      RL : Add "unwrap" method, and add documents.
    '''
    output = np.copy(A[::-1, ::-1])
    maxval = 2 * np.pi
    tol = 0.25 * maxval
    if len(np.shape(A)) == 2:
        *_, s2, s1 = np.shape(A)
        mid2 = s2 // 2
        mid1 = s1 // 2
        for i in range(s2):
            output[i, :] = unwrap_phase(
                output[i, :], tolerance=thres, maxval=maxval)
        for i in range(s1):
            output[:, i] = unwrap_phase(
                output[:, i], tolerance=thres, maxval=maxval)
        linex = output[:, mid1]
        liney = output[mid2, :]
        dphx = np.diff(linex)
        dphy = np.diff(liney)

        dphx[np.where(np.abs(dphx) < tol)] = 0
        dphx[np.where(dphx < -tol)] = 1
        dphx[np.where(dphx > tol)] = -1

        dphy[np.where(np.abs(dphy) < tol)] = 0
        dphy[np.where(dphy < -tol)] = 1
        dphy[np.where(dphy > tol)] = -1

        for i in range(s2):
            output[i, 1:] += 2*np.pi * np.cumsum(dphy)
        for i in range(s1):
            output[1:, i] += 2*np.pi * np.cumsum(dphx)
        return output[::-1, ::-1]

#6 - unwrap_phase
def unwrap_phase(ph, tolerance=None, maxval=None):
    maxval = 2 * np.pi if maxval is None else maxval
    tol = 0.25*maxval if tolerance is None else tolerance*maxval
    if len(ph) < 2:
        return ph

    dph = np.diff(ph)
    dph[np.where(np.abs(dph) < tol)] = 0
    dph[np.where(dph < -tol)] = 1
    dph[np.where(dph > tol)] = -1
    ph[1:] += maxval * np.cumsum(dph)
    return ph

def unwrap_phase_2d(A, thres=None):
    output = np.copy(A[::-1, ::-1])
    if len(np.shape(A)) == 2:
        n = np.shape(A)[-1]
        for i in range(n):
            output[i, :] = unwrap_phase(output[i, :], tolerance=thres)
        for i in range(n):
            output[:, i] = unwrap_phase(output[:, i], tolerance=thres)
        return output[::-1, ::-1]

#7 - driftmap
def driftmap(phix=None, phiy=None, Q1=None, Q2=None, method="lockin"):
    '''
    Calculate drift fields based on phase shift maps, with Q1 and Q2 generated by phasemap.

    Inputs:
        phix        - Optional : 2D arrays of phase shift map in x direction with phase slips corrected
        phiy        - Optional : 2D arrays of phase shift map in y direction with phase slips corrected
        Q1          - Optional : Coordinates of 1st Bragg peak, generated by phasemap
        Q2          - Optional : Coordinates of 2nd Bragg peak, generated by phasemap
        method      - Optional : Specifying which method to use.
                                    "lockin": Used for phase shift map generated by lockin method
                                    "convolution": Used for phase shift map generated by lockin method

    Returns:
        ux          - 2D array of drift field in x direction
        uy          - 2D array of drift field in y direction

    Usage:
        import stmpy.driftcorr as dfc
        ux, uy = dfc.driftmap(thetaxf, thetayf, Q1, Q2, method='lockin')

    History:
        04/28/2017      JG : Initial commit.
        04/29/2019      RL : Add "lockin" method, and add documents.
        11/30/2019      RL : Add support for non-square dataset
    '''
    if method is "lockin":
        tx = np.copy(phix)
        ty = np.copy(phiy)
        ux = -(Q2[1]*tx - Q1[1]*ty) / (Q1[0]*Q2[1]-Q1[1]*Q2[0])
        uy = -(Q2[0]*tx - Q1[0]*ty) / (Q1[1]*Q2[0]-Q1[0]*Q2[1])
        return ux, uy
    elif method is "convolution":
        #s = np.shape(thetax)[-1]
        Qx_mag = np.sqrt((Q1[0])**2 + (Q1[1])**2)
        Qy_mag = np.sqrt((Q2[0])**2 + (Q2[1])**2)
        Qx_ang = np.arctan2(Q1[1], Q1[0])  # in radians
        Qy_ang = np.arctan2(Q2[1], Q2[0])  # in radians
        Qxdrift = 1/(Qx_mag) * phix  # s/(2*np.pi*Qx_mag) * thetax
        Qydrift = 1/(Qy_mag) * phiy  # s/(2*np.pi*Qy_mag) * thetay
        ux = Qxdrift * np.cos(Qx_ang) - Qydrift * np.sin(Qy_ang-np.pi/2)
        uy = Qxdrift * np.sin(Qx_ang) + Qydrift * np.cos(Qy_ang-np.pi/2)
        return -ux, -uy
    else:
        print("Only two methods are available now:\n1. lockin\n2. convolution")

#8. - driftcorr
def driftcorr(A, ux=None, uy=None, method="lockin", interpolation='cubic'):
    '''
    Correct the drift in the topo according to drift fields

    Inputs:
        A           - Required : 2D or 3D arrays of topo to be drift corrected
        ux          - Optional : 2D arrays of drift field in x direction, generated by driftmap()
        uy          - Optional : 2D arrays of drift field in y direction, generated by driftmap()
        method      - Optional : Specifying which method to use.
                                    "lockin": Interpolate A and then apply it to a new set of coordinates,
                                                    (x-ux, y-uy)
                                    "convolution": Used inversion fft to apply the drift fields
        interpolation - Optional : Specifying which method to use for interpolating

    Returns:
        A_corr      - 2D or 3D array of topo with drift corrected

    Usage:
        import stmpy.driftcorr as dfc
        A_corr = dfc.driftcorr(ux, uy, method='interpolate', interpolation='cubic')

    History:
        04/28/2017      JG : Initial commit.
        04/29/2019      RL : Add "invfft" method, and add documents.
        11/30/2019      RL : Add support for non-square dataset
    '''
    if method is "lockin":
        A_corr = np.zeros_like(A)
        *_, s2, s1 = np.shape(A)
        t1 = np.arange(s1, dtype='float')
        t2 = np.arange(s2, dtype='float')
        x, y = np.meshgrid(t1, t2)
        xnew = (x - ux).ravel()
        ynew = (y - uy).ravel()
        tmp = np.zeros(s1*s2)
        if len(A.shape) is 2:
            tmp_f = interp2d(t1, t2, A, kind=interpolation)
            for ix in range(tmp.size):
                tmp[ix] = tmp_f(xnew[ix], ynew[ix])
            A_corr = tmp.reshape(s2, s1)
            return A_corr
        elif len(A.shape) is 3:
            for iz, layer in enumerate(A):
                tmp_f = interp2d(t1, t2, layer, kind=interpolation)
                for ix in range(tmp.size):
                    tmp[ix] = tmp_f(xnew[ix], ynew[ix])
                A_corr[iz] = tmp.reshape(s2, s1)
                print('Processing slice %d/%d...' %
                      (iz+1, A.shape[0]), end='\r')
            return A_corr
        else:
            print('ERR: Input must be 2D or 3D numpy array!')
    elif method is "convolution":
        A_corr = np.zeros_like(A)
        if len(A.shape) is 2:
            return _apply_drift_field(A, ux=ux, uy=uy, zeroOut=True)
        elif len(A.shape) is 3:
            for iz, layer in enumerate(A):
                A_corr[iz] = _apply_drift_field(
                    layer, ux=ux, uy=uy, zeroOut=True)
                print('Processing slice %d/%d...' %
                      (iz+1, A.shape[0]), end='\r')
            return A_corr
        else:
            print('ERR: Input must be 2D or 3D numpy array!')

# help function: apply drift field using inverse FT method
def _apply_drift_field(A, ux, uy, zeroOut=True):
    A_corr = np.copy(A)
    *_, s2, s1 = np.shape(A)
    t1 = np.arange(s1, dtype='float')
    t2 = np.arange(s2, dtype='float')
    x, y = np.meshgrid(t1, t2)
    xshifted = x - ux
    yshifted = y - uy
    if zeroOut is True:
        A_corr[np.where(xshifted < 0)] = 0
        A_corr[np.where(yshifted < 0)] = 0
        A_corr[np.where(xshifted > s1)] = 0
        A_corr[np.where(yshifted > s2)] = 0
    qcoordx = (2*np.pi/s1)*(np.arange(s1)-int(s1/2))
    qcoordy = (2*np.pi/s2)*(np.arange(s2)-int(s2/2))
    #qcoord = (2*np.pi/s)*(np.arange(s)-(s/2))
    xshifted = np.reshape(xshifted, [1, s1*s2])
    yshifted = np.reshape(yshifted, [1, s1*s2])
    qcoordx = np.reshape(qcoordx, [s1, 1])
    qcoordy = np.reshape(qcoordy, [s2, 1])
    xphase = np.exp(-1j*(np.matmul(xshifted.T, qcoordx.T).T))
    yphase = np.exp(-1j*(np.matmul(yshifted.T, qcoordy.T).T))
    avgData = np.mean(A_corr)
    A_corr -= avgData
    A_corr = np.reshape(A_corr, s1*s2)
    data_temp = np.zeros([s2, s1*s2])
    for i in range(s2):
        data_temp[i] = A_corr
    FT = np.matmul(data_temp * xphase, yphase.T).T
    invFT = np.fft.ifft2(np.fft.fftshift(FT)) + avgData
    return np.real(invFT)

#9
def generate_bp(A, bp, angle=np.pi/2, orient=np.pi/4, even_out=False, obj=None):
    '''
    Generate Bragg peaks with given q-vectorss

    Input:
        A           - Required : 2D array of topo in real space, or FFT in q space.
        bp          - Required : Bragg peaks associated with A, to be checked
        angle       - Optional : Angle of the lattice. If the lattice is n-fold symmetry, then angle = 2*pi/n
        orient      - Optional : Initial angle of Bragg peak, or orientation of the scan. Default is np.pi/4
        obj         - Optional : Data object that has bp_parameters with it,

    Return:
        bp_new      : new Bragg peak generated from q-vectors

    Usage:
        bp_new = dfc.check_bp(A, qx=[qx1,qx2], qy=[qy1,qy2], obj=obj)

    History:
        05-25-2020      RL : Initial commit.
        06-08-2020      RL : Add the ability to compute correct Bragg peaks automatically
    '''
    *_, s2, s1 = np.shape(A)
    bp = sortBraggs(bp, s=np.shape(A))
    center = (np.array([s1, s2])-1) // 2
    Q1, Q2, Q3, Q4, *_ = bp
    Qx_mag = compute_dist(Q1, center)
    Qy_mag = compute_dist(Q2, center)
    Q_corr = np.mean([Qx_mag, Qy_mag])
    Qc1 = np.array([int(k) for k in Q_corr*np.array([np.cos(orient+np.pi), np.sin(orient+np.pi)])])
    Qc2 = np.array([int(k) for k in Q_corr*np.array([np.cos(-angle+orient+np.pi), np.sin(-angle+orient+np.pi)])])
    bp_out = np.array([Qc1, Qc2, -Qc1, -Qc2]) + center
    if even_out is not False:
        bp_out = __even_bp(bp_out, s=np.shape(A))
                
    if obj is not None:
        pixels = np.shape(A)[::-1]
        __update_parameters(obj, a0=obj.parameters['a0'], bp=bp_out, pixels=pixels,
                            size=obj.parameters['size'], use_a0=obj.parameters['use_a0'])
    return sortBraggs(bp_out, s=np.shape(A))

##################################################################################
####################### Useful functions in the processing #######################
##################################################################################

def sortBraggs(bp, s):
    ''' Sort the Bragg peaks in the order of "lower left, lower right, upper right, and upper left" '''
    *_, s2, s1 = s
    center = np.array([(s1 - 1) // 2, (s2 - 1) // 2])
    out = np.array(sorted(bp-center, key=lambda x: np.arctan2(*x))) + center
    return out


def Gaussian2d(x, y, sigma_x, sigma_y, theta, x0, y0, Amp):
    '''
    x, y: ascending 1D array
    x0, y0: center
    '''
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)**2/4/sigma_x**2 + np.sin(2*theta)**2/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    z = np.zeros((len(x), len(y)))
    X, Y = np.meshgrid(x, y)
    z = Amp * np.exp(-(a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    return z


def FTDCfilter(A, sigma1, sigma2):
    '''
    Filtering DC component of Fourier transform and inverse FT, using a gaussian with one parameter sigma
    A is a 2D array, sigma is in unit of px
    '''
    *_, s2, s1 = A.shape
    m1, m2 = np.arange(s1, dtype='float'), np.arange(s2, dtype='float')
    c1, c2 = np.float((s1-1)/2), np.float((s2-1)/2)
    # sigma1 = sigma
    # sigma2 = sigma * s1 / s2
    g = Gaussian2d(m1, m2, sigma1, sigma2, 0, c1, c2, 1)
    ft_A = np.fft.fftshift(np.fft.fft2(A))
    ft_Af = ft_A * g
    Af = np.fft.ifft2(np.fft.ifftshift(ft_Af))
    return np.real(Af)


def compute_dist(x1, x2, p=None):
    '''
    Compute the distance between point x1 and x2.
    '''
    if p is None:
        p1, p2 = 1, 1
    else:
        p1, p2 = p
    return np.sqrt(((x1[0]-x2[0])*p1)**2+((x1[1]-x2[1])*p2)**2)


def bp_to_q(bp, A):
    '''
    Convert the Bragg peaks to Q vectors by subtracting the center of the image.
    Input:
    bp      - Required : Array of Bragg peaks
    A       - Required : 
    '''
    center = (np.array(np.shape(A)[::-1])-1) // 2
    return bp - center


#15. - display


def display(A, B=None, sigma=3, clim_same=True):
    '''
    Display or compare images in both real space and q-space.

    Inputs:
        A           - Required : Real space image to display.
        B           - Optional : Another real space image to be compared with A.
        sigma       - Optional : sigma for the color limit.
        clim_same   - Optional : If True, then both FT of A and B will be displayed under the
                                    same color limit (determined by A).

    Returns:
        N/A

    Usage:
        import stmpy.driftcorr as dfc
        dfc.display(topo.z)

    '''
    if B is None:
        A_fft = stmpy.tools.fft(A, zeroDC=True)
        c = np.mean(A_fft)
        s = np.std(A_fft)
        fig, ax = plt.subplots(1, 2, figsize=[8, 4])
        ax[0].imshow(A, cmap=stmpy.cm.blue2, origin='lower')
        ax[1].imshow(A_fft, cmap=stmpy.cm.gray_r,
                     origin='lower', clim=[0, c+sigma*s])
        for ix in ax:
            ix.set_aspect(1)
    else:
        A_fft = stmpy.tools.fft(A, zeroDC=True)
        B_fft = stmpy.tools.fft(B, zeroDC=True)
        c1 = np.mean(A_fft)
        s1 = np.std(A_fft)
        if clim_same is True:
            c2 = c1
            s2 = s1
        else:
            c2 = np.mean(B_fft)
            s2 = np.std(B_fft)
        fig, ax = plt.subplots(2, 2, figsize=[8, 8])
        ax[0, 0].imshow(A, cmap=stmpy.cm.blue2, origin='lower')
        ax[0, 1].imshow(A_fft, cmap=stmpy.cm.gray_r,
                        origin='lower', clim=[0, c1+sigma*s1])
        ax[1, 0].imshow(B, cmap=stmpy.cm.blue2, origin='lower')
        ax[1, 1].imshow(B_fft, cmap=stmpy.cm.gray_r,
                        origin='lower', clim=[0, c2+sigma*s2])
        for ix in ax.flatten():
            ix.set_aspect(1)

def quick_linecut(A, width=2, n=4, bp=None, ax=None, thres=3):
    """
    Take four linecuts automatically, horizontal, vertical, and two diagonal.
    Inputs:
        A           - Required : FT space image to take linecuts.
        width       - Optional : Number of pixels for averaging.
        bp          - Optional : Bragg peaks
        thres       - Optional : threshold for displaying FT

    Returns:
        N/A

    Usage:
        import stmpy.driftcorr as dfc
        r, cut = dfc.quick_linecut(A)

    """
    Y = np.shape(A)[-2] / 2
    X = np.shape(A)[-1] / 2
    r = []
    cut = []
    start = [[0, Y], [X, 0], [0, 0], [0, Y*2]]
    end = [[X*2, Y], [X, Y*2], [X*2, Y*2], [X*2, 0]]
    color = ['r', 'g', 'b', 'k']

    plt.figure(figsize=[4, 4])
    if len(np.shape(A)) == 3:
        if bp is None:
            bp_x = np.min(findBraggs(np.mean(A, axis=0), rspace=False))
        else:
            bp_x = bp
        cm = np.mean(np.mean(A, axis=0))
        cs = np.std(np.mean(A, axis=0))
        plt.imshow(np.mean(A, axis=0), clim=[0, cm+thres*cs])
    elif len(np.shape(A)) == 2:
        if bp is None:
            bp_x = np.min(findBraggs(A, rspace=False))
        else:
            bp_x = bp
        cm = np.mean(A)
        cs = np.std(A)
        plt.imshow(A, clim=[0, cm+thres*cs])

    qscale = X*2 / (X*2 - bp_x * 2)

    for i in range(n):
        r1, cut1 = stmpy.tools.linecut(A, start[i], end[i],
                                       width=width, show=True, ax=plt.gca(), color=color[i])
        r.append(r1)
        cut.append(cut1)
    plt.gca().set_xlim(-1, X*2+1)
    plt.gca().set_ylim(-1, Y*2+1)
    return qscale, cut


def quick_show(A, en, thres=5, rspace=True, saveon=False, qlimit=1.2, imgName='', extension='png'):
    layers = len(A)
    if rspace is False:
        imgsize = np.shape(A)[-1]
        bp_x = np.min(findBraggs(np.mean(A, axis=0),
                                 min_dist=int(imgsize/10), rspace=rspace))
        ext = imgsize / (imgsize - 2*bp_x)
    if layers > 12:
        skip = layers // 12
    else:
        skip = 1
    fig, ax = plt.subplots(3, 4, figsize=[16, 12])
    try:
        for i in range(12):
            c = np.mean(A[i*skip])
            s = np.std(A[i*skip])
            if rspace is True:
                ax[i//4, i % 4].imshow(A[i*skip], clim=[c -
                                                        thres*s, c+thres*s], cmap=stmpy.cm.jackyPSD)
            else:
                ax[i//4, i % 4].imshow(A[i*skip], extent=[-ext, ext, -ext, ext, ],
                                       clim=[0, c+thres*s], cmap=stmpy.cm.gray_r)
                ax[i//4, i % 4].set_xlim(-qlimit, qlimit)
                ax[i//4, i % 4].set_ylim(-qlimit, qlimit)
            stmpy.image.add_label("${}$ mV".format(
                int(en[i*skip])), ax=ax[i//4, i % 4])
    except IndexError:
        pass
    if saveon is True:
        plt.savefig("{}.{}".format(imgName, extension), bbox_inches='tight')


def quick_show_cut(A, en, qscale, thres=5, thres2=None, saveon=False, qlimit=1.2, imgName='', extension="png"):
    fname = ["M-0", "M-90", "X-45", "X-135"]
    X1, Y1 = np.shape(A[0])
    X2, Y2 = np.shape(A[-1])
    q1 = np.linspace(-qscale, qscale, num=Y1)
    q2 = np.linspace(-qscale*np.sqrt(2), qscale*np.sqrt(2), num=Y2)
    if thres2 is None:
        thres2 = thres
    for i, ix in enumerate(A):
        plt.figure(figsize=[6, 3])
        c = np.mean(ix)
        s = np.std(ix)
        if i in [0, 1]:
            plt.pcolormesh(q1, en, ix, cmap=stmpy.cm.gray_r,
                           vmin=0, vmax=c+thres*s)
        else:
            plt.pcolormesh(q2, en, ix, cmap=stmpy.cm.gray_r,
                           vmin=0, vmax=c+thres2*s)
        plt.gca().set_xlim(-qlimit, qlimit)
        plt.axvline(-1, linestyle='--')
        plt.axvline(1, linestyle='--')
        if saveon is True:
            plt.savefig(
                imgName + " along {}.{}".format(fname[i], extension), facecolor='w')

# Quick show single images


def quick_show_single(A, en, thres=5, fs=4, qscale=None, rspace=False, saveon=False, qlimit=1.2, imgName='', extension='png', dpi=400):
    layers = len(A)
    if rspace is False:
        if qscale is None:
            imgsize = np.shape(A)[-1]
            if len(np.shape(A)) == 3:
                A_topo = np.mean(A, axis=0)
            else:
                A_topo = A
            bp_x = np.min(findBraggs(
                A_topo, min_dist=int(imgsize/10), rspace=rspace))
            ext = imgsize / (imgsize - 2*bp_x)
        else:
            ext = qscale
    if len(np.shape(A)) == 3:
        for i in range(layers):
            plt.figure(figsize=[fs, fs])
            c = np.mean(A[i])
            s = np.std(A[i])
            if rspace is True:
                plt.imshow(A[i], clim=[c-thres*s, c+thres*s],
                           cmap=stmpy.cm.jackyPSD)
            else:
                plt.imshow(A[i], extent=[-ext, ext, -ext, ext, ],
                           clim=[0, c+thres*s], cmap=stmpy.cm.gray_r)
                plt.xlim(-qlimit, qlimit)
                plt.ylim(-qlimit, qlimit)
            #stmpy.image.add_label("${}$ mV".format(int(en[i])), ax=plt.gca())
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().set_frame_on(False)
            plt.gca().set_aspect(1)
            if saveon is True:
                if extension == 'png':
                    plt.savefig("{} at {} mV.{}".format(imgName, int(
                        en[i]), extension), dpi=dpi, bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig("{} at {} mV.{}".format(imgName, int(
                        en[i]), extension), bbox_inches='tight', pad_inches=0)
    elif len(np.shape(A)) == 2:
        plt.figure(figsize=[fs, fs])
        c = np.mean(A)
        s = np.std(A)
        if rspace is True:
            plt.imshow(A, clim=[c-thres*s, c+thres*s], cmap=stmpy.cm.jackyPSD)
        else:
            plt.imshow(A, extent=[-ext, ext, -ext, ext, ],
                       clim=[0, c+thres*s], cmap=stmpy.cm.gray_r)
            plt.xlim(-qlimit, qlimit)
            plt.ylim(-qlimit, qlimit)
        #stmpy.image.add_label("${}$ mV".format(int(en)), ax=plt.gca())
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().set_frame_on(False)
        plt.gca().set_aspect(1)
        if saveon is True:
            if extension == 'png':
                plt.savefig("{} at {} mV.{}".format(imgName, int(
                    en), extension), dpi=dpi, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig("{} at {} mV.{}".format(imgName, int(
                    en), extension), bbox_inches='tight', pad_inches=0)

from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.ndimage as snd
from scipy.interpolate import interp1d, interp2d
from skimage import transform as tf
from skimage.feature import peak_local_max

'''
Local drift correction of square/triangular lattice. (Please carefully rewrite if a rectangular lattice version is needed.)

Usage:
0. please import stmpy.driftcorr (Package 'skimage' required, run 'pip install -U scikit-image' in terminal);

1. findBraggs: FT the topo image (please interpolate to be 2D square array if not), then find all Bragg peaks by peak_local_max, plot result with abs(FT) to check validity (sometimes points in the center should be removed);

2. gshearcorr: global shear correction, outputs corrected image and positions of the corrected Bragg peaks;

3. phasemap: use the new Bragg peaks to generate local phase shift (theta) maps. Use chkphasemap get phase (phi) maps and check if pi lines in those matches with raw image. If not, please check your Bragg peaks or adjust sigma in the FT DC filter;

4. fixphaseslip: fix phase slips of 2d phase maps in a 1d spiral way.(this is not well developed yet. Phase slips could still be present 
near edges after processing, please crop images and DOS maps accordingly and manually AFTER applying driftmap.) Output is 2d array in the same shape as the input array;

5. driftmap: calculate drift fields u(x,y) in both x and y directions;

6. driftcorr: local drift correction using u(x,y) on 2D (topo) or 3D(DOS map) input.

7. (OPTIONAL) chkperflat: check perfect lattice generated by Q1, Q2 (to be improved)

REFERENCES: 
[1] MH Hamidian, et al. "Picometer registration of zinc impurity states in Bi2Sr2CaCu2O8+d for phase determination in intra-unit-cell Fourier transform STM", New J. Phys. 14, 053017 (2012).
[2] JA Slezak, PhD thesis (Ch. 3), http://davisgroup.lassp.cornell.edu/theses/Thesis_JamesSlezak.pdf

20170428 CREATED BY JIANFENG GE

'''

def findBraggs(A, min_distance=2, threshold_rel=0.5, norm='linear', rspace=True, zero_center=True):
    '''
    find Bragg peaks of topo image A using peak_local_max, will plot modulus of FT of A with result points
    NOTE: PLEASE REMOVE NON-BRAGG PEAKS MANUALLY BY 'Bragg = np.delete(coords, [indices], axis=0)'

    Parameters: 
    min_distance - peaks are separated by at least min_distance in pixel;
    threshold_rel - minimum intensity of peaks, calculated as max(image) * threshold_rel
    norm: 'linear' or 'log', scale of display
    rspace: True - real space; False - reciprocal space
    zero_center - whether or not zero the DC point of Fourier transform
    '''
    if rspace:
        ft = np.fft.fft2(A)
        if zero_center:
            ft[0,0] = 0
        F = np.absolute(np.fft.fftshift(ft))
    else:
        F = np.copy(A)
    cnorm = mpl.colors.Normalize(vmin=F.min(), vmax=F.max())
    if norm is 'log':
        cnorm = mpl.colors.LogNorm(vmin=F.mean(), vmax=F.max())
    coords = peak_local_max(F, min_distance=min_distance, threshold_rel=threshold_rel)
    coords = np.fliplr(coords)
    plt.imshow(F, cmap=plt.cm.gray, origin='lower', norm=cnorm, aspect=1)
    plt.plot(coords[:, 0], coords[:, 1], 'r.')
    plt.gca().set_aspect(1)
    plt.axis('tight')
    print('#:\t[x y]')
    for ix, iy in enumerate(coords):
        print(ix, end='\t')
        print(iy)
    return coords
    

def gshearcorr(A, Bragg, rspace=True, slow_scan='None'):
    '''
    Shear correction based on FT of 2D array A
    Inputs:
    A - 2D or 3D array to be shear corrected;
    Bragg - 2D array of the shape (N, 2), e.g. Bragg = np.array([[x1, y1], [x2, y2], ...]), coordinates in pixel from FT image;
    rspace: True - correction in real space, False - in reciprocal space.

    Output: A_corr - corrected 2D or 3D array; Bragg_M - model points for Bragg peaks

    '''
    
    A_corr = np.zeros_like(A)
    if len(Bragg.shape) is 2 and Bragg.shape[1] is 2:
        if len(A.shape) is 2:
            tform, Bragg_M = calctform(Bragg, A.shape[-1], slow_scan=slow_scan)
            A_corr = corr2d(A, tform, rspace=rspace)
            return A_corr, Bragg_M
        if len(A.shape) is 3:
            tform, Bragg_M = calctform(Bragg, A.shape[-1], slow_scan=slow_scan)
            for ix, layer in enumerate(A):
                A_corr[ix] = corr2d(layer, tform, rspace=rspace)
            return A_corr, Bragg_M
        else:
            print('ERR: Input must be 2D or 3D numpy array!')
    else:
        print('Bragg peak coordinates should be 2D array of the shape (N, 2)')

def phasemap(A, Br_c, sigma=10):
    '''
    calculate local phase and phase shift maps
    Input:
    A - global shear corrected 2D (Topo.) array (cropped)
    Br_c: Bragg peaks of FT(A), N x 2 array, find them using findBraggs(A), remember to delete wrong points.
    sigma: width of DC filter.
    
    NOTE: PHASE SLIPS ARE VERY SENSITIVE TO SIGMA.
    TRY A SMALL VALUE FIRST AND THEN DO LOCAL DRIFT WITH PROPER VALUE AGAIN TO AVOID PHASE SLIPS
    
    Output: phix, phiy - phase maps; thetax, thetay - phase shift maps
    use plt.contour(x, y, np.cos(phix), [-0.9], colors='m', origin='lower', linewidths=0.5, nchunk=0)
    with imshow to check pi phase lines
    use np.diff(thetax, axis=0) to see phase slips, do not use gradient
    '''
    if A.shape[1] != A.shape[0]:
        print('Input is not a square 2D array!')
    else:
        s = A.shape[-1]
        Br_c = sortBraggs(Br_c, s)
        t = np.arange(s, dtype='float')
        x, y = np.meshgrid(t, t)
        Q1 = 2*np.pi*np.array([Br_c[0][0]-s/2, Br_c[0][1]-s/2])/s
        Q2 = 2*np.pi*np.array([Br_c[1][0]-s/2, Br_c[1][1]-s/2])/s
        Axx = A * np.sin(Q1[0]*x+Q1[1]*y)
        Axy = A * np.cos(Q1[0]*x+Q1[1]*y)
        Ayx = A * np.sin(Q2[0]*x+Q2[1]*y)
        Ayy = A * np.cos(Q2[0]*x+Q2[1]*y)
        Axxf = FTDCfilter(Axx, sigma)
        Axyf = FTDCfilter(Axy, sigma)
        Ayxf = FTDCfilter(Ayx, sigma)
        Ayyf = FTDCfilter(Ayy, sigma)
        thetax = np.arctan2(Axxf, Axyf)
        thetay = np.arctan2(Ayxf, Ayyf)
        #thetax -= thetax.mean()
        #thetay -= thetay.mean()
        return thetax, thetay, Q1, Q2

def chkphasemap(A, thetax, thetay, Q1, Q2, l=-0.95):
    '''
    Compare phase map with phi=pi lines.
    A - 2D array used to calculate phasemap
    l - a value a bit large than cos(pi)=-1 to draw the contour
    '''
    s = A.shape[-1]
    t = np.arange(s)
    x, y = np.meshgrid(t, t)
    phix = np.mod(-thetax + Q1[0]*x + Q1[1]*y, 2*np.pi)
    phiy = np.mod(-thetay + Q2[0]*x + Q2[1]*y, 2*np.pi)
    
    plt.figure(figsize=[8, 4])
    plt.subplot(121)
    plt.imshow(A, cmap=plt.cm.gray, interpolation='None', origin='lower left')
    plt.contour(x, y, np.cos(phix), [l], colors='m', origin='lower', linewidths=0.5, nchunk=0)
    plt.gca().set_aspect(1)
    plt.subplot(122)
    plt.imshow(A, cmap=plt.cm.gray, interpolation='None', origin='lower left')
    plt.contour(x, y, np.cos(phiy), [l], colors='y', origin='lower', linewidths=0.5, nchunk=0)
    plt.tight_layout()
    return phix, phiy

def fixphaseslip(A, thres=np.pi, method='spiral', orient=0):
    '''
    fix phase slip in phase shift maps by flattening A into a 1D array in a spiral way orientation is clockwise(0) or conter-clockwise(1).
    Fails when discontiuity between lines presents. If so, edges need to be cropped.
    '''
    def fixphaseslip1d(A, thres=np.pi):
        dA = np.diff(A, 1)
        slips = np.where(np.absolute(dA)>thres)[0]
        sliprm = np.zeros_like(A)
        for slip in slips:
            sliprm[:slip+1] += 2 * np.pi * np.sign(dA[slip])
        return A+sliprm
        ##################
        
        #B = np.copy(A)
        #for ix in range(len(A)-1):
        #    tmp = B[ix] - B[ix+1]
        #    if abs(tmp) > thres:
        #        B[:ix+1] -= np.sign(tmp) * np.around(abs(tmp/np.pi)) * np.pi
        #return B

    if method is 'spiral':
        if A.shape[0] != A.shape[1]:
            print('ERR: Input must be a square 2D array!')
        else:
            C = np.array([], dtype=A.dtype)
            n = A.shape[0]
            B = np.copy(A)
            if orient is 1:
                B = B.T
            for ix in range(int((n-1)/2)):
                C = np.append(C, B[0, :-1])
                C = np.append(C, B[:-1, -1])
                C = np.append(C, B[-1, -1:0:-1])
                C = np.append(C, B[-1:0:-1, 0])
                B = B[1:-1, 1:-1]
            if n%2:
                C = np.append(C, B[0, 0])
            else:
                C = np.append(C, B[0, :])
                C = np.append(C, B[-1, ::-1])
            
            D = fixphaseslip1d(C, thres=thres)
        
            D = D[::-1]
            if n%2:
                E, D = np.split(D, [1])
                E = E.reshape(1,1)
                start = 2
            else:
                E, D = np.split(D, [4])
                E = E[::-1].reshape(2,2)
                E[-1] = E[-1, ::-1]
                start = 3
            for ix in range(start, n, 2):
                E = np.pad(E, ((1, 1), (1, 1)), mode='constant')
                D1, D2, D3, D4, D = np.split(D, [ix, ix*2, ix*3, ix*4])
                E[1:, 0] = D1
                E[-1, 1:] = D2
                E[-2::-1, -1] = D3
                E[0, -2::-1] = D4
            if orient is 1:
                E = E.T
            return E
    else:
        print('Method not implemented!')

def driftmap(thetax, thetay, Q1, Q2):
    '''calculate drift maps based on phase shift maps, use Q1 and Q2 generated by phasemap'''
    tx = np.copy(thetax)
    ty = np.copy(thetay)
    #tx -= tx.mean()
    #ty -= ty.mean()
    ux = -(Q2[1]*tx - Q1[1]*ty) / (Q1[0]*Q2[1]-Q1[1]*Q2[0])
    uy = -(Q2[0]*tx - Q1[0]*ty) / (Q1[1]*Q2[0]-Q1[0]*Q2[1])
    return ux, uy

def driftcorr(A, ux, uy, interpolation='cubic'):
    ''' 
    drift correction on 2D image or 3D DOS map, use ux, uy calculated by driftmap. Crop edges of A_corr if needed.
    interpolation: 'linear', 'cubic' or 'quintic'. Default is 'cubic'
    '''

    A_corr = np.zeros_like(A)
    s = A.shape[-1]
    t = np.arange(s, dtype='float')
    x, y = np.meshgrid(t, t)
    xnew = (x - ux).ravel()
    ynew = (y - uy).ravel()
    tmp = np.zeros(s**2)

    if len(A.shape) is 2:
        tmp_f = interp2d(t, t, A, kind=interpolation)
        for ix in range(tmp.size):
            tmp[ix] = tmp_f(xnew[ix], ynew[ix])
        A_corr = tmp.reshape(s, s)
        return A_corr
    elif len(A.shape) is 3:
        for iz, layer in enumerate(A):
            tmp_f = interp2d(t, t, layer, kind=interpolation)
            for ix in range(tmp.size):
                tmp[ix] = tmp_f(xnew[ix], ynew[ix])
            A_corr[iz] = tmp.reshape(s, s)
            print('Processing slice %d/%d...'%(iz+1, A.shape[0]), end='\r')
        return A_corr
    else:
        print('ERR: Input must be 2D or 3D numpy array!')

def chkperflat(A_corr, Q1, Q2, shiftx=0, shifty=0):
    '''check corrected topo image with perfect lattice. Manually find smallest shifts of x and y in corrected image to align atoms'''
    s = A_corr.shape[-1]
    t = np.arange(s*10, dtype='float')
    x, y = np.meshgrid(t, t)
    Q3 = Q1-Q2 #2*np.pi*np.array([Bragg_c[2][0]-s/2, Bragg_c[2][1]-s/2])/s
    P = np.cos(Q1[0]/10*x+Q1[1]/10*y)+np.cos(Q2[0]/10*x+Q2[1]/10*y)+np.cos(Q3[0]/10*x+Q3[1]/10*y)
    P = np.roll(P, int(shifty*10), axis=0)
    P = np.roll(P, int(shiftx*10), axis=1)
    perflat = peak_local_max(P, min_distance=5, threshold_rel=0.7)
    perflat = np.fliplr(perflat)
    plt.imshow(A_corr, cmap=plt.cm.afmhot, interpolation='None', origin='lower left')
    plt.plot(perflat[:, 0]/10, perflat[:, 1]/10, 'b.')
    plt.gca().set_aspect(1)
    plt.axis('tight')

##################################################################################
####################### Useful functions in the processing #######################
##################################################################################

def squareinterp(A, kind='linear'):
    '''
    Square rectangular 2D arrays (For 3D arrays, sqaure each slice), by interpolating short axis to the same length of the long axis
    kind can be 'linear', 'cubic' or 'quintic'
    '''
    x = A.shape[-1]
    y = A.shape[-2]
    xx = np.linspace(0, 1, x)
    yy = np.linspace(0, 1, y)
    s = max(x, y)
    ss = np.linspace(0, 1, s)
    if len(A.shape) is 2:
        B = np.zeros((s, s))
        f = interp2d(xx, yy, A, kind=kind)
        B = f(ss, ss)
        return B
    elif len(A.shape) is 3:
        B = np.zeros((A.shape[0], s, s))
        for ix, layer in enumerate(A):
           f = interp2d(xx, yy, layer, kind=kind)
           B[ix] = f(ss, ss)
        return B
    else:
        print('ERR: Input must be 2D or 3D numpy array!')

def sortBraggs(Br, s):
    ''' sort Bragg peaks in conter-clockwise way around center, image shape is (s, s) '''
    Br_s = np.copy(Br)
    c = int((s+1)/2)
    xn = Br_s[:, 0]/c-1
    yn = Br_s[:, 1]/c-1
    theta = np.arctan2(yn, xn)
    atansorted = np.asarray(sorted((atanval, ix) for ix, atanval in enumerate(theta)))# sort angles of vertices in rad
    tanseq = np.int_(atansorted[:, 1])
    Br_s = Br_s[tanseq] # sort Bragg peaks accordingly
    return Br_s

def calctform(Br, s, slow_scan):
    ''' 
    calculate projective transform matrix by Bragg peaks Br with coordinates in the range of F.shape
    slow_scan: None

    '''
    N = Br.shape[0]
    c = s*0.5
    Br = sortBraggs(Br, s)
    xn = Br[:, 0]/c-1
    yn = Br[:, 1]/c-1
    R = np.mean(np.sqrt(xn**2+yn**2))
    theta = np.arctan2(yn, xn)
    theta_M = (np.arange(N)*2./N-1)*np.pi # Model angles of vertices, in rad, range (-pi, pi)
    dtheta = np.mean(theta - theta_M)
    xn_M = R * np.cos(theta_M+dtheta) # Generate Model coordinates
    yn_M = R * np.sin(theta_M+dtheta) # based on first point

    # rescale for fast scan direction, i.e. trust the coordinate of the fast scan direction
    if slow_scan is 'y':
        R *= np.mean(abs(yn/yn_M))
        xn_M = R * np.cos(theta_M+dtheta) # Generate Model coordinates
        yn_M = R * np.sin(theta_M+dtheta)
    if slow_scan is 'x':
        R *= np.mean(abs(xn/xn_M))
        xn_M = R * np.cos(theta_M+dtheta) # Generate Model coordinates
        yn_M = R * np.sin(theta_M+dtheta)
    Br_M = np.concatenate(((xn_M+1)*c, (yn_M+1)*c)).reshape(2, N).T # Back to original coordinates
    tform = tf.ProjectiveTransform()
    if tform.estimate(Br_M, Br):
        return tform, Br_M
            
def corr2d(A, tform, rspace=True):
    ''' shear correction of 2d image in real or reciprocal space, called by gshearcorr'''
    matrix = np.copy(tform.params)
    A_corr2d = np.zeros_like(A)
    # calctform is used to estimate transformation matrix in reciprocal space
    # NOTE: warp function use inverse matrix to do forward transformation
    # in real space the inverse matrix does not equal to tform.params, because translation operation after inverse FT vanishes
    # Due to fftshift the origin of FT differs from that in real space, which should be changed before applying transformation

    if rspace:
        matrix[-1, :] = np.array([0., 0., 1.])
        matrix[:, -1] = np.array([0., 0., 1.])
        A_corr2d = tf.warp(np.flipud(A.T), matrix, preserve_range=True, order=5)
        A_corr2d = np.flipud(A_corr2d).T
    else:
        A_corr2d = tf.warp(A, matrix, preserve_range=True, order=5)
    return A_corr2d

def cropedge(A, n):
    ''' crop off n pixel from edges of Topo image or each layer of DOS map accordingly, useful after shear correction and fixphaseslip'''
    B = np.copy(A)
    if len(B.shape) is 2:
        B = B[n:-n, n:-n]
    elif len(B.shape) is 3:
        B = B[:, n:-n, n:-n]
    else:
        print('ERR: Input must be 2D or 3D numpy array!')
    print('Shape before crop:', end=' ')
    print(A.shape)
    print('Shape after crop:', end=' ')
    print(B.shape)
    return B

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

def FTDCfilter(A, sigma):
    '''
    Filtering DC component of Fourier transform and inverse FT, using a gaussian with one parameter sigma
    A is a 2D array, sigma is in unit of px
    '''
    if A.shape[1] != A.shape[0]:
        print('ERR: not a sqare FFT!')
    else:
        n = A.shape[1]
        m = np.arange(n, dtype='float')
        c = np.float((n-1)/2)
    g = Gaussian2d(m, m, sigma, sigma, 0, c, c, 1)
    ft_A = np.fft.fftshift(np.fft.fft2(A))
    ft_Af = ft_A * g
    Af = np.fft.ifft2(np.fft.ifftshift(ft_Af))
    return np.real(Af)

def chkphaseslip(thetax, thetay):
    ''' check phase slips by differentiate on both axes '''
    dthetax0 = np.diff(thetax, axis=0)
    dthetax1 = np.diff(thetax, axis=1)
    dthetay0 = np.diff(thetay, axis=0)
    dthetay1 = np.diff(thetay, axis=1)
    dtheta = [dthetax0, dthetax1, dthetay0, dthetay1]
    s = thetax.shape[0]
    plt.figure(figsize=[8, 8])
    for ix in range(4):
        plt.subplot(2, 2, ix+1)
        plt.imshow(dtheta[ix]/np.pi, cmap=plt.cm.bwr, interpolation='None', origin='lower left')
        plt.gca().set_aspect(1)
        plt.clim(-2, 2)
        plt.xlim(0, s)
        plt.ylim(0, s)
        plt.colorbar(fraction=0.03)
    plt.tight_layout()

def spiralflatten(A, orient=0):
    '''
    Spiral flattening a square(?) 2D array
    orient: orientation, 0 for clockwise, 1 for counter-clockwise
    '''
    C = np.array([], dtype=A.dtype)
    n = A.shape[-1]
    B = np.copy(A)
    if orient is 1:
        B = B.T
    for ix in range(int((n-1)/2)):
        C = np.append(C, B[0, :-1])
        C = np.append(C, B[:-1, -1])
        C = np.append(C, B[-1, -1:0:-1])
        C = np.append(C, B[-1:0:-1, 0])
        B = B[1:-1, 1:-1]
    if n%2:
        C = np.append(C, B[0, 0])
    else:
        C = np.append(C, B[0, :])
        C = np.append(C, B[-1, ::-1])
    return C

def spiralroll(B, orient=1):
    ''' undo spiral flatten '''
    k = int(np.sqrt(B.size))
    if k**2-B.size != 0:
        print('ERR: unable to form a square 2D array!')
    else:
        C = np.copy(B)
        C = C[::-1]
        if k%2:
            A, C = np.split(C, [1])
            A = A.reshape(1,1)
            start = 2
        else:
            A, C = np.split(C, [4])
            A = A[::-1].reshape(2,2)
            A[-1] = A[-1, ::-1]
            start = 3
        for ix in range(start, k, 2):
            A = np.pad(A, ((1, 1), (1, 1)), mode='constant')
            C1, C2, C3, C4, C = np.split(C, [ix, ix*2, ix*3, ix*4])
            A[1:, 0] = C1
            A[-1, 1:] = C2
            A[-2::-1, -1] = C3
            A[0, -2::-1] = C4
        if orient is 0:
            A = A.T
        return A
def findBZvertices(A, Br):
    '''
    Find vertices of Brillouin zone (BZ) based on coordinates of Bragg peaks in Fourier space image.
    
    A: 2-D array, Fourier transformed image
    Br: (N, 2) array, coordinates of all Bragg peaks
    output: coordinates of BZ vertices
    usage: K = findBZvertices(FT, Br)
    plot together with FT image: imshow(FT, origin='lower left'); gca().add_patch(Polygon(V, fill=None, color='b'))
    '''
    def findV(C, B1, B2):
        C = np.array(C)
        B1 = np.array(B1-C)
        B2 = np.array(B2-C)
        theta1 = np.arctan2(B1[1], B1[0]) + np.pi/2.
        theta2 = np.arctan2(B2[1], B2[0]) + np.pi/2.
        V = np.linalg.solve(np.array([[-np.tan(theta1), 1], [-np.tan(theta2), 1]]), np.array([B1[1]/2.-np.tan(theta1)*B1[0]/2., B2[1]/2.-np.tan(theta2)*B2[0]/2.]))
        return C + V

    C = np.array(A.shape[-2:])/2.
    Br = np.array(Br)
    Br = sortBraggs(Br, A.shape[0])
    N = Br.shape[0]
    V = np.zeros_like(Br)
    for ix in range(N-1):
        V[ix] = findV(C, Br[ix], Br[ix+1])
    V[-1] = findV(C, Br[-1], Br[0])
    return V

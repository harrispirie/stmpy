# -*- coding: utf-8 -*-

from __future__ import print_function
import stmpy
import sys
import numpy as np
import matplotlib as mpl
#import scipy.interpolate as sin  #this is a stupid name for this package...
from scipy.interpolate import interp1d
import scipy.optimize as opt
import scipy.ndimage as snd
from scipy.signal import butter, filtfilt, fftconvolve, hilbert, correlate


def interp2d(x, y, z, kind='nearest', **kwargs):
    '''
    An extension for scipy.interpolate.interp2d() which adds a 'nearest'
    neighbor interpolation.

    See help(scipy.interpolate.interp2d) for details.

    Inputs:
        x       - Required : Array contining x values for data points.
        y       - Required : Array contining y values for data points.
        z       - Required : Array contining z values for data points.
        kind    - Optional : Sting for interpolation scheme. Options are:
                             'nearest', 'linear', 'cubic', 'quintic'.  Note
                             that 'linear', 'cubic', 'quintic' use spline.
        **kwargs - Optional : Keyword arguments passed to
                              scipy.interpolate.interp2d

    Returns:
        f(x,y) - Callable function which will return interpolated values.

    History:
        2017-08-24  - HP : Initial commit.
    '''
    from scipy.interpolate import NearestNDInterpolator
    if kind is 'nearest':
        X, Y = np.meshgrid(x ,y)
        points = np.array([X.flatten(), Y.flatten()]).T
        values = z.flatten()
        fActual = NearestNDInterpolator(points, values)
        def fCall(x, y):
            if type(x) is not np.ndarray:
                lx = 1
            else:
                lx = x.shape[0]
            if type(y) is not np.ndarray:
                ly = 1
            else:
                ly = y.shape[0]
            X, Y = np.meshgrid(x ,y)
            points = np.array([X.flatten(), Y.flatten()]).T
            values = fActual(points)
            return values.reshape(lx, ly)
        return fCall
    else:
        from scipy.interpolate import interp2d as scinterp2d
        return scinterp2d(x, y, z, kind=kind, **kwargs)


def azimuthalAverage(F, x0, y0, r, theta=np.linspace(0,2*np.pi,500),
        kind='linear'):
    ''' Uses 2D interpolation to average F over an arc defined by theta
    for every r value starting from x0,y0.

    History:
        2017-08-24  - HP : Modified to use stmpy.tools.interp2d().
    '''
    f = interp2d(np.arange(F.shape[1]), np.arange(F.shape[0]), F, kind=kind)
    Z = np.zeros_like(r); fTheta = np.zeros_like(theta)
    for ix, r0 in enumerate(r):
        x = r0*np.cos(theta) + x0
        y = r0*np.sin(theta) + y0
        for iy, (xn,yn) in enumerate(zip(x,y)):
            fTheta[iy] = f(xn,yn)
        Z[ix] = np.mean(fTheta)
    return Z


def azimuthalAverageRaw(F,x0,y0,rmax):
    ''' Azimuthally average beginning at x0,y0 to a maximum distance rmax'''
    f=[]; p=[]; R=[]; FAvg=[]
    for x in range(F.shape[1]):
        for y in range(F.shape[0]):
            r = np.sqrt((x-x0)**2 + (y-y0)**2)
            if r <= rmax:
                p.append(r)
                f.append(F[y,x])
    for r0 in set(np.sort(p)):
        R.append(r0)
        allFVals = [f0 for ix,f0 in enumerate(f) if p[ix] == r0]
        FAvg.append(np.mean(allFVals))
    R = np.array(R); ixSorted = R.argsort()
    return R[ixSorted], np.array(FAvg)[ixSorted]


def arc_linecut(data, p0, length, angle, width=20, dl=0, dw=100, kind='linear',
        show=False, ax=None, **kwarg):
    '''A less cumbersome wrapper for stmpy.tools.azimuthalAverage.  Computes an
    arc-averaged linecut on 2D data, or on each layer in 3D data.

    Inputs:
        data    - Required : A 2D or 3D numpy array.
        p0      - Required : A tuple containing indicies for the start of the
                             linecut: p0=(x0,y0)
        length  - Required : Float containing length of linecut to compute.
        angle   - Required : Angle (IN DEGREES) to take the linecut along.
        width   - Optional : Angle (IN DEGREES) to average over.
        dl      - Optional : Extra pixels for interpolation in the linecut
                             direction.
        dw      - Optional : Number of pixels for interpolation in the
                             azimuthal direction: default 100.
        kind    - Optional : Sting for interpolation scheme. Options are:
                             'nearest', 'linear', 'cubic', 'quintic'.  Note
                             that 'linear', 'cubic', 'quintic' use spline.
        show    - Optional : Boolean determining whether to plot where the
                             linecut was taken.
        ax      - Optional : Matplotlib axes instance to plot where linecut is
                             taken.  Note, if show=True you MUST provide and
                             axes instance as plotting is done using ax.plot().
        **kwarg - Optional : Additional keyword arguments passed to ax.plot().

    Returns:
        r   -   1D numpy array which goes from 0 to the length of the cut.
        cut -   1D or 2D numpy array containg the linecut.

    Usage:
        r, cut = arc_linecut(data, cen, length, angle, width=20, dl=0, dw=100,
                             show=False, ax=None, **kwarg):

    History:
        2017-07-20  - HP : Initial commit.
        2017-08-24  - HP : Modified to use stmpy.tools.interp2d() for
                           interpolation, which allows for 'nearest'.
    '''
    theta = np.radians(angle)
    dtheta = np.radians(width/2.0)
    r = np.linspace(0, length, round(length+dl))
    t = np.linspace(theta-dtheta, theta+dtheta, round(dw))
    if len(data.shape) == 2:
        cut = azimuthalAverage(data, p0[0], p0[1], r, t, kind=kind)
    elif len(data.shape) == 3:
        cut = np.zeros([data.shape[0], len(r)])
        for ix, layer in enumerate(data):
            cut[ix] = azimuthalAverage(layer, p0[0], p0[1], r, t, kind=kind)
    else:
        raise TypeError('Data must be 2D or 3D numpy array.')
    if show:
        ax.plot([p0[0], p0[0]+length*np.cos(theta-dtheta)],
                [p0[1], p0[1]+length*np.sin(theta-dtheta)], 'k--', lw=1, **kwarg)
        ax.plot([p0[0], p0[0]+length*np.cos(theta+dtheta)],
                [p0[1], p0[1]+length*np.sin(theta+dtheta)], 'k--', lw=1, **kwarg)
    return r, cut


def binData(x,y,nBins):
    ''' For any randomly sampled data x,y, return a histogram with linear bin spacing'''
    # Issues: What if there are no elements in a bin?
    binSize = max(x)/nBins; X=[];Y=[]
    for n in range(nBins):
        allBinYVal = []
        minR = n * binSize; maxR = (n+1) * binSize;
        for ix,R in enumerate(x):
            if R >= minR and R < maxR:
                allBinYVal.append(y[ix])
        X.append( (minR + maxR) / 2.0 )
        Y.append( np.mean(allBinYVal) )
    return X,Y


def linecut_old(F, x1, y1, x2, y2, n):
    ''' Use linear interpolation on a 2D data set F, sample along a line from (x1,y1) to (x2,y2) in n points

Usage:  x_linecut, y_linecut = linecut(image, x1, y1, x2, y2, n)

History:
    2017-06-19  - HP : Changed name to linecut_old (will be replaced by
                       linecut)
    '''
    x = np.arange(F.shape[0])
    y =  np.arange(F.shape[1])
    cen = np.sqrt((x1-x2)**2 + (y1-y2)**2) / 2.0
    r = np.linspace(-1*cen, cen, n)
    f = interp2d(x, y, F, kind = 'linear')
    xval = np.linspace(x1, x2, n)
    yval = np.linspace(y1, y2, n)
    z = [f(xval[ix],yval[ix])[0] for ix in range(n)]
    return r, np.array(z)


def squareCrop(image,m=None):
    ''' Crops a 2D image to be mxm, where m is an even integer. '''
    image = np.array(image)
    a,b =image.shape
    if m is None: m = min(a,b)
    if m%2 != 0: m-=1
    imageCrop = image[:m,:m]
    return imageCrop


def lineCrop(x, y, cropRange):
    ''' Crops a 1D line using a list of start, stop values. Can delete sections of data
        Usage: xCrop,yCrop = lineCrop(x, y, [start,stop,start,stop,...])'''
    cropIndex = []; xCrop = []; yCrop = []
    for cropVal in cropRange:
        a = [ix for ix,x0 in enumerate(x) if x0 >= cropVal]
        if a == []:
            if cropVal <= x[0]: a = [0]
            else: a = [len(x)-1]
        cropIndex.append(a[0])
    for ix in range(0,len(cropIndex),2):
        xCrop += x.tolist()[cropIndex[ix]:cropIndex[ix+1]+1]
        yCrop += y.tolist()[cropIndex[ix]:cropIndex[ix+1]+1]
    return np.array(xCrop), np.array(yCrop)


def removePolynomial1d(y, n, x=None, fitRange=None):
    ''' Removes a background polynomial of degree n to the line y(x) in the range fitRange (optional).
        Usage: yCorrected = removePolynomial1d(y, n)'''
    if x is None: x=np.linspace(0, 1, len(y))
    if fitRange is None: fitRange = [x[0],x[-1]]
    xBkg,yBkg = lineCrop(x,y,fitRange)
    polyCoeff = np.polyfit(xBkg,yBkg,n)
    polyBackgroundFunction = np.poly1d(polyCoeff)
    return y - polyBackgroundFunction(x)

def lineSubtract(data, n=1, maskon=False, thres=4, M=4, normalize=True, colSubtract=False):
    '''
    Remove a polynomial background from the data line-by-line, with
    the option to skip pixels within certain distance away from
    impurities.  If the data is 3D (eg. 3ds) this does a 2D background
    subtract on each layer independently.  Input is a numpy array.

    Inputs:
        data    -   Required : A 1D, 2D or 3D numpy array.
        n       -   Optional : Degree of polynomial to subtract from each line.
                               (default : 1).
        maskon  -   Optional : Boolean flag to determine if the impurty areas are excluded.
        thres   -   Optional : Float number specifying the threshold to determine
                               if a pixel is impurity or bad pixels. Any pixels with intensity greater
                               than thres*std will be identified as bad points.
        M       -   Optional : Integer number specifying the box size where all pixels will be excluded
                               from poly fitting.
        normalize - Optional : Boolean flag to determine if the mean of a layer
                               is set to zero (True) or preserved (False).
                               (default : True)
        colSubtract - Optional : Boolean flag (False by default) to determine if polynomial background should also be subtracted column-wise

    Returns:
        subtractedData  -   Data after removing an n-degree polynomial

    Usage:
        dataObject.z = lineSubtract(dataObject.Z, n=1, normalize=True)
        dataObject.z = lineSubtract(dataObject.Z, n=1, mask=True, thres=1.5, M=4, normalize=True)

    History:
        2017-07-19  - HP : Updated to work for 1D data.
        2018-06-07  - MF : Updated to do a background subtract in the orthogonal direction (ie. column-wise)
        2018-11-04  - RL : Updated to add mask to exclude impurity and bad pixels in polyfit.
        2018-11-05  - RL : Update to add support for 1D and 3D data file.
    '''
    # Polyfit for lineSubtraction excluding impurities.
    def filter_mask(data, thres, M, D):
        filtered = data.copy()
        if D == 1:
            temp = np.gradient(filtered)
            badPts = np.where(np.abs(temp-np.mean(temp))>thres*np.std(temp))
            for ix in badPts[0]:
                filtered[max(0, ix-M) : min(data.shape[0], ix+M+1)] = np.nan
            return filtered
        elif D == 2:
            temp = np.gradient(filtered)[1]
            badPts = np.where(np.abs(temp-np.mean(temp))>thres*np.std(temp))
            for ix, iy in zip(badPts[1], badPts[0]):
                filtered[max(0, iy-M) : min(data.shape[0], iy+M+1),
                                  max(0, ix-M) : min(data.shape[1], ix+M+1)] = np.nan
            return filtered

    def subtract_mask(data, n, thres, M, D):
        d = data.shape[0]
        x = np.linspace(0,data.shape[-1]-1,data.shape[-1])
        filtered = filter_mask(data, thres, M, D)
        output = data.copy()
        if D == 1:
            index = np.isfinite(filtered)
            try:
                popt = np.polyfit(x[index], data[index], n)
                output = data - np.polyval(popt, x)
            except TypeError:
                raise TypeError('Empty x-array encountered. Please use a larger thres value.')
            return output
        if D == 2:
            for i in range(d):
                index = np.isfinite(filtered[i])
                try:
                    popt = np.polyfit(x[index], data[i][index], n)
                    output[i] = data[i] - np.polyval(popt, x)
                except TypeError:
                    raise TypeError('Empty x-array encountered. Please use a larger thres value.')
            return output

    if maskon is not False:
        if len(data.shape) == 3:
            output = np.zeros_like(data)
            for ix, layer in enumerate(data):
                output[ix] = subtract_mask(layer, n, thres, M, 2)
            return output
        elif len(data.shape) == 2:
            return subtract_mask(data, n, thres, M, 2)
        elif len(data.shape) == 1:
            return subtract_mask(data, n, thres, M, 1)
        else:
            raise TypeError('Data must be 1D, 2D or 3D numpy array.')

    # The original lineSubtract code.
    def subtract_1D(data, n):
        x = np.linspace(0,1,len(data))
        popt = np.polyfit(x, data, n)
        return data - np.polyval(popt, x)
    def subtract_2D(data, n):
        if normalize:
            norm = 0
        else:
            norm = np.mean(data)
        output = np.zeros_like(data)
        for ix, line in enumerate(data):
            output[ix] = subtract_1D(line, n)
        if colSubtract:
            temp = np.zeros_like(data)
            for ix, line in enumerate(np.transpose(output)):
                temp[ix] = subtract_1D(line, n)
            output = np.transpose(temp)
        return output + norm

    if len(data.shape) == 3:
        output = np.zeros_like(data)
        for ix, layer in enumerate(data):
            output[ix] = subtract_2D(layer, n)
        return output
    elif len(data.shape) == 2:
        return subtract_2D(data, n)
    elif len(data.shape) == 1:
        return subtract_1D(data, n)
    else:
        raise TypeError('Data must be 1D, 2D or 3D numpy array.')

def fitGaussian2d(data, p0):
    ''' Fit a 2D gaussian to the data with initial parameters p0. '''
    data = np.array(data)
    def gauss(xy,amplitude,x0,y0,sigmaX,sigmaY,theta,offset):
        x,y = xy
        x0=float(x0);y0=float(y0)
        a =  0.5*(np.cos(theta)/sigmaX)**2 + 0.5*(np.sin(theta)/sigmaY)**2
        b = -np.sin(2*theta)/(2*sigmaX)**2 + np.sin(2*theta)/(2*sigmaY)**2
        c =  0.5*(np.sin(theta)/sigmaX)**2 + 0.5*(np.cos(theta)/sigmaY)**2
        g = offset+amplitude*np.exp(-( a*(x-x0)**2 -2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ))
        return g.ravel()
    x = range(data.shape[0]);  y = range(data.shape[1])
    X,Y = np.meshgrid(x,y)
    popt, pcov = opt.curve_fit(gauss, (X,Y), data.ravel(), p0=p0)
    return gauss((X,Y),*popt).reshape(data.shape)


def findOtherBraggPeaks(FT, bpx, bpy, n = 1):
    '''Once one bragg peak is found this retruns n other bragg peak harmonics by symmetry of the Fourier transform.'''
    N  = range(-n,0)+range(1,n+1)
    Bpx = [];  Bpy = []
    cenX = FT.shape[1]/2.0;  cenY = FT.shape[0]/2.0
    rScale = np.sqrt((bpx-cenX)**2 + (bpy-cenY)**2)
    dx = bpx - cenX;  dy = bpy - cenY
    for i in N:
        Bpx.append(cenX + i*dx)
        Bpy.append(cenY + i*dy)
    return Bpx,Bpy,rScale


def findPeaks(x,y,n=1,nx=1e3):
    ''' Finds n peaks in a given 1D-line by finding where the derivative crosses zero. '''
    f = interp1d(x,y,kind='linear')
    xx = np.linspace(x[0],x[-1],nx); fx = f(xx)
    df = np.gradient(fx)
    x0=[]; y0=[]
    for ix, df0 in enumerate(df):
        if ix != 0:
            if df0/df[ix-1] < 0:
                x0.append(xx[ix])
                y0.append(fx[ix])
    xn = [x for (y,x) in sorted(zip(y0,x0))][::-1][:n]
    yn = sorted(y0)[::-1][:n]
    if xn == []: xn = [0] * n
    if yn == []: yn = [0] * n
    return xn, yn


def locmax1d(x, min_distance=1, thres_rel=0, thres_abs=-np.inf):
    '''Return indices of local maxima in an 1d array x'''
    d = int(min_distance) # forced to be integer number of points
    tr = thres_rel * (x.max() - x.min()) + x.min() # threshold
    t = max(thres_abs, tr)
    mask = np.full_like(x, False, dtype=bool) # mask array for output indices
    x2 = np.concatenate((np.repeat(x[0], d), x, np.repeat(x[-1], d))) # constant values added to the edge
    for ix, value in enumerate(x):
        if value > t: # condition 1: value exceeds threshold
            x_seg = np.take(x2, range(ix, ix+2*d+1))
            if value == x_seg.max(): # condition 2: x[ix] is maximum of x_seg
                mask[ix] =  True
    return np.where(mask == True)[0]


def removeGaussian2d(image, x0, y0, sigma):
    '''Removes a isotropic gaussian of width sigma centered at x0,y0 from image.'''
    x0 = float(x0); y0 = float(y0)
    a = - 0.5 / sigma**2
    x = np.arange(image.shape[0]); y = np.arange(image.shape[1])
    X,Y = np.meshgrid(x,y)
    g = np.exp( a*(X-x0)**2 + a*(Y-y0)**2 )
    return image * (1 - g)


def fitGaussians1d(x, y, p0):
    ''' Fits n gaussians to the line y(x), with guess parameters p0 = [amp_1,mu_1,sigma_1,amp_2,mu_2,sigma_2,...]'''
    def gaussn(x, *p):
        g = np.zeros_like(x)
        for i in range(0,len(p),3):
            amp = abs(float( p[i] ))
            mu = float( p[i+1] )
            sigma = float( p[i+2] )
            g += amp * np.exp(- (x-mu)**2 / (2.0*sigma**2) )
        return g
    p,cov = opt.curve_fit(gaussn, x, y, p0)
    return p,gaussn(x,*p)


def foldLayerImage(layerImage,bpThetaInRadians=0,n=4):
    ''' Returns a n-Fold symmetric layer image, where the fold directions are defined by bpTheta. '''
    B = np.zeros_like(layerImage)
    bpTheta = bpThetaInRadians * (180.0/np.pi) + 45
    for ix,layer in enumerate(layerImage):
        if bpTheta != 0: layer = snd.interpolation.rotate(layer,bpTheta,reshape = False)
        step1 = (layer + layer.T) / 2.0
        step2 = (step1 + np.flipud((np.fliplr(step1)))) / 2.0
        step3 = (step2 + np.fliplr(step2)) / 2.0
        options = { 1 : step1,
                2 : step2,
                4 : step3  }
        B[ix] = options.get(n,0)
        if bpTheta != 0: B[ix] = snd.interpolation.rotate(B[ix],-bpTheta,reshape = False)
    if n not in options.keys(): print('{:}-fold symmetrization not yet implemented'.format(n))
    return B


def quickFT(data, n=None, zero_center=True, bp=(1.,1.), diag=True):
    '''
    A hassle-free FFT for 2D or 3D data.  Useful for quickly computing the QPI
    patterns from a DOS map. Returns the absolute value of the FFT for each
    layer in the image. Has the option of setting the center pixel to zero and
    the option to n-fold symmetrize the output.

    Usage: A.qpi = quickFT(A.LIY, zero_center=True, n=None)
    '''
    def ft2(data):
        ft = np.fft.fft2(data)
        if zero_center:
            ft[0,0] = 0
        return np.absolute(np.fft.fftshift(ft))
    if len(data.shape) is 2:
        if n is None:
            return ft2(data)
        else:
            return symmetrize(ft2(data), n, bp=bp, diag=diag)
    if len(data.shape) is 3:
        output = np.zeros_like(data)
        for ix, layer in enumerate(data):
            output[ix] = ft2(layer)
        if n is None:
            return output
        else:
            return symmetrize(output, n, bp=bp, diag=diag)
    else:
        print('ERR: Input must be 2D or 3D numpy array.')

def GMKhexagon(br, s, ec='k', lw=1):
    '''generate Gamma, M, K points and hexagon by one Bragg point (image should be symmetrized).'''
    def rotatexy(p1, p0, alpha, zoom=1):
        '''rotate point pi:(xi, yi) around p0:(x0, y0) for an angle alpha'''
        x1, y1 = p1[0], p1[1]
        x0, y0 = p0[0], p0[1]
        xf = x0 + zoom * ((x1 - x0)*np.cos(alpha) - (y1 - y0)*np.sin(alpha))
        yf = y0 + zoom * ((y1 - y0)*np.cos(alpha) + (x1 - x0)*np.sin(alpha))
        return xf, yf


    x0, y0 = int(s/2), int(s/2)
    x1, y1 = br[0], br[1]
    x4, y4 = s-x1, s-y1

    a1, b1 = rotatexy((x1, y1), (x0, y0), np.pi/6, 2/np.sqrt(3))
    a4, b4 = rotatexy((x4, y4), (x0, y0), np.pi/6, 2/np.sqrt(3))
    a2, b2 = rotatexy((a1, b1), (x0, y0), np.pi/3)
    a3, b3 = rotatexy((a2, b2), (x0, y0), np.pi/3)
    a5, b5 = rotatexy((a4, b4), (x0, y0), np.pi/3)
    a6, b6 = rotatexy((a5, b5), (x0, y0), np.pi/3)

    G = np.array([x0, y0])
    K = np.array([a1, b1])
    M = np.array([(a1+a2)/2, (b1+b2)/2])
    hexagon = mpl.patches.Polygon(xy=[(a1, b1),(a2, b2),(a3, b3),(a4, b4),(a5, b5),(a6, b6)],\
                                  closed=True, fill=None, ec=ec, lw=lw)
    return G, M, K, hexagon

def symmetrize(data, n, bp=(1.,1.), diag=False):
    '''
    Applies n-fold symmetrization to the image by rotating clockwise and
    anticlockwise by an angle 2pi/n, then applying a mirror line.  Works on 2D
    and 3D data sets, in the case of 3D each layer is symmetrzed.
    p is the location of one Bragg peak.

    Inputs:
        data    - Required : A 2D or 3D numpy array.
        n       - Required : Integer describing the degree of symmetrization.
        bp      - Optional : Pixel coordinates of the Bragg peak to define the
                             mirror line.
        diag    - Optional : Boolean to assert whether the mirror line is left
                             on the diagonal.

    Returns:
        dataSymm - A 2D or 3D numpy array containing symmetrized data.

    History:
        2017-05-04  - JG : Initial commit.
        2017-06-05  - HP : Modified default bp-value to be on diagonal.
        2017-08-15  - HP : Added flag to leave mirror line on the diagonal.
                           Code will not line mirror unsquare data.
     '''
    def sym2d(F, n):
        angle = 360.0/n
        out = np.zeros_like(F)
        for ix in range(n):
            out += snd.rotate(F, angle*ix, reshape=False)
            out += snd.rotate(F, -angle*ix, reshape=False)
        out /= 2*n
        return out

    def linmirr(F, x1, y1):
        x0 = int(F.shape[0]/2.)
        y0 = int(F.shape[1]/2.)
        if x0 == y0:
            # angle between mirror line and diagonal line, unit in rad
            alpha = 3*np.pi/4-np.arctan((y1-y0)/(x1-x0))
            # rotate the mirror line to be diagonal
            Fr = snd.rotate(F, -alpha/np.pi*180, reshape=False)
            Ff = Fr.T # diagnoal mirror
            if diag:
                return (Ff+Fr)/2.0
            else:
                Ffr = snd.rotate(Ff, alpha/np.pi*180, reshape=False) # rotate back
                return (Ffr+F)/2.0
        else:
            return F
    p = np.array(bp, dtype=np.float64)
    if len(data.shape) is 2:
            return linmirr(sym2d(data, n), p[0], p[1])
    if len(data.shape) is 3:
        out = np.zeros_like(data)
        for ix, layer in enumerate(data):
            out[ix] = linmirr(sym2d(layer, n), p[0], p[1])
        return out
    else:
        print('ERR: Input must be 2D or 3D numpy array.')


def gauss2d(x, y, p, symmetric=False):
    '''Create a two dimensional Gaussian.

    Inputs:
        x   - Required : 1D array containing x values
        y   - Required : 1D array containing y values.  The funciton will
                         create a meshgrid from x and y, but should be called
                         like f(x, y, *args).
        p   - Required : List of parameters that define the gaussian, in the
                         following order: [x0, y0, sigmax, sigmay, Amp, theta]
        symmetric   - Optional : Boolean, if True this will add another
                                 Gaussian at (cenx - x0, ceny - y0), which is
                                 useful in frequency space.

    Returns:
        G   -   2D array containing Gaussian.

    History:
        2018-03-30  - HP : Initial commit.
    '''
    x0, y0, sx, sy, A, theta = [float(val) for val in p]
    X, Y = np.meshgrid(x, y);
    theta = np.radians(theta)
    a = np.cos(theta)**2/(2*sx**2) + np.sin(theta)**2/(2*sy**2)
    b = -np.sin(2*theta)/(4*sx**2) + np.sin(2*theta)/(4*sy**2)
    c = np.sin(theta)**2/(2*sx**2) + np.cos(theta)**2/(2*sy**2)
    G = A*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    if symmetric:
        x1 = x[-1] + x[0] - x0
        y1 = y[-1] + y[0] - y0
        G += A*np.exp( -(a*(X-x1)**2 + 2*b*(X-x1)*(Y-y1) + c*(Y-y1)**2))
    return G

def gauss_ring(x, y, major, sigma, minor=None, theta=0, x0=None, y0=None):
    '''
    Create a 2D ring with a gaussian cross section.

    Inputs:
        x   - Required : 1D array containing x values
        y   - Required : 1D array containing y values.  The funciton will
                         create a meshgrid from x and y, but should be called
                         like f(x, y, *args).
        major   - Required : Float. Radius of ring or major axis of ellipse.
        sigma   - Required : Float. Width of gaussian cross section
        minor   - Optional : Float. Radius of minor axis of ellipse
                             (default: major)
        theta   - Optional : Float. Angle in degrees to rotate ring.
        x0      - Optional : Float. Center point of ring (default: center of x)
        y0      - Optional : Float. Center point of ring (default: center of y)


    Returns:
        G   -   2D array containing Gaussian ring.

    History:
        2018-05-09  - HP : Initial commit.
        2018-05-10  - HP : Added center point.
    '''
    if minor is None:
        minor = 1
    if x0 is None:
        x0 = (x[-1] + x[0])/2.0
    if y0 is None:
        y0 = (y[-1] + y[0])/2.0
    x, y = x[:,None], y[None,:]
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    T = np.arctan2(x-x0,y-x0) - np.radians(theta)
    R = major*minor / np.sqrt((minor*np.cos(T))**2 + (major*np.sin(T))**2)
    return np.exp(-(r-R)**2 / (2*sigma**2))


def gauss_theta(x, y, theta, sigma, x0=None, y0=None, symmetric=1):
    '''
    Create a radial wedge with a gaussian profile. For theta-dependent
    amplitude modulation of a signal.

    Inputs:
        x   - Required : 1D array containing x values
        y   - Required : 1D array containing y values.  The funciton will
                         create a meshgrid from x and y, but should be called
                         like f(x, y, *args).
        theta   - Required : Float. Angle in degrees for center of wedge.
        sigma   - Required : Float. Width of gaussian cross section.
        x0      - Optional : Float. Center point of arc wedge (default: center of x)
        y0      - Optional : Float. Center point of arc wedge (default: center of y)
        symmetric - Optional : Integer.  Gives the wedge n-fold rotational
                               symmetry (default:1).


    Returns:
        G   -   2D array containing Gaussian wedge.

    History:
        2018-05-10  - HP : Initial commit.

    '''
    def reduce_angle(theta):
        '''Maps any angle in degrees to the interval -180 to 180'''
        t =  theta % 360
        if t > 180:
            t -= 360
        return t

    if x0 is None:
        x0 = (x[-1] + x[0])/2.0
    if y0 is None:
        y0 = (y[-1] + y[0])/2.0
    t = np.radians(reduce_angle(theta))
    sig = np.radians(reduce_angle(sigma))
    T = np.arctan2(x[:,None]-x0, y[None,:]-y0)
    if -np.pi/2.0 < t <= np.pi/2.0:
        amp = np.exp(-(T-t)**2/(2*sig**2))
    else:
        amp = np.exp(-(T-(np.sign(t)*np.pi-t))**2/(2*sig**2))[:,::-1]
    for deg in np.linspace(360.0/symmetric, 360, int(symmetric)-1, endpoint=False):
        amp += gauss_theta(x, y, np.degrees(t)+deg, sigma, x0=x0, y0=y0, symmetric=1)
    return amp


class ngauss1d(object):
    '''
    Fits a combination of n gaussians to 1d data. Output is an object
    containing various attributs pertaining to the fit. Includes the option to
    fix a number of parameters in the fit by providing an array of 1 and 0
    corresponding to each parameter: 1 - vary, 0 - fix.

    Inputs:
        x - x data
        y - normalized y data: divide by the end point and subtract 1:
            y = y_data / y_data[-1] - 1
        p0 - array of initial guess parameters in the form:
            [amp, mu, sigma, amp, mu, sigma, ...]
            len(p0) must be divisible by 3.
        vary - array with same lengh as p0 describing whether to vary or fix
            each parameter. Defaults to varying all.
        kwarg - additional keyword arguments passed to scipy.optimize.minimize

    Usage: result = ngauss1d(x, y, p0, vary=None, **kwarg)
    '''
    def __init__(self, x, y, p0, vary=None, **kwarg):
        if vary is None:
            vary = np.zeros(len(p0)) + 1
        if len(vary) != len(p0):
            print('Warning - Vary not specified for each parameter.')
        self._x = x
        self._yf = y
        self._ix = np.where(vary == 1)[0]
        self._p0 = p0
        self.output = opt.minimize(self._chix, p0[self._ix], **kwarg)
        p = self._find_p(self.output.x)
        self.fit = self.gaussn(*p)
        self.p_unsrt = p.reshape(int(len(p0)/3), 3).T
        mu = self.p_unsrt[1]
        self.p = self.p_unsrt[:, mu.argsort()]
        self.peaks = np.zeros([self.p.shape[1], len(self._x)])
        self.peaks_unsrt = np.zeros([self.p.shape[1], len(self._x)])
        for ix, (peak, peak_u) in enumerate(zip(self.p.T, self.p_unsrt.T)):
            self.peaks[ix] = self.gaussn(*peak)
            self.peaks_unsrt[ix] = self.gaussn(*peak_u)

    def gaussn(self, *p):
        g = np.zeros_like(self._x)
        for i in range(0,len(p),3):
            amp = abs(float(p[i]))
            mu = float(p[i+1])
            sigma = float(p[i+2])
            g += amp * np.exp(-(self._x-mu)**2 / (2.0*sigma**2))
        return g

    def _find_p(self, p_vary):
        p = np.zeros([len(self._p0)])
        vix = 0
        for ix in range(len(self._p0)):
            if ix in self._ix:
                p[ix] = p_vary[vix]
                vix += 1
            else:
                p[ix] = self._p0[ix]
        return p

    def _chix(self, p_vary):
        p = self._find_p(p_vary)
        gf = self.gaussn(*p)
        err = np.abs(gf - self._yf)
        return np.log(sum(err**2))


def track_peak(x, z, p0, **kwarg):
    '''
    Simple interface for ngauss1d that tracks peaks on a 2d map in the y
    direction.

    Inputs:
        x - x data
        z - 2d map with peaks in the y direction
        p0 - initial guess parameters for peaks
        kwarg - additional keyword arguments passed to ngauss1d.  Check
        ngauss1d.__doc__ for details.

    Usage: mu = track_peak(x, z, p0, vary=vary, bounds=bounds)
    '''
    mu = np.zeros([len(p0)/3, z.shape[0]])
    for ix, yv in enumerate(z):
        y = yv/yv[-1] - 1
        result = ngauss1d(x, y, p0, **kwarg)
        mu[:,ix] = result.p_unsrt[1,:]
    return mu


def plane_subtract(data, deg, X0=None):
    '''
    Subtracts a polynomial plane from an image. The polynomial does not keep
    any cross terms, i.e. not xy, only x^2 and y*2.  I think this is fine and
    just doesn't keep any hyperbolic-like terms.

    Inputs:
        data    - Required : A 2D or 3D numpy array containing data
        deg     - Required : Degree of polynomial to be removed.
        X0      - Optional : Guess optimization parameters for
                             scipy.optimize.minimize.

    Returns:
        subtractedData - Data with a polynomial plane removed.

    History:
        2017-07-13  - HP : Fixed so that it works up to at least 3rd order.
    '''
    def plane(a):
        x = np.arange(subtract2D.norm.shape[1])
        y = np.arange(subtract2D.norm.shape[0])
        x = x[None,:]
        y = y[:,None]
        z = np.zeros_like(subtract2D.norm) + a[0]
        N = int((len(a)-1)/2)
        for k in range(1, N+1):
            z += a[2*k-1] * x**k + a[2*k] * y**k
        return z
    def chi(X):
        chi.fit = plane(X)
        res = subtract2D.norm - chi.fit
        err = np.sum(np.absolute(res))
        return err
    def subtract2D(layer):
        vx = np.linspace(-1, 1, layer.shape[0])
        vy = np.linspace(-1, 1, layer.shape[1])
        x, y = vx[:, None], vy[None, :]
        subtract2D.norm = (layer-np.mean(layer)) / np.max(layer-np.mean(layer))
        result = opt.minimize(chi, X0)
        return subtract2D.norm - chi.fit
    if X0 is None:
        X0 = np.zeros([2*deg+1])
    if len(data.shape) == 2:
        return subtract2D(data)
    elif len(data.shape) == 3:
        output = np.zeros_like(data)
        for ix, layer in enumerate(data):
            output[ix] = subtract2D(layer)
        return output

def butter_lowpass_filter(data, ncutoff=0.5, order=1, method='pad', padtype='odd', irlen=None):
    '''
    Low-pass filter applied for an individual spectrum (.dat) or every spectrum in a DOS map (.3ds)

    Parameters:
    data: data to be filtered, could be A.didv or A.LIY
    ncutoff: unitless cutoff frequency normaled by Nyquist frequency (half of sampling frequency),
    note that ncutoff <=1, i.e., real cutoff frequency should be less than Nyquist frequency
    order: degree of high frequency attenuation, see Wikipedia item "Butterworth filter".
    method : “pad” or “gust”. When method is “pad”, the signal is padded. When method is “gust”, Gustafsson’s method is used. [F. Gustaffson, “Determining the initial states in forward-backward filtering”, Transactions on Signal Processing, Vol. 46, pp. 988-992, 1996.]
    padtype : ‘odd’, ‘even’, ‘constant’, or None. This determines the type of extension to use for the padded signal to which the filter is applied. If padtype is None, no padding is used. The default is ‘odd’.
    irlen : When method is “gust”, irlen specifies the length of the impulse response of the filter. If irlen is None, no part of the impulse response is ignored. For a long signal, specifying irlen can significantly improve the performance of the filter.

    Usage: A_didv_filt = butter_lowpass_filter(A.didv, ncutoff=0.5, order=1)
           A_LIY_filt = butter_lowpass_filter(A.LIY, ncutoff=0.5, order=1)
    '''

    b, a = butter(order, ncutoff, btype='low', analog=False)
    y = np.zeros_like(data)
    if len(data.shape) is 1:
        y = filtfilt(b, a, data, method=method, padtype=padtype, irlen=irlen)
        return y
    elif len(data.shape) is 3:
        for ic in np.arange(data.shape[1]):
            for ir in np.arange(data.shape[2]):
                didv = data[:, ic, ir]
                y[:, ic, ir] = filtfilt(b, a, didv, method=method, padtype=padtype, irlen=irlen)
        return y
    else:
        print('ERR: Input must be 1D or 3D numpy array.')


def highpass(data, ncutoff=0.5, order=1, method='pad', padtype='odd', irlen=None):
    """Simple 1D highpass filter

    Inputs:
        data    - Required : 1D array containing data, eg. A.LIY.flatten()
        ncutoff - Optional : Unitless cutoff frequency normaled by Nyquist
                             frequency (half of sampling frequency). Note that
                             ncutoff <=1, ie. real cutoff frequency should be
                             less than Nyquist frequency.
        order   - Optional : degree of high frequency attenuation, see Wikipedia
                             article for "Butterworth filter".
        method  - Optional : “pad” or “gust”. When method is “pad”, the signal
                             is padded. When method is “gust”, Gustafsson’s
                             method is used. [F. Gustaffson, “Determining the
                             initial states in forward-backward filtering”,
                             Transactions on Signal Processing, Vol. 46, pp.
                             988-992, 1996.]
        padtype - Optional : ‘odd’, ‘even’, ‘constant’, or None. This determines
                             the type of extension to use for the padded signal
                             to which the filter is applied. If padtype is None,
                             no padding is used. The default is ‘odd’.
        irlen   - Optional : When method is “gust”, irlen specifies the length
                             of the impulse response of the filter. If irlen is
                             None, no part of the impulse response is ignored.
                             For a long signal, specifying irlen can
                             significantly improve the performance of the filter.

    Returns:
        out - 1D array containting filtered data.

    History:
    2020-04-25  - HP : Initial commit.
    """
    b, a = butter(order, ncutoff, btype='high', analog=False)
    y = filtfilt(b, a, data, method=method, padtype=padtype, irlen=irlen)
    return y


def gradfilter(A, x, y, genvec=False):
    '''
    Minimum gradient filter for dispersive features (Ref. arXiv:1612.07880), returns filtered image
    with optional gradient components for pseudo-vector-field and gradient modulus maps,
    e.g. grad[I(k, E)].

    A is a 2D array composed of two axes x and y representing two independent experimental variables
    x and y should be both equally spaced 1D array but may not be same increment dx and dy

    Usage: x = np.linspace(-1, 1, 40)
           y = np.linspace(0, 1, 20)
           A = np.array([...])
           # simple filtering
           A_gfl = gradfilter(A, x, y)

           # with gradient mapped
           A_gfl, A_grad_x, A_grad_y = gradfilter(A, x, y, genvec=True)

           X, Y = np.meshgrid(x, y)
           quiver(X, Y, A_grad_x, A_grad_Y, facecolors='b' ,width=0.005, pivot='mid') # vector-field
           A_grad_map = np.sqrt(A_grad_x**2 + A_grad_y**2) # modulus map
           pcolormesh(X, Y, A_gfl) # filtered image
    '''
    # Use built-in np.gradient() function
    #A_grad_row, A_grad_col = np.gradient(A, edge_order=1)
    #A_grad = np.sqrt(A_grad_row**2 + A_grad_col**2)
    #A_grad_filtered = A / np.sqrt(A_grad_row**2 + A_grad_col**2)

    # 8-component method
    col, row = A.shape
    norm = np.sqrt(1/8.) # normalize boundaries such that boundary values of modulus map are 1 to be divided.
    dx = x[1]-x[0] # increment of W, E
    dy = y[1]-y[0] # increment of N, S
    dxy = np.sqrt(dx**2 + dy**2) # increment of NW, NE, SW, SE

    A_grad_N = np.ones_like(A)*norm
    A_grad_S = np.ones_like(A)*norm
    A_grad_W = np.ones_like(A)*norm
    A_grad_E = np.ones_like(A)*norm
    A_grad_NW = np.ones_like(A)*norm
    A_grad_NE = np.ones_like(A)*norm
    A_grad_SW = np.ones_like(A)*norm
    A_grad_SE = np.ones_like(A)*norm

    for i in np.arange(1,col-1):
        for j in np.arange(1,row-1):
            A_grad_N[i, j] = (A[i, j] - A[i-1, j]) / dy
            A_grad_S[i, j] = (A[i, j] - A[i+1, j]) / dy
            A_grad_W[i, j] = (A[i, j] - A[i, j-1]) / dx
            A_grad_E[i, j] = (A[i, j] - A[i, j+1]) / dx
            A_grad_NW[i, j] = (A[i, j] - A[i-1, j-1]) / dxy
            A_grad_NE[i, j] = (A[i, j] - A[i-1, j+1]) / dxy
            A_grad_SW[i, j] = (A[i, j] - A[i+1, j-1]) / dxy
            A_grad_SE[i, j] = (A[i, j] - A[i+1, j+1]) / dxy

    A_grad_col = A_grad_W + (A_grad_NW + A_grad_SW) /np.sqrt(2) - A_grad_E - (A_grad_NE + A_grad_SE)/ np.sqrt(2)
    A_grad_row = A_grad_N + (A_grad_NW + A_grad_NE) / np.sqrt(2) - A_grad_S - (A_grad_SW + A_grad_SE)/ np.sqrt(2)
    A_grad_mod = np.sqrt(A_grad_N**2 + A_grad_S**2 + A_grad_W**2 + A_grad_E**2 + A_grad_NW**2 + A_grad_NE**2 \
                         + A_grad_SW**2 + A_grad_SE**2)
    A_grad_filtered = A / A_grad_mod
    A_grad_filtered = (A_grad_filtered * (A.max() - A.min()) + (A.min() * A_grad_filtered.max() - A.max() * A_grad_filtered.min())) \
    /(A_grad_filtered.max() - A_grad_filtered.min()) # optional: normalize amplitude to fit original range
    if genvec:
        return A_grad_filtered, A_grad_col, A_grad_row
    else:
        return A_grad_filtered

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1,
        length = 70, fill = '█'):
    """
    Copied straight from stackoverflow:
    Call in a loop to create terminal progress bar
    Inputs:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    Outputs:
        None
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print('\n Process completed!')


def nsigma_global(data, n=5, M=2, repeat=1):
    '''Replace bad pixels that have a value n-sigma greater than the global
    mean with the average of their neighbors.

    Inputs:
        data    - Required : 1D, 2D or 3D numpy array containing bad pixels.
                             If 3D, a 2D global filter is applied to each layer
                             by iterating over the first index.
        n       - Optional : Number of standard deviations away from mean for
                             filter to identify bad pixels (default : 5).
        M       - Optional : Size of box for calculating replacement value.
        repeat  - Optional : Number of times to repeat the filter.

    Returns:
        filteredData    :  Data with bad pixels set to the local average
                           value.

    Usage:
        filteredData = nsigma_global(data, n=5, M=2)

    History:
        2017-06-07  - HP : Initial commit
        2017-06-18  - HP : Added support for 1D data.
        2017-07-12  - HP : Added repeat flag.
    '''
    def filter_1D(line, n, M):
        filtered = line.copy()
        badPixels = np.where((line > np.mean(line) + n*np.std(line)) |
                             (line < np.mean(line) - n*np.std(line)) )
        for ix in badPixels[0]:
            neighbors = line[max(0, ix-M) : min(line.shape[0], ix+M+1)]
            mask = (neighbors != line[ix])
            replacement = np.sum(mask*neighbors) / (neighbors.size - 1.0)
            filtered[ix] = replacement
        return filtered

    def filter_2D(layer, n, M):
        filtered = layer.copy()
        badPixels = np.where((layer > np.mean(layer) + n*np.std(layer)) |
                             (layer < np.mean(layer) - n*np.std(layer)) )
        for ix, iy in zip(badPixels[1], badPixels[0]):
            neighbors = layer[max(0, iy-M) : min(layer.shape[0], iy+M+1),
                              max(0, ix-M) : min(layer.shape[1], ix+M+1)]
            mask = (neighbors != layer[iy,ix])
            replacement = np.sum(mask*neighbors) / (neighbors.size - 1.0)
            filtered[iy, ix] = replacement
        return filtered

    filteredData = data.copy()
    if len(data.shape) == 1:
        for iz in range(repeat):
            filteredData = filter_1D(filteredData, n, M)

    elif len(data.shape) == 2:
        for iz in range(repeat):
            filteredData = filter_2D(filteredData, n, M)

    elif len(data.shape) == 3:
        for iz in range(repeat):
            for ix, layer in enumerate(filteredData):
                filteredData[ix] = filter_2D(layer, n, M)
    else:
        print('ERR: Input must be 1D, 2D or 3D numpy array')

    return filteredData


def nsigma_local(data, n=4, N=4, M=4, repeat=1):
    '''
    Removes bad pixels that have a value n-sigma greater than their neighbors.
    Works computes sigma and replacement values locally.

    Inputs:
        data    - Required  :  A 1D, 2D or 3D numpy array containing bad pixels.
        n       - Optional  :  Number of local standard deviations away from
                               local mean for filter to identify bad pixels
                               (default : 4).
        N       - Optional  :  Size of local neighborhood for determining
                               standard deviation and local mean (default : 4).
        M       - Optional  :  Size of local neighborhood for determining
                               replacement value.  The replacement is the mean
                               of a (2M+1) x (2M+1) square that excludes the
                               bad pixel (default : 4)
        repeat  - Optional  :  Number of times to repeat the filter
                               (default : 1)

   Returns:
        filteredData    :  Data with bad pixels set to the average value of
                           neighbors.

    Usage:
        filteredData = nsigma_local(data, n=4, N=4, M=4, repeat=1)

    History:
        2017-06-07  - HP : Initial commit
        2017-06-18  - HP : Added support for 1D data.

    '''
    def nsigma_local_1D(line, n, N, M):
        filtered = line.copy()
        for IX in range(N, line.shape[0], 2*N+1):
                local = filtered[max(0, IX-N) : min(line.shape[0], IX+N+1)]
                badPixels = np.where((local > np.mean(local) + n*np.std(local)) |
                                     (local < np.mean(local) - n*np.std(local)) )
                for ix in badPixels[0]:
                    neighbors = local[max(0, ix-M) : min(line.shape[0], ix+M+1)]
                    mask = (neighbors != local[ix])
                    replacement = np.sum(mask*neighbors) / (neighbors.size - 1.0)
                    filtered[IX-N+ix] = replacement
        return filtered

    def nsigma_local_2D(layer, n, N, M):
        filtered = layer.copy()
        for IY in range(N, layer.shape[0], 2*N+1):
            for IX in range(N, layer.shape[1], 2*N+1):
                local = filtered[max(0, IY-N) : min(layer.shape[0], IY+N+1),
                                 max(0, IX-N) : min(layer.shape[1], IX+N+1)]
                badPixels = np.where((local > np.mean(local) + n*np.std(local)) |
                                     (local < np.mean(local) - n*np.std(local)) )
                for ix, iy in zip(badPixels[1], badPixels[0]):
                    neighbors = local[max(0, iy-M) : min(layer.shape[0], iy+M+1),
                                      max(0, ix-M) : min(layer.shape[1], ix+M+1)]
                    mask = (neighbors != local[iy,ix])
                    replacement = np.sum(mask*neighbors) / (neighbors.size - 1.0)
                    filtered[IY-N+iy, IX-N+ix] = replacement
        return filtered

    filteredData = data.copy()
    if len(data.shape) == 1:
        for iz in range(repeat):
            filteredData = nsigma_local_1D(filteredData, n, N, M)

    elif len(data.shape) == 2:
        for iz in range(repeat):
            filteredData = nsigma_local_2D(filteredData, n, N, M)

    elif len(data.shape) == 3:
        for iz in range(repeat):
            for ix, layer in enumerate(filteredData):
                filteredData[ix] = nsigma_local_2D(layer, n, N, M)
    else:
        print('ERR: Input must be 1D, 2D or 3D numpy array')

    return filteredData

def radial_linecut(data, length, angle, width, reshape=True):
    '''
    Computes a retangular linecut radially from the center of an image.
    Designed for QPI linecuts.

    Inputs:
        data    - Required : A 2D or 3D numpy array.
        length  - Required : Integer length of linecut in pixels.
        angle   - Required : Float used to specify the angle in degrees
                             relative to the x-axis for the linecut.
        width   - Required : Integer (>0) that specified the perpendicular
                             width to average over.
        reshape - Optional : Boolean to define whether to reshape the image
                             during rotation. True seems better unless it cuts
                             off.

    Returns:
        linecut - numpy array containing 1D or 2D linecut

    Usage:
        linecut = radial_linecut(data, length, angle, width, reshape=length)

    History:
         2017-06-15  - HP : Initial commit.
    '''
    def linecut2D(layer):
        layerRot = snd.rotate(layer, angle, reshape=reshape)
        cen = np.array(layerRot.shape)/2
        layerCrop = layerRot[cen[1]-int(width) : cen[1]+int(width),
                             cen[0] : cen[0]+int(length)]
        return np.mean(layerCrop, axis=0)
    if len(data.shape) == 2:
        return linecut2D(data)
    elif len(data.shape) == 3:
        linecut = np.zeros([data.shape[0], int(length)])
        for ix, layer in enumerate(data):
            linecut[ix] = linecut2D(layer)
        return linecut
    else:
        print('ERR: Input must be 2D or 3D numpy array')


def fft(dataIn, window='None', output='absolute', zeroDC=False, beta=1.0,
        units='None'):
    '''
    Compute the fast Frouier transform of a data set with the option to add
    windowing.

    Inputs:
        dataIn    - Required : A 1D, 2D or 3D numpy array
        window  - Optional : String containing windowing function used to mask
                             data.  The options are: 'None' (or 'none'), 'bartlett',
                             'blackman', 'hamming', 'hanning' and 'kaiser'.
        output  - Optional : String containing desired form of output.  The
                             options are: 'absolute', 'real', 'imag', 'phase'
                             or 'complex'.
        zeroDC  - Optional : Boolean indicated if the centeral pixel of the
                                FFT will be set to zero.
        beta    - Optional : Float used to specify the kaiser window.  Only
                               used if window='kaiser'.
        units   - Optional : String containing desired units for the FFT.
                             Options: 'None', or 'amplitude' (in the future, I
                             might add "ASD" and "PSD".

    Returns:
        fftData - numpy array containing FFT of data

    Usage:
        fftData = fft(data, window='None', output='absolute', zeroDC=False,
                      beta=1.0)

    History:
        2017-06-15  - HP : Initial commit.
        2017-06-22  - HP : Added support for 1D data and complex output.
        2017-10-31  - HP : Improved zeroDC to subtact the mean before FFT.
        2017-11-19  - HP : Fixed a bug in calculating the mean of 3D data.
    '''
    def ft2(data):
        ftData = np.fft.fft2(data)
        if zeroDC:
            ftData[0,0] = 0
        return np.fft.fftshift(ftData)

    outputFunctions = {'absolute':np.absolute, 'real':np.real,
                       'imag':np.imag, 'phase':np.angle, 'complex':(lambda x:x) }

    windowFunctions = {'None':(lambda x:np.ones(x)), 'none':(lambda x:np.ones(x)),
                       'bartlett':np.bartlett, 'blackman':np.blackman,
                       'hamming':np.hamming, 'hanning':np.hanning,
                       'kaiser':np.kaiser }

    outputFunction = outputFunctions[output]
    windowFunction = windowFunctions[window]

    data = dataIn.copy()
    if zeroDC:
        if len(data.shape) == 3:
            for ix, layer in enumerate(data):
                data[ix] -= np.mean(layer)
        else:
            data -= np.mean(data)

    if len(data.shape) != 1:
        if window == 'kaiser':
            wX = windowFunction(data.shape[-2], beta)[:,None]
            wY = windowFunction(data.shape[-1], beta)[None,:]
        else:
            wX = windowFunction(data.shape[-2])[:,None]
            wY = windowFunction(data.shape[-1])[None,:]
        W = wX * wY
        if len(data.shape) == 2:
            wData = data * W
            ftData = outputFunction(ft2(wData))
        elif len(data.shape) == 3:
            wTile = np.tile(W, (data.shape[0],1,1))
            wData = data * wTile
            if output == 'complex':
                ftData = np.zeros_like(data, dtype=np.complex)
            else:
                ftData = np.zeros_like(data)
            for ix, layer in enumerate(wData):
                ftData[ix] = outputFunction(ft2(layer))
        else:
            print('ERR: Input must be 1D, 2D or 3D numpy array')

    else:
        if window == 'kaiser':
            W = windowFunction(data.shape[0], beta)
        else:
            W = windowFunction(data.shape[0])
        wData = data * W
        ftD = np.fft.fft(wData)
        ftData = outputFunction(np.fft.fftshift(ftD))
    if units == 'amplitude':
        if len(data.shape) == 3:
            datashape = data[0].shape
        else:
            datashape = data.shape
        for size in datashape:
            ftData /= size
            if window == 'hanning':
                ftData *= 2
            elif window == 'None' or window == 'none':
                pass
            else:
                print('WARNING: The window function "%s" messes up the FT units' %
                        window)
    return ftData


def ifft(data, output='real', envelope=False):
    '''
    Compute the inverse Fourier transform with the option to detect envelope.

    Inputs:
        data    - Required : A 1D or 2D numpy array. (3D not yet supported)
        output  - Optional : String containing desired form of output.  The
                             options are: 'absolute', 'real', 'imag', 'phase'
                             or 'complex'.
        envelope - Optional : Boolen, when True applies the Hilbert transform
                              to detect the envelope of the IFT, which is the
                              absolute values.
        **kwarg - Optional : Passed to scipy.signal.hilbert()

    See docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    for more information about envelope function.

    Outputs:
        ift - A numpy array containing inverse Fourier transform.

    History:
        2018-03-30  - HP : Initial commit.
    '''
    outputFunctions = {'absolute':np.absolute, 'real':np.real,
                       'imag':np.imag, 'phase':np.angle, 'complex':(lambda x:x)}
    out = outputFunctions[output]
    if len(data.shape) == 2:
        ift = np.fft.ifft2(np.fft.ifftshift(data))
    elif len(data.shape) == 1:
        ift = np.fft.ifft(np.fft.ifftshift(data))
    if envelope:
        ift = hilbert(np.real(ift))
    return out(ift)


def fftfreq(px, nm):
    '''Get frequnecy bins for Fourier transform.'''
    freqs = np.fft.fftfreq(px, float(nm)/(px))
    return np.fft.fftshift(freqs)


def normalize(data, axis=0, condition='mean'):
    '''
    Normalize a 2D image line by line in the x or y direction.

    Inputs:
        data    - Required : A 2D numpy array to be normalized
        axis    - Optional : The axis along which each line is normalized.
        condition - Optional : Function to use for normalization.  The line is
                               divided by the condition.  Options are: 'max',
                               'min', 'mean'.

    Returns:
        normData - 2D array containing normalized data

    Usage:
        normData = normalize(data, axis=0, condition='mean')

    History:
        2017-06-16  - HP : Initial commit
    '''
    conditionOptions = {'max':np.max, 'mean':np.mean,
                        'min':np.min}
    cond = conditionOptions[condition]
    dataT = np.moveaxis(data, axis, 0)
    outputT = np.zeros_like(dataT)
    for ix, line in enumerate(dataT):
        outputT[ix] = line/cond(line)
    output = np.moveaxis(outputT, 0, axis)
    return output


def linecut(data, p0, p1, width=1, dl=0, dw=0, kind='linear',
                show=False, ax=None, **kwarg):
    '''Linecut tool for 2D or 3D data.

    Inputs:
        data    - Required : A 2D or 3D numpy array.
        p0      - Required : A tuple containing indicies for the start of the
                             linecut: p0=(x0,y0)
        p1      - Required : A tuple containing indicies for the end of the
                             linecut: p1=(x1,y1)
        width   - Optional : Float for perpendicular width to average over.
        dl      - Optional : Extra pixels for interpolation in the linecut
                             direction.
        dw      - Optional : Extra pixels for interpolation in the
                             perpendicular direction.
        kind    - Optional : Sting for interpolation scheme. Options are:
                             'nearest', 'linear', 'cubic', 'quintic'.  Note
                             that 'linear', 'cubic', 'quintic' use spline.
        show    - Optional : Boolean determining whether to plot where the
                             linecut was taken.
        ax      - Optional : Matplotlib axes instance to plot where linecut is
                             taken.  Note, if show=True you MUST provide and
                             axes instance as plotting is done using ax.plot().
        **kwarg - Optional : Additional keyword arguments passed to ax.plot().

    Returns:
        r   -   1D numpy array which goes from 0 to the length of the cut.
        cut -   1D or 2D numpy array containg the linecut.

    Usage:
        r, cut = linecut(data, (x0,y0), (x1,y1), width=1, dl=0, dw=0,
                         show=False, ax=None, **kwarg)

    History:
        2017-06-19  - HP : Initial commit.
        2017-06-22  - HP : Python 3 compatible.
        2017-08-24  - HP : Modified to use stmpy.tools.interp2d()
    '''
    def calc_length(p0, p1, dl):
        dx = float(p1[0]-p0[0])
        dy = float(p1[1]-p0[1])
        l = np.sqrt(dy**2 + dx**2)
        if dx == 0:
            theta = np.pi/2
        else:
            theta = np.arctan(dy / dx)
        xtot = np.linspace(p0[0], p1[0], int(np.ceil(l+dl)))
        ytot = np.linspace(p0[1], p1[1], int(np.ceil(l+dl)))
        return l, theta, xtot, ytot

    def get_perp_line(x, y, theta, w):
        wx0 = x - w/2.0*np.cos(np.pi/2 - theta)
        wx1 = x + w/2.0*np.cos(np.pi/2 - theta)
        wy0 = y + w/2.0*np.sin(np.pi/2 - theta)
        wy1 = y - w/2.0*np.sin(np.pi/2 - theta)
        return (wx0, wx1), (wy0, wy1)

    def cutter(F, p0, p1, dw):
        l, __, xtot, ytot = calc_length(p0, p1, dw)
        cut = np.zeros(int(np.ceil(l+dw)))
        for ix, (x,y) in enumerate(zip(xtot, ytot)):
            cut[ix] = F(x,y)
        return cut

    def linecut2D(layer, p0, p1, width, dl, dw):
        xAll, yAll = np.arange(layer.shape[1]), np.arange(layer.shape[0])
        F = interp2d(xAll, yAll, layer, kind=kind)
        l, theta, xtot, ytot = calc_length(p0, p1, dl)
        r = np.linspace(0, l, int(np.ceil(l+dl)))
        cut = np.zeros(int(np.ceil(l+dl)))
        for ix, (x,y) in enumerate(zip(xtot,ytot)):
            (wx0, wx1), (wy0, wy1) = get_perp_line(x, y, theta, width)
            wcut = cutter(F, (wx0,wy0), (wx1,wy1), dw)
            cut[ix] = np.mean(wcut)
        return r, cut

    if len(data.shape) == 2:
        r, cut = linecut2D(data, p0, p1, width, dl, dw)
    if len(data.shape) == 3:
        l, __, __, __ = calc_length(p0, p1, dl)
        cut = np.zeros([data.shape[0], int(np.ceil(l+dl))])
        for ix, layer in enumerate(data):
            r, cut[ix] = linecut2D(layer, p0, p1, width, dl, dw)
    if show:
        __, theta, __, __ = calc_length(p0, p1, dl)
        (wx00, wx01), (wy00, wy01) = get_perp_line(p0[0], p0[1], theta, width)
        (wx10, wx11), (wy10, wy11) = get_perp_line(p1[0], p1[1], theta, width)
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], 'k--', **kwarg)
        ax.plot([wx00,wx01], [wy00,wy01], 'k:', **kwarg)
        ax.plot([wx10,wx11], [wy10,wy11], 'k:', **kwarg)
    return r, cut


def crop(data, cen, width=15):
    '''Crops data to be square.

    Inputs:
        data    - Required : A 2D or 3D numpy array.
        cen     - Required : A tuple containing location of the center
                             pixel.
        width   - Optional : Integer containing the half-width of the
                             square to crop.

    Returns:
        croppedData - A 2D or 3D numpy square array of specified width.

    Usage:
        croppedData = crop(data, cen, width=15)

    History:
        2017-07-11  - HP : Initial commit.
    '''
    imcopy = data.copy()
    if len(data.shape) == 2:
        return imcopy[cen[1]-width : cen[1]+width,
                      cen[0]-width : cen[0]+width]
    elif len(data.shape) == 3:
        return imcopy[:, cen[1]-width : cen[1]+width,
                      cen[0]-width : cen[0]+width]


def curve_fit(f, xData, yData, p0=None, vary=None, **kwarg):
    '''Fit a function to data allowing parameters to be fixed.

    Inputs:
        f       - Required : Fitting function callable as f(xData, *args).
        xData   - Required : 1D array containing x values.
        yData   - Required : 1D array containing y values.
        p0      - Optional : Initial guess for parameter values, defauts to 1.
        vary    - Optional : List of booleans describing which parameters to
                             vary (True) and which to keep fixed (False).
        **kwarg - Optional : Passed to scipy.optimize.minimize(). Example: option={'disp':True} to display convergence messages.

    Returns:
        popt    - 1D array containing the optimal values.
        The full result is accessible as curve_fit.result, which contains the
        covariance matix and fitting details.

    History:
        2017-07-13  - HP : Initial commit.
        2017-08-14  - HP : Added python 3 compatibility.
        2017-08-27  - HP : Set default method to 'SLSQP'
                           Will print warning if no iterations are evalued.
        2017-09-06  - JG : Set dtype of p0 to be float
        2019-04-05  - JG : Force the size of vary the same with p0 if vary is None
    '''
    if 'method' not in kwarg.keys():
        kwarg['method'] = 'SLSQP'
    def chi(pv):
        p0[vary == True] = pv
        fit = f(xData, *p0)
        err = np.absolute(yData - fit)
        return np.log(np.sum(err**2))
    if sys.version_info[0] == 2:
        nargs = f.func_code.co_argcount - 1
    elif sys.version_info[0] == 3:
        from inspect import signature
        sig = signature(f)
        nargs = len(sig.parameters) - 1
    if p0 is None:
        p0 = np.ones(nargs)
    if vary is None:
        vary = np.ones_like(p0, dtype=bool)
    p0 = np.array(p0, dtype=float)
    vary = np.array(vary)
    curve_fit.result = opt.minimize(chi, p0[vary == True], **kwarg)
    if curve_fit.result.nit == 0:
        print('WARNING - Optimization did not iterate, check for failure:\n' +
                'Try a different starting guess, or method (Powell can work' +
                ' well).')
    p0[vary == True] =  curve_fit.result.x
    return p0


def boxcar_average1D(data, N):
    '''Averages 1D data in a moving retangular window.

    Inputs:
        data    - Required : A 1D or 3D numpy array.  If 3D the filter is
                             applied along the first axis, e.g. the energy
                             direction in a DOS map.
        N       - Required : Integer describing the width of the boxcar window.

    Returns:
        averagedData - Data with filter applied

    History:
        2017-07-14  - HP : Initial commit.
    '''
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / N
    if type(N) != int:
        raise TypeError('N must be an integer.')
    if len(data.shape) == 1:
        return running_mean(data, N)
    elif len(data.shape) == 3:
        getShape = running_mean(data[:,0,0], N)
        output = np.zeros([data.shape[0] - N + 1, data.shape[1], data.shape[2]])
        for ix in range(data.shape[2]):
            for iy in range(data.shape[1]):
                output[:,iy,ix] = running_mean(data[:,iy,ix], N)
        return output
    else:
        print('ERR - Data must be 1D or 3D numpy array.')


def gaussn(x, p):
    '''Return a linear combination of n gaussians.

    Inputs:
        x   - Required : A 1D numpy array of x values.
        p   - Required : A list of parameters for the n gaussians in the form:
                         [amplitude_1, mu_1, sigma_1, amplitude_2, mu_2, ...].
                         The amplitudes are taken as |amplitude| and cannot be
                         negative.

    Returns:
        g(x) - A 1D numpy array

    History:
        2017-08-14  - HP : Initial commit.
    '''
    g = np.zeros_like(x, dtype=np.float64)
    for i in range(0, len(p), 3):
        amp = abs(float(p[i]))
        mu = float(p[i+1])
        sigma = float(p[i+2])
        g += amp * np.exp(-(x-mu)**2 / (2.0*sigma**2))
    return g


def find_extrema(data, n=(1,0), minDist=10, thres=(0.01,0.01),
                 exclBorder=False, **kwarg):
    '''
    Get the coordinates of n local maxima and minima in a 2D image, or in each
    layer in a 3D map.
    Note: this function requires the package `skimage` to be installed.
    Run `pip install -U scikit-image` in terminal, or see scikit-image.org for
    details.

    Inputs:
        data    - Required : A 2D or 3D numpy array.
        n       - Optional : A tuple containing the number of maxima and minima
                             to find in the form: (nMax, nMin).
        minDist - Optional : Float to describe the minimum distance between
                             maxima or minima.
        thres   - Optional : Tuple describing the relative theshold for maxima
                             and minima in the form: (thresMax, thresMin).
        exclBorder - Optional : Boolean dictating whether to exclude points at
                                the boundary.

    Returns:
        coords - A 2D or 3D numpy array containng the coordinates of the
                 extrema.  Note: this uses the numpy convention [y,x].

    History:
        2017-08-14  - HP : Initial commit.
    '''
    try:
        from skimage.feature import peak_local_max
    except ImportError:
        raise ImportError('This function needs the package `skimage` to be installed.\n' +
                  'Run `pip install -U scikit-image` in terminal, or see scikit-'+
                  'image.org for details.')

    def find_extrema2D(layer, n=(1, 0), minDist=10, thres=(0.01, 0.01),
                       exclBorder=False, **kwarg):
        cmax = np.array([[np.nan, np.nan]])
        cmin = np.array([[np.nan, np.nan]])
        n = [int(val) for val in n]
        if n[0] is not 0:
            cmax = peak_local_max(layer, min_distance=minDist, threshold_rel=thres[0],
                              num_peaks=n[0], exclude_border=exclBorder, **kwarg)
        if n[1] is not 0:
            cmin = peak_local_max(np.max(layer)-layer, min_distance=minDist,
                              threshold_rel=thres[1], num_peaks=n[1],
                              exclude_border=exclBorder, **kwarg)
        coords = np.concatenate([cmax, cmin])
        mask = ~np.isnan(coords)[:,0]
        return coords[mask]

    if len(data.shape) == 3:
        output = np.zeros([data.shape[0], n[0]+n[1], 2])
        output.fill(np.nan)
        for ix, layer in enumerate(data):
            coords = find_extrema2D(layer, n=n, minDist=minDist, thres=thres,
                                    exclBorder=exclBorder, **kwarg)
            output[ix, :coords.shape[0]] = coords
    elif len(data.shape) == 2:
        output = find_extrema2D(data, n=n, minDist=minDist, thres=thres,
                                    exclBorder=exclBorder, **kwarg)
    else:
        raise ValueError('Data must be 2D or 3D numpy array')
    return output


def remove_extrema(data, coords=None, sigma=4, replSigma=None, replDist=None,
        **kwarg):
    '''Remove extrema by gaussian-smearing to a local backgound value.

    Inputs:
        data    - Required : A 2D or 3D numpy array.
        coords  - Optional : A 2D or 3D numpy array containing the coordinates
                             of the extrema to be removed. The list is given in
                             numpy convention: [(ie), iy, ix].  Note: if not
                             provided coordinates are found automatically using
                             stmpy.tools.find_extrema(data, **kwarg), see docs
                             for infomation.
        sigma   - Optional : Float for the FWHM value of the gaussian area
                            replaced.
        replSigma - Optional : Float for the FWHM of the weighting gaussian for
                               finding the replacement value.
        replDist - Optional : Float for the distance away from the defect to
                              average over when finding a replacement value.
        **kwarg - Optional : Sent to stmpy.tools.find_extrema() if coords is
                             not provided

    Returns:
        output  -   A 2D or 3D numpy array containg data with gaussian removed
                    extrema

    History:
        2017-08-14  - HP : Initial commit.
    '''
    def remove_peak2D(layer, coords):
        output = layer.copy()
        for (IY, IX) in coords:
            r, cut = arc_linecut(layer, (IX,IY), replDist, 0, width=360)
            g = 1 - gaussn(r, (1, 0, replSigma))
            x = np.arange(layer.shape[1])[None, :]
            y = np.arange(layer.shape[0])[:, None]
            gx = gaussn(x, (1,IX,sigma))
            gy = gaussn(y, (1,IY,sigma))
            G = gx * gy
            fill = G * np.average(cut, weights=g)
            output = output * (1-G) + fill
        return output
    if coords is None:
        coords = find_extrema(data, **kwarg)
    if replSigma is None:
        replSigma = sigma
    if replDist is None:
        replDist = 5*replSigma
    out = np.zeros_like(data)
    if len(data.shape) == 3:
        if len(coords.shape) == 3:
            for ix, (layer, coord) in enumerate(zip(data, coords)):
                out[ix] = remove_peak2D(layer, coord)
        else:
            for ix, layer in enumerate(data):
                out[ix] = remove_peak2D(layer, coords)
    elif len(data.shape) ==2:
        out = remove_peak2D(data, coords)
    else:
        raise ValueError('Data must be 2D or 3D numpy array')
    return out


def shift_DOS_en(en, LIY, shift, enNew=None, **kwargs):
    '''Resample LIY data at shifted energy values.

    Inputs:
        en      - Required : 1D array containing measured energies.
        LIY     - Required : 3D array containing measured DOS.
        shift   - Required : Float, or 2D array containing amount to shift
                             energies before resampling.  If 2D array, it must
                             have the same xy shape as LIY.
        eNew    - Optional : Resampled energy values, if not provided en will
                             be used.  Note that energy values outside the
                             original range will have a NaN value.
        **kwargs - Optional : Passed to scipy.interpolate.interp1d(), e.g.
                              kind='linear'.

    Returns:
        LIYshift - 3D array contining the LIY values once the shift has been
                   applied.

    History:
        2017-08-24  - HP : Initial commit.
    '''
    if type(shift) is not np.ndarray:
        shift = np.zeros_like(LIY[0]) + shift
    if enNew is None:
        enNew = en.copy()
    output = np.zeros([len(enNew), LIY.shape[1], LIY.shape[2]])
    for ix in range(LIY.shape[2]):
        for iy in range(LIY.shape[1]):
            f = interp1d(en-shift[iy,ix], LIY[:,iy,ix], bounds_error=False,
                    **kwargs)
            output[:, iy, ix] = f(enNew)
    return output


def get_qscale(data, isReal=True, cix=1, n=(3,0), thres=(1e-10,1), show=False,
        ax=None, **kwarg):
    '''
    Find the radial coordinate of the Bragg peak in a 2D FFT. This defines
    the scale in q-space.

    Inputs:
        data    - Required : a 2D numpy array containing the real-space data,
                             can be the topography or an LIY layer.
        isReal  - Optional : Boolean to specify is data is in real-space
                             (default) or in q-space.
        cix     - Optional : Integer to choose between peaks to find the Bragg
                             peak.
        n       - Optional : Tuple of integers that defines the number of peaks
                             and dipe to find in q-space as: (nPeaks, nDips)
        thres   - Optional : Tuple for relative threshold to search for peaks.
                             See help(stmpy.tools.find_extrema) for more details.
        show    - Optional : Boolean. If true will plot the detected peaks and
                             circle the one being used. Must supply ax.
        ax      - Optional : Matplotlib axes instance.  Must be supplied if
                             show=True.
        **kwarg - Optional : Passed to stmpy.tools.find_extrema

    Returns:
        r, phi - Floats containing the angular coordinates of a Bragg peak
                 (usually lower left in q-space).
                 Note: The angle is in degrees.

    History:
        2017-10-20  - HP : Initial commit.
        2017-11-19  - HP : Added manual way to choose which peak is the Bragg
                           peak.
    '''
    if len(data.shape) != 2:
        raise ValueError('Data must be 2D numpy array')

    if isReal:
        f = fft(data, window='none')
        ftData = f/np.max(f)
    else:
        ftData = data/np.max(data)
    cen = np.array(ftData.shape)/2.0
    coords = find_extrema(ftData, n=n, thres=thres, **kwarg)
    bp = coords[cix]
    r = np.linalg.norm(cen-bp, 2)
    if cen[1]-bp[1] != 0:
        phi = np.arctan((cen[0]-bp[0]) / float(cen[1]-bp[1]))
    else:
        phi = np.pi/2.0
    if show:
        ax.imshow(ftData, origin='lower', rasterized=True)
        for ix, coord in enumerate(coords):
            if ix == cix:
                label = str(cix) + ' - BP'
                ax.plot(coord[1], coord[0], 'o', mfc='none', ms=8, mew=1, label=label)
            else:
                label = str(ix)
                ax.plot(coord[1], coord[0], 'x', ms=8, mew=1, label=label)
        ax.legend()
    return r, np.degrees(phi)


def find_intersect(x, y1, y2):
    '''
    Find the intersection between two curves. Note that the curves must be of
    the form y1(x), y2(x), that is, they must share the same x values.

    Inputs:
        x   - Required : 1D array containing shared x values.
        y1  - Required : 1D array containing y values of the first curve.
        y2  - Required : 1D array containing y values of the second curve.

    Returns:
        intersect - Float for the intersection of the two curves.

    History:
        2017-10-31  - HP : Initial commit.
    '''
    diff = y1 - y2
    for ix, (yL, yH) in enumerate(zip(diff[:-1], diff[1:])):
        if np.sign(yL) != np.sign(yH):
            xL, xH = x[ix], x[ix+1]
            intersect = xH - yH * (xH-xL) / float(yH - yL)
    return intersect


def thermal_broaden(en, didv, T, N=10000, mode='reflect', offset=None):
    '''
    Temperature causes spectral features to become broader.  This function
    mimiks the effect of higher temperatures by convoluting a didv data
    with the derivatice of the Fermi-Dirac distribution. This function works to
    temperatures as low as 1mK.  To go lower in temperature is risky, because
    the kernel become sharply peaked and requires and extremely dense set of
    data points to accurately compute it.  To avoid long computation times this
    function refuses to use more than 1e7 points, though you're welcome to try
    and trick it by reducing 'N'. I wouldn't recommend ridiculously high (>300)
    temperatures either.

    To smooth edge effects, the data is reflected onto itself at both ends.

    Inputs:
        en      - Required : Array containing energy values.
        didv    - Required : 1D array with the spectrum to be smeared.
        T       - Required : Float for temperature (in Kelvin) to smear to.
        N       - Optional : Integer for number of points for temperatures
                             greater than 1K.  Temperature less than 1K will
                             use N*1/T points.
        mode    - Optional : String describing mode of convolution. Only option
                             available is 'reflect'.
        offset  - Optional : Integer to shift the convoluted spectra to the
                             left or right. No, I do not know why you need to
                             do that, but you do.

    Returns:
        smearedData - Array same size as didv containing broadened data.

    History:
        2018-02-01  - HP : Initial commit.


    '''

    def fermi_derivative(en, T):
        '''This is the Kernel for thermal smearing.'''
        kT = 8.617330350e-5 * T * 1e3 # in meV
        y = (1 - np.tanh((en/(2*kT)))**2) / (4*kT)
        return y * (en[1]-en[0])

    if mode == 'reflect':
        dv =  np.concatenate([didv[::-1][:-1], didv, didv[::-1][1:]])
        De = en[-1]-en[0]
        ev = np.linspace(en[0]-De, en[-1]+De, 3*len(en)-2)
    else:
        raise(ValueError('Mode must be reflect. (I have not coded any others).'))

    if offset is None:
        if len(en)%2 == 0:
            offset = -1
        else:
            offset = 1

    n = int(N*max(1/T, 1))
    if n>1e7:
        raise(ValueError('Too many points for interpolation.  Probably caused by '
                        + 'the temperature being too extreme. '))

    x = np.linspace(ev[0], ev[-1], n)
    dx = x[1] - x[0]
    f = fermi_derivative(x, T)
    norm = sum(f)
    if norm < 0.99:
        print('Warning: The Kernal is not normalized, which may cause the calculation '
              + 'to be inaccurate. This is probably because the temperature is too '
             + 'high for the input energy range.  Norm = {:2.2f}'.format(norm))

    g = interp1d(ev, dv, kind='linear')
    fg = fftconvolve(f, g(x), 'same')
    y = interp1d(x, fg, kind='linear')
    sampled = y(ev)
    return sampled[len(en)+offset:2*len(en)+offset]



def find_edges(img, sigma=1, mult=1, thresL=None, thresH=None, ax=None):
    from skimage import feature
    data = img - np.mean(img)
    edges = feature.canny(data/np.max(data)*mult, sigma=sigma, low_threshold=thresL, high_threshold=thresH)
    if ax is not None:
        ax.imshow(edges, cmap=stmpy.cm.gray_r, alpha=0.3)
    return edges

def remove_edges(data, edges=None, sigmaRemove=2.0, sigmaFind=2.0, **kwargs):
    from skimage import filters
    if edges is None:
        edges = find_edges(data, sigma=sigmaFind, **kwargs)
    smooth = filters.gaussian(edges, sigma=sigmaRemove)
    edgeSmooth = smooth/np.max(smooth)
    if len(data.shape) == 3:
        out = np.zeros_like(data)
        for ix, layer in enumerate(data):
            out[ix]  = (1-edgeSmooth)*layer + edgeSmooth*np.mean(layer)
    else:
        out = (1-edgeSmooth)*data + edgeSmooth*np.mean(data)

    return out


def narrow2oct(freq, asd, n=3, fbase=1.0):
    '''
    narrow band ASD data to 3rd octave ASD data

    Input:
    freq, asd - frequency and amplitude spectral density data, e.g. in Hz and V/sqrt(Hz)
    n - order of octave band, e.g. n=1 octave bands; n=3 one-third octave bands
    fbase - reference frequency, default is 1.0 Hz

    Output:
    f_center - center frequency of nth octave bands
    asd_oct - amplitude spectral density array in each octave bands, in e.g. V/sqrt(Hz)
    [you need to integrate to get band RMS by sum(asd_oct**2 * bw)]
    bw - width array of each octave band

    Usage:
    f_center, V_octave, bw = narrow2oct(freq, asd, n=3, fbase=1.0)
    plot(freq, asd, 'b.')
    bar(f_center, V_octave, bw, color='r', edgecolor='k')
    xscale('log', basex=2) # show it in 2-base log spacing so you can tell n th order from plot
    yscale('log')

    History:
    2018-05-08 - Jacky: initial commit
    '''
    N = int(np.log(max(freq))/np.log(2)*3)+1 # number of bands available
    df = np.diff(freq)[0] # narrow band width
    f_center = fbase * np.power(2, np.arange(N)/n)
    f_lower = f_center * np.power(2, -1/n/2)
    f_upper = f_center * np.power(2, 1/n/2)
    bw = f_upper - f_lower
    asd_oct = np.zeros(N, dtype='float')
    for ix, low in enumerate(f_lower):
        up = low * np.power(2, 1/3)
        u, ru = int(up/df), np.mod(up, df)
        l, rl = int(low/df), np.mod(low, df)
        # assuming asd constant in each narrow band
        if u == l: # leading octave bands may fall inside a narraow band
            asd_oct[ix] = np.sqrt(asd[l]**2 * (up - low))
        else:
            asd_oct[ix] = np.sqrt(asd[l]**2 * (df-rl) + sum(asd[l+1:u]**2) * df + asd[u]**2 * ru)
    asd_oct = asd_oct/np.sqrt(bw)
    return f_center, asd_oct, bw


def xcorr(data1, data2, norm=True):
    '''
    Compute the cross correlation of two ndarrays.

    Inputs:
        data1   - Required : Numpy ndarray containing variable 1.
        data2   - Required : Numpy ndarray containing variable 2.  For auto
                             correlation use data2=data1.
        norm    - Optional : Normalize the output so that the autocorrelation
                             is 1 in the center.  Not 100% sure this works...

    Returns:
        out     - Numpy ndarray containing correlation coefficients.

    History:
        2019-03-10  - HP : Initial commit.

    '''
    out = correlate(data1-np.mean(data1), data2-np.mean(data2), mode='same')
    if norm:
        if len(data1.shape) == 1:
            out /= (out.shape[0] * data1.std() * data2.std())
        elif len(data1.shape) == 2:
            out /= (out.shape[0] * out.shape[1] * data1.std() * data2.std())
        else:
            print('ERR - Norm not implemented for {:2.0f}Darrays.'.format(len(data1.shape)))
    return out

def remove_piezo_drift(data):
    '''
    Removes vertical axis piezo drift for 2D image files (.sxm) by fitting
    average y values to a logistic function (fundamental model for piezo
    drift).

    Inputs:
        data    - Required : Numpy array containing image data (must be 2D).

    Returns:
        out     - Numpy 2D array with drift removed. The mean is set to zero,
                  but the units are retained.

    History:
        2019-10-02  - HP : Initial commit.

    '''
    datan = (data-np.min(data)) / np.max(data-np.min(data))
    y = np.mean(datan, axis=1)
    x = np.linspace(0,1,data.shape[1])
    def logistic(x, L, k, x0, c):
        return L / (1 + np.exp(-k*(x-x0))) + c
    p0 = curve_fit(logistic, x, y)
    ls = logistic(x, *p0)
    out = np.zeros_like(data)
    for ix, line in enumerate(datan):
        out[ix] = line - ls[ix]
    outn = out * np.max(data-np.min(data)) + np.min(data)
    return outn - np.mean(outn)

def bias_offset_map(en, I, I2=None):
    '''
    Calculate zero-bias offset for I(V) map or for a two-setpoint map. If only
    one map is provided, the zero-bais point is when the I(V) curve crosses
    0pA. If two maps are provided, it is the intersection of the two I(V)
    curves.

    Inputs:
        en      - Required : Numpy array containing voltage data (must be 1D).
        I       - Required : Numpy array containing current data (must be 3D).
        I2      - Optional : Numpy array containing current data from second
                             map (must be same size as I).

    Returns:
        mu      - Numpy 2D array of the zero-bias voltage point at each point
                  in space.
        g       - Numpy 2D array of the slope at each point.
        g2      - (optional) If I2 is not None, also return the slope for the
                  second I(V) map.

    History:
        2019-10-15  - HP : Initial commit.
        2019-11-04  - HP : Add compatibility for a two-setpoint map.

    '''
    mu = np.zeros_like(I[0])
    g = np.zeros_like(mu)
    g2 = np.zeros_like(g)
    for ix in range(I.shape[-1]):
        for iy in range(I.shape[1]):
            p = np.polyfit(en, I[:,iy,ix], 1)
            g[iy,ix] = p[0]
            if I2 is not None:
                p2 = np.polyfit(en, I2[:,iy,ix], 1)
                g[iy,ix] = p2[0]
                p -= p2
            mu[iy,ix] =  (-p[1]) / (p[0])
    if I2 is None:
        return mu, g
    else:
        return mu, g, g2

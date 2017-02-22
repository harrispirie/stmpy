import numpy as np
import matplotlib as mpl
import scipy.interpolate as sin
import scipy.optimize as opt
import scipy.ndimage as snd
from scipy.signal import butter, filtfilt

def saturate(level_low=0, level_high=None, im=None):
    '''
    Adjusts color axis of in current handle.  Calculates a probablility density function for the data in current axes handle.  Uses upper and lower thresholds to find sensible c-axis limits.  Thresholds are between 0 and 100.  If unspecified the upper threshold is assumed to be 100 - lower threshold.
    
    Usage:  pcolormesh(image)
            saturate(10)
    '''
    level_low = float(level_low) / 200.0
    if level_high is None:
        level_high = 1-level_low
    else:
        level_high = (float(level_high)+100) / 200.0
    if im is not None:
        images = [im]
        data = im.get_array().ravel()
    else:
        imageObjects = mpl.pyplot.gca().get_children()
        data = []
        images = []
        for item in imageObjects:
            if isinstance(item, (mpl.image.AxesImage, mpl.collections.QuadMesh)):
                images.append(item)
                data.append(item.get_array().ravel())
    y = sorted(np.array(data).ravel())
    y_density = np.absolute(y) / sum(np.absolute(y))
    pdf = np.cumsum(y_density)
    y_low = np.absolute(level_low - pdf)
    y_high = np.absolute(level_high - pdf)
    c_low = y[np.argmin(y_low)]
    c_high = y[np.argmin(y_high)]
    for image in images:
        image.set_clim(c_low, c_high)


def azimuthalAverage(F,x0,y0,r,theta = np.linspace(0,2*np.pi,500)):
    ''' Uses 2d interpolation to average F over an arc defined by theta for every r value starting from x0,y0. '''
    f = sin.interp2d(np.arange(F.shape[1]), np.arange(F.shape[0]), F, kind='linear')
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


def linecut(F, x1, y1, x2, y2, n):
    ''' Use linear interpolation on a 2D data set F, sample along a line from (x1,y1) to (x2,y2) in n points

Usage:  x_linecut, y_linecut = linecut(image, x1, y1, x2, y2, n)
    '''
    x = np.arange(F.shape[0])
    y =  np.arange(F.shape[1])
    cen = np.sqrt((x1-x2)**2 + (y1-y2)**2) / 2.0
    r = np.linspace(-1*cen, cen, n)
    f = sin.interp2d(x, y, F, kind = 'linear')
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

def lineSubtract(data, n):
    '''
    Remove a polynomial background from the data line-by-line.  If the data is
    3D (eg. 3ds) this does a 2D background subtract on each layer
    independently.  Input is a numpy array. 
    
    Usage: A.z = lineSubtract(A.Z, 2)
    '''
    def subtract_2D(data, n):
        output = np.zeros_like(data)
        for ix, line in enumerate(data):
            output[ix] = removePolynomial1d(line, n)
        return output
    if len(data.shape) is 3:
        output = np.zeros_like(data)
        for ix, layer in enumerate(data):
            output[ix] = subtract_2D(layer, n)
        return output
    elif len(data.shape) is 2:
        return subtract_2D(data, n)


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
    f = sin.interp1d(x,y,kind='linear')
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

def quickFT(data, n=None, zero_center=True):
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
            return symmetrize(ft2(data), n)
    if len(data.shape) is 3:
        output = np.zeros_like(data)
        for ix, layer in enumerate(data):
            output[ix] = ft2(layer)
        if n is None:
            return output
        else:
            return symmetrize(output, n)
    else:
        print('ERR: Input must be 2D or 3D numpy array.')


def symmetrize(F, n):
    '''
    Applies n-fold symmetrization to the image by rotating clockwise and
    anticlockwise by an angle 2pi/n, then applying a mirror line.  Works on 2D
    and 3D data sets, in the case of 3D each layer is symmetrzed.

    Usage: A.sym = symmetrize(A.qpi, 2)
    '''
    def sym2d(F, n):

        angle = 360.0/n
        out = np.zeros_like(F)
        for ix in range(n):
            out += snd.rotate(F, angle*ix, reshape=False)
            out += snd.rotate(F, -angle*ix, reshape=False)
        out /= 4*n
        return out
    if len(F.shape) is 2:
        return sym2d(F, n)
    if len(F.shape) is 3:
        out = np.zeros_like(F)
        for ix, layer in enumerate(F):
            out[ix] = sym2d(layer, n)
        return out
    else:
        print('ERR: Input must be 2D or 3D numpy array.')

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
        self.p_unsrt = p.reshape(len(p0)/3, 3).T
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

def shearcorr(FT, Bragg):
    '''
        Shear correction on FT, based on coordinates of Bragg peaks.
        Inputs: FT - 2D image or 3D layers;
        Bragg - 2D array of the shape (N, 2), e.g. Bragg = np.array([[x1, y1], [x2, y2], ...])
        
        Note: 1. only works on symmetric (like hexagonal or square) lattices.
        2. should install the package 'skimage' first:   'pip install -U scikit-image'
        3. another function 'peak_local_max' in this package helps you find Bragg peak locations.
        Usage: 'from skimage.feature import peak_local_max'
        '''
    from skimage import transform as tf
    
    def calctform(F, Br):
        ''' calculate transform matrix based on 2D image F and Bragg peaks Br'''
        N = Br.shape[0]
        sx, sy = F.shape
        xn = Br[:, 0]/sx*2-1
        yn = Br[:, 1]/sy*2-1
        R = np.mean(np.sqrt(xn**2+yn**2))
        theta = np.arctan2(yn, xn)
        atansorted = np.asarray(sorted((atanval, ix) for ix, atanval in enumerate(theta)))# sort angles of vertices in rad
        theta = atansorted[:, 0]
        tanseq = np.int_(atansorted[:, 1])
        Br = Br[tanseq] # sort Bragg peaks accordingly
        theta_M = (np.arange(N)*2/N-1)*np.pi # Model angles of vertices, in rad, range (-pi, pi)
        dtheta = np.mean(theta - theta_M)
        xn_M = R * np.cos(theta_M+dtheta) # Generate Model coordinates
        yn_M = R * np.sin(theta_M+dtheta) # based on first point
        Br_M = np.concatenate(((xn_M+1)*sx/2, (yn_M+1)*sy/2)).reshape(2, N).T # Back to original coordinates
        tform = tf.ProjectiveTransform()
        if tform.estimate(Br, Br_M):
            return tform
        else:
            print('failed in calculating transform matrix')

    if len(Bragg.shape) is 2 and Bragg.shape[1] is 2:
        if len(FT.shape) is 2:
            tform = calctform(FT, Bragg)
            FT_corr = tf.warp(FT, inverse_map=tform.inverse, preserve_range=True)
            return FT_corr
        if len(FT.shape) is 3:
            tform = calctform(FT[0], Bragg)
            FT_corr = np.zeros_like(FT)
            for ix, layer in enumerate(FT):
                FT_corr[ix] = tf.warp(layer, inverse_map=tform.inverse, preserve_range=True)
            return FT_corr
        else:
            print('ERR: Input must be 2D or 3D numpy array')
    else:
        print('Bragg peak coordinates should be 2D array of the shape (N, 2)')



def planeSubtract(image, deg, X0=None):
    '''
    Subtracts a polynomial plane from an image. The polynomial does not keep
    any cross terms.
    '''
    def plane(a):
        z = np.zeros_like(image) + a[0]
        N = (len(a)-1)/2
        for k in range(1, N+1):
            z += a[2*k-1] * x**k + a[2*k] * y**k
        return z
    def chi(X):
        chi.fit = plane(X)
        res = norm - chi.fit
        err = np.sum(np.absolute(res))
        return err
    if X0 is None:
        X0 = np.zeros([2*deg+1])
    vx = np.linspace(-1, 1, image.shape[0])
    vy = np.linspace(-1, 1, image.shape[1])
    x, y = vx[:, None], vy[None, :]
    norm = (image-np.mean(image)) / np.max(image-np.mean(image))
    result = opt.minimize(chi, X0)
    return norm - chi.fit

def butter_lowpass_filter(data, ncutoff=0.5, order=1):
    '''
    Low-pass filter applied for an individual spectrum (.dat) or every spectrum in a DOS map (.3ds)
    
    Parameters:
    data: data to be filtered, could be A.didv or A.LIY
    ncutoff: unitless cutoff frequency normaled by Nyquist frequency (half of sampling frequency),
    note that ncutoff <=1, i.e., real cutoff frequency should be less than Nyquist frequency
    order: degree of high frequency attenuation, see Wikipedia item "Butterworth filter".
    
    Usage: A_didv_filt = butter_lowpass_filter(A.didv, ncutoff=0.5, order=1)
           A_LIY_filt = butter_lowpass_filter(A.LIY, ncutoff=0.5, order=1)
    '''
    
    b, a = butter(order, ncutoff, btype='low', analog=False)
    y = np.zeros_like(data)
    if len(data.shape) is 1:
        y = filtfilt(b, a, data)
        return y
    elif len(data.shape) is 3:
        for ic in np.arange(data.shape[1]):
            for ir in np.arange(data.shape[2]):
                didv = data[:, ic, ir]
                y[:, ic, ir] = filtfilt(b, a, didv)
        return y
    else:
        print('ERR: Input must be 1D or 3D numpy array.')


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
    norm = np.sqrt(1/8) # normalize boundaries such that boundary values of modulus map are 1 to be divided.
    dx = x[1]-x[0] # increment of W, E
    dy = y[1]-y[0] # increment of N, S
    dxy = np.sqrt(dx**2 + dy**2) # increment of NW, NE, SW, SE
    
    A_grad_N = ones_like(A)*norm
    A_grad_S = ones_like(A)*norm
    A_grad_W = ones_like(A)*norm
    A_grad_E = ones_like(A)*norm
    A_grad_NW = ones_like(A)*norm
    A_grad_NE = ones_like(A)*norm
    A_grad_SW = ones_like(A)*norm
    A_grad_SE = ones_like(A)*norm
    
    for i in arange(1,col-1):
        for j in arange(1,row-1):
            A_grad_N[i, j] = (A[i, j] - A[i-1, j]) / dy
            A_grad_S[i, j] = (A[i, j] - A[i+1, j]) / dy
            A_grad_W[i, j] = (A[i, j] - A[i, j-1]) / dx
            A_grad_E[i, j] = (A[i, j] - A[i, j+1]) / dx
            A_grad_NW[i, j] = (A[i, j] - A[i-1, j-1]) / dxy
            A_grad_NE[i, j] = (A[i, j] - A[i-1, j+1]) / dxy
            A_grad_SW[i, j] = (A[i, j] - A[i+1, j-1]) / dxy
            A_grad_SE[i, j] = (A[i, j] - A[i+1, j+1]) / dxy
    
    A_grad_col = A_grad_W + (A_grad_NW + A_grad_SW) / np.sqrt(2) - A_grad_E - (A_grad_NE + A_grad_SE)/ np.sqrt(2)
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
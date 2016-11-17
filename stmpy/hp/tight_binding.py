import numpy as np
import matplotlib as mpl
import pylab as plt

k = np.linspace(0, 1, 5e3)

def tight_binding_model_1D(p, en, greens_functions=False,
        anisotropy=(True,True), constrained=False):
    if constrained:
        #af1, af2, v1, v2, g1, g2 = p
        ef1, ef2, c0, t1, t2 = -1.5, -25.5, 0.55, 0.032, -0.020
        ef1, ef2, c0, t1, t2 = -0.9, -26.5, 0.54, 0.0361, -0.0142
        ef1, ef2, c0, t1, t2 = -0.3, -23.5, 0.56, 0.0383, -0.0189
        
        ef1, ef2, c0, t1, t2 = -1.17, -23.25, 0.537, 0.0385, -0.0033
        af1, af2 = 10., -8.
        v1, v2, g1, g2 = p

    else:
        af1, ef1, af2, ef2, c0, v1, v2, g1, g2, t1, t2 = p

    g0 = 2.0 # Take the c-electron self-energy out of optimization
    b = 1600.0 # Take the band minimum out of optimization
    c0 = float(c0)
    f1 = af1*np.cos(k*np.pi) + ef1
    f2 = af2*np.cos(k*np.pi) + ef2
    c = (k**2.-c0**2) * b/c0**2
    s1, s2 = v1, v2
    if anisotropy[0]:
        s1 *= np.sin(k*np.pi)
    if anisotropy[1]:
        s2 *= np.sin(k*np.pi)
    
    G = np.zeros([6, len(en), len(k)], dtype=np.complex128)
    for ix, enval in enumerate(en):
        w0 = enval + 1j*g0
        w1 = enval + 1j*g1
        w2 = enval + 1j*g2
        denominator = s2**2 * (f1-w1) + (f2-w2) * (s1**2 - (c-w0) * (f1-w1))
        G[0,ix] = ((f1 - w1) * (f2 - w2)) / denominator
        G[1,ix] = ((c-w0) * (f2-w2) - s2**2) / denominator
        G[2,ix] = ((c-w0) * (f1-w1) - s1**2) / denominator
        G[3,ix] = (s1 * (f2 - w2)) / denominator
        G[4,ix] = (s2 * (f1 - w1)) / denominator
        G[5,ix] = (s1 * s2) / denominator
    if greens_functions:
        return G
    
    N = 1/(2*np.pi)*np.sum(k*G, axis=2)
    dIdV = -(np.imag(N[0]) +  t1**2*np.imag(N[1]) + t2**2*np.imag(N[2]) 
            + 2*t1*np.imag(N[3]) + 2*t2*np.imag(N[4]) + 2*t1*t2*np.imag(N[5]))
    return dIdV


def tight_binding_model_1F(p, en, greens_functions=False, anisotropy=True):
    af1, ef1, b, c, v1, g0, g1, t1 = p
    f1 = af1*np.cos(k*np.pi) + ef1
    c = (k**2.-c**2.) * b/c**2
    if anisotropy:
        s1 = v1*np.sin(k*np.pi)
    else:
        s1 = v1
    
    G = np.zeros([3, len(en), len(k)], dtype=np.complex128)
    for ix, enval in enumerate(en):
        w0 = enval + 1j*g0
        w1 = enval + 1j*g1
        denominator = s1**2 - (c-w0) * (f1-w1)
        G[0,ix] = (f1-w1) / denominator
        G[1,ix] = (c-w0) / denominator
        G[2,ix] = s1 / denominator
    
    if greens_functions:
        return G

    N = 1/(2*np.pi)*np.sum(k*G, axis=2)
    dIdV = -(np.imag(N[0]) + t1**2*np.imag(N[1]) + 2*t1*np.imag(N[2]))
    return dIdV

def dIdV_1F(G, t1):
    N = 1/(2*np.pi)*np.sum(k*G, axis=2)
    dIdV = -(np.imag(N[0]) + t1**2*np.imag(N[1]) + 2*t1*np.imag(N[2]))
    return dIdV


def bands_1D(p, en):
    af1, ef1, af2, ef2, c0, v1, v2 = p
    b = 1600.0  # Taken out of optimization
    c0 = float(c0)
    s1 = v1*np.sin(k*np.pi)
    s2 = v2*np.sin(k*np.pi)

    H = np.zeros([len(k), 3, 3])
    H[:,0,0] = (k**2.-c0**2) * b/c0**2 
    H[:,1,1] = af1*np.cos(k*np.pi) + ef1
    H[:,2,2] = af2*np.cos(k*np.pi) + ef2
    H[:,0,1] = -s1
    H[:,1,0] = -s1
    H[:,0,2] = -s2
    H[:,2,0] = -s2

    bands = np.zeros([len(k), 3])
    for ix, h in enumerate(H):
        bands[ix] = np.linalg.eigvalsh(h)
    return bands

def fbands_1D(p, en, anisotropy=(True,True), constrained=False):
    if constrained:
        # af1, af2, v1, v2 = p
        ef1, ef2, c0 = -1.5, -25.5, 0.55
        
        ef1, ef2, c0 = -1.17, -23.25, 0.537
        af1, af2 = 10., -8.
        v1, v2 = p
    else:
        af1, ef1, af2, ef2, c0, v1, v2 = p
    b = 1600.0
    c0 = float(c0)
    s1, s2 = v1, v2
    if anisotropy[0]:
        s1 *= np.sin(k*np.pi)
    if anisotropy[1]:
        s2 *= np.sin(k*np.pi)
    
    u = np.zeros([3, len(k)])
    H = np.zeros([3, 3, len(k)])
    H[0,0,:] = (k**2.-c0**2) * b/c0**2 
    H[1,1,:] = af1*np.cos(k*np.pi) + ef1
    H[2,2,:] = af2*np.cos(k*np.pi) + ef2
    H[0,1,:] = -s1
    H[1,0,:] = -s1
    H[0,2,:] = -s2
    H[2,0,:] = -s2
    
    p1 = H[0,1]**2 + H[0,2]**2 + H[1,2]**2
    pix = np.where(p1 == 0)
    nix = np.where(p1 != 0)

    u[0, pix] = H[0, 0, pix]
    u[1, pix] = H[1, 1, pix]
    u[2, pix] = H[2, 2, pix]

    q = (H[0,0,nix] + H[1,1,nix] + H[2,2,nix]) / 3.0
    p2 = ((H[0,0,nix] - q)**2 + (H[1,1,nix] - q)**2 
         + (H[2,2,nix] - q)**2 + 2 * p1[nix])
    p = np.sqrt(p2 / 6.0)

    # Construct an Identity matrix at each point. 
    I = np.zeros_like(H)
    for ix in range(3):
        I[ix,ix] = 1.0

    B = np.zeros_like(H)
    B[:,:,nix] = (1.0 / p) * (H[:,:,nix] - q*I[:,:,nix]) 
    detB = (B[0,0]*B[1,1]*B[2,2] + B[0,1]*B[1,2]*B[2,0]
            + B[0,2]*B[1,0]*B[2,1] - B[2,0]*B[1,1]*B[0,2]
            - B[2,1]*B[1,2]*B[0,0] - B[2,2]*B[1,0]*B[0,1])
    r = detB / 2.0
    phi = np.zeros_like(r)
    rsmallix = np.where(r <= 1)
    rsafeix = np.where(np.absolute(r) < 1) 
    phi[rsmallix] = np.pi/3.0
    phi[rsafeix] = np.arccos(r[rsafeix])/3.0

    u[0,nix] = q + 2*p*np.cos(phi[nix])
    u[1,nix] = q + 2*p*np.cos(phi[nix] + (2*np.pi/3.0))
    u[2,nix] = q + 2*p*np.cos(phi[nix] + (4*np.pi/3.0))
    return np.sort(u, axis=0)

def bands_1F(p, en):
    af1, ef1, b, c, v1 = p
    c = float(c)
    s1 = v1*np.sin(k*np.pi)
    
    u = np.zeros([2, len(k)])
    H = np.zeros([2, 2, len(k)])
    H[0,0] = (k**2.-c**2.) * b/c**2
    H[1,1] = af1*np.cos(k*np.pi) + ef1
    H[1,0] = -s1
    H[0,1] = -s1

    trH = H[0,0] + H[1,1]
    detH = H[0,0]*H[1,1] - s1**2
    gapH = np.sqrt(trH**2 - 4*detH)
    u[0] = (trH + gapH) / 2.0
    u[1] = (trH - gapH) / 2.0

    return np.sort(u, axis=0) 

def fBand1(k): return 9*np.cos(k*np.pi)
def fBand2(k): return -9*np.cos(k*np.pi)-21.0
def cBand(k): return (k**2.-0.54**2)*1600/0.54**2
def H(k,v1,v2): return np.array([[cBand(k), -np.sin(k*np.pi)*v1, -np.sin(k*np.pi)*v2],
                                 [-np.sin(k*np.pi)*v1, fBand1(k), 0],
                                 [-np.sin(k*np.pi)*v2, 0, fBand2(k)]])

def Hamiltonian(k, p, anisotropy=(True,False)):
    af1, ef1, af2, ef2, c0, v1, v2 = p
    cBand = (k**2.-c0**2)*1600/c0**2
    fBand1 = af1*np.cos(k*np.pi) + ef1
    fBand2 = af2*np.cos(k*np.pi) + ef2
    if anisotropy[0]:
        v1 *= np.sin(k*np.pi)
    if anisotropy[1]:
        v2 *= np.sin(k*np.pi)
    return np.array([[cBand, -v1, v2],
                     [-v1, fBand1, 0],
                     [-v2, 0, fBand2]])

def hybridize(k, p, anisotropy=(True,False)):
    bands = np.zeros([3, len(k)])
    character = np.zeros([3, 3, len(k)])
    for ix, kval in enumerate(k):
        u,w = np.linalg.eigh(Hamiltonian(kval, p, anisotropy=anisotropy))
        bands[:,ix], character[:,:,ix] = u, np.abs(w)
    return bands, character

def plot_bands(k,  p, anisotropy=(True,False), label=False):
    u, w = hybridize(k, p, anisotropy=(True,False))
    for ix,(kval,uval) in enumerate(zip(k,u.T)):
        rgb0 = mpl.cm.coolwarm_r([w[0,0,ix]**(0.32)])
        rgb1 = mpl.cm.coolwarm_r([w[0,1,ix]**(0.32)])
        rgb2 = mpl.cm.coolwarm_r([w[0,2,ix]**(0.32)])
        plt.plot(kval, uval[0], 'o', ms=2, mec='None', color=rgb0[0,:3])
        plt.plot(kval, uval[1], 'o', ms=2, mec='None', color=rgb1[0,:3])
        plt.plot(kval, uval[2], 'o', ms=2, mec='None', color=rgb2[0,:3])
    if label:
        plt.plot(-10, -10, color=my_cmap(0), label='4f')
        plt.plot(-10, -10, color=my_cmap(255), label='5d')
    return u 


def hybBands(k,v1,v2=None):
    if v2 is None: v2 = 1.8*v1
    bands = np.zeros([3, len(k)])
    character = np.zeros([3, 3, len(k)])
    for ix, kval in enumerate(k):
        u,w = np.linalg.eigh(H(kval, v1, v2))
        bands[:,ix], character[:,:,ix] = u, np.abs(w)
    return bands, character

def plot_band_character(k, v, label=False):
    u,w = hybBands(k, v[0], v[1])
    for ix,(kval,uval) in enumerate(zip(k,u.T)):
        rgb0 = mpl.cm.coolwarm_r([w[0,0,ix]**(0.32)])
        rgb1 = mpl.cm.coolwarm_r([w[0,1,ix]**(0.32)])
        rgb2 = mpl.cm.coolwarm_r([w[0,2,ix]**(0.32)])
        plt.plot(kval, uval[0], 'o', ms=2, mec='None', color=rgb0[0,:3])
        plt.plot(kval, uval[1], 'o', ms=2, mec='None', color=rgb1[0,:3])
        plt.plot(kval, uval[2], 'o', ms=2, mec='None', color=rgb2[0,:3])
    if label:
        plt.plot(-10, -10, color=my_cmap(0), label='4f')
        plt.plot(-10, -10, color=my_cmap(255), label='5d')
    return u 



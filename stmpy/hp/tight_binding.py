import numpy as np
import matplotlib as mpl
import pylab as plt
from scipy.optimize import minimize
from scipy.special import j0
import scipy.constants as const

k = np.linspace(0, 1, 5e3)
enh = np.linspace(-100,100,500)

def nonlocal_shift(p,en, get_G=False, get_N=False):
    af1, ef1, af2, ef2, c0, v1, v2, g1, g2, t1, t2 = p
    g0 = 3.0
    c0 = float(c0)
    b = 1600
    f1 = af1*np.cos(k*np.pi) + ef1
    f2 = af2*np.cos(k*np.pi) + ef2
    c = (k**2.-c0**2) * b/c0**2

    G = np.zeros([6, len(en), len(k)], dtype=np.complex128)
    for ix, enval in enumerate(en):
        w0 = enval + 1j*g0
        w1 = enval + 1j*g1
        w2 = enval + 1j*g2
        denominator =  (c - w0) * (f1 - w1) * (f2 - w2) - (v2**2 * (f1 - w1) + 
                v1**2 * (f2 - w2)) * (np.sin(np.pi*k)**2)
        G[0,ix] = (-(f1 - w1) * (f2 - w2)) /denominator
        G[1,ix] = (-(c - w0) * (f2 - w2) + v2**2 * np.sin(np.pi*k)**2) /denominator
        G[2,ix] = (-(c - w0) * (f1 - w1) + v1**2 * np.sin(np.pi*k)**2) /denominator
        G[3,ix] = -(v1 * (f2 - w2)) /denominator
        G[4,ix] = -(v2 * (f1 - w1)) /denominator 
        G[5,ix] = -v1*v2*np.sin(np.pi*k)**2 /denominator
    if get_G:
        return G
    N = np.zeros([9,len(en)],  dtype=np.complex128)
    N[0] = (0.5/np.pi)*np.sum((1-k)*G[0], axis=1)
    N[1] = (0.5/np.pi)*np.sum((1-k)*G[1], axis=1)
    N[2] = (0.5/np.pi)*np.sum((1-k)*G[2], axis=1)
    N[3] = (0.5/np.pi)*np.sum((1-k)*G[5], axis=1)
    N[4] = (0.5/np.pi)*np.sum((1-k)*G[1]*j0(2*(1-k)), axis=1)
    N[5] = (0.5/np.pi)*np.sum((1-k)*G[2]*j0(2*(1-k)), axis=1)
    N[6] = (0.5/np.pi)*np.sum((1-k)*G[5]*j0(2*(1-k)), axis=1)
    N[7] = (0.25/np.pi)*np.sum((1-k)*G[3]*(1-j0(2*(1-k))), axis=1)
    N[8] = (0.25/np.pi)*np.sum((1-k)*G[4]*(1-j0(2*(1-k))), axis=1)
    if get_N:
        return N
    didv = -1*(np.imag(N[0]) + 2*t1**2*np.imag(N[1]) + 
            2*t2**2*np.imag(N[2]) + 8*t1*t2*np.imag(N[3]) -
            4*t1**2*np.imag(N[4]) - 4*t2**2*np.imag(N[5]) -
            8*t1*t2*np.imag(N[6]) + 8*t1*np.imag(N[7]) + 8*t2*np.imag(N[8]))
    return didv


def nonlocal_model(p, en, get_G=False, get_N=False, pristine=False):
    af1, ef1, af2, ef2, c0, v1, v2, g1, g2, t1, t2 = p
    g0 = 3.0
    c0 = float(c0)
    b = 1600
    f1 = af1*np.cos(k*np.pi) + ef1
    f2 = af2*np.cos(k*np.pi) + ef2
    c = (k**2.-c0**2) * b/c0**2

    G = np.zeros([6, len(en), len(k)], dtype=np.complex128)
    for ix, enval in enumerate(en):
        w0 = enval + 1j*g0
        w1 = enval + 1j*g1
        w2 = enval + 1j*g2
        denominator =  (c - w0) * (f1 - w1) * (f2 - w2) - (v2**2 * (f1 - w1) + 
                v1**2 * (f2 - w2)) * (np.sin(np.pi*k)**2)
        G[0,ix] = (-(f1 - w1) * (f2 - w2)) /denominator
        G[1,ix] = (-(c - w0) * (f2 - w2) + v2**2 * np.sin(np.pi*k)**2) /denominator
        G[2,ix] = (-(c - w0) * (f1 - w1) + v1**2 * np.sin(np.pi*k)**2) /denominator
        G[3,ix] = -(v1 * (f2 - w2)) /denominator
        G[4,ix] = -(v2 * (f1 - w1)) /denominator 
        G[5,ix] = -v1*v2*np.sin(np.pi*k)**2 /denominator
    if get_G:
        return G
    N = np.zeros([9,len(en)],  dtype=np.complex128)
    N[0] = (0.5/np.pi)*np.sum(k*G[0], axis=1)
    N[1] = (0.5/np.pi)*np.sum(k*G[1], axis=1)
    N[2] = (0.5/np.pi)*np.sum(k*G[2], axis=1)
    N[3] = (0.5/np.pi)*np.sum(k*G[5], axis=1)
    N[4] = (0.5/np.pi)*np.sum(k*G[1]*j0(2*k), axis=1)
    N[5] = (0.5/np.pi)*np.sum(k*G[2]*j0(2*k), axis=1)
    N[6] = (0.5/np.pi)*np.sum(k*G[5]*j0(2*k), axis=1)
    N[7] = (0.25/np.pi)*np.sum(k*G[3]*(1-j0(2*k)), axis=1)
    N[8] = (0.25/np.pi)*np.sum(k*G[4]*(1-j0(2*k)), axis=1)
    if get_N:
        return N
    if pristine:
        didv = -1*(np.imag(N[0]) + 4*t1**2*np.imag(N[1]) + 
            4*t2**2*np.imag(N[2]) + 8*t1*t2*np.imag(N[3]) -
            4*t1**2*np.imag(N[4]) - 4*t2**2*np.imag(N[5]) -
            8*t1*t2*np.imag(N[6]) + 8*t1*np.imag(N[7]) + 8*t2*np.imag(N[8]))

    else:
        didv = -1*(np.imag(N[0]) + 2*t1**2*np.imag(N[1]) + 
            2*t2**2*np.imag(N[2]) + 8*t1*t2*np.imag(N[3]) -
            4*t1**2*np.imag(N[4]) - 4*t2**2*np.imag(N[5]) -
            8*t1*t2*np.imag(N[6]) + 8*t1*np.imag(N[7]) + 8*t2*np.imag(N[8]))
    return didv


def nonlocal_bands_slow(p, en):
    af1, ef1, af2, ef2, c0, v1, v2 = p
    c0 = float(c0)
    b = 1600
    f1 = af1*np.cos(k*np.pi) + ef1
    f2 = af2*np.cos(k*np.pi) + ef2
    c = (k**2.-c0**2) * b/c0**2

    H = np.zeros([len(k), 3, 3], dtype=np.complex128)
    H[:,0,0] = c
    H[:,1,1] = f1
    H[:,2,2] = f2
    H[:,0,1] = -1j*v1*np.sin(k*np.pi)
    H[:,1,0] = 1j*v1*np.sin(k*np.pi)
    H[:,0,2] = -1j*v2*np.sin(k*np.pi)
    H[:,2,0] = 1j*v2*np.sin(k*np.pi)

    bands = np.zeros([len(k), 3])
    for ix, h in enumerate(H):
        bands[ix] = np.linalg.eigvalsh(h)
    return bands
     
def antitunnel_model(p, en, greens_functions=False,
        anisotropy=(True,True), constrained=False, antitunnel=False, get_N=False):
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
    if antitunnel:
        T = np.array([np.ones_like(k), (t1*np.sin(k*np.pi))**2, (t2*np.sin(k*np.pi))**2,
                  2*t1*np.sin(k*np.pi),  2*t2*np.sin(k*np.pi),
                  2*t1*t2*(np.sin(k*np.pi))**2])
    else:
        t = np.array([1, t1**2, t2**2, 2*t1, 2*t2, 2*t1*t2])
        T = t[:,None]*np.ones_like(k)
    Gt = np.ones_like(G)
    for ix, t in enumerate(T):
        Gt[ix] = t*G[ix]
    N = 1/(2*np.pi)*np.sum(k*Gt, axis=2)
    if get_N:
        return N
    dIdV = -np.imag(np.sum(N, axis=0))
    return dIdV



def tight_binding_model_1D(p, en, greens_functions=False,
        anisotropy=(True,True), constrained=False, antitunnel=False, get_N=False):
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
    if antitunnel:
        T = np.array([np.ones_like(k), (t1*np.sin(k*np.pi))**2, (t2*np.sin(k*np.pi))**2,
                  2*t1*np.sin(k*np.pi),  2*t2*np.sin(k*np.pi),
                  2*t1*t2*(np.sin(k*np.pi))**2])
        Gt = np.ones_like(G)
        for ix, t in enumerate(T):
            Gt[ix] = t*G[ix]
        N = 1/(2*np.pi)*np.sum(k*Gt, axis=2)
        if get_N:
            return N
        dIdV = -np.imag(np.sum(N, axis=0))
    else:
        N = 1/(2*np.pi)*np.sum(k*G, axis=2)
        if get_N:
            return N
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


def fitData(data, X0=None, bounds=None, nix=None, add_constant=True,
        anisotropy=(True, False), antitunnel=False):
    if nix is None:
        nix = np.where((data.en<-9) | ((data.en>=-5)&(data.en<-3)) | (data.en>3)) 
    if add_constant:
        def chi_data(X):
            p = X[:-3]
            didv_FIT = tight_binding_model_1D(p, data.en[nix],
                    anisotropy=anisotropy, antitunnel=antitunnel)
            data.fit = didv_FIT * X[-3] + X[-2]*(data.en[nix]) + X[-1]
            err = np.abs(data.fit - data.didv[nix])
            return np.log(np.sum(err**2))
        if X0 is None:
            X0 = 7, -1, -5, -28,  0.55, 36, 45, 3.5, 8, 0.04, -0.01, 3.3, 0.002, 0
        if bounds is None:
            no = (None, None)
            pos = (0, None)
            bounds = [(8,17), (-5,1), (-10,5), (-25,-20), (0.535,0.555), 
                      (20,50), (50,100), pos, pos, no, no, no, no, no]
        data.result = minimize(chi_data, X0, bounds=bounds, method='SLSQP')
        p = data.result.x[:-3]
        fit = tight_binding_model_1D(p, enh, anisotropy=anisotropy, antitunnel=antitunnel)
        data.G = tight_binding_model_1D(p, enh, greens_functions=True,
                anisotropy=anisotropy, antitunnel=antitunnel)
        data.didvf = fit * data.result.x[-3] + data.result.x[-2]*enh + data.result.x[-1]
        fit = tight_binding_model_1D(p, data.en, anisotropy=anisotropy, antitunnel=antitunnel)
        data.ss = data.didv - fit * data.result.x[-3] - \
                data.result.x[-2]*data.en - data.result.x[-1]
    else:
        def chi_data(X):
            p = X[:-2]
            fit = tight_binding_model_1D(p, data.en[nix],
                    anisotropy=anisotropy, antitunnel=antitunnel)
            data.fit = fit * X[-2] + X[-1]*(data.en[nix])
            err = np.abs(data.fit - data.didv[nix])
            return np.log(np.sum(err**2))
        if X0 is None:
            X0 = 7, -1, -5, -28,  0.55, 36, 45, 3.5, 8, 0.04, -0.01, 3.3, 0.002
        if bounds is None:
            no = (None, None)
            pos = (0, None)
            bounds = [(8,17), (-5,1), (-10,5), (-25,-20), (0.535,0.555), 
                      (20,50), (50,100), pos, pos, no, no, no, no]
        data.result = minimize(chi_data, X0, bounds=bounds, method='SLSQP')
        p = data.result.x[:-2]
        fit = tight_binding_model_1D(p, enh, anisotropy=anisotropy, antitunnel=antitunnel)
        data.G = tight_binding_model_1D(p, enh, greens_functions=True,
                anisotropy=anisotropy, antitunnel=antitunnel)
        data.didvf = fit * data.result.x[-2] + data.result.x[-1]*enh
        fit = tight_binding_model_1D(p, data.en, anisotropy=anisotropy, antitunnel=antitunnel)
        data.ss = data.didv - fit * data.result.x[-2] - data.result.x[-1]*data.en
    data.bands = fbands_1D(data.result.x[:7], data.en, anisotropy=anisotropy)


def plot_bands2(ax=None, bulk=False, xBand=True, gBand=True, bragg=False, 
               c1='lime', c2='r', c3='b', p=None):
    '''Add guidelines for the calculated electronic structure of SmB6.

    Inputs:
        ax      - Optional: Axis for plotting.  Default is gca().
        bulk    - Optional: Boolean flag to add bulk band guidelines.
        xBand   - Optional: Boolean flag to add x SS band guidelines.
        gBand   - Optional: Boolean flag to add g SS band guidelines.
        Bragg   - Optional: Boolean flag to add Bragg relection of x SS
                            band guidelines.
        c1      - Optional: String for x SS band color.
        c2      - Optional: String for g SS band color.
        c3      - Optional: String for bulk band color.
        p       - Optional: parameters used for calculating bulk bands.

    Returns:
        None

    History:
        2017-08-06  - HP: Initial commit
    '''
    xx0 = np.linspace(0, 0.5, 10)
    xx1 = np.linspace(-0.5, 0, 10)
    xx2 = np.linspace(0.05, 0.15, 10)
    d = np.array([[   9.98869445,   -4.38151444],
                  [ 118.35067879,   -7.71964218]])
    if p is None:
        p = 16, 1, -9, -24.5, 0.54 , 45, 100
    if ax is None:
        ax = plt.gca()
    bands = fbands_1D(p, 0, anisotropy=(True, False))
    if bulk:
        for band in bands:
            ax.plot(k[::-1], band, '--', color=c2, lw=0.7, alpha=1)
            ax.plot(k, band, '-', color=c2, lw=0.7, alpha=1)
    if xBand:
        ax.plot(xx0, np.polyval(d[0], xx0), '--', color=c1, lw=0.9)
        ax.plot(-xx1, np.polyval(d[0], xx1), '--', color=c1, lw=0.9)
    if gBand:
        ax.plot(xx2, np.polyval(d[1], xx2), '--', color=c3, lw=0.9)
    if bragg:
        ax.plot(xx0+0.5, np.polyval(d[0], xx0), '--', color=c1, lw=0.7, alpha=0.5)
        ax.plot(-xx1+0.5, np.polyval(d[0], xx1), '--', color=c1, lw=0.7, alpha=0.5)

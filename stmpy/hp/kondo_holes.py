import stmpy
import numpy as np
from scipy.optimize import minimize
import pylab as plt


def fano(E, g, ef, q, a, b, c):
    EPrime = 2.0*(E - ef)/g
    y = (q+EPrime)**2 / (EPrime**2 + 1.0)
    return a*y + b*E + c


def fano_fit(xData, yData):
    def chi(X):
        yFit = fano(xData, X[0], X[1], X[2], X[3], X[4], X[5])
        err = np.absolute(yData - yFit)
        return np.log(np.sum(err**2))
    X0 = [8.75, -3.6, -0.6, -5.6, 0.04, 10]
    result = minimize(chi, X0)
    return result


def fano_gapmap(LIY, en): 
    gapmap = np.zeros([6, LIY.shape[1], LIY.shape[2]])
    for iy in range(LIY.shape[1]):
        for ix in range(LIY.shape[2]):
            result = fano_fit(en, LIY[:,iy,ix])
            gapmap[:,iy,ix] = result.x
        stmpy.tools.print_progress_bar(iy, LIY.shape[1]-1, fill='>')
    return gapmap

from matplotlib.colors import ListedColormap
from scipy.io import loadmat
import os

_file = loadmat(os.path.dirname(__file__) + '/YH.mat')
yanghe = ListedColormap(_file['YH'])



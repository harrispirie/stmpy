import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
#For Python 3.8+, to avoid SyntaxWarning: "is" with a literal. Did you mean "=="?

__all__ = ['tools']

from stmpy.io import load, save
from stmpy import tools
from stmpy import matio
from stmpy import image
try:
    from stmpy import driftcorr as drift
except ModuleNotFoundError:
    print(
    '''
    WARNING: Drift correciton has not been imported due to missing modules.
    Please install opencv and scikit-image using:
    \npip3 install opencv-python scikit-image\n
    '''
    )
from stmpy.image import saturate
from stmpy.color.colormap import cm
import stmpy.color.palette as palette

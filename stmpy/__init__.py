__all__ = ['tools']

from stmpy.read_all import load, save
from stmpy.color.colormap import cm 
import stmpy.color.palette as palette
from stmpy import tools
from stmpy import matio
from stmpy import image
from stmpy.image import saturate


delattr(cm, 'ScalarMappable')




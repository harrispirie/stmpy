from matplotlib.colors import to_rgba as _to_rgba

'''
Create color palettes.

Color pallets created here are imported to the main package and can be called
as: stmpy.palette.<palette_name>.<color>.  Or more conveniently as:
    from stmpy.color.palette import <palette_name>
    <palette_name>.<color>.

All colors are attributes of a palette class. All color values should be
specified as RGBA in the range[0,1], or converted from hex using the funtion 
_to_rgba(<hex_string>). Class definitions shoud be at the top of the file and 
hidden. An instance of the class should be created at the bottom of the file.

History:
    2017-06-17  - HP : Intial commit with solarized and hp palette
'''

# ___ CLASS DEFINITIONS ___

class _SolarizedPalette(object):
    '''
    From ethanschoonover.com:

    Solarized is a sixteen color palette (eight monotones, eight accent colors)
    designed for use with terminal and gui applications.   
   '''
    def __init__(self):
        self.base03 = _to_rgba('#002b36')
        self.base02 = _to_rgba('#073642')
        self.base01 = _to_rgba('#586e75')
        self.base00 = _to_rgba('#657b83')
        self.base0  = _to_rgba('#839496')
        self.base1  = _to_rgba('#93a1a1')
        self.base2  = _to_rgba('#eee8d5')
        self.base3  = _to_rgba('#fdf6e3')
        self.yellow = _to_rgba('#b58900')
        self.orange = _to_rgba('#cb4b16')
        self.red    = _to_rgba('#dc322f')
        self.magenta= _to_rgba('#d33682')
        self.violet = _to_rgba('#6c71c4')
        self.blue   = _to_rgba('#268bd2')
        self.cyan   = _to_rgba('#2aa198')
        self.green  = _to_rgba('#859900')


class _HpPalette(object):
    '''
    Just a place to store nice colors as I come across them...
    '''
    def __init__(self):
        self.blue   = _to_rgba('#24226f')
        self.red    = _to_rgba('#ca0222')
        self.orange = _to_rgba('#EB8C2D')
        self.green  = _to_rgba('#68b959')
        self.vblue  = _to_rgba('#377eb8')
        self.vorange = _to_rgba('#ff7f00')
        self.vgreen = _to_rgba('#4daf4a')
        self.vpink  = _to_rgba('#f781bf')
        self.vbrown = _to_rgba('#a65628')
        self.vpurple = _to_rgba('#984ea3')
        self.vgray  = _to_rgba('#999999')
        self.vred   = _to_rgba('#e41a1c')
        self.vyellow = _to_rgba('#dede00')


# ___ CREATE COLOR PALETTES ___

solarized = _SolarizedPalette()
hpalette = _HpPalette()

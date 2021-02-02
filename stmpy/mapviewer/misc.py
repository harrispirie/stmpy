import numpy as np
import math

def volumetype(volume):
    '''for ScrollableMap class'''
    if type(volume) == list:
        if all((type(item) == np.ndarray) for item in volume):
            if all((item.ndim == 2) for item in volume):
                return 'list of 2D arrays'
            else:
                return 'list of arrays'
        else:
            return 'not list of arrays'
    elif type(volume) == np.ndarray:
        if volume.ndim == 3:
            return 'a 3D array'
        elif volume.ndim == 2:
            return 'a 2D array'
        else:
            return 'an array'
    else:
        return 'unknown type'

# Below are blitting methods to update faster than canvas.draw()
# For details of blitting refer to this tutorial https://matplotlib.org/3.3.3/tutorials/advanced/blitting.html
# Modified based on https://gist.github.com/joferkington/6e5bdf8600be2cf4ac79
def grab_bg(canvas, artists):
    for artist in artists:
        artist.set_visible(False)
    canvas.draw_idle()
    bg = canvas.copy_from_bbox(canvas.figure.bbox)
    for artist in artists:
        artist.set_visible(True)
    # self.blit(artists) # this was an extra, problematic line in the original gist
    return bg

def blit_bg(canvas, bg, artists):
    canvas.restore_region(bg)
    for artist in artists:
        artist.axes.draw_artist(artist)
    canvas.blit(canvas.figure.bbox)
    canvas.flush_events()

### copy from matplotlib.ticker.EngFormatter ###
def format_eng(num, places=None, sep='', _usetex=False, _useMathText=False):
    """
    Format a number in engineering notation, appending a letter
    representing the power of 1000 of the original number.
    Some examples:

    >>> format_eng(0)       # for places = 0
    '0'

    >>> format_eng(1000000) # for places = 1
    '1.0 M'

    >>> format_eng("-1e-6") # for places = 2
    '-1.00 \N{MICRO SIGN}'
    """
    ENG_PREFIXES = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
         -9: "n",
         -6: "\N{MICRO SIGN}",
         -3: "m",
          0: "",
          3: "k",
          6: "M",
          9: "G",
         12: "T",
         15: "P",
         18: "E",
         21: "Z",
         24: "Y"
    }

    sign = 1
    fmt = ".6g" if places is None else ".{:d}g".format(places)

    if num < 0:
        sign = -1
        num = -num

    if num != 0:
        pow10 = int(math.floor(math.log10(num) / 3) * 3)
    else:
        pow10 = 0
        # Force num to zero, to avoid inconsistencies like
        # format_eng(-0) = "0" and format_eng(0.0) = "0"
        # but format_eng(-0.0) = "-0.0"
        num = 0.0

    pow10 = np.clip(pow10, min(ENG_PREFIXES), max(ENG_PREFIXES))

    mant = sign * num / (10.0 ** pow10)
    # Taking care of the cases like 999.9..., which may be rounded to 1000
    # instead of 1 k.  Beware of the corner case of values that are beyond
    # the range of SI prefixes (i.e. > 'Y').
    if (abs(float(format(mant, fmt))) >= 1000
            and pow10 < max(ENG_PREFIXES)):
        mant /= 1000
        pow10 += 3

    prefix = ENG_PREFIXES[int(pow10)]
    if _usetex or _useMathText:
        formatted = "${mant:{fmt}}${sep}{prefix}".format(
            mant=mant, sep=sep, prefix=prefix, fmt=fmt)
    else:
        formatted = "{mant:{fmt}}{sep}{prefix}".format(
            mant=mant, sep=sep, prefix=prefix, fmt=fmt)

    return formatted
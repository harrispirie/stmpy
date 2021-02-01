import numpy as np
import math

# def helpfile():
#     helptext =
#         "1. Hit Cmd+Q (Mac) or ? (Windows) to quit Python process after use; Otherwise the cell in jupyter notebook would not finish execution.\n 2. Scroll mouse in Topo window to iterate between channels.
# 3. Scroll mouse in Map window to iterate through bias layers. The vertical green line in Spectrum window will follow and inicate the current bias.
# 4. Click and drag either one of the yellow cross cursors to show spectra at different locations.
# 5. Click "Cursor" button to show/hide yellow cursors.
# 6. Use the native navigation toolbar in the window to zoom, pan, and reset field of views. (Reset somehow freezes the limits of the spectrum plot...)
# 7. Move mouse over windows, current xy coordinates (and intensity value) will be shown on the native navigation toolbar
# 8. Click and drag indicators in the colorbars to adjust the contrast of the corresponding image.
# 9. Click "Fit" button to show/hide fit data (if provided when calling the function)."
    
#     return helptext

def volumetype(volume):
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
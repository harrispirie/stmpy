### Modified based on matplotlib.widgets.Slider ###

import matplotlib as mpl
import numpy as np
from matplotlib import ticker
from matplotlib.colors import Colormap
from matplotlib.widgets import AxesWidget
from misc import format_eng

class RangeSlider(AxesWidget):
    def __init__(self, ax, label, valmin, valmax, valinit=None, valfmt=None,
                 closedmin=True, closedmax=True, slidermin=None,
                 slidermax=None, valstep=None, pixeltol=10,
                 orientation='horizontal', cmap=None, **kwargs):
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : tuple or list of floats, default: (0, 1)
            The slider initial positions.

        valfmt : str, default: None
            %-format string used to format the slider value.  If None, a
            `.EngFormatter` is used instead.

        closedmin : bool, default: True
            Whether the slider interval is closed on the bottom.

        closedmax : bool, default: True
            Whether the slider interval is closed on the top.

        slidermin : Slider, default: None
            Do not allow the current slider to have a value less than
            the value of the Slider *slidermin*.

        slidermax : Slider, default: None
            Do not allow the current slider to have a value greater than
            the value of the Slider *slidermax*.

        pixeltol : int, default: 5
            Tolerance of mouse position in pixels with respect to slider position.

        valstep : float, default: None
            If given, the slider will snap to multiples of *valstep*.

        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The orientation of the slider.

        cmap : matplotlib.colors.Colormap, defalt: None
            Whether a colormap is showed instead of a colored range between slider poisitions

        kwargs : Line2D properties (cmap is specified) or Rectangle properties (cmap is None)
        """
        if ax.name == '3d':
            raise ValueError('Sliders cannot be added to 3D Axes')

        AxesWidget.__init__(self, ax)

        if slidermin is not None and not hasattr(slidermin, 'val'):
            raise ValueError("Argument slidermin ({}) has no 'val'".format(type(slidermin)))

        if slidermax is not None and not hasattr(slidermax, 'val'):
            raise ValueError("Argument slidermax ({}) has no 'val'".format(type(slidermax)))
        if orientation not in ['horizontal', 'vertical']:
            raise ValueError("Argument orientation ({}) must be either""'horizontal' or 'vertical'".format(orientation))
        self.kwargs = kwargs
        self.orientation = orientation
        self.closedmin = closedmin
        self.closedmax = closedmax
        self.slidermin = slidermin
        self.slidermax = slidermax
        self.bound = None
        self.drag_active = False
        self.set_valmin_valmax(valmin, valmax)
        if valinit is None:
            valinit = (valmin, valmax)
        if len(valinit) != 2:
            raise ValueError("Argument valinit ({}) must have length of 2"
                             .format(type(valinit)))
        else:
            if valinit[0] >= valinit[1]:
                raise ValueError("First value of argument valinit ({}) must be smaller than the second one".format(type(valinit)))

        self.valstep = valstep
        self.pixeltol = pixeltol
        if self._value_in_bounds(valinit[0]) is None:
            valinit[0] = valmin
        if self._value_in_bounds(valinit[1]) is None:
            valinit[1] = valmax
        self.vals = valinit
        self.valinit = valinit
        self.cmap = cmap

        if isinstance(cmap, Colormap):
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            if orientation == 'vertical':
                gradient = np.flipud(gradient.transpose())
                extent = [0, 1, self.vals[0], self.vals[1]]
            else:
                extent = [self.vals[0], self.vals[1], 0, 1]
            self.im = ax.imshow(gradient, aspect='auto', cmap=self.cmap, extent=extent)
            clow, chigh = cmap(0.0), cmap(1.0)
            if orientation == 'vertical':
                self.low = ax.axhspan(self.valmin, valinit[0], 0, 1, fc=clow)
                self.high = ax.axhspan(valinit[1], self.valmax, 0, 1, fc=chigh)
            else:
                self.low = ax.axvspan(self.valmin, valinit[0], 0, 1, fc=clow)
                self.high = ax.axvspan(valinit[1], self.valmax, 0, 1, fc=chigh)

        else:
            self.cmap = None
            if cmap is not None:
                raise TypeError("cmap must be a Colormap instance! Use None instead.")
        
        if cmap is not None:
            polykwargs = {}
            linekwargs = self.kwargs
        else:
            polykwargs = self.kwargs
            linekwargs = {'lw':0.5, 'color':'k'}

        if orientation == 'vertical':
            self.poly = ax.axhspan(valinit[0], valinit[1], 0, 1, **polykwargs)
            self.hline1 = ax.axhline(valinit[0], 0, 1, **linekwargs)
            self.hline2 = ax.axhline(valinit[1], 0, 1, **linekwargs)
        else:
            self.poly = ax.axvspan(valinit[0], valinit[1], 0, 1, **polykwargs)
            self.vline1 = ax.axvline(valinit[0], 0, 1, **linekwargs)
            self.vline2 = ax.axvline(valinit[1], 0, 1, **linekwargs)

        self.valfmt = valfmt

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)
        if cmap is not None:
            self.poly.set_visible(False) # hide poly but still use its verts

        self.cid_press = self.connect_event('button_press_event', self._pressed)
        self.label = label
        if orientation == 'vertical':
            self.labeltext = ax.text(0.5, 1.02, label+'\n'+self._format(valinit[1]), transform=ax.transAxes,
                                 verticalalignment='bottom', horizontalalignment='center')

            self.valtext = ax.text(0.5, -0.02, self._format(valinit[0]), transform=ax.transAxes,
                                   verticalalignment='top', horizontalalignment='center')
        else:
            self.labeltext = ax.text(-0.02, 0.5, label+' '+self._format(valinit[0]), transform=ax.transAxes,
                                 verticalalignment='center', horizontalalignment='right')

            self.valtext = ax.text(1.02, 0.5, self._format(valinit[1]), transform=ax.transAxes,
                                   verticalalignment='center', horizontalalignment='left')

        self.cnt = 0
        self.observers = {}

        # self.set_val(valinit[0], 0)
        # self.set_val(valinit[1], 1)

    def _value_in_bounds(self, val):
        """Makes sure *val* is with given bounds."""
        if self.valstep:
            val = (self.valmin
                   + round((val - self.valmin) / self.valstep) * self.valstep)

        if val <= self.valmin:
            if not self.closedmin:
                return
            val = self.valmin
        elif val >= self.valmax:
            if not self.closedmax:
                return
            val = self.valmax

        if self.slidermin is not None and val <= self.slidermin.val:
            if not self.closedmin:
                return
            val = self.slidermin.val

        if self.slidermax is not None and val >= self.slidermax.val:
            if not self.closedmax:
                return
            val = self.slidermax.val
        return val

    def _pressed(self, event):
        """Update the slider position."""
        if event.button != 1 or event.inaxes != self.ax:
            return

        if self.orientation == 'vertical':
            d0 = abs(self.ax.transData.transform((0, self.vals[0]))[1] - event.y)
            d1 = abs(self.ax.transData.transform((0, self.vals[1]))[1] - event.y)
            if min(d0, d1) > self.pixeltol:
                return
            else:
                self.bound = np.argmin([d0, d1])
                self.drag_active = True
                event.canvas.grab_mouse(self.ax)
                self.cid_release = self.connect_event('button_release_event', self._released)
                self.cid_motion = self.connect_event('motion_notify_event', self._update)
        else:
            d0 = abs(self.ax.transData.transform((self.vals[0], 0))[0] - event.x)
            d1 = abs(self.ax.transData.transform((self.vals[1], 0))[0] - event.x)
            if min(d0, d1) > self.pixeltol:
                return
            else:
                self.bound = np.argmin([d0, d1])
                self.drag_active = True
                event.canvas.grab_mouse(self.ax)
                self.cid_release = self.canvas.mpl_connect('button_release_event', self._released)
                self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self._update)

    def _update(self, event):
        if event.inaxes != self.ax:
            self.drag_active = False
            return
        if self.drag_active:
            if self.orientation == 'vertical':
                    val = self._value_in_bounds(event.ydata)
            else:
                val = self._value_in_bounds(event.xdata)
            if (val not in [None, self.vals[0], self.vals[1]]) and (self.bound is not None):
                self.set_val(val, self.bound)

    def _released(self, event):
        if event.button != 1:
            return
        self.drag_active = False
        self.bound = None
        event.canvas.release_mouse(self.ax)
        self.canvas.mpl_disconnect(self.cid_motion)
        self.canvas.mpl_disconnect(self.cid_release)


    def _format(self, val):
        """Pretty-print *val*."""
        if self.valfmt is not None:
            return self.valfmt % val
        else:
            return format_eng(val, places=4)

    def set_val(self, val, bound):
        """
        Set slider value to *val*

        Parameters
        ----------
        val : float
        bound : 0 (min) or 1 (max), changed
        """
        vals = list(self.vals)
        vals[bound] = val
        if vals[0] >= vals[1]:
            raise ValueError("Lower bound must be smaller than upper bound!")
        xy = self.poly.xy
        ## xy = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]
        if self.orientation == 'vertical':
            if bound: # ymax changed
                xy[1] = 0, val
                xy[2] = 1, val
            else:     # ymin changed
                xy[0] = 0, val
                xy[3] = 1, val
        else:
            if bound: # xmax changed
                xy[2] = val, 1
                xy[3] = val, 0
            else:     # xmin changed
                xy[0] = val, 0
                xy[1] = val, 1
        xy[-1] = xy[0] # ensure a closed rectangle
        self.poly.xy = xy

        if self.cmap is not None:
            # extent = xmin, xmax, ymin, ymax
            extent = xy[0][0], xy[2][0], xy[0][1], xy[2][1]
            self.im.set_extent(extent)
            if self.orientation == 'vertical':
                self.low.xy[1][1] = extent[2]
                self.low.xy[2][1] = extent[2]
                self.high.xy[0][1] = extent[3]
                self.high.xy[3][1] = extent[3]
            else:
                self.low.xy[2][0] = extent[0]
                self.low.xy[3][0] = extent[0]
                self.high.xy[0][0] = extent[1]
                self.high.xy[1][0] = extent[1]

            self.low.xy[-1] = self.low.xy[0] # ensure a closed rectangle
            self.high.xy[-1] = self.high.xy[0]


        if self.orientation == 'vertical':
            label = self.labeltext.get_text().split('\n')[0]
            if bound: # max changed
                self.labeltext.set_text(label+'\n'+self._format(val))
                self.hline2.set_ydata(val)
            else:
                self.valtext.set_text(self._format(val))
                self.hline1.set_ydata(val)
        else:
            label = self.labeltext.get_text().split(' ')[0]
            if bound: # max changed
                self.valtext.set_text(self._format(val))
                self.vline2.set_xdata(val)
            else:
                self.labeltext.set_text(label+' '+self._format(val))
                self.vline1.set_xdata(val)

        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.vals = vals
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(vals)

    def set_valmin_valmax(self, valmin, valmax, extra=0.01, update=False):
        valrange = valmax - valmin
        if valrange <= 0:
            raise ValueError("valmax must be greater than valmin!")
        self.valmin = valmin - extra * valrange
        self.valmax = valmax + extra * valrange
        if self.orientation == 'vertical':
            self.ax.set_ylim(self.valmin, self.valmax)
        else:
            self.ax.set_xlim(self.valmin, self.valmax)
        if update:
            if self.cmap is not None:
                if self.orientation == 'vertical':
                    self.high.xy[1][1] = self.valmax
                    self.high.xy[2][1] = self.valmax
                    self.low.xy[0][1] = self.valmin
                    self.low.xy[3][1] = self.valmin
                else:
                    self.high.xy[2][0] = self.valmax
                    self.high.xy[3][0] = self.valmax
                    self.low.xy[0][0] = self.valmin
                    self.low.xy[1][0] = self.valmin
                self.low.xy[-1] = self.low.xy[0] # ensure a closed rectangle
                self.high.xy[-1] = self.high.xy[0]

            vals = []
            vals.append(self._value_in_bounds(self.vals[0]))
            vals.append(self._value_in_bounds(self.vals[1]))
            if vals[0] != self.vals[0]:
                vals[0] = valmin # reset
            if vals[1] != self.vals[1]:
                vals[1] = valmax # reset
            self.vals = vals
            self.set_val(vals[0], 0)
            self.set_val(vals[1], 1)


    def on_changed(self, func):
        """
        When the slider value is changed call *func* with the new
        slider value

        Parameters
        ----------
        func : callable
            Function to call when slider is changed.
            The function must accept a single float as its arguments.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*)
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid


    def disconnect(self, cid):
        """
        Remove the observer with connection id *cid*

        Parameters
        ----------
        cid : int
            Connection id of the observer to be removed
        """
        try:
            del self.observers[cid]
        except KeyError:
            pass

    def reset(self):
        """Reset the slider to the initial value"""
        if self.vals != self.valinit:
            self.set_val(self.valinit[0], 0)
            self.set_val(self.valinit[1], 1)


# import matplotlib.pyplot as plt
# from matplotlib.widgets import Button, RadioButtons
# from matplotlib.ticker import EngFormatter
# from matplotlib.ticker import MaxNLocator

# fig, ax = plt.subplots()
# fig.set_tight_layout(False)
# plt.subplots_adjust(left=0.25, bottom=0.25)
# t = np.arange(0.0, 1.0, 0.001)
# s = 5000 * np.sin(2 * np.pi * 3 * t)
# l, = plt.plot(t, s, lw=2)
# ax.margins(x=0)
# ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# axcolor = 'lightgoldenrodyellow'
# axamp = plt.axes([0.25, 0.05, 0.65, 0.03],)# facecolor=axcolor)
# # axamp = plt.axes([0.25, 0.05, 0.03, 0.65],)# facecolor=axcolor)
# samp = RangeSlider(axamp, 'YLim', valmin=-10000, valmax=10000, orientation='horizontal', cmap=mpl.cm.bwr, lw=0.5, color='orange', alpha=0.7)
# samp.labeltext.set_fontsize(6)
# samp.valtext.set_fontsize(6)


# def update(vals):
#     ymin, ymax = samp.vals[0], samp.vals[1]
#     l.axes.set_ylim(ymin, ymax)
#     fig.canvas.draw_idle()
# update(samp.vals)
# samp.on_changed(update)
# samp.set_valmin_valmax(-20000, 20000, update=True)
# # samp.set_val(-18000, 0)
# # samp.set_val(-16000, 1)

# resetax = plt.axes([0.025, 0.5, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
# button.label.set_fontsize(6)
# def reset(event):
#     # sfreq.reset()
#     samp.reset()
# button.on_clicked(reset)
# plt.show()

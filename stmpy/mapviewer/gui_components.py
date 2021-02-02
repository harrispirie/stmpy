import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import Colormap
from matplotlib.widgets import AxesWidget
from .misc import format_eng, grab_bg, blit_bg, volumetype

class SpecLinkedCursors:
    '''
    Two linked cursors in two separate images/axes. Image shapes/extents must be the same.
    Created with the help of matplotlib.widget.MultiCursor
    '''
    epsilon = 10 # tolerance in pixels for enabling cursor dragging
    cursor_color = 'gold'
    bias_color = 'forestgreen'
    cursor_lw = 0.5
    title_fontdict = dict(fontsize=12)

    
    def __init__(self, axtopo, axmap, axspec, tcbar, mcbar, fit_didv=None, use_blit=True):
        '''
        Inputs:
            axtopo, axmap, axspec - Axes instances in the figure
            tcbar, mcbar - RangeSlider instances, ajustable colorbars for the images
            fit_didv - 3d array, to store fit data for spectrum plot overlay
            use_blit - bool, whether to use blitting, default is True
        '''
        # axmap has attributes: volume = map array; index = bias index to show; ch_names = list of channel names(default = [])
        self.axtopo = axtopo
        self.axmap = axmap
        self.axspec = axspec
        self.tcbar = tcbar # RangeSlider instance, to update color limits after mouse scroll
        self.mcbar = mcbar # RangeSlider instance, to update color limits after mouse scroll
        self.fig = axtopo.figure
        self.canvas = self.fig.canvas
        self.im1 = axtopo.get_images()[-1] # image on top
        self.im2 = axmap.get_images()[-1]
        self.spec = axspec.get_lines()[0] # bottom plot is spectrum plot
        self.en = self.spec.get_data()[0]
        self.en_ind = axmap.index # for bias scroll
        self.z_ind = axtopo.index # for topo channel scroll
        extent = self.im1.get_extent() # set by scan info
        shape = self.im1.get_array().shape
        if not np.allclose(extent, self.im2.get_extent()):
            print('WARNING: two images have different extents! Please check input arrays!')
        if not shape == self.im2.get_array().shape:
            print('WARNING: two images have different shapes! Please check input arrays!')
        ### initialize plots ###
        self.dx = (extent[1]-extent[0])/shape[-1]
        self.dy = (extent[3]-extent[2])/shape[-2]
        self.x = np.arange(extent[0]+self.dx/2, extent[1], self.dx) # half pixel shift in imshow
        self.y = np.arange(extent[2]+self.dy/2, extent[3], self.dy)
        self._last_indices = [int(shape[-1]/2), int(shape[-2]/2)] # save cursor position in xy indices (snap to data points)
        self.spec.set_ydata(axmap.volume[:, self._last_indices[1], self._last_indices[0]])
        self.fit_didv = fit_didv
        if self.fit_didv is not None:
            self.spec_fit, = self.axspec.plot(self.en, self.fit_didv[:, self._last_indices[1], self._last_indices[0]], 'tomato', lw=1, alpha=0.7,)
        self.bline = axspec.axvline(self.en[self.en_ind], color=self.bias_color, lw=self.cursor_lw)
        self.btext = axmap.set_title('Bias='+format_eng(self.en[self.en_ind])+'V', fontdict=self.title_fontdict)
        self.btext.set_color(self.bias_color)
        self.ttext = axtopo.set_title(self.axtopo.ch_names[self.z_ind], fontdict=self.title_fontdict)
        axspec.relim()
        axspec.autoscale_view()
        ### plot cursors ###
        xc, yc = self.x[self._last_indices[0]], self.y[self._last_indices[1]]
        self.hline1 = axtopo.axhline(yc, color=self.cursor_color, lw=self.cursor_lw)
        self.vline1 = axtopo.axvline(xc, color=self.cursor_color, lw=self.cursor_lw)
        self.hline2 = axmap.axhline(yc, color=self.cursor_color, lw=self.cursor_lw)
        self.vline2 = axmap.axvline(xc, color=self.cursor_color, lw=self.cursor_lw)
        ### events initialization ###
        self.drag = False
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.cid_bscroll = self.canvas.mpl_connect('scroll_event', self.bias_scroll)
        if self.axtopo.volume.shape[0] > 1: # suppress scroll event if voloume has only one image
            self.cid_tscroll = self.canvas.mpl_connect('scroll_event', self.topo_scroll)
        self.use_blit = use_blit
        if self.use_blit:
            self._bg = None

    def on_button_press(self, event):
        if event.inaxes not in [self.axtopo, self.axmap]:
            return
        if event.button != 1:
            return
        index = self._last_indices
        xy_disp = event.inaxes.transData.transform((self.x[index[0]], self.y[index[1]])) # transform cursor position in display pixels
        d = np.sqrt((xy_disp[0] - event.x)**2 + (xy_disp[1] - event.y)**2) # event.x, event.y - mouse position in display coordinate
        if d <= self.epsilon: # if mouse position is within the display pixel range
            self.drag = True
            # Only connect to mouse movement when the left mouse button REMAINS pressed
            self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_button_release)
            self.cid_move = self.canvas.mpl_connect('motion_notify_event', self.drag_cursors)

    def on_button_release(self, event):
        '''Disconnect events when release the left mouse button'''
        if event.button != 1:
            return
        self.drag = False
        self.canvas.mpl_disconnect(self.cid_move)
        self.canvas.mpl_disconnect(self.cid_release)

    def bias_scroll(self, event):
        '''mouse scroll events in the map window'''
        if event.inaxes != self.axmap:
            return
        else:
            if event.button == 'up': 
                ind = self.en_ind - 1
                if ind >= 0:
                    self.bias_scroll_update(ind)
            elif event.button == 'down':
                ind = self.en_ind + 1
                if ind < len(self.en):
                    self.bias_scroll_update(ind)

    def bias_scroll_update(self, ind):
        '''
        Lots to do when scroll:
            1. move axvline correspondingly in the spectrum window
            2. update slice in the map window
            3. update clims of the map window, and colorbar indicators
            4. update title of the map window
        '''
        if self.use_blit:
            self._bg = grab_bg(self.canvas, [self.bline, self.btext, self.im2, self.vline2, self.hline2])
            self.canvas.restore_region(self._bg)
        self.bline.set_xdata(self.en[ind])
        self.im2.set_array(self.axmap.volume[ind])
        newmin, newmax = self.axmap.volume[ind].min(), self.axmap.volume[ind].max()
        self.im2.set_clim(newmin, newmax)
        self.mcbar.set_val(newmin, 0)
        self.mcbar.set_val(newmax, 1)
        self.btext = self.axmap.set_title('Bias='+format_eng(self.en[ind])+'V', fontdict=self.title_fontdict)
        self.en_ind = ind
        if self.use_blit:
            blit_bg(self.canvas, self._bg, [self.bline, self.btext, self.im2, self.vline2, self.hline2])
        else:
            self.canvas.draw_idle()

    def map_clim_update(self, vals):
        '''update clims of the map window.'''
        if self.use_blit:
            self._bg = grab_bg(self.canvas, [self.im2, self.vline2, self.hline2])
            self.canvas.restore_region(self._bg)
        newmin, newmax = vals[0], vals[1]
        self.im2.set_clim(newmin, newmax)
        if self.use_blit:
            blit_bg(self.canvas, self._bg, [self.im2, self.vline2, self.hline2])
        else:
            self.canvas.draw_idle()

    def topo_scroll(self, event):
        '''mouse scroll events in the topo window'''
        if event.inaxes != self.axtopo:
            return
        else:
            if event.button == 'up':
                ind = self.z_ind - 1
                if ind >= 0:
                    self.topo_scroll_update(ind)
            elif event.button == 'down':
                ind = self.z_ind + 1
                if ind < self.axtopo.volume.shape[0]:
                    self.topo_scroll_update(ind)

    def topo_scroll_update(self, ind):
        '''
        Lots to do when scroll:
            1. update display channel in the topo window
            2. update clims of the topo window, and colorbar indicators
            3. update title of the topo window
        '''
        if self.use_blit:
            self._bg = grab_bg(self.canvas, [self.ttext, self.im1, self.vline1, self.hline1])
            self.canvas.restore_region(self._bg)
        self.im1.set_array(self.axtopo.volume[ind])
        newmin, newmax = self.axtopo.volume[ind].min(), self.axtopo.volume[ind].max()
        newrange = newmax - newmin
        self.im1.set_clim(newmin, newmax)
        self.tcbar.set_valmin_valmax(newmin, newmax, update=True)
        self.tcbar.set_val(newmin, 0)
        self.tcbar.set_val(newmax, 1)
        self.ttext = self.axtopo.set_title(self.axtopo.ch_names[ind], fontdict=self.title_fontdict)
        self.z_ind = ind
        if self.use_blit:
            blit_bg(self.canvas, self._bg, [self.ttext, self.im1, self.vline1, self.hline1])
        else:
            self.canvas.draw_idle()        

    def topo_clim_update(self, vals):
        '''update clims of the topo window'''
        if self.use_blit:
            self._bg = grab_bg(self.canvas, [self.im1, self.vline1, self.hline1])
            self.canvas.restore_region(self._bg)
        newmin, newmax = vals[0], vals[1]
        self.im1.set_clim(newmin, newmax)
        if self.use_blit:
            blit_bg(self.canvas, self._bg, [self.im1, self.vline1, self.hline1])
        else:
            self.canvas.draw_idle()

    def drag_cursors(self, event):
        '''
        update cursor position (always snapped to data points), and spectrum plot(s)
        Condition: mouse in the topo or map window, clicked on/near the current cursor position, while no navigation tools are enabled.
        '''
        if event.inaxes not in [self.axtopo, self.axmap]:
            self.drag = False
            return
        else:
            for ax in self.fig.axes:
                navmode = ax.get_navigate_mode()
                # avoid possible conflicts between drag and navigation toolbar
                if navmode is not None:
                    self.drag = False
                    break
            if self.drag:   
                x, y = event.xdata, event.ydata # mouse position in data coordinate
                indices = [min(np.searchsorted(self.x+self.dx/2, x), len(self.x) - 1), min(np.searchsorted(self.y+self.dy/2, y), len(self.y) - 1)]
                if np.allclose(indices, self._last_indices):
                    return  # still on the same data point. Nothing to do.
                self._last_indices = indices
                x = self.x[indices[0]]
                y = self.y[indices[1]]
                # update the cursor positions
                if self.use_blit:
                    artists = [self.hline1, self.hline2, self.vline1, self.vline2, self.spec, self.bline,]
                    if self.fit_didv is not None:
                        artists.append(self.spec_fit)
                    self._bg = grab_bg(self.canvas, artists)
                    self.canvas.restore_region(self._bg)
                self.hline1.set_ydata(y)
                self.vline1.set_xdata(x)
                self.hline2.set_ydata(y)
                self.vline2.set_xdata(x)
                self.spec.set_ydata(self.axmap.volume[:, indices[1], indices[0]])
                if self.fit_didv is not None:
                    self.spec_fit.set_ydata(self.fit_didv[:, self._last_indices[1], self._last_indices[0]])
                self.axspec.autoscale(enable=True, axis='both')
                self.axspec.relim()
                self.axspec.autoscale_view()
                if self.use_blit:
                    blit_bg(self.canvas, self._bg, artists)
                else:
                    self.canvas.draw_idle()


class ScrollableMap:
    '''an extention to imshow on 3d data. its volume attribute stores the 3d array. events are defined in the SpecLinkedCursors class'''
    def __init__(self, volume, ax, ch_names=[], **kwargs):
        self.ax = ax
        self.canvas = ax.figure.canvas
        ### check volume type and channel names ###
        if volumetype(volume) == 'a 3D array':
            self.ax.volume = volume
            self.ax.ch_names = [] # kill channel name for map (you should know the channel name)
        elif volumetype(volume) == 'list of 2D arrays':
            self.ax.volume = np.asarray(volume)
            if ch_names:
                if len(volume) == len(ch_names):
                   self.ax.ch_names = ch_names
                else:
                    raise ValueError('ERROR: number of ch_names does not match number of channels in volume!')
            else:
                if len(volume) == 1:
                    self.ax.ch_names = ['Topo']
                else:
                    raise ValueError('ERROR: please provide %d channel names!'%len(volume))
        elif volumetype(volume) == 'a 2D array':
            self.ax.volume = volume.reshape(1, volume.shape[0], volume.shape[1])
            if ch_names:
                self.ax.ch_names = ch_names[0] # only use the first channel name
            else:
                self.ax.ch_names = ['Topo']
                print('WARNING: channel name not provided, use \'Topo\' as default.')
        else:
            raise TypeError('ERROR: volume argument only supports input types of a 2D/3D array or a list of 2D arrays!')

        ### draw the first image in volume ###
        self.ax.index = 0
        self.im = ax.imshow(self.ax.volume[self.ax.index], **kwargs)

class LimLinkedAxes:
    '''Link xy limits of two axes so they update together'''
    def __init__(self, ax1, ax2):
        self.ax1 = ax1
        self.ax2 = ax2
        self.canvas = ax1.figure.canvas
        self.oxy = [self._get_limits(ax) for ax in [ax1, ax2]] # old limits
        self.cid1 = self.canvas.mpl_connect('motion_notify_event', self.re_zoom)  # for right-click pan/zoom
        self.cid2 = self.canvas.mpl_connect('button_release_event',self.re_zoom)  # for rectangle-select zoom

    def _get_limits(self, ax):
        return [list(ax.get_xlim()), list(ax.get_ylim())]

    def _set_limits(self, ax, lims):
        ax.set_xlim(*(lims[0]))
        ax.set_ylim(*(lims[1]))

    def re_zoom(self, event):
        for ax in [self.ax1, self.ax2]:
            navmode = ax.get_navigate_mode()
            if navmode is not None:
                break
        if navmode not in ['PAN', 'ZOOM']:
            return
        else:
            for ix, ax in enumerate([self.ax1, self.ax2]):
                nxy = self._get_limits(ax) # new limits
                if self.oxy[ix] != nxy:
                    self._set_limits(self.ax1, nxy)
                    self._set_limits(self.ax2, nxy)
                    self.canvas.draw_idle()
                    self.oxy = [self._get_limits(ax) for ax in [self.ax1, self.ax2]]

class LinesVisible:
    '''Button that controls visibility of lines'''
    ind = 1 # default is ON
    def __init__(self, button, lines):
        self.button = button
        self.lines = lines
        self.bcolor = button.ax.get_facecolor()

    def vis(self, event):
        self.ind = (self.ind + 1) % 2
        bcolors = ['0.85', self.bcolor]
        for line in self.lines:
            line.set_visible(self.ind)
        self.button.ax.set_facecolor(bcolors[self.ind]) # this changes color immediately
        self.button.color = bcolors[self.ind] # this only changes color when mouse move
        self.button.hovercolor = bcolors[self.ind]
        line.figure.canvas.draw_idle()

class HelpWindow:
    '''Button that pops up helptext in the help window '''
    def __init__(self, button, helptext):
        self.button = button 
        self.helptext = helptext     

    def helpwindow(self, event):
        exist = 99 in plt.get_fignums() # 99 will be the identifier number for help window
        fig_help = plt.figure(num=99, figsize=(4, 3), dpi=120, tight_layout=False)
        if not exist: # if help window 99 exsits, plt.figure above pops it to front
            fig_help.canvas.manager.set_window_title('Instructions')
            fig_help.set_size_inches(6, 3)
            fig_help.set_dpi(120)
            axhelp = fig_help.add_subplot()
            axhelp.set_axis_off()
            axhelp.annotate(self.helptext, xy=(0.02, 0.98), xycoords='figure fraction', fontsize=10, verticalalignment='top', wrap=True)
            fig_help.canvas.draw()
        fig_help.show()        

class RangeSlider(AxesWidget):
    '''
    Colorbar/value-range slider, has two indicating lines to set the range.
    Modified based on matplotlib.widgets.Slider
    '''
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
                print("WARNING: cmap must be a Colormap instance! Use None instead.")
        
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
        '''activate dragging of indicating lines'''
        if event.button != 1 or event.inaxes != self.ax:
            return

        if self.orientation == 'vertical':
            d0 = abs(self.ax.transData.transform((0, self.vals[0]))[1] - event.y)
            d1 = abs(self.ax.transData.transform((0, self.vals[1]))[1] - event.y)
            if min(d0, d1) > self.pixeltol:
                return
            else:
                self.bound = np.argmin([d0, d1])  # min (0) or max (1), to be used in set_val
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
                self.bound = np.argmin([d0, d1]) # min (0) or max (1), to be used in set_val
                self.drag_active = True
                event.canvas.grab_mouse(self.ax)
                self.cid_release = self.canvas.mpl_connect('button_release_event', self._released)
                self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self._update)

    def _update(self, event):
        '''update vals'''
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
        '''deactivate dragging'''
        if event.button != 1:
            return
        self.drag_active = False
        self.bound = None # min (0) or max (1), to be used in set_val
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
            if bound == 1: # ymax changed
                xy[1] = 0, val
                xy[2] = 1, val
            else:     # ymin changed
                xy[0] = 0, val
                xy[3] = 1, val
        else: # self.orientation == 'horizontal'
            if bound == 1: # xmax changed
                xy[2] = val, 1
                xy[3] = val, 0
            else:     # xmin changed
                xy[0] = val, 0
                xy[1] = val, 1
        xy[-1] = xy[0] # ensure a closed rectangle
        self.poly.xy = xy
        ### update extent of the colorbar ###
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
        ### update labels ###
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
        ### update slider ###
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.vals = vals
        ### activate events connected to the slider ###
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(vals)

    def set_valmin_valmax(self, valmin, valmax, extra=0.01, update=False):
        '''
        Set new valmin, valmax of the slider, leaving extra space (normalized by valmax-valmin) for dragging the indicators.
        update: False - used for initial setting in __init__; True - for scroll events in the SpecLinkedCursors class
        This funciton was not originally in the Slider class.
        '''
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

import matplotlib as mpl
# PyQt5 package required for mac users 
# mpl.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from matplotlib.ticker import EngFormatter, AutoMinorLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import stmpy
from .gui_components import *
from .misc import grab_bg, blit_bg
import os.path as _ospath

def view3ds(data, topos=[], ch_names=[], didv=None, fit_didv=None, extent=[],
            cmap_topo=stmpy.cm.blue1, cmap_didv=stmpy.cm.bone, cbar_topo='salmon', cbar_didv='salmon',
             use_blit=False):
    '''
    Interactive visulization tools of any dI/dV map.
    Requires `%matplotlib qt` to open a new window. Run `%matplotlib inline` to switch back to inline jupyter notebook.
    Inputs:
          data - Spy object returned from stmpy.load().
          topos - list of channels (2d arrays of the same shape) to be shown in the topo window. If not given, line-subtracted topography (data.Z) will be shown.
          ch_names - list of channel names (str) to be shown in the topo window. If not given, ['Topo',] will be used.
          didv - A 3d array, each slice of which has a same shape as a topo channel, to be shown in the map window. If not given, data.LIY will be shown.
          fit_didv - An additional 3d array, having a same shape as didv, whose same location spectrum to be overlayed in the spectrum window.
          extent - list of four floats, same the extent argument in plt.imshow, defining the extent of topo and map windows. If not given, scan fields info will be used.
          cmap_topo, cmap_didv - Colormap instances for topo and map window, respectively.
          cbar_topo, cbar_didv - colors to be assigned to the color limit indicators in the colorbars
          use_blit - boolean, use blitting or not. Defaut is False, try switching it to True if interactive tools are sluggish.
    Example in jupyter notebook:
          ...
          %matplotlib qt # switch to qt backend
          from stmpy.mapviewer.view3ds import view3ds
          view3ds(data, topos=[data.Z,], ch_names=['Topo',], didv=data.liy, fit_didv=data.fit)
          %matplotlib inline # switch back to inline
          ...
    '''
    # ----- Default rc Parameters -----
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['xtick.minor.width'] = 0.3
    mpl.rcParams['ytick.minor.width'] = 0.3
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['ytick.major.size'] = 2
    mpl.rcParams['xtick.minor.size'] = 1
    mpl.rcParams['ytick.minor.size'] = 1

    # ----- Figure Settings -----
    ### Create axes ###
    fig_main = plt.figure(num=1, figsize=(8, 6), dpi=120, tight_layout=False)
    fig_main.canvas.manager.set_window_title('3DS Viewer')
    gs = GridSpec(3, 6, left=0.07, right=0.98, bottom=0.08, top=0.98, hspace=0.1, wspace=0.1, height_ratios=[2.5, 0.5, 3], width_ratios=[5, 0.2, 0.5, 5, 0.2, 0.5])
    axctrl = fig_main.add_subplot(gs[0, 0])
    axspec = fig_main.add_subplot(gs[0, 1:])
    axtopo = fig_main.add_subplot(gs[2, 0])
    axtcbar = fig_main.add_subplot(gs[2, 1])
    axmap = fig_main.add_subplot(gs[2, 3])
    axmcbar = fig_main.add_subplot(gs[2, 4])
    ### Set axis labels ###
    axspec.set_xlabel('Bias [V]')
    axspec.set_ylabel('dI/dV [A]')
    axtopo.set_xlabel('x [m]')
    axtopo.set_ylabel('y [m]')
    axmap.set_xlabel('x [m]')
    fig_main.align_labels()
    # axmap.set_ylabel('y [m]')
    ### Set ticks and tick labels ###
    axes = [axspec, axtopo, axmap]
    for ax in axes:
        ax.xaxis.set_major_formatter(EngFormatter(sep=''))
        ax.yaxis.set_major_formatter(EngFormatter(sep=''))
    axspec.xaxis.set_minor_locator(AutoMinorLocator(n=5))
    axspec.yaxis.set_minor_locator(AutoMinorLocator(n=5))
    axtopo.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axtopo.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axmap.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axmap.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # ----- Load Data -----
    if not topos:
        topo = stmpy.tools.lineSubtract(data.Z, n=1)
        topos = [topo,]
    if didv is None:
        didv = data.LIY
    # ----- Spectrum Plot -----
    axspec.plot(data.en, didv[:, 0, 0], 'b', lw=1,) # will be updated by cursor, HAS to be here to pass data.en to SpecLinkedCursors object
    axspec.grid(which='major', alpha=0.3, color='k', linewidth=0.5)
    axspec.grid(which='minor', alpha=0.3, color='k', linewidth=0.3)
    # ----- 2D Images -----
    if not extent:
        fov = [float(val) for val in data.header['Scan>Scanfield'].split(';')]
        extent = [0, fov[2], 0, fov[3]]
    sm2 = ScrollableMap(topos, axtopo, ch_names=ch_names, cmap=cmap_topo, origin='lower', aspect=1, extent=extent)
    rs2 = RangeSlider(axtcbar, '', topos[0].min(), topos[0].max(),
     orientation='vertical', cmap=cmap_topo, color=cbar_topo, lw=0.8, alpha=0.7)
    # ----- 3D Slice -----
    sm3 = ScrollableMap(didv, axmap, cmap=cmap_didv, origin='lower', aspect=1, extent=extent)
    rs3 = RangeSlider(axmcbar, '', didv.min(), didv.max(), valinit=(didv[0].min(), didv[0].max()),
     orientation='vertical', cmap=cmap_didv, color=cbar_didv, lw=0.8, alpha=0.7)
    lla = LimLinkedAxes(axtopo, axmap)
    slc = SpecLinkedCursors(axtopo, axmap, axspec, rs2, rs3, fit_didv=fit_didv, use_blit=use_blit)
    rs2.on_changed(slc.topo_clim_update)
    rs3.on_changed(slc.map_clim_update)
    # ----- Control Aera -----
    axctrl.axis('off')
    axctrl.set_navigate(False)
    ### Cursor button ###
    axcur = plt.axes([0.1, 0, 1, 1])
    ipcur = InsetPosition(axctrl, [0.1, 0.1, 0.3, 0.2])
    axcur.set_axes_locator(ipcur)
    bvis = Button(axcur, 'Cursor', color='gold', hovercolor='gold')
    bvis.label.set_fontsize(12)
    cursorswitch = LinesVisible(bvis, [slc.vline1, slc.hline1, slc.vline2, slc.hline2])
    bvis.on_clicked(cursorswitch.vis)
    ### Fit button ###
    axfit = plt.axes([0.2, 0, 1, 1])
    ipfit = InsetPosition(axctrl, [0.1, 0.4, 0.3, 0.2])
    axfit.set_axes_locator(ipfit)
    fvis = Button(axfit, 'Fit', color='tomato', hovercolor='tomato')
    fvis.label.set_fontsize(12)
    if fit_didv is None:
        fvis.color = '0.85'
        fvis.hovercolor = '0.85'
    else:
        fitswitch = LinesVisible(fvis, [slc.spec_fit,])
        fvis.on_clicked(fitswitch.vis)
    ### Help button ###
    axhelp = plt.axes([0.3, 0, 1, 1])
    iphelp = InsetPosition(axctrl, [0.1, 0.7, 0.3, 0.2])
    axhelp.set_axes_locator(iphelp)
    helpbtn = Button(axhelp, 'Help', color='0.9', hovercolor='0.9')
    helpbtn.label.set_fontsize(12)
    helpfile_path = _ospath.join(_ospath.dirname(__file__), 'helptext.txt')
    with open(helpfile_path) as helpfile:
        helptext = helpfile.read()
    helpwin = HelpWindow(helpbtn, helptext)
    helpbtn.on_clicked(helpwin.helpwindow)
    ### Buttons not in use ###
    # axpts = plt.axes([0.1, 0.1, 1, 1])
    # ippts = InsetPosition(axctrl, [0.5, 0.1, 0.3, 0.2])
    # axpts.set_axes_locator(ippts)
    # scatter = Button(axpts, 'Scatter', color='skyblue', hovercolor='skyblue')
    # scatter.label.set_fontsize(12)

    # axlcut = plt.axes([0.2, 0.1, 1, 1])
    # iplcut = InsetPosition(axctrl, [0.5, 0.4, 0.3, 0.2])
    # axlcut.set_axes_locator(iplcut)
    # linecut = Button(axlcut, 'Linecut', color='skyblue', hovercolor='skyblue')
    # linecut.label.set_fontsize(12)

    # axrcut = plt.axes([0.3, 0.1, 1, 1])
    # iprcut = InsetPosition(axctrl, [0.5, 0.7, 0.3, 0.2])
    # axrcut.set_axes_locator(iprcut)
    # radcut = Button(axrcut, 'Radialcut', color='skyblue', hovercolor='skyblue')
    # radcut.label.set_fontsize(12)

    plt.show(block=True)

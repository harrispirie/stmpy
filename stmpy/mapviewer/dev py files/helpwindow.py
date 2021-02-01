import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class HelpWindow:
    def __init__(self, button):
        self.button = button      

    def helpwindow(self, event):
        fig_help = plt.figure(num='Help', figsize=(3, 2), dpi=120)
        axhelp = fig_help.add_subplot()
        axhelp.set_axis_off()
        axhelp.text(0.1, 0.1, 'Help text line1\nHelp text line 2.', fontsize=8)
        fig_help.canvas.draw()
# helpwindow()
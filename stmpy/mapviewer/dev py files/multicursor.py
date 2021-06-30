from matplotlib.widgets import MultiCursor
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(nrows=2,)# sharex=True)
t = np.arange(0.0, 2.0, 0.01)
ax1.imshow(np.random.rand(20, 20))
ax2.imshow(np.random.rand(20, 20))

multi = MultiCursor(fig.canvas, (ax1, ax2), color='r', lw=1,
                    horizOn=True, vertOn=True)
plt.show()
import temporal_filters
from temporal_filters import *
from matplotlib.pyplot import *
import numpy as np

fps = 60
secs = 3
n = fps * secs
ts = np.linspace(0, secs, n)
noise = np.random.rand(n) * 1
fq1 = np.sin(ts * 2 * np.pi * 1)
fq2 = np.sin(ts * 2 * np.pi * 7)
fq3 = np.sin(ts * 2 * np.pi * 12)
data = fq1*1 + fq2*1 + fq3 + noise

win = IdealFilterWindowed(60, 4, 9, fps, outfun=lambda x: x[0])
#win = ButterBandpassFilter(1, 4, 9, fps)
#win = ButterFilter(5, 9, fps)
win.update(data)
out = win.collect()

if 1:
    # create plot
    clf()
    plot(data, 'k:')
    plot(fq2, 'k-')
    plot(out, 'r', linewidth=2)
    #out2 = scipy.signal.lfilter(win.b, win.a, data)
    #plot(out2, '--')

    show()
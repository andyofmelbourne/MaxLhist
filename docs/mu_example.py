import sys
sys.path.insert(0, '../../')

from MaxLhist import utils as ut
from MaxLhist import forward_model as fm

import matplotlib.pyplot as plt
import scipy
import numpy as np

#hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 10, sigma_f = 5., sigma_mu = 20.)
M = 2
I = 250
sigma_f = 5.

# the "adu" range
i      = np.arange(0, I, 1)
i_bins = np.arange(0, I+1, 1)

# the probability function or "background"
f = np.exp( - (i - 100).astype(np.float64)**2 / (2. * sigma_f**2)) 
f = f / np.sum(f)
F = scipy.stats.rv_discrete(name='background', values = (i, f))

# make some histograms
hists = []
mus = np.array([-20., 20.])
for n in range(M):
    mu = mus[n]
    
    # create a new random variable with the shifted background
    f_shift = ut.roll_real(f, mu)
    F_shift = scipy.stats.rv_discrete(name='background', values = (i, f_shift))
    ff = F_shift.rvs(size = 1000)
    hist, bins = np.histogram(ff, bins = i_bins)
    hists.append(hist)
    
hists = np.array(hists)

# plot the histograms
ax = plt.subplot(111)
ax.bar(i, hists[0], alpha = 0.5, color='r', width=1.0, label='pixel 0')
ax.bar(i, hists[1], alpha = 0.5, color='b', width=1.0, label='pixel 1')
ax.set_xlabel('adus')
ax.set_ylabel('frequency')
ax.legend()
fig = plt.gcf()
fig.set_size_inches(100,5)
fig.show()

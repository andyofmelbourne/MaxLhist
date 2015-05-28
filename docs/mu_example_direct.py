import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

from MaxLhist import utils as ut
from MaxLhist import forward_model as fm

#from cgls.cgls.line_search import line_search_secant

import matplotlib.pyplot as plt
import scipy
import numpy as np


# forward model 
#--------------
hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 2, sigma_f = 5., sigma_mu = 50., size = 200)

# truncate to non-zero measurements
#----------------------------------
i_range = np.arange(I)

# inital guess
#-------------
f_god = F.pmf(i_range)
f     = f_god

"""
f_grad = ut.grad_shift_f_real(f, 0)

F = f_grad / (1.0e-10 + f)

# calculate the cross-correlation of hists and F
cor   = np.fft.rfftn(hists.astype(np.float64), axes=(-1, ))
check = cor * np.conj(np.fft.rfft(f))
cor   = cor * np.conj(np.fft.rfft(F))

# interpolation factor
padd_factor = 1
if padd_factor != 0 and padd_factor != 1 :
    cor   = np.concatenate((cor, np.zeros( (cor.shape[0], cor.shape[1] -1), dtype=cor.dtype)), axis=-1)
    check = np.concatenate((check, np.zeros( (check.shape[0], check.shape[1] -1), dtype=check.dtype)), axis=-1)

cor   = np.fft.irfftn(cor, axes=(-1, ))
check = np.fft.irfftn(check, axes=(-1, ))

mus = []
for m in range(cor.shape[0]):
    i_s = np.where(check[m] > 1)
    # search within the subset 
    mu = np.argmin(np.abs(cor[m, i_s[0]])) 
    # map to absolute pixel coord
    mu = i_s[0][mu]
    # map to shift coord
    mu = cor.shape[1] * np.fft.fftfreq(cor.shape[1])[mu]
    mus.append(mu / float(padd_factor))

"""

# calculate the cross-correlation of hists and F
cor   = np.fft.rfftn(hists.astype(np.float64), axes=(-1, ))
cor   = cor * np.conj(np.fft.rfft(np.log(f)))

# interpolation factor
padd_factor = 1
if padd_factor != 0 and padd_factor != 1 :
    cor   = np.concatenate((cor, np.zeros( (cor.shape[0], cor.shape[1] -1), dtype=cor.dtype)), axis=-1)
    check = np.concatenate((check, np.zeros( (check.shape[0], check.shape[1] -1), dtype=check.dtype)), axis=-1)

cor  = np.fft.irfftn(cor, axes=(-1, ))

mus = []
for m in range(cor.shape[0]):
    mu = np.argmax(cor[m])
    # map to shift coord
    mu = cor.shape[1] * np.fft.fftfreq(cor.shape[1])[mu]
    mus.append(mu / float(padd_factor))

if False :
    # plot the histograms
    ax = plt.subplot(111)
    ax.bar(i, hists[0], alpha = 0.5, color='r', width=1.0, label='pixel 0')
    ax.bar(i, hists[1], alpha = 0.5, color='b', width=1.0, label='pixel 1')
    ax.set_xlabel('adus')
    ax.set_ylabel('frequency')
    ax.legend()
    plt.show()


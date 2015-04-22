"""
Given an estimate for f can we retrieve the mu values at each pixel?
Is this any better than taking the maximum value?
"""

import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut

# jacobian
def jacobian_mus_manual_calc(f, mus, hists, error_func, step = 0.1):
    """
    Calculate d error / d mus using the central difference algorithm:
    d error / d mus_i = [error(mus_i + 1) - error(mus_i - 1)] / 2
    REF:
        http://en.wikipedia.org/wiki/Numerical_differentiation
        and
        http://en.wikipedia.org/wiki/Finite_difference_coefficient
    """
    J_mus = np.zeros_like(mus, dtype=np.float64)
    for i in range(len(mus)):
        mus_temp     = mus
        mus_temp[i] += step
        e1 = error_func(f, mus_temp, hists)
        
        mus_temp     = mus
        mus_temp[i] -= step
        e2 = error_func(f, mus_temp, hists)
        
        d = (e1 - e2) / (2. * step)
        
        J_mus[i] = d
    return J_mus

def jacobian_mus_calc_old(f, mus, hists, prob_tol = 1.0e-5, continuity = 0):
    out = np.zeros_like(mus, dtype=np.float64)
    
    # calculate the derivative of f
    fprime = np.gradient(f)
    if continuity != 0 :
        fprime = np.convolve(fprime, np.ones((continuity), dtype=np.float)/ float(continuity), mode='same')
    
    for n in range(len(hists)) :
        Is      = np.where(hists[n] > 0)[0]
        fs      = f[Is - int(np.rint(mus[n]))]
        fprimes = fprime[Is - int(np.rint(mus[n]))]
        
        out[n] = np.sum(hists[n, Is] * fprimes / (fs + prob_tol))
    return out

def jacobian_mus_calc(f, mus, hists, prob_tol = 0.0):
    # this could be more efficient
    f_prime_shift = np.zeros_like(hists, dtype=np.float64)
    for m in range(len(mus)) :
        f_prime_shift = ut.grad_shift_f_real(f, mus[m]) / (prob_tol + ut.roll_real(f, mus[m]))
    
    temp  = hists * f_prime_shift 
    temp  = np.sum(temp, axis=1)
    mus_f = ut.mu_transform(temp)
    mus_d = ut.make_f_real(mus_f, f_norm = 0.0)
    return mus_d

# update
def update_mus(f, mus0, hists, grad_calc, iters = 1):
    """
    Follow the line of steepest descent.
        mus_i = mus_i-1 - 0.5 * grad_calc(mus_i-1)
    """
    mus = mus0.copy()
    for i in range(iters):
        # print the error
        print i, 'log likelihood error', ut.log_likelihood_calc(f, mus, hists)

        # calculate the gradient
        grad = grad_calc(f, mus, hists)

        # take a step proportional to the gradient
        mus = mus - 0.5 * grad
    return mus


# forward model 
hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 10, sigma_f = 5., sigma_mu = 20.)

# inital guess
f      = F.pmf(np.arange(I))

mus0   = np.zeros_like(mus_god) 
for m in range(hists.shape[0]):
    mus0[m]   = np.argmax(hists[m]) - np.argmax(f)

# update the guess
#mus = update_mus(f, mus0, hists, jacobian_mus_calc, iters=10)
mus = update_mus(f, mus0, hists, lambda x,y,z: jacobian_mus_manual_calc(x,y,z,ut.log_likelihood_calc), iters=10)

# derivates
J_mus_manual = jacobian_mus_manual_calc(f, mus, hists, ut.log_likelihood_calc)
J_mus_calc   = jacobian_mus_calc(f, mus, hists)

hists0 = fm.forward_hists(f, mus.astype(np.int), np.sum(hists[0]))


# display
if True :
    import pyqtgraph as pg
    import PyQt4.QtGui
    import PyQt4.QtCore
    # Always start by initializing Qt (only once per application)
    app = PyQt4.QtGui.QApplication([])
    # Define a top-level widget to hold everything
    win = pg.GraphicsWindow(title="forward model")
    pg.setConfigOptions(antialias=True)
    
    # show f and the mu values
    p1 = win.addPlot(title="probablity function", x = np.arange(I), y = F.pmf(np.arange(I)), name = 'f')
    
    p2 = win.addPlot(title="shifts", y = mus_god, name = 'shifts')
    p2.plot(mus0, pen=(255, 0, 0), name = 'mus0')
    p2.plot(mus, pen=(0, 255, 0), name = 'mus')
    
    win.nextRow()
    
    p2 = win.addPlot(title="shifts jacobian", y = J_mus_manual, name = 'shifts jacobian')
    p2.plot(J_mus_calc, pen=(255, 0, 0))
    
    win.nextRow()

    # now plot the histograms
    hplots = []
    for i in range(M / 2):
        for j in range(2):
            m = 2 * i + j
            hplots.append(win.addPlot(title="histogram pixel " + str(m), y = hists[m], name = 'hist' + str(m)))
            hplots[-1].plot(hists0[m], pen = (255, 0, 0))
            hplots[-1].setXLink('f')
        win.nextRow()

"""
Given an estimate for f can we retrieve the mu values at each pixel?
Is this any better than taking the maximum value?
"""

import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut
from scipy.ndimage import gaussian_filter1d

# update
def update_fs(f0, mus, hists):
    """
    Follow the line of steepest descent.
        mus_i = mus_i-1 - 0.5 * grad_calc(mus_i-1)
    """
    f = f0.copy()

    print 0, 'log likelihood error', ut.log_likelihood_calc(f, mus, hists)
    
    h = np.zeros( f.shape, dtype=f.dtype)
    for m in range(len(mus)) :
        h += ut.roll_real(hists[m].astype(np.float64), - mus[m])
    # generate a mask for the gradient vector
    mask = (h >= 1.0)
    f = mask * h
    f = f / np.sum(f)

    error = ut.log_likelihood_calc(f, mus, hists)
    print 1, 'log likelihood error', error
    return f, error

def update_mus(f, mus0, hists, grad_calc, iters = 1):
    """
    Follow the line of steepest descent.
        mus_i = mus_i-1 - 0.5 * grad_calc(mus_i-1)
    """
    mus = mus0.copy()
    
    # go to Fourier space
    muhat = np.fft.rfft(mus)[1 :]
    
    for i in range(iters):
        # print the error
        print i, 'log likelihood error', ut.log_likelihood_calc(f, mus, hists)

        # calculate the descent direction
        grad = -grad_calc(f, muhat, hists)
        grad = grad / np.sqrt( np.sum( np.abs(grad)**2 ) ) 
        
        # perform the line minimisation in this direction
        eprime_alpha = lambda alpha : ut.mu_directional_derivative(grad, f, muhat + grad * alpha, hists, prob_tol = 1.0e-10)
        alpha        = ut.mu_bisection(eprime_alpha)

        muhat = muhat + grad * alpha
        mus   = ut.make_f_real(muhat)

    error = ut.log_likelihood_calc(f, mus, hists)
    print i+1, 'log likelihood error', error
    return mus, error

if __name__ == '__main__':
    # forward model 
    #--------------
    hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 100, sigma_f = 5., sigma_mu = 20., size = 200)
    #
    # f smoothness constraint sigma (pixels)
    f_sig = 0.

    # truncate to non-zero measurements
    #----------------------------------
    i_range = np.arange(I)
    
    # inital guess
    #-------------
    f_god  = F.pmf(i_range)
    f      = np.sum(hists.astype(np.float64), axis=0) 
    f      = f / np.sum(f)
    f0     = f.copy()

    mus0   = np.zeros_like(mus_god) 
    # uncomment for a better starting guess
    for m in range(hists.shape[0]):
        mus0[m]   = np.argmax(hists[m]) - np.argmax(f)
    mus = mus0.copy()

    # update the guess
    #-------------
    errors = []
    for i in range(10):
        mus_t, e = update_mus(f, mus, hists, ut.jacobian_mus_calc, iters=2)
        f_t, e   = update_fs(f, mus, hists)
        #
        mus = mus_t
        f   = f_t
        errors.append(e)
        # smoothness constraint
        #f = gaussian_filter1d(f, f_sig)

    hists0 = fm.forward_hists(f0, mus0, np.sum(hists[0]))
    hists1 = fm.forward_hists(f, mus, np.sum(hists[0]))

    # display
    if False :
        import pyqtgraph as pg
        import PyQt4.QtGui
        import PyQt4.QtCore
        # Always start by initializing Qt (only once per application)
        app = PyQt4.QtGui.QApplication([])
        # Define a top-level widget to hold everything
        win = pg.GraphicsWindow(title="forward model")
        pg.setConfigOptions(antialias=True)
        
        # show f and the mu values
        p1 = win.addPlot(title="probablity function", x = i_range, y = f_god, name = 'f_god')
        p1.plot(x = i_range, y = f0, pen=(255, 0, 0), name = 'f0')
        p1.plot(x = i_range, y = f, pen=(0, 255, 0), name = 'f')
        
        p2 = win.addPlot(title="shifts", y = mus_god, name = 'shifts')
        p2.plot(mus0, pen=(255, 0, 0), name = 'mus0')
        p2.plot(mus, pen=(0, 255, 0), name = 'mus')
        
        win.nextRow()
        
        #p2 = win.addPlot(title="shifts jacobian", y = J_mus_manual, name = 'shifts jacobian')
        #p2.plot(J_mus_calc, pen=(255, 0, 0))
        
        #win.nextRow()

        # now plot the histograms
        hplots = []
        for i in range(4 / 2):
            for j in range(2):
                m = 2 * i + j
                hplots.append(win.addPlot(title="histogram pixel " + str(m), y = hists[m], name = 'hist' + str(m), fillLevel = 0.0, fillBrush = 0.7, stepMode = True))
                hplots[-1].plot(hists0[m], pen = (255, 0, 0))
                hplots[-1].plot(hists1[m], pen = (0, 255, 0))
                hplots[-1].setXLink('f')
            win.nextRow()

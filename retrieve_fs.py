"""
Given an estimate for f can we retrieve the mu values at each pixel?
Is this any better than taking the maximum value?
"""

import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut

# update
def update_fs(f0, mus, hists, grad_calc, iters = 1):
    """
    Follow the line of steepest descent.
        mus_i = mus_i-1 - 0.5 * grad_calc(mus_i-1)
    """
    f = f0.copy()

    h = np.zeros( f.shape, dtype=f.dtype)
    for m in range(len(mus)) :
        h += ut.roll_real(hists[m].astype(np.float64), - mus[m])
    # generate a mask for the gradient vector
    mask = (h >= 1.0)
    f = mask * f
    f = f / np.sum(f)
    
    for i in range(iters):
        # print the error
        print i, 'log likelihood error', ut.log_likelihood_calc(f, mus, hists)

        # calculate the descent direction
        grad = -grad_calc(f, mus, hists)
        
        # make into a unit vector
        grad = grad / np.sqrt( np.sum( np.abs(grad)**2 ) ) 

        # perform the line minimisation in this direction
        eprime_alpha = lambda alpha : ut.f_directional_derivative(grad, f + grad * alpha, mus, hists, prob_tol = 1.0e-10)
        alpha        = ut.mu_bisection(eprime_alpha)

        f = f + grad * alpha
        print f, '\n\n\n'

    print i+1, 'log likelihood error', ut.log_likelihood_calc(f, mus, hists), np.array(mus, dtype=np.int)
    return f

if __name__ == '__main__':
    # forward model 
    #--------------
    hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 10, sigma_f = 5., sigma_mu = 20.)

    # truncate to non-zero measurements
    #----------------------------------
    i_range = np.arange(I)
    
    # inital guess
    #-------------
    f_god  = F.pmf(i_range)
    f      = np.sum(hists.astype(np.float64), axis=0) 
    f      = f / np.sum(f)
    f0     = f.copy()

    mus = mus_god

    # update the guess
    #-------------
    #f = update_fs(f, mus, hists, ut.jacobian_fs_calc, iters=5)
    h = np.zeros( f.shape, dtype=f.dtype)
    for m in range(len(mus)) :
        h += ut.roll_real(hists[m].astype(np.float64), - mus[m])
    # generate a mask for the gradient vector
    mask = (h >= 1.0)
    f = mask * h
    f = f / np.sum(f)

    hists0 = fm.forward_hists(f0, mus, np.sum(hists[0]))
    hists1 = fm.forward_hists(f, mus, np.sum(hists[0]))

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
        p1 = win.addPlot(title="probablity function", x = i_range, y = f_god, name = 'f_god')
        p1.plot(x = i_range, y = f0, pen=(255, 0, 0), name = 'f0')
        p1.plot(x = i_range, y = f, pen=(0, 255, 0), name = 'f')
        
        p2 = win.addPlot(title="shifts", y = mus_god, name = 'shifts')
        
        win.nextRow()
        
        #p2 = win.addPlot(title="shifts jacobian", y = J_mus_manual, name = 'shifts jacobian')
        #p2.plot(J_mus_calc, pen=(255, 0, 0))
        
        #win.nextRow()

        # now plot the histograms
        hplots = []
        for i in range(M / 2):
            for j in range(2):
                m = 2 * i + j
                hplots.append(win.addPlot(title="histogram pixel " + str(m), y = hists[m], name = 'hist' + str(m), fillLevel = 0.0, fillBrush = 0.7, stepMode = True))
                hplots[-1].plot(hists0[m], pen = (255, 0, 0))
                hplots[-1].plot(hists1[m], pen = (0, 255, 0))
                hplots[-1].setXLink('f')
            win.nextRow()

"""
Given an estimate for f can we retrieve the mu values at each pixel?
Is this any better than taking the maximum value?
"""

import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut

# update
def update_fs(f, mus0, hists, grad_calc, iters = 1):
    """
    Follow the line of steepest descent.
        mus_i = mus_i-1 - 0.5 * grad_calc(mus_i-1)
    """
    mus = mus0.copy()
    
    # go to Fourier space
    muhat = np.fft.rfft(mus)[1 :]
    
    for i in range(iters):
        # print the error
        print i, 'log likelihood error', ut.log_likelihood_calc(f, mus, hists), np.array(mus, dtype=np.int)

        # calculate the descent direction
        grad = -grad_calc(f, muhat, hists)
        grad = grad / np.sqrt( np.sum( np.abs(grad)**2 ) ) 
        
        # perform the line minimisation in this direction
        eprime_alpha = lambda alpha : ut.mu_directional_derivative(grad, f, muhat + grad * alpha, hists, prob_tol = 1.0e-10)
        alpha        = ut.mu_bisection(eprime_alpha)

        muhat = muhat + grad * alpha
        mus   = ut.make_f_real(muhat)

    print i+1, 'log likelihood error', ut.log_likelihood_calc(f, mus, hists), np.array(mus, dtype=np.int)
    return mus

if __name__ == '__main__':
    # forward model 
    #--------------
    hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 10, sigma_f = 5., sigma_mu = 20.)

    # inital guess
    #-------------
    f_god  = F.pmf(np.arange(I))
    f      = hists[0]
    f0     = f.copy()

    mus = mus_god

    # update the guess
    #-------------
    f = update_fs(f, mus, hists, ut.jacobian_fs_calc, iters=5)

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
        p1 = win.addPlot(title="probablity function", x = np.arange(I), y = F.pmf(np.arange(I)), name = 'f')
        
        p2 = win.addPlot(title="shifts", y = mus_god, name = 'shifts')
        p2.plot(mus0, pen=(255, 0, 0), name = 'mus0')
        p2.plot(mus, pen=(0, 255, 0), name = 'mus')
        
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

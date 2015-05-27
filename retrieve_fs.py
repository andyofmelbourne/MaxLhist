"""
Given an estimate for f can we retrieve the mu values at each pixel?
Is this any better than taking the maximum value?
"""

import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut

# update
def update_fs(f0, mus, hists):
    """
    Follow the line of steepest descent.
        mus_i = mus_i-1 - 0.5 * grad_calc(mus_i-1)
    """
    f = f0.copy()

    print 0, 'log likelihood error', ut.log_likelihood_calc(f, mus, hists, prob_tol = 1.0e-5)
    
    h = np.zeros( f.shape, dtype=f.dtype)
    for m in range(len(mus)) :
        h += ut.roll_real(hists[m].astype(np.float64), - mus[m])
    # generate a mask for the gradient vector
    mask = (h >= 1.0)
    f = mask * h
    f = f / np.sum(f)

    print 1, 'log likelihood error', ut.log_likelihood_calc(f, mus, hists, prob_tol = 1.0e-5)
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
    f = update_fs(f, mus, hists)

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

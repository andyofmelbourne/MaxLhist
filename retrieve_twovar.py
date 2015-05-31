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

if __name__ == '__main__':
    # forward model 
    #--------------
    hists, M, I, mus_god, nms_god, D, S = fm.forward_model_twovars(I = 250, M = 10, sigma_d = 5., sigma_s = 5., ds = 20., sigma_nm = 0.4, sigma_mu = 10., size = 1000)

    # truncate to non-zero measurements
    #----------------------------------
    i_range = np.arange(I)
    
    # inital guess
    #-------------
    d_god  = D.pmf(i_range)
    #d      = np.sum(hists.astype(np.float64), axis=0) 
    #d      = d / np.sum(d)
    d      = d_god.copy()
    d0     = d.copy()

    s_god  = S.pmf(i_range)
    #s      = np.sum(hists.astype(np.float64), axis=0) 
    #s      = s / np.sum(s)
    s      = s_god.copy()
    s0     = s.copy()

    nms0   = nms_god.copy()
    nms    = nms0.copy()
    
    mus0   = np.zeros_like(mus_god) 
    # uncomment for a better starting guess
    for m in range(hists.shape[0]):
        mus0[m]   = np.argmax(hists[m]) - np.argmax(d)
    mus = mus0.copy()

    # update the guess
    #-------------
    errors = [0]

    hists0 = fm.forward_hists_twovar(d0, s0, nms0, mus0, np.sum(hists[0]))
    hists1 = fm.forward_hists_twovar(d, s, nms, mus, np.sum(hists[0]))

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
        p1 = win.addPlot(title="probablity function", x = i_range, y = d_god, name = 'd_god')
        p1.plot(x = i_range, y = s_god, name = 's_god')
        p1.plot(x = i_range, y = s, pen=(255, 0, 0), name = 's')
        p1.plot(x = i_range, y = d, pen=(0, 255, 0), name = 'd')
        
        p2 = win.addPlot(title="shifts", y = mus_god, name = 'shifts')
        p2.plot(mus0, pen=(255, 0, 0), name = 'mus0')
        p2.plot(mus, pen=(0, 255, 0), name = 'mus')
        
        win.nextRow()
        
        #p2 = win.addPlot(title="shifts jacobian", y = J_mus_manual, name = 'shifts jacobian')
        #p2.plot(J_mus_calc, pen=(255, 0, 0))
        
        #win.nextRow()

        # now plot the histograms
        hplots = []
        for i in range(10 / 2):
            for j in range(2):
                m = 2 * i + j
                hplots.append(win.addPlot(title="histogram pixel " + str(m), y = hists[m], name = 'hist' + str(m), fillLevel = 0.0, fillBrush = 0.7, stepMode = True))
                hplots[-1].plot(hists0[m], pen = (255, 0, 0))
                hplots[-1].plot(hists1[m], pen = (0, 255, 0))
                hplots[-1].setXLink('f')
            win.nextRow()
        
        win.nextRow()
        
        p3 = win.addPlot(title="log likelihood error", y = errors)
        p3.showGrid(x=True, y=True) 

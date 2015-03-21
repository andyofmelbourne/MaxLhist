import numpy as np
import scipy.stats

def forward_model(I = 250, M = 10, sigma_f = 5., sigma_mu = 20.):
    """
    The forward model should produce:
        h^m_i   the histograms or "data"
        M       the number of histograms or "pixels"
        I       the number of bins for the histogram

    In addition we may want the God mode variables:
        mu_m    the true offsets for the histograms
        f_i     the true probability function for the histograms

    And a ploting funciton to look at the data
    """
    # the "adu" range
    i = np.arange(0, I, 1)
    
    # the probability function or "background"
    f = np.exp( - (i - 100).astype(np.float64)**2 / (2. * sigma_f**2)) 
    f = f / np.sum(f)
    F = scipy.stats.rv_discrete(name='background', values = (i, f))
    
    # the probability function for the offsets or "dark values"
    shifts = np.arange(-100, 100, 1)
    mu = np.exp( - shifts.astype(np.float64)**2 / (2. * sigma_mu**2)) 
    mu = mu / np.sum(mu)
    MU = scipy.stats.rv_discrete(name='mu', values = (shifts, mu))
    
    # make some histograms
    hists = []
    mus = []
    for n in range(M):
        mu = MU.rvs(1)
        f  = F.rvs(size = 50)
        hist, bins = np.histogram(f, bins = i)
        hists.append(np.roll(hist, mu))
        mus.append(mu)
    hists = np.array(hists)
    mus   = np.array(mus)
    
    return hists, M, I, mus, F

def forward_hists(f, mus, N):
    hists = np.zeros(mus.shape + f.shape, dtype=f.dtype)
    for i in range(hists.shape[0]):
        hists[i] = np.roll(f, mus[i]) * N
    return hists

if __name__ == '__main__':
    hists, M, I, mus, F = forward_model(I = 250, M = 10, sigma_f = 5., sigma_mu = 20.)
    
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

    p2 = win.addPlot(title="shifts", y = mus, name = 'shifts')

    win.nextRow()

    # now plot the histograms
    hplots = []
    for i in range(M / 2):
        for j in range(2):
            m = 2 * i + j
            hplots.append(win.addPlot(title="histogram pixel " + str(m), y = hists[m], name = 'hist' + str(m)))
            hplots[-1].setXLink('f')
        win.nextRow()
    """
    ## Start Qt event loop unless running in interactive mode or using pyside.
    if __name__ == '__main__':
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'): PyQt4.QtGui.QApplication.instance().exec_()
    """
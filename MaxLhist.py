import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut
from scipy.ndimage import gaussian_filter1d
import sys

def refine(datas, iterations=1):
    # Is there only one dataset?
    if len(datas) == 0 :
        print 'no data: nothing to do...'
        sys.exit()
    elif len(datas) != 1 :
        print 'I only support one dataset for now...'
        sys.exit()

    data = datas[0]
    
    # Is there any histogram data?
    if not data.has_key('histograms'):
        print 'Error: no histogram data.'
        sys.exit()
    
    hists = data['histograms']

    # Is there only one random variable?
    if not data.has_key('vars'):
        print 'Error: no histogram data.'
        sys.exit()
    elif len(data['vars']) != 1 :
        print 'Error: I only support one random variable for now.'
        sys.exit()

    var = data['vars'][0]

    # The random variable must have a function element
    if not var.has_key('elements'):
        print 'Error: every random variable must have at least a function element', var['name']
        sys.exit()
    
    for e in var['elements']:
        if e['type'] == 'function':
            func = e
        elif e['type'] == 'offset':
            offsets = e
        else :
            print 'Error: type of', e['name'] ,'not understood:', e['type']
            sys.exit()

    # inital guess
    #-------------
    f = func['initial guess']
    if f is not None :
        if f.shape[0] != hists.shape[1]:
            print 'Error:', f['name'],'s initial guess does not have the right shape:', f.shape[0], ' hists.shape[1]=', hists.shape[1]
            sys.exit()
        else :
            f = func['initial guess']
    else :
        print 'initialising', func['name'], 'with the sum of the histogram data'
        f = np.sum(hists.astype(np.float64), axis=0) 
        f = f / np.sum(f)

    mus = offsets['initial guess']
    if mus is not None :
        if mus.shape[0] != hists.shape[0]:
            print 'Error:', offsets['name'],'s initial guess does not have the right shape:', offsets.shape[0], ' hists.shape[0]=', hists.shape[0]
            sys.exit()
        else :
            mus = offsets['initial guess']
    else :
        print 'initialising', offsets['name'], 'with the argmax of the histograms.'
        mus = np.zeros((hists.shape[0]), dtype=np.float64)
        for m in range(hists.shape[0]):
            mus[m]   = np.argmax(hists[m]) - np.argmax(f)
        mus = mus - np.sum(mus) / float(len(mus))

    mus0 = mus.copy()
    f0   = f.copy()
    
    # update the guess
    #-------------
    errors = []
    mus_t  = mus.copy()
    f_t    = f.copy()
    for i in range(iterations):
        if offsets['update'] :
            mus_t = ut.update_mus(f, mus, hists)
        
        if func['update'] :
            f_t   = ut.update_fs(f, mus, hists)
         
        mus = mus_t
        f   = f_t
        e   = ut.log_likelihood_calc(f, mus, hists)
        errors.append(e)
        print i, 'log likelihood error:', e
    
    errors = np.array(errors)
    
    # return the results
    func['values']           = f 
    func['initial guess']    = f0 
    offsets['values']        = mus
    offsets['initial guess'] = mus0 

    var['elements'][0] = offsets
    var['elements'][1] = func
    
    result = Result(var, data, errors)
    return result

class Result():
    def __init__(self, var, data, errors):
        self.result = {}
        self.result[var['name']]     = var
        self.result['error vs iter'] = errors

        pixel_map_errors = ut.log_likelihood_calc(var['elements'][1]['values'], \
                var['elements'][0]['values'], data['histograms'], pixelwise=True)

        self.result['error vs pixel'] = pixel_map_errors

    def dump_to_h5(self, fnam):
        import h5py
        f = h5py.File(fnam, 'w')
        
        def recurse(elem, tree):
            if type(elem) is dict :
                for k in elem.keys():
                    v = elem[k]
                    try :
                        tree.create_dataset(k, data=v)
                        print 'writing: ', k, 'at ', tree
                    except :
                        print 'creating group ', k, 'under', tree
                        tree2 = tree.create_group(k)
                        recurse(v, tree2)
            elif type(elem) is list :
                for v in elem:
                    if v.has_key('name'):
                        print 'creating group ', v['name'], 'under', tree
                        tree2 = tree.create_group(v['name'])
                    recurse(v, tree2)
        recurse(self.result, f)

        f.close()
    
    def show_fit(self, varname, hists):
        errors = self.result['error vs iter']
        
        for e in self.result[varname]['elements'] :
            if e['type'] == 'function':
                f_name = e['name']
                f0 = e['initial guess']
                f  = e['values']
            
            if e['type'] == 'offset':
                mus_name = e['name']
                mus0 = e['initial guess']
                mus  = e['values']

        hists0 = fm.forward_hists(f0, mus0, np.sum(hists[0]))
        hists1 = fm.forward_hists(f, mus, np.sum(hists[0]))
        
        i_range = np.arange(hists.shape[1])
        
        import pyqtgraph as pg
        import PyQt4.QtGui
        import PyQt4.QtCore
        # Always start by initializing Qt (only once per application)
        app = PyQt4.QtGui.QApplication([])
        # Define a top-level widget to hold everything
        win = pg.GraphicsWindow(title="results")
        pg.setConfigOptions(antialias=True)
        
        # show f and the mu values
        p1 = win.addPlot(title=f_name)
        p1.plot(x = i_range, y = f0, pen=(255, 0, 0), name = 'f0')
        p1.plot(x = i_range, y = f, pen=(0, 255, 0), name = 'f')
        
        p2 = win.addPlot(title=mus_name)
        p2.plot(mus0, pen=(255, 0, 0), name = 'mus0')
        p2.plot(mus, pen=(0, 255, 0), name = 'mus')
        
        win.nextRow()
        
        # now plot the histograms
        hplots = []
        for i in range(8 / 2):
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
        sys.exit(app.exec_())
            

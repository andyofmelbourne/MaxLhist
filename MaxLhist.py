import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut
from scipy.ndimage import gaussian_filter1d
import sys

def process_input(datas):
    """ """
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
    if not var.has_key('function'):
        print 'Error: every random variable must have at least a function element', var['name']
        sys.exit()
    elif not var['function'].has_key('update') :
        print 'Error: the function of', var['name'], 'must have an update variable.'
        sys.exit()

    # The random variable must have a shift element
    if not var.has_key('offset'):
        print 'Error: every random variable must have at least an offset element', var['name']
        sys.exit()
    elif not var['offset'].has_key('update') :
        print 'Error: the function of', var['name'], 'must have an update variable.'
        sys.exit()
    
    # inital guess
    #-------------
    if var['function'].has_key('init') :
        f = var['function']['init']
    else :
        f = None
    
    if f is not None :
        if f.shape[0] != hists.shape[1]:
            print 'Error:', var['name'],'s initial guess for the function does not have the right shape:', f.shape[0], ' hists.shape[1]=', hists.shape[1]
            sys.exit()
    else :
        print 'initialising', var['name'], 's function with the sum of the histogram data'
        f = np.sum(hists.astype(np.float64), axis=0) 
        f = f / np.sum(f)

    if var['offset'].has_key('init') :
        mus = var['offset']['init']
    else :
        mus = None
    
    if mus is not None :
        if mus.shape[0] != hists.shape[0]:
            print 'Error:', var['name'],'s initial guess for the offsets does not have the right shape:', mus.shape[0], ' hists.shape[0]=', hists.shape[0]
            sys.exit()
    else :
        print 'initialising', var['name'], ' offset with the argmax of the histograms.'
        mus = np.zeros((hists.shape[0]), dtype=np.float64)
        for m in range(hists.shape[0]):
            mus[m]   = np.argmax(hists[m]) - np.argmax(f)
        mus = mus - np.sum(mus) / float(len(mus))

    return mus, f, hists, data, var 


def refine(datas, iterations=1):
    mus, f, hists, data, var = process_input(datas)

    mus0 = mus.copy()
    f0   = f.copy()
    
    # update the guess
    #-------------
    errors = []
    mus_t  = mus.copy()
    f_t    = f.copy()
    for i in range(iterations):
        if var['offset']['update'] :
            mus_t = ut.update_mus(f, mus, hists)
        
        if var['function']['update'] :
            f_t   = ut.update_fs(f, mus, hists)
         
        mus = mus_t
        f   = f_t
        e   = ut.log_likelihood_calc(f, mus, hists)
        errors.append(e)
        print i, 'log likelihood error:', e
    
    errors = np.array(errors)
    
    # return the results
    var['function']['values'] = f 
    var['function']['init']   = f0 
    var['offset']['values']   = mus
    var['offset']['init']     = mus0 

    result = Result(var, data, errors)
    return result


class Result():
    def __init__(self, var, data, errors):
        self.result = {}
        self.result[data['name']] = {}
        self.result[data['name']]['vars']    = var['name']
        self.result[data['name']]['comment'] = data['comment']

        pixel_map_errors = ut.log_likelihood_calc_pixelwise(var['function']['values'], \
                var['offset']['values'], data['histograms'])

        self.result[data['name']]['pix_errors'] = pixel_map_errors

        self.result[var['name']]         = var

        self.result['error vs iter'] = errors

    def dump_to_h5(self, fnam):
        import h5py
        f = h5py.File(fnam, 'w')
        
        def recurse(elem, tree):
            if type(elem) is dict :
                for k in elem.keys():
                    v = elem[k]
                    try :
                        tree.create_dataset(k, data=v)
                        #print 'writing: ', k, 'at ', tree
                    except :
                        #print 'creating group ', k, 'under', tree
                        tree2 = tree.create_group(k)
                        recurse(v, tree2)
            elif type(elem) is list :
                for v in elem:
                    if v.has_key('name'):
                        #print 'creating group ', v['name'], 'under', tree
                        tree2 = tree.create_group(v['name'])
                    recurse(v, tree2)
        recurse(self.result, f)

        f.close()
    
    def show_fit(self, data):
        errors   = self.result['error vs iter']
        p_errors = self.result[data['name']]['pix_errors']
        
        var = self.result[data['vars'][0]['name']]
        f_name = var['name'] + ' function'
        f0 = var['function']['init']
        f  = var['function']['values']
            
        mus_name = var['name'] + ' offset'
        mus0 = var['offset']['init']
        mus  = var['offset']['values']

        N = np.sum(data['histograms'], axis=1)
        hists1 = fm.forward_hists(f, mus, N)

        m_sort = np.argsort(p_errors)
        
        i_range = np.arange(data['histograms'].shape[1])
        
        import pyqtgraph as pg
        import PyQt4.QtGui
        import PyQt4.QtCore
        # Always start by initializing Qt (only once per application)
        app = PyQt4.QtGui.QApplication([])
        # Define a top-level widget to hold everything
        win = pg.GraphicsWindow(title="results")
        pg.setConfigOptions(antialias=True)
        
        # show f and the mu values
        p1 = win.addPlot(title=f_name, name = 'f')
        p1.plot(x = i_range, y = f0, pen=(255, 0, 0))
        p1.plot(x = i_range, y = f, pen=(0, 255, 0))
        
        p2 = win.addPlot(title=mus_name, name = 'mus')
        p2.plot(mus0[m_sort], pen=(255, 0, 0))
        p2.plot(mus[m_sort],  pen=(0, 255, 0))
        
        win.nextRow()
        
        # now plot the histograms
        m      = 0
        title  = "histogram pixel " + str(m) + ' error ' + str(int(p_errors[m])) + ' offset {0:.1f}'.format(mus[m])
        hplot  = win.addPlot(title = title)
        curve_his = hplot.plot(data['histograms'][m], fillLevel = 0.0, fillBrush = 0.7, stepMode = False)
        curve_fit = hplot.plot(hists1[m], pen = (0, 255, 0))
        hplot.setXLink('f')
        def replot():
            m = hline.value()
            m = m_sort[m]
            title  = "histogram pixel " + str(m) + ' error ' + str(int(p_errors[m])) + ' offset {0:.1f}'.format(mus[m])
            hplot.setTitle(title)
            curve_his.setData(data['histograms'][m])
            curve_fit.setData(hists1[m])
        
        p3 = win.addPlot(title='pixel errors', name = 'p_errors')
        p3.plot(p_errors[m_sort], pen=(255, 255, 255))
        p3.setXLink('mus')

        hline = pg.InfiniteLine(angle=90, movable=True, bounds = [0, mus.shape[0]-1])
        #hline.sigPositionChangeFinished.connect(replot)
        hline.sigPositionChanged.connect(replot)
        p3.addItem(hline)

        win.nextRow()
        
        p4 = win.addPlot(title="log likelihood error", y = errors)
        p4.showGrid(x=True, y=True) 
        sys.exit(app.exec_())
            
if __name__ == '__main__':
    print 'executing :', sys.argv[1]
    execfile(sys.argv[1])

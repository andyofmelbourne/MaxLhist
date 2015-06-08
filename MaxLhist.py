import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut
from scipy.ndimage import gaussian_filter1d
import sys

def process_input(datas):
    """
    """
    if len(datas) == 0 :
        print 'no data: nothing to do...'
        sys.exit()
    
    data = datas
    
    for data in datas:
        if not data.has_key('histograms'):
            print 'Error: no histogram data. For', data['name']
            sys.exit()
        
    vars = []
    for data in datas:
        if not data.has_key('vars'):
            print 'Error: no random variable in data', data['name']
            sys.exit()
        for var in data['vars']:
            # check if we already have this in the list
            add = True
            for v in vars :
                if v is var :
                    add = False
            # if not then add it
            if add :
                vars.append(var)

    # The random variable must have a function element
    for var in vars :
        if not var.has_key('function'):
            print 'Error: every random variable must have at least a function element', var['name']
            sys.exit()
        elif not var['function'].has_key('update') :
            print 'Error: the function of', var['name'], 'must have an update variable.'
            sys.exit()

    # get the offsets
    offsets = []
    for data in datas:
        if data.has_key('offset'):
            offset = data['offset']
            # check if we already have this in the list
            add = True
            for o in offsets :
                if o is offset :
                    print data['name'] +"'s offset is linked to a previous one. Will update together..."
                    add = False
            # if not then add it
            if add :
                offsets.append(offset)
            
            if not data['offset'].has_key('update') :
                print 'Error: the offset of', data['name'], 'must have an update variable.'
                sys.exit()
        else :
            print 'Error: every dataset must have an offset element ', data['name']
            sys.exit()

    # get the gain
    gains = []
    for data in datas:
        if data.has_key('gain'):
            gain = data['gain']
            # check if we already have this in the list
            add = True
            for o in gains :
                if o is gain :
                    print data['name'] +"'s gain is linked to a previous one. Will update together..."
                    add = False
            # if not then add it
            if add :
                gains.append(gain)
            
            if not data['gain'].has_key('update') :
                print 'Error: the gain of', data['name'], 'must have an update variable.'
                sys.exit()
        else :
            print 'Error: every dataset must have an gain element ', data['name']
            sys.exit()

    # inital guess
    # Loop over the value elements of vars and 
    # offsets to initialise the random variables
    #-------------------------------------------
    f = np.sum(datas[0]['histograms'].astype(np.float64), axis=0) 
    f = f / np.sum(f)
    
    for var in vars:
        if var['function']['value'] is None :
            print "initialising", var['name'] + "'s function with the sum of the histogram data"
            var['function']['value'] = f
        else :
            if var['function']['value'].shape[0] != datas[0]['histograms'].shape[1]:
                print "Error:", var['name']+"'s initial guess for the function does not have the right shape:", var['function']['value'].shape[0], ' hists.shape[1]=', f.shape[0]
                sys.exit()
    
    for data in datas :
        if data['offset']['value'] is None :
            print 'initialising', data['name'] + "'s offset with the argmax of the histograms."
            hist = data['histograms']
            mus = np.zeros((hist.shape[0]), dtype=np.float64)
            for m in range(hist.shape[0]):
                mus[m]   = np.argmax(hist[m]) - np.argmax(f)
            mus = mus - np.sum(mus) / float(len(mus))
            data['offset']['value'] = mus
        else :
            if data['offset']['value'].shape[0] != data['histograms'].shape[0]:
                print "Error:", data['name']+"'s initial guess for the offsets does not have the right shape:", data['offset']['value'].shape[0], ' hists.shape[0]=', data['histograms'].shape[0]
                sys.exit()

    for gain in gains :
        if gain['value'] is None :
            gain['value'] = np.ones((datas[0]['histograms'].shape[0]), dtype=np.float64)
        else :
            if gain['value'].shape[0] != datas[0]['histograms'].shape[0] :
                print "Error:", data['name']+"'s initial guess for the gain does not have the right shape:", data['gain']['value'].shape[0], ' hists.shape[0]=', data['histograms'].shape[0]
                sys.exit()

    # get the counts
    for data in datas:
        init = False
        if not data.has_key('counts'):
            init = True
        if init == False :
            if data['counts']['value'] is None :
                init = True
        if init :
            print 'initialising the counts for', data['name'] + "'s variables with the number of counts / number of vars."
            nvars  = len(data['vars'])
            counts = np.sum(data['histograms'], axis=-1).astype(np.float64) / float(nvars) 
            if nvars == 1 :
                update = False 
            else :
                update = True
            data['counts'] = {'update' : update, 'value' : np.array([counts.copy() for v in data['vars']])}
        else :
            for i in range(data['counts']['value'].shape[0]):
                count = data['counts']['value'][i]
                if count.shape[0] != data['histograms'].shape[0] :
                    print "Error:", data['name']+"'s initial guess for the counts does not have the right shape:", count.shape[0], ' hists.shape[0]=', data['histograms'].shape[0]
                    sys.exit()
    
    return offsets, gains, vars, datas


def refine(datas, iterations=1):
    offsets, gains, vars, datas = process_input(datas)
    
    # update the guess
    #-----------------
    errors = []
    offsets_temp   = list(offsets)
    gains_temp     = list(gains)
    vars_temp      = list(vars)
    
    e   = ut.log_likelihood_calc_many(datas)
    errors.append(e)
    print 0, 'log likelihood error:', e
    
    for i in range(iterations):
        # new gain / offsets 
        for j in range(len(gains_temp)):
            if gains[j]['update'] :
                # grab all the data that has this gain
                ds              = [d for d in datas if gains[j] is d['gain']]
                print 'updating the offset and gain of: ', [d['name'] for d in ds]
                offsets_temp[j]['value'], gains_temp[j]['value'] = ut.update_mus_gain(ds) 
                
        # now check if there are any offsets without gain updates
        for j in range(len(offsets_temp)):
            if offsets[j]['update'] :
                # grab all the data that has this offset
                ds              = [d for d in datas if (offsets[j] is d['offset']) and (d['gain']['update']==False)]
                if len(ds) > 0 :
                    print 'updating the offset: ', [d['name'] for d in ds]
                    offsets_temp[j]['value'] = ut.update_mus_not_gain(ds)
        
        # new functions 
        print 'updating the functions:', [v['name'] for v in vars if v['function']['update']] 
        Xv = ut.update_fs_new(vars, datas)
            
        for v in range(len(vars)):
            vars_temp[v]['function']['value'] = Xv[v]
        
        # new counts 
        print 'updating the counts for ', [d['name'] for d in datas if d['counts']['update']]
        counts_temp    = []
        for d in datas:
            if d['counts']['update']: 
                counts = ut.update_counts(d)
            else :
                counts = d['counts']['update']
            counts_temp.append(counts)
        
        # update the current guess
        for j in range(len(offsets_temp)):
            if offsets[j]['update'] :
                offsets[j]['value'] = offsets_temp[j]['value']
        
        for j in range(len(gains_temp)):
            if gains[j]['update'] :
                gains[j]['value'] = gains_temp[j]['value']
        
        for j in range(len(vars)):
            if vars[j]['function']['update'] :
                vars[j]['function']['value'] = vars_temp[j]['function']['value']
        
        for j in range(len(datas)):
            if datas[j]['counts']['update']: 
                datas[j]['counts']['value'] = np.array(counts_temp[j])

        # minimise overlap on the X's
        #----------------------------
        Xv = ut.minimise_overlap(vars)
        for v in range(len(vars)):
            vars_temp[v]['function']['value'] = Xv[v]
        for j in range(len(vars)):
            if vars[j]['function']['update'] :
                vars[j]['function']['value'] = vars_temp[j]['function']['value']
        # new counts 
        counts_temp    = []
        for d in datas:
            if d['counts']['update']: 
                counts = ut.update_counts(d)
            else :
                counts = d['counts']['update']
            counts_temp.append(counts)
        for j in range(len(datas)):
            if datas[j]['counts']['update']: 
                datas[j]['counts']['value'] = np.array(counts_temp[j])
        
        e   = ut.log_likelihood_calc_many(datas)
        errors.append(e)
        print i+1, 'log likelihood error:', e

    
    errors = np.array(errors)
    
    result = Result(vars, datas, errors)
    return result


class Result():
    def __init__(self, vars, datas, errors):
        self.result = {}
        for d in datas:
            self.result[d['name']] = {}
            self.result[d['name']]['vars']    = np.array( [var['name'] for var in d['vars']] )
            self.result[d['name']]['comment'] = d['comment']
            self.result[d['name']]['offset']  = d['offset']['value']
            self.result[d['name']]['counts']  = d['counts']['value']
            self.result[d['name']]['gain']    = d['gain']['value']

            pixel_map_errors = ut.log_likelihood_calc_pixelwise_many(d)
            
            self.result[d['name']]['pix_errors'] = pixel_map_errors

        for var in vars :
            self.result[var['name']] = {}
            self.result[var['name']]['function'] = var['function']['value']

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
    
    def show_fit(self, dataname, hists):
        errors   = self.result['error vs iter']
        p_errors = self.result[dataname]['pix_errors']
        
        varnames  = self.result[dataname]['vars']
        f_names   = [i + ' function' for i in varnames]
        fs        = [self.result[i]['function'] for i in varnames]
            
        mus_name = dataname + ' offset'
        mus      = self.result[dataname]['offset']
        
        gs_name = dataname + ' gain'
        gs      = self.result[dataname]['gain']

        counts       = self.result[dataname]['counts']
        counts_names = self.result[dataname]['vars']
        
        hists1 = fm.forward_hists_nvar(fs, mus, gs, counts)

        # get the sum of the unshifted and ungained histograms
        hist_proj    = np.sum(ut.ungain_unshift_hist(hists, mus, gs), axis=0)
        total_counts = np.sum(hist_proj)
        hist_proj    = hist_proj / total_counts
         
        m_sort = np.argsort(p_errors)
        
        i_range = np.arange(hists.shape[1])
        
        import pyqtgraph as pg
        import PyQt4.QtGui
        import PyQt4.QtCore
        app = PyQt4.QtGui.QApplication([])
        win = pg.GraphicsWindow(title="results")
        pg.setConfigOptions(antialias=True)
        
        # show f and the mu values
        Xplot = win.addPlot(title='functions')
        Xplot.plot(x = i_range, y = hist_proj + 1.0e-10, fillLevel = 0.0, fillBrush = 0.7, stepMode = False)
        f_tot = np.zeros_like(fs[0])
        for i in range(len(fs)):
            Xplot.plot(x = i_range, y = fs[i] * np.sum(counts[i]) / total_counts + 1.0e-10, pen=(i, len(fs)+1), width = 10)
        Xplot.plot(x = i_range, y = np.sum(fs * np.sum(counts, axis=-1)[:, np.newaxis] / total_counts, axis=0) + 1.0e-10, pen=(len(fs), len(fs)+1), width = 10)
        
        # now plot the histograms
        m      = 0
        title  = "histogram pixel " + str(m) + ' error ' + str(int(p_errors[m])) + ' offset {0:.1f}'.format(mus[m])
        hplot  = win.addPlot(title = title)
        curve_his = hplot.plot(hists[m], fillLevel = 0.0, fillBrush = 0.7, stepMode = False)
        curve_fit = hplot.plot(hists1[m], pen = (0, 255, 0))
        hplot.setXLink('f')
        def replot():
            m = hline.value()
            m = m_sort[m]
            title = "histogram pixel " + str(m) + ' error ' + str(int(p_errors[m])) + ' offset {0:.1f}'.format(mus[m])
            hplot.setTitle(title)
            curve_his.setData(hists[m])
            curve_fit.setData(hists1[m])
        
        p_error_plot = win.addPlot(title='pixel errors', name = 'p_errors')
        p_error_plot.plot(p_errors[m_sort], pen=(255, 255, 255))
        p_error_plot.setXLink('mus')

        hline = pg.InfiniteLine(angle=90, movable=True, bounds = [0, mus.shape[0]-1])
        #hline.sigPositionChangeFinished.connect(replot)
        hline.sigPositionChanged.connect(replot)
        p_error_plot.addItem(hline)
        
        win.nextRow()
        
        muplot = win.addPlot(title=mus_name, name = 'mus')
        muplot.plot(mus[m_sort],  pen=(0, 255, 0))
        
        gplot = win.addPlot(title=gs_name, name = 'gs')
        gplot.plot(gs[m_sort],  pen=(0, 255, 0))
        gplot.setXLink('mus')
        
        p4 = win.addPlot(title="log likelihood error", y = errors)
        p4.showGrid(x=True, y=True) 

        win.nextRow()

        cplots = []
        for c in range(counts.shape[0]):
            cplots.append(win.addPlot(title=counts_names[c] + ' counts: ' + str(int(np.sum(counts[c]))), name = counts_names[c]))
            cplots[-1].plot(counts[c],  pen=(c, counts.shape[0]+1))
            cplots[-1].setXLink('mus')

        sys.exit(app.exec_())

            
if __name__ == '__main__':
    print 'executing :', sys.argv[1]
    execfile(sys.argv[1])

import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln
import sys
import copy
from mpi4py import MPI

"""
The basic data structures that we work with: 
the pixel map:

    pix     hist                hist_cor           gain gain_up offset  offset_up   ns                   ns_up              valid
    0       0 0 100 120 ... I   1.1 2 110.4 ... I  1.0  True    -1      True        [0.01    0.09    0]  [True True False]  True
    1       ...                 ...                1.1  True    1       True        [0.1     0.9     0]  [True True False]  False
    2                                              ...  False   0       False       [0.0     1.0     0]  [True True False]  True
    .                                                                                    
    M                                                                                    

The Xs:
    adus 0 1 2   ... I
    X0   0 0 0.1 ... 0
    X1   0 0 0.2 ... 0
    .    ...
    XV   0 0 0.2 ... 0

"""
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Histograms():

    def __init__(self, datas):
        # set the array dimensions
        #-------------------------
        if rank == 0 :
            vars, M, I, V = self.check_input(datas)
        else :
            M = I = V = 0
        
        comm.barrier()
        M, I, V = comm.bcast([M, I, V], root=0)
        
        # set the numpy dtypes
        #---------------------
        dt_n  = np.dtype([('v', np.float128, (V,)), ('up', np.bool, (V,))]) 
        dt_g  = np.dtype([('v', np.float32), ('up', np.bool)]) 
        dt_pm = np.dtype([('pix', np.int64), ('hist', np.uint64, (I,)), ('hist_cor', np.float128, (I,)),\
                          ('g', dt_g), ('mu', dt_g), ('n', dt_n), ('valid', np.bool), ('m', np.float128), ('e', np.float128)])
        dt_Xs = np.dtype([('v', np.float128, (I,)), ('up', np.bool, (I,)), ('name', np.str_, 128),  ('type', np.str_, 128)])
        self.dt_pm = dt_pm
        self.dt_Xs = dt_Xs
        
        # check input, initialise, broadcast data to workers
        #---------------------------------------------------
        if rank == 0 :
            pix_map, Xs  = self.process_input(datas, vars, M, I, V)
            pix_map, Xs  = self.init_input(datas, vars, pix_map, Xs)
            
            self.pix_map = chunkIt(pix_map, size)
            self.Xs      = Xs

            # I need to remember which dataset belongs to 
            # to which pixels. As well as the names and comments
            self.datas = datas
            # remove the raw data and put in a reference to the 
            # pixel numbers
            start = 0
            for d in range(len(datas)):
                Md = self.datas[d]['histograms'].shape[0]
                del self.datas[d]['histograms']
                
                self.datas[d]['histograms'] = np.arange(start, start + Md, 1)
                start += Md
        else :
            self.adus    = chunkIt(range(I), size-1)
            self.pix_map = chunkIt(range(M), size-1)
            self.Xs      = None
            self.pix_map = None
        
        if rank == 0 : print '\n broadcasting the Xs to everyone...'
        comm.barrier()
        self.Xs = comm.bcast(self.Xs, root=0)
        
        if rank == 0 : print '\n scattering the pixel maps to everyone...'
        comm.barrier()
        self.pix_map = comm.scatter(self.pix_map, root=0)

        self.pix_map = self.unshift_ungain(self.pix_map)
        self.pix_map = self.pixel_multiplicity(self.pix_map)
        self.pix_map = self.pixel_errors(self.Xs, self.pix_map)
        comm.barrier()


    def check_input(self, datas):
        """
        conditions:
        -----------
        len(datas) > 0
        data[i] must have the structure:
        data:
            histograms 
            vars
                function
                    update
                    value
            offset
                update
                value
            gain 
                update
                value
            counts 
                update
                value
        """
        # check that we have data
        if len(datas) == 0 :
            print 'no data: nothing to do...'
            sys.exit()
        
        for data in datas:
            if not data.has_key('histograms'):
                print 'Error: no histogram data. For', data['name']
                sys.exit()
            
        # get the unique vars
        vars = []
        for data in datas:
            if not data.has_key('vars'):
                print 'Error: no random variable in data', data['name']
                sys.exit()
                
            for var in data['vars']:
                #if var not in vars: vars.append(var) # this fails
                if id(var) not in [id(v) for v in vars]: vars.append(var)

        def check_name_key_update_value(dic_list, key):
            for d in dic_list:
                if not d.has_key('name'):
                    print 'Error: every dataset and variable must have a name', d.keys()
                    sys.exit()
                
                if not d.has_key(key):
                    print 'Error:', d['name'], 'must have a', key, 'element'
                    sys.exit()
                
                if not d[key].has_key('update') :
                    print 'Error:', d['name'], "'s", key, 'must have an update variable.'
                    sys.exit()
                
                if not d[key].has_key('value') :
                    print 'Error:', d['name'], "'s", key, 'must have an value variable.'
                    sys.exit()

        # The random variable must have a function element
        check_name_key_update_value(vars, 'function')

        # check the offsets
        check_name_key_update_value(datas, 'offset')

        # check the gains
        check_name_key_update_value(datas, 'gain')

        # get the counts
        check_name_key_update_value(datas, 'counts')

        # check the dimensions of the input
        I  = datas[0]['histograms'].shape[1]
        M  = 0
        for d in datas :
            M += d['histograms'].shape[0]
            if d['histograms'].shape[1] != I :
                print 'Error:', d['name'], 'does not have the same adu range as', datas[0]['name'], d['histograms'].shape[1], I
                sys.exit()
        
        for var in vars :
            if var['function']['value'] is not None :
                if var['function']['value'].shape != (I,):
                    print 'Error:', var['name'], 'does not have the same adu range as', datas[0]['name'], var['function']['value'].shape, (I,)
                    sys.exit()

        for d in datas :
            if d['offset']['value'] is not None :
                if d['offset']['value'].shape != (d['histograms'].shape[0],):
                    print 'Error: the offset of', d['name'], 'does not have the same number of pixels as the histogram', (d['histograms'].shape[0],), d['offset']['value'].shape
                    sys.exit()

            if d['gain']['value'] is not None :
                if d['gain']['value'].shape != (d['histograms'].shape[0],):
                    print 'Error: the gain of', d['name'], 'does not have the same number of pixels as the histogram', (d['histograms'].shape[0],), d['gain']['value'].shape
                    sys.exit()

            if d['counts']['value'] is not None :
                if d['counts']['value'].shape[1] != d['histograms'].shape[0] :
                    print "Error:", d['name']+"'s initial guess for the counts does not have the right shape:", d['counts']['value'].shape, ' hists.shape=', d['histograms'].shape
                    sys.exit()
                
                if d['counts']['value'].shape[0] != len(d['vars']) :
                    print "Error:", d['name']+"'s initial guess for the counts does not have the right shape:", d['counts']['value'].shape, ' no. of vars', len(d['vars'])
                    sys.exit()
        
        Xs = []
        V = len(vars)
        return vars, M, I, V
     
    def process_input(self, datas, vars, M, I, V):
        """
        Fill the pixel_map and Xs with datas
        """
        pix_map = np.zeros((M,), self.dt_pm)
        Xs      = np.zeros((V,), self.dt_Xs)
        
        # pixel numbers
        pix_map['pix']      = np.arange(M)
        pix_map['valid'][:] = True
        
        start = 0
        for d in datas:
            # histogram data 
            hist                                              = d['histograms']
            
            print '\n Setting histogram data for', d['name'], 'with pixel ids:', start, start + hist.shape[0]
            pix_map['hist'][start : start + hist.shape[0], :] = hist
            
            # inverse gain
            if d['gain']['value'] is not None :
                print '\n Setting gain values for', d['name'], 'with the input values. In the range', start, start + hist.shape[0]
                pix_map['g']['v'][start : start + hist.shape[0]] = d['gain']['value']
            
            if d['gain']['update'] == True :
                pix_map['g']['up'][start : start + hist.shape[0]][:] = True
            
            # adu offsets
            if d['offset']['value'] is not None :
                print '\n Setting offsets for', d['name'], 'with the input values. In the range', start, start + hist.shape[0]
                pix_map['mu']['v'][start : start + hist.shape[0]] = d['offset']['value']
            
            if d['offset']['update'] == True :
                pix_map['mu']['up'][start : start + hist.shape[0]][:] = True
            
            # errant counts
            counts_perpix = np.sum(hist, axis=-1)
            max           = np.max(counts_perpix)
            sig           = 0.1 * max
            bad_pix       = np.where(np.abs(counts_perpix - max) > sig)[0] 
            
            if len(bad_pix) > 0 :
                print '\n Warning: found', len(bad_pix), 'pixels with counts more than 10% from the max', int(max-sig), int(max+sig)
                print 'masking...'
                pix_map['valid'][start + bad_pix] = False
                        
            
            # counts
            initialised_counts = []
            if d['counts']['value'] is not None :
                if d['counts']['value'].shape[0] == 1 :
                    varno = np.where( [id(var) == id(d['vars'][0]) for var in vars] )[0][0]
                    
                    print '\n only one random variable for', d['name'], 'setting the count fractions to one. In the range', start, start + hist.shape[0]
                    pix_map['n']['v'][start : start + hist.shape[0], varno][:] = 1.
                    print ' and seting the update to False...', vars[varno]['name']
                    pix_map['n']['up'][start : start + hist.shape[0], :]       = False
                else :
                    for v in range(d['counts']['value'].shape[0]):
                        # which var number is this count ?
                        varno = np.where( [id(var) == id(d['vars'][v]) for var in vars] )[0][0]
                        
                        if varno not in initialised_counts :
                            print '\n Setting count fractions for', d['name']+"'s", vars[varno]['name'], 'with the input values. In the range', start, start + hist.shape[0]
                            ns = d['counts']['value'][v].astype(np.float128)
                            ms = np.where(pix_map['valid'][start : start + hist.shape[0]])
                            pix_map['n']['v'][start : start + hist.shape[0], varno][ms] = ns[ms] / counts_perpix.astype(np.float128)[ms]
                            
                            initialised_counts.append(varno)
                        
                    if d['counts']['update'] :
                        print '\n will update the count fractions for', d['name']
                        pix_map['n']['up'][start : start + hist.shape[0], :] = True
            
            start += hist.shape[0]

        print '\n masking the offsets, gains and counts for invalid pixels...'
        pix_map['mu']['up'][np.where(pix_map['valid'] == False)]      = False
        pix_map['g']['up'][np.where(pix_map['valid'] == False)]       = False
        pix_map['n']['up'][np.where(pix_map['valid'] == False)[0], :] = False
        
        for v in range(len(vars)) :
            if vars[v]['function']['value'] is not None :
                print '\n setting', vars[v]['name'], 'var number',v,'to the input value'
                Xs[v]['v'][:]  = vars[v]['function']['value'].astype(np.float128)
            
            if vars[v]['function'].has_key('adus') and vars[v]['function']['update']:
                print '\n applying the adu mask of', vars[v]['name'], 'var number',v,'to the input value'
                Xs[v]['up'][:]  = vars[v]['function']['adus']
            
            elif not vars[v]['function']['update']:
                print '\n will not update ', vars[v]['name']
                Xs[v]['up'][:]  = False
                    
            else :
                print '\n will update all adu values that have signal for', vars[v]['name']
                Xs[v]['up'][:]  = True
            
            Xs[v]['name'] = vars[v]['name']
            Xs[v]['type'] = vars[v]['type']
        
        return pix_map, Xs

    def init_input(self, datas, vars, pix_map, Xs):
        """
        Don't touch anything that has been initialised.
        Otherwise:
            mus    = argmax(hist) - argmax(sum(hist))
            gs     = 1
            Xs     = (f,f,f)
            counts = (1,0,0)
        """
        start = 0
        for d in datas:
            hist = d['histograms']
            # inverse gain
            if d['gain']['value'] is None :
                print "\n initialising the inverse gain with 1's..."
                pix_map['g']['v'][start : start + hist.shape[0]].fill(1)
            
            # adu offsets
            if d['offset']['value'] is None :
                print '\n summing the histgram of', d['name']
                f = np.sum(d['histograms'].astype(np.float64), axis=0) 
                f = f / np.sum(f)
                
                print ' setting the offset to armax(hist[m]) - argmax(sum(hist)):'
                mus = np.argmax(hist, axis=-1) - np.argmax(f)
                pix_map['mu']['v'][start : start + hist.shape[0]] = mus
            
            # count fractions
            if d['counts']['value'] is None :
                # which var number is this count ?
                varno = np.where( [id(var) == id(d['vars'][0]) for var in vars] )[0][0]
                print '\n setting the count fractions for', d['name'], 'to 1. In range:', start, start + hist.shape[0]
                print ' for variable', vars[varno]['name']
                pix_map['n']['v'][start : start + hist.shape[0], varno].fill(1.)
            
                print '\n will update the count fractions for', d['name'], '. In range:', start, start + hist.shape[0]
                print ' for variable', vars[varno]['name']
                
                print '\n will update the count fractions for', d['name'], '. In range:', start, start + hist.shape[0]
                pix_map['n']['up'][start : start + hist.shape[0], :] = True
            start += hist.shape[0]

        # the adu distributions
        for v in range(len(vars)) :
            if vars[v]['function']['value'] is None :
                print '\n setting', vars[v]['name'],' to the sum of the histograms'
                Xs[v]['v'] = f 
        return pix_map, Xs

    def unshift_ungain(self, pix_map):
        if rank == 0 : print '\n calcuating the unshifted and ungained histograms...'
        i = np.arange(pix_map['hist'].shape[1]).astype(np.float)
        M = float(pix_map['hist'].shape[0])
        
        for m, p in enumerate(pix_map):
            if rank == 0 : update_progress(float(m + 1) / M)
             
            if p['valid'] :
                pix_map[m]['hist_cor'][:] = np.interp(i/p['g']['v'] + p['mu']['v'], i, p['hist'].astype(np.float64), left=0.0, right=0.0) / p['g']['v'] 
        return pix_map
    
    def pixel_multiplicity(self, pix_map):
        if rank == 0 : print '\n calcuating the log(multiplicity) for each pixel...'
        M = float(pix_map['hist'].shape[0])
        
        for m, p in enumerate(pix_map):
            if rank == 0 : update_progress(float(m + 1) / M)
             
            pix_map[m]['m'] = gammaln(np.sum(p['hist']) + 1) - np.sum(gammaln(p['hist'] + 1))
        
        return pix_map

    def pixel_errors(self, Xs, pix_map, prob_tol = 1.0e-10):
        if rank == 0 : print '\n calcuating the log(likelihood) error for each pixel...'
        i = np.arange(pix_map['hist'].shape[1]).astype(np.float)
        M = float(pix_map['hist'].shape[0])
        
        for m, p in enumerate(pix_map):
            if rank == 0 : update_progress(float(m + 1) / M)
            
            if p['valid'] :
                # do not evaluate for pixels where hist == 0
                Is = np.where(p['hist'] > prob_tol)
                
                # evaluate the shifted gained probability function
                f = np.sum( Xs['v'] * p['n']['v'][:, np.newaxis], axis=0)
                f = np.interp(i*p['g']['v'] - p['mu']['v'], i, f.astype(np.float64), left=0.0, right=0.0) * p['g']['v'] 
                
                # sum the log liklihood errors for this pixel
                e  = p['hist'][Is] * np.log(prob_tol + f[Is])
                
                pix_map[m]['e'] = - np.sum(e) - p['m']
                if pix_map[m]['e'] < 0.0 :
                    print 'Error: pixel', p['pix'], 'has a negative log likelihood (prob > 1). sum(f)', np.sum(f)
                    sys.exit()
            else :
                pix_map[m]['e'] = 0.0
        
        return pix_map

    def hist_fit(self, p, Xs):
        i = np.arange(p['hist'].shape[0]).astype(np.float)
        f = np.sum( Xs['v'] * p['n']['v'][:, np.newaxis], axis=0)
        f = np.interp((i - p['mu']['v'])*p['g']['v'], i, f.astype(np.float64), left=0.0, right=0.0) * p['g']['v'] 
        f *= np.sum(p['hist'])
        return f
        
    def update_counts(self):
        """
        """
        for m, p in enumerate(self.pix_map):
            if np.any(p['n']['up']):
                hgm = p['hist_cor']
                
                # we don't need the zeros...
                Is  = np.where(hgm > 0)[0]
                hgm = hgm[Is]
                vs  = np.where(p['n']['up'])[0]
                Xv2 = self.Xs['v'][:, Is]
                Xv2 = Xv2[vs]
                ns0 = p['n']['v'][vs]
                
                def fun(ns):
                    ns2 = ns / np.sum(ns)
                    error = - np.sum( hgm * np.log( np.sum( ns2[:, np.newaxis] * Xv2, axis=0) + 1.0e-10)) - p['m']
                    return error
                
                bounds = []
                for v in vs:
                    bounds.append( (0.0, 1.0) )
                
                res    = scipy.optimize.minimize(fun, ns0, bounds=bounds, tol = 1.0e-10)
                self.pix_map['n']['v'][m][vs] = res.x / np.sum(res.x) 
                #print 'updating pixel', p['pix'], res.x/ np.sum(res.x), self.pix_map['n']['v'][m][vs][:], rank, np.sum(p['hist_cor']), np.sum(p['hist'])

    def update_gain_offsets(self):
        pass
    def update_Xs(self):
        pass
    
    def gather_pix_map(self):
        """
        Gather the results from everyone
        """
        comm.barrier()
        if rank == 0 : print '\n gathering the pixel maps from everyone...'
        self.pix_map = comm.gather(self.pix_map, root=0)
        
        # self.pix_map is now None for all the workers
        # and a list of numpy arrays of dtype self.dt_pm
        # we just need to concatenate everything...
        
        if rank == 0 :
            self.pix_map = np.concatenate( tuple(self.pix_map) )
            print ' recieved the pixel map of shape:', self.pix_map.shape

    def show(self, dname = None):
        if rank != 0 :
            return 
        
        if dname is None :
            pixels       = self.pix_map['pix']
            pixels_valid = np.where(self.pix_map['valid'])[0]
            dataname = 'all'
        else :
            for d in self.datas:
                if d['name'] == dname :
                    pixels = d['histograms']
                    pixels_valid = np.where(self.pix_map['valid'][pixels])
                    pixels_valid = pixels[pixels_valid]
            dataname = dname
        
        #errors   = self.result['error vs iter']
        # get the sum of the unshifted and ungained histograms
        total_counts = np.sum(self.pix_map['hist_cor'][pixels_valid])
        hist_proj    = np.sum(self.pix_map['hist_cor'][pixels_valid], axis=0) / total_counts
         
        mus_name  = dataname + ' offset'
        gs_name   = dataname + ' gain'
        p_errors  = self.pix_map['e'][pixels]
        m_sort    = np.argsort(p_errors)
        mus       = self.pix_map['mu']['v'][pixels]
        gs        = self.pix_map['g']['v'][pixels]
        hists     = self.pix_map['hist'][pixels]
        hists_cor = self.pix_map['hist_cor'][pixels]
        
        import pyqtgraph as pg
        import PyQt4.QtGui
        import PyQt4.QtCore
        app = PyQt4.QtGui.QApplication([])
        win = pg.GraphicsWindow(title="results")
        pg.setConfigOptions(antialias=True)
        
        # show f and the mu values
        ns     = self.pix_map['n']['v'][pixels_valid]
        counts = np.sum(self.pix_map['hist'][pixels_valid], axis=-1)
        total_counts = np.sum(counts)
        
        print ns.shape
        fi   = lambda f: self.Xs['v'][i] * np.sum(ns[:, i] * counts) / float(total_counts)
        ftot = np.sum([fi(i) for i in range(self.Xs.shape[0])], axis=0) 

        Xplot = win.addPlot(title='functions')
        Xplot.plot(y = hist_proj + 1.0e-10, fillLevel = 0.0, fillBrush = 0.7, stepMode = False)
        f_tot = np.zeros_like(self.Xs['v'][0])
        for i in range(self.Xs.shape[0]):
            Xplot.plot(y = fi(i) + 1.0e-10, pen=(i, len(self.Xs)+1), width = 10)
        Xplot.plot(y = ftot + 1.0e-10, pen=(len(self.Xs), len(self.Xs)+1), width = 10)
        
        # now plot the histograms
        m      = 0
        title  = "histogram pixel " + str(m) + ' error ' + str(int(p_errors[m])) + ' offset {0:.1f}'.format(mus[m]) + ' inv. gain {0:.1f}'.format(gs[m])
        hplot  = win.addPlot(title = title)
        curve_his = hplot.plot(hists[m], fillLevel = 0.0, fillBrush = 0.7, stepMode = False)
        curve_fit = hplot.plot(self.hist_fit(self.pix_map[pixels][m], self.Xs), pen = (0, 255, 0))
        hplot.setXLink('f')
        def replot():
            m = hline.value()
            m = m_sort[m]
            title = "histogram pixel " + str(m) + ' error ' + str(int(p_errors[m])) + ' offset {0:.1f}'.format(mus[m]) + ' inv. gain {0:.1f}'.format(gs[m])
            hplot.setTitle(title)
            curve_his.setData(hists[m])
            curve_fit.setData(self.hist_fit(self.pix_map[pixels][m], self.Xs))
        
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
        
        #p4 = win.addPlot(title="log likelihood error", y = errors)
        #p4.showGrid(x=True, y=True) 

        win.nextRow()

        counts = np.sum(self.pix_map['hist'][pixels], axis=-1)
        ns     = self.pix_map['n']['v'][pixels]
        cplots = []
        for c in range(self.Xs.shape[0]):
            cplots.append(win.addPlot(title=self.Xs[c]['name'] + ' counts: ' + str(int(np.sum(counts * ns[:, c]))), name = self.Xs[c]['name']))
            cplots[-1].plot(ns[m_sort][:, c],  pen=(c, self.Xs.shape[0]+1))
            cplots[-1].setXLink('mus')

        sys.exit(app.exec_())

def chunkIt(seq, num):
    splits = np.mgrid[0:len(seq):(num+1)*1J].astype(np.int)
    out    = []
    for i in range(splits.shape[0]-1):
        out.append(seq[splits[i]:splits[i+1]])
    return out


def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.1f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

if __name__ == '__main__':
    print 'executing :', sys.argv[1]
    execfile(sys.argv[1])

import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln
import sys
import copy
from mpi4py import MPI
import h5py


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
    def __init__(self, datas = None, fnam_h5 = None):
        """
        """
        if datas is not None :
            # set the array dimensions
            #-------------------------
            if rank == 0 :
                vars, M, I, V = self.check_input(datas)
            else :
                M = I = V = 0
            
            M, I, V = comm.bcast([M, I, V], root=0)
            self.M, self.I, self.V = M, I, V
            
            # set the numpy dtypes
            #---------------------
            dt_n  = np.dtype([('v', np.float32, (V,)), ('up', np.bool, (V,))]) 
            dt_g  = np.dtype([('v', np.float32), ('up', np.bool)]) 
            dt_pm = np.dtype([('pix', np.uint32), ('hist', np.uint32, (I,)), ('hist_cor', np.float32, (I,)),\
                              ('g', dt_g), ('mu', dt_g), ('n', dt_n), ('valid', np.bool), ('m', np.float32), ('e', np.float32)])
            dt_Xs = np.dtype([('v', np.float64, (I,)), ('up', np.bool, (I,)), ('name', np.str_, 128),  ('type', np.str_, 128)])
            self.dt_pm = dt_pm
            self.dt_Xs = dt_Xs
            
            # check input, initialise, broadcast data to workers
            #---------------------------------------------------
            if rank == 0 :
                self.errors  = []
                pix_map, Xs  = self.process_input(datas, vars, M, I, V)
                pix_map, Xs  = self.init_input(datas, vars, pix_map, Xs)
                
                self.pix_map = chunkIt(pix_map, size)
                self.Xs      = Xs

                # I need to remember which dataset belongs to 
                # to which pixels. As well as the names and comments
                #self.datas = datas
                self.datas = []
                # remove the raw data and put in a reference to the 
                # pixel numbers
                start = 0
                data = {}
                for d in range(len(datas)):
                    data['name'] = datas[d]['name']
                    data['comment'] = datas[d]['comment']
                    Md = datas[d]['histograms'].shape[0]
                    data['histograms'] = np.arange(start, start + Md, 1)
                    #Md = self.datas[d]['histograms'].shape[0]
                    #del self.datas[d]['histograms']
                    start += Md
                    self.datas.append(copy.deepcopy(data))
            else :
                self.Xs      = None
                self.pix_map = None
            
            self.adus    = chunkIt(np.arange(I), size)
            
            self.scatter_bcast()
            self.unshift_ungain()
            self.pixel_multiplicity()
            #self.pixel_errors()
        
        else :
            self.load_h5(fnam_h5)


    def scatter_bcast(self):
        if rank == 0 : print '\n broadcasting the Xs to everyone...'
        self.Xs = comm.bcast(self.Xs, root=0)
        
        if rank == 0 : print '\n scattering the pixel maps to everyone...'
        #if rank == 0 : print ' len(self.pix_map)', len(self.pix_map), [len(p) for p in self.pix_map]
        if rank == 0 :
            for i in range(1, size):
                update_progress(float(i+1) / float(size))
                comm.send(self.pix_map[i], dest=i, tag=i)
            
            del self.pix_map[1 :]
            self.pix_map = self.pix_map[0]
        else :
            self.pix_map = comm.recv(source = 0, tag=rank)
        

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
                Xs[v]['up'][vars[v]['function']['adus']] = True
            
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
                f = np.sum(pix_map['hist'], axis=0).astype(np.float64)
                f = f / np.sum(f)
                Xs[v]['v'][:] = f
        return pix_map, Xs


    def unshift_ungain(self):
        if rank == 0 : print '\n calcuating the unshifted and ungained histograms...'
        i = np.arange(self.pix_map['hist'].shape[1]).astype(np.float)
        M = float(self.pix_map['hist'].shape[0])
        
        for m, p in enumerate(self.pix_map):
            if rank == 0 : update_progress(float(m + 1) / M)
             
            if p['valid'] :
                self.pix_map[m]['hist_cor'][:] = np.interp(i/p['g']['v'] + p['mu']['v'], i, p['hist'].astype(np.float64), left=0.0, right=0.0) / p['g']['v'] 
    

    def pixel_multiplicity(self):
        if rank == 0 : print '\n calcuating the log(multiplicity) for each pixel...'
        M = float(self.pix_map['hist'].shape[0])
        
        for m, p in enumerate(self.pix_map):
            if rank == 0 : update_progress(float(m + 1) / M)
             
            self.pix_map[m]['m'] = gammaln(np.sum(p['hist']) + 1) - np.sum(gammaln(p['hist'] + 1))


    def pixel_errors(self, prob_tol = 1.0e-10):
        if rank == 0 : print '\n calcuating the log(likelihood) error for each pixel...'
        i = np.arange(self.pix_map['hist'].shape[1]).astype(np.float)
        M = float(self.pix_map['hist'].shape[0])
        
        neg_count = 0
        for m, p in enumerate(self.pix_map):
            if rank == 0 : update_progress(float(m + 1) / M)
            
            if p['valid'] :
                # do not evaluate for pixels where hist == 0
                Is = np.where(p['hist'] > prob_tol)
                
                # evaluate the shifted gained probability function
                f = np.sum( self.Xs['v'] * p['n']['v'][:, np.newaxis], axis=0)
                f = np.interp((i - p['mu']['v'])*p['g']['v'], i, f.astype(np.float64), left=0.0, right=0.0) * p['g']['v'] 
                
                # sum the log liklihood errors for this pixel
                e  = p['hist'][Is] * np.log(prob_tol + f[Is])
                
                self.pix_map[m]['e'] = - np.sum(e) - p['m']
                if self.pix_map[m]['e'] < 0.0 and self.pix_map[m]['valid']:
                    neg_count += 1
            else :
                self.pix_map[m]['e'] = 0.0
        
        # reduce the errors to the master
        valid = np.where(self.pix_map['valid'])
        error = comm.reduce(np.sum(self.pix_map['e'][valid]), op=MPI.SUM, root = 0)
        if rank == 0 :
            self.errors.append(error)
    
        if neg_count > 0 and rank == 0 :
            print 'rank:', rank, 'Error: ', neg_count, 'pixels have a negative log likelihood.'


    def hist_fit(self, p, Xs):
        i = np.arange(p['hist'].shape[0]).astype(np.float)
        f = np.sum( Xs['v'] * p['n']['v'][:, np.newaxis], axis=0)
        f = np.interp((i - p['mu']['v'])*p['g']['v'], i, f.astype(np.float64), left=0.0, right=0.0) * p['g']['v'] 
        f *= np.sum(p['hist'])
        return f
        

    def update_counts(self):
        """
        """
        if rank == 0 : print '\n updating the count fractions...'
        M = float(self.pix_map['hist'].shape[0])
        
        for m, p in enumerate(self.pix_map):
            if rank == 0 : update_progress(float(m + 1) / M)
            
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


    def update_gain_offsets(self, quadfit=False, gmin=0.5, gmax=1.5, N=10, iters=3):
        if rank == 0 : print '\n updating gain and offset values...', len(self.pix_map)
        I       = self.pix_map['hist'].shape[1]
        i       = np.arange(I).astype(np.float)
        fftfreq = I * np.fft.fftfreq(I)
        M       = float(self.pix_map['hist'].shape[0])
        
        for m, p in enumerate(self.pix_map):
            if rank == 0 : update_progress(float(m + 1) / M)
            
            if not p['valid']:
                continue
            
            if (not p['mu']['up']) and (not p['g']['up']):
                continue
            
            h_hat = np.array(np.fft.rfft(p['hist'].astype(np.float64)))
            
            f  = np.sum( self.Xs['v'] * p['n']['v'][:, np.newaxis], axis=0).astype(np.float64)
            
            # zoom
            for j in range(iters):
                if j == 0 :
                    gs, gstep = np.linspace(gmin, gmax, N, endpoint=True, retstep=True)
                else :
                    gs, gstep = np.linspace(np.max([gmin, g0-gstep]), np.min([gmax, g0+gstep]), N, endpoint=True, retstep=True)
                
                fs     = [np.interp(i*g, i, f, left=0.0, right=0.0) * g for g in gs] 
                fs     = np.array(fs)
                fs     = np.log(fs + 1.0e-10)
                fs_hat = np.fft.rfftn(fs, axes=(-1,))
                    
                cor = np.fft.irfftn( np.conj(fs_hat) * h_hat, (I,), axes=(-1,) )
                
                gij     = np.unravel_index(np.argmax(cor), dims=cor.shape)
                gi, mui = gij[0], gij[1]
                
                g0      = gs[gi] 
                
            if quadfit :
                cor_max0 = cor[gi, mui]
                
                # map to shift coord
                mus_0 = np.array([mui - 1, mui, mui + 1]) % cor.shape[1]
                mus_t = fftfreq[mus_0]
                vs    = cor[gi, mus_0]
                poly  = np.polyfit(mus_t, vs, 2)
                # evaluate the new arg maximum 
                mu      = - poly[1] / (2. * poly[0])
                if (mu > mus_t[0]) and (mu < mus_t[-1]) and (poly[0] < 0) :
                    #cor_max = poly[0] * mu**2 + poly[1] * mu + poly[2]
                    pass
                else :
                    #print 'quadratic fit failed', mu_0, mu, mus_t, cor.shape, poly, vs
                    mu      = fftfreq[mui]
            else :
                mu      = fftfreq[mui]
            
            self.pix_map[m]['mu']['v'] = mu
            self.pix_map[m]['g']['v']  = g0


        # we need to ensure that sum mus = 0
        #                        sum gs  = M
        # pretty anoying -------------------
        self.normalise_gain_offsets()


    def normalise_gain_offsets(self):
        if rank == 0 : print '\n gathering gain and offsets for normalisation...'
        
        gs  = comm.gather(self.pix_map['g'], root=0)
        if rank == 0 : gs  = np.concatenate( tuple(gs) )
        
        mus = comm.gather(self.pix_map['mu'], root=0)
        if rank == 0 : mus = np.concatenate( tuple(mus) )
        
        if rank == 0 :
            i = np.where(mus['up'])
            if len(i[0]) > 0 :
                mus['v'][i] = mus[i]['v'] - np.sum(mus[i]['v']) / float(len(i[0]))
                mus         = chunkIt(mus, size)
            else :
                mus         = [False for r in range(size)]
            
            j = np.where(gs['up'])
            if len(j[0]) > 0 :
                gs['v'][j] = gs[j]['v'] / np.mean(gs[j]['v']) 
                gs         = chunkIt(gs, size)
            else :
                gs         = [False for r in range(size)]
             
        if rank == 0 : print ' scattering the new gain and offset values...'
        gs = comm.scatter(gs, root=0)
        if gs is not False :
            i  = np.where(gs['up'])
            self.pix_map['g']['v'][i] = gs['v'][i]

        mus = comm.scatter(mus, root=0)
        if mus is not False :
            i   = np.where(mus['up'])
            self.pix_map['mu']['v'][i] = mus['v'][i]
        # ----------------------------------

        self.unshift_ungain()


    def update_Xs(self, verb=False):
        if rank == 0 : print '\n updating the Xs...'
        
        I  = self.Xs[0]['v'].shape[0] 
        up = self.Xs['up']
        Xs = self.Xs['v']

        # allgather valid pixels
        valid_pix = self.pix_map['valid']
        valid_pix = comm.allgather(valid_pix)
        valid_pix = np.concatenate( tuple(valid_pix) )
        valid_pix = np.where(valid_pix)[0]

        # allgather pixel counts
        my_valid = np.where(self.pix_map['valid'])[0]
        counts   = np.sum(self.pix_map[my_valid]['hist_cor'], axis=1)
        counts   = comm.allgather(counts)
        counts   = np.concatenate( tuple(counts) )
        
        total_counts = np.sum(counts)
        
        # allgather count fractions
        ns = self.pix_map['n']['v'][my_valid]
        ns = comm.allgather(ns)
        ns = np.concatenate( tuple(ns) )

        counts_n = counts[:, np.newaxis] * ns

        # get the hist_cor's for our adu's
        hist_cor = []
        for i in range(self.I):
            rank_i = [np.any(self.adus[j] == i) for j in range(size)].index(True)
            h      = comm.gather(self.pix_map['hist_cor'][my_valid, i], root = rank_i)
            if h is not None :
                h = np.concatenate( tuple(h) )
            hist_cor.append(h)

        hist_proj = []
        for h in hist_cor:
            if h is not None :
                hist_proj.append(np.sum(h))
            else :
                hist_proj.append(None)


        # grab hist_cor for our adu value from everyone
        #hist_cor     = self.pix_map['hist_cor'][valid_pix]
        #hist_proj    = np.sum(hist_cor, axis=0) 

        my_X                     = np.zeros_like(self.Xs['v'])
        my_X[:, self.adus[rank]] = self.Xs['v'][:, self.adus[rank]]
        
        I = len(self.adus[rank])
        count = 0
        for i in self.adus[rank]:
            if rank == 0 : 
                update_progress(float(count + 1) / float(len(self.adus[rank])))
                count += 1

            vs_up      = np.where(up[:, i])[0]
            vs_nup     = np.where(up[:, i] == False)[0]
            nonzero    = self.Xs['v'][:, i] > 0.0
            

            if np.sum(up[:, i]) == 0 :
                if rank == 0 and verb : print ' no Xs to update at adu value', i,'skipping'
                continue

            if hist_proj[i] == 0.0 :
                if rank == 0 and verb : print ' no counts at adu value', i,'setting to zero'
                for v in vs_up:
                    my_X[v, i] = 0.0
                continue
            
            # if we only have one var to update 
            # and it's the only non zero var
            if np.sum(up[:, i]) == 1 and np.all((up[:, i] | nonzero) == up[:,i]) :
                if rank == 0 and verb : print ' only one X to update at adu value', i,' setting to the sum of the corrected hist'
                my_X[vs_up[0], i] = hist_proj[i] / total_counts
            
            # if we only have one var to update 
            # and there are other vars that are nonzero
            if np.sum(up[:, i]) == 1 and np.any(nonzero[vs_nup]) :
                if rank == 0 and verb : print '\n only one X to update, but other vars are nonzero, at adu value', i
                if np.any(counts_n[:, vs_up[0]] > 0):
                    my_X[vs_up[0], i]  = hist_proj[i] - np.sum(np.sum(counts_n[:, vs_nup], axis=0) * self.Xs['v'][vs_nup, i])
                    my_X[vs_up[0], i] /= np.sum(counts_n[:, vs_up[0]])
                    if self.Xs['v'][vs_up[0], i] < 0.0 :
                        my_X[vs_up[0], i] = 0.0
                    if rank == 0 and verb : print ' setting to the sum of the residual corrected hist', my_X[vs_up[0], i]

            # if we have more than one var to update
            # and they are the only vars
            if np.sum(up[:, i]) > 1 and np.all((up[:, i] | nonzero) == up[:,i]) :
                if rank == 0 and verb : print ' more than one var to update and they are the only vars, at adu', i
                A      = np.sum(counts_n[:, vs_up], axis=0)
                b      = hist_proj[i]
                bounds = [(0.0, 1.0) for v in range(len(vs_up))]
                def err(xs):
                    e = -np.sum( hist_cor[i] * np.log(np.sum(ns * xs[np.newaxis, :], axis=1) + 1.0e-10) )
                    return e
                
                xs = grid_condition_boundaries(err, A, b, bounds, N=10, iters=10)
                
                my_X[vs_up, i] = xs
        
        if rank == 0 : print '\n reducing the Xs to everyone...'
        self.Xs['v'][:] = comm.allreduce(my_X, op=MPI.SUM)
        
        # normalise
        for v in range(len(self.Xs)):
            self.Xs['v'][v][:] = self.Xs['v'][v][:] / np.sum(self.Xs['v'][v][:])
                

    def gather_pix_map(self):
        """
        Gather the results from everyone
        """
        if rank == 0 : print '\n gathering the pixel maps from everyone...'
        #self.pix_map = comm.gather(self.pix_map, root=0)
        
        if rank == 0 :
            self.pix_map = [self.pix_map]
            for i in range(1, size):
                update_progress(float(i+1) / float(size))
                self.pix_map.append([])
                self.pix_map[-1] = comm.recv(source = i, tag=i)
        else :
            comm.send(self.pix_map, dest=0, tag=rank)

        if rank == 0 :
            self.pix_map = np.concatenate( tuple(self.pix_map) )
            print ' recieved the pixel map of shape:', self.pix_map.shape


    def show(self, dname = None, subsample=10000):
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
        
        if subsample is not None and subsample < len(pixels_valid):
            subsample = np.random.random((subsample)) * len(pixels_valid)
            subsample = subsample.astype(np.int)
            pixels_valid = pixels_valid[subsample]

        errors   = self.errors
        # get the sum of the unshifted and ungained histograms
        total_counts = np.sum(self.pix_map['hist_cor'][pixels_valid])
        hist_proj    = np.sum(self.pix_map['hist_cor'][pixels_valid], axis=0) / total_counts
         
        mus_name  = dataname + ' offset'
        gs_name   = dataname + ' gain'
        p_errors  = self.pix_map['e'][pixels_valid]
        m_sort    = np.argsort(p_errors)
        mus       = self.pix_map['mu']['v'][pixels_valid]
        gs        = self.pix_map['g']['v'][pixels_valid]
        hists     = self.pix_map['hist'][pixels_valid]
        hists_cor = self.pix_map['hist_cor'][pixels_valid]
        
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
        curve_fit = hplot.plot(self.hist_fit(self.pix_map[pixels_valid][m], self.Xs), pen = (0, 255, 0))
        hplot.setXLink('f')
        def replot():
            m = hline.value()
            m = m_sort[m]
            title = "histogram pixel " + str(m) + ' error ' + str(int(p_errors[m])) + ' offset {0:.1f}'.format(mus[m]) + ' inv. gain {0:.1f}'.format(gs[m])
            hplot.setTitle(title)
            curve_his.setData(hists[m])
            curve_fit.setData(self.hist_fit(self.pix_map[pixels_valid][m], self.Xs))
        
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

        counts = np.sum(self.pix_map['hist'][pixels_valid], axis=-1)
        ns     = self.pix_map['n']['v'][pixels_valid]
        cplots = []
        for c in range(self.Xs.shape[0]):
            cplots.append(win.addPlot(title=self.Xs[c]['name'] + ' counts: ' + str(int(np.sum(counts * ns[:, c]))), name = self.Xs[c]['name']))
            cplots[-1].plot(ns[m_sort][:, c],  pen=(c, self.Xs.shape[0]+1))
            cplots[-1].setXLink('mus')

        sys.exit(app.exec_())


    def dump_h5(self, fnam):
        if rank == 0 :
            f = h5py.File(fnam, 'w')
            f.create_dataset('pix_map', data=self.pix_map)
            f.create_dataset('Xs'     , data=self.Xs)
            f.create_dataset('errors' , data=np.array(self.errors))
            g = f.create_group('hist_pixels')
            for d in self.datas : 
                g.create_dataset(d['name'] + ' pixels', data=d['histograms'])
                g.create_dataset(d['name'] + ' comment', data=d['comment'])
            f.close()


    def load_h5(self, fnam):
        if rank == 0 :
            f = h5py.File(fnam, 'r')
            self.pix_map = f['pix_map'].value
            self.Xs      = f['Xs'].value
            self.errors  = list(f['errors'].value)
            #
            # get the dataset pixels name and comment
            g          = f['hist_pixels']
            self.datas = []
            data       = {}
            for k in g.keys():
                if k.find('pixels') != -1 :
                    data['name']       = k[: k.find('pixels')-1]
                    data['histograms'] = g[k].value
                    data['comment']    = g[data['name'] + ' comment'].value
                    self.datas.append(data)
                    data = {}
            f.close()


    def mask_bad_pixels(self, error_thresh=None, sigma=None):

        if error_thresh is not None :
            mask_pix = np.where(self.pix_map['e']>error_thresh)[0]
            print '\n ',rank,'masking', len(mask_pix),'pixels...'
            self.pix_map['valid'][mask_pix]      = False
            self.pix_map['n']['up'][mask_pix, :] = False
            self.pix_map['g']['up'][mask_pix]    = False
            self.pix_map['mu']['up'][mask_pix]   = False
        
        elif sigma is not None :
            errors  = comm.gather(self.pix_map['e'], root=0)
            
            if rank == 0 : 
                errors = np.concatenate( tuple(errors) )
                print errors.shape, errors.dtype
                sig    = np.std(errors)
                mean   = np.mean(errors)
            else :
                sig, mean = None, None
            
            sig, mean = comm.bcast([sig, mean], root=0)
            
            mask_pix = np.where((self.pix_map['e'] - mean) > sigma * sig)[0]
             
            print '\n ',rank,'masking', len(mask_pix),'pixels, sig, mean, thresh', sig, mean, sigma * sig + mean
            self.pix_map['valid'][mask_pix]      = False
            self.pix_map['n']['up'][mask_pix, :] = False
            self.pix_map['g']['up'][mask_pix]    = False
            self.pix_map['mu']['up'][mask_pix]   = False


def load_display(fnam):
    H = Histograms(fnam_h5 = fnam)
    H.show()


def chunkIt(seq, num):
    splits = np.mgrid[0:len(seq):(num+1)*1J].astype(np.int)
    out    = []
    for i in range(splits.shape[0]-1):
        out.append(seq[splits[i]:splits[i+1]])
    return out


def grid_condition_boundaries(err, A, b, bounds, N=10, iters=10):
    """
    find the minimum of err(X) such that:
    A . X = b
    bounds[i][0] <= Xi <= bounds[i][1]
    by grid searching.

    A = list or 1d array of length N
    b = scalar
    bounds = list (N) of tuples or lists (2) 
    err must take an array of X values of length N
    e.g. err(np.array([X0, X1, ... XN-1]))
    
    -set X0 = b - Aj . Xj , j>0
    -make a grid of values for Xj in the bounds
    -manually evaluate that the bounds are satisfied at each grid point
    -find minimum of err at these points
    -then zoom and repeat
    """
    # zoom
    for n in range(iters):
        if n == 0 :
            dom   = []
            steps = []
            for bound in bounds[1 :]:
                x, step = np.linspace(bound[0], bound[1], N, endpoint=True, retstep=True)
                dom.append(x)
                steps.append(step)
        else :
            # make the domain within the new bounds
            steps_old = list(steps)
            dom   = []
            steps = []
            for step_old, xmin, bound in zip(steps_old, X[1 :], bounds[1 :]):
                x, step = np.linspace(max([xmin-step_old, bound[0]]), min([xmin+step_old, bound[1]]), N, endpoint=True, retstep=True)
                dom.append(x)
                steps.append(step)
        
        # make the grid:
        if len(dom) > 1 :
            grid_X = np.meshgrid(*dom, copy=False, indexing='ij')
        else :
            grid_X = dom

        # evaluate X0 
        if A[0] <= 0.0 :
            print '\n Warning A[0] <= 0.0, setting to zero'
            return [0.0 for i in range(len(A))]
        
        X0  = (b - np.sum( [A[i+1] * grid_X[i] for i in range(0, len(A)-1)], axis=0)) / A[0]

        # evaluate the mask 
        mask = (X0 > bounds[0][0]) * (X0 < bounds[0][1])
        if not np.any(mask):
            print 'Error: A . X not == b for any values inside the bounds...'

        # this generates a list of N dimensional X coords [X0, X1, X2, ..., XN-1, M]
        # and the mask
        grid_X = [X0] + grid_X + [mask]
        errors = []
        for xs in zip(*[X.ravel() for X in grid_X]):
            if xs[-1]:
                errors.append(err(np.array(xs[:-1])))
            else :
                errors.append(np.inf)

        i = np.argmin(errors)
        i = np.unravel_index(i, dims=X0.shape)
        X = [grid_X[j][i] for j in range(len(A))]
    return X


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

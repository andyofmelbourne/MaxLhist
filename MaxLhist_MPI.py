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
        if rank == 0 :
            vars, M, I, V = self.check_input(datas)
        else :
            M = I = V = 0

        comm.barrier()
        M, I, V = comm.bcast([M, I, V], root=0)

        # set the numpy dtypes
        dt_n  = np.dtype([('v', np.float128, (V,)), ('up', np.bool, (V,))]) 
        dt_g  = np.dtype([('v', np.float32), ('up', np.bool)]) 
        dt_pm = np.dtype([('pix', np.int64), ('hist', np.uint64, (I,)), ('hist_cor', np.float128, (I,)),\
                          ('g', dt_g), ('mu', dt_g), ('n', dt_n), ('valid', np.bool), ('m', np.float128), ('e', np.float128)])
        dt_Xs = np.dtype([('v', np.float128, (I,)), ('up', np.bool, (I,))])
        self.dt_pm = dt_pm
        self.dt_Xs = dt_Xs
        
        if rank == 0 :
            pix_map, Xs  = self.process_input(datas, vars, M, I, V)
            pix_map, Xs  = self.init_input(datas, vars, pix_map, Xs)
            
            self.pix_map = chunkIt(pix_map, size)
            self.Xs      = Xs
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
            pix_map['hist'][start : start + hist.shape[0], :] = hist
            
            # inverse gain
            if d['gain']['value'] is not None :
                pix_map['g']['v'][start : start + hist.shape[0]] = d['gain']['value']
            
            if d['gain']['update'] == True :
                pix_map['g']['up'][start : start + hist.shape[0]][:] = True
            
            # adu offsets
            if d['offset']['value'] is not None :
                pix_map['mu']['v'][start : start + hist.shape[0]] = d['offset']['value']
            
            if d['offset']['update'] == True :
                pix_map['mu']['up'][start : start + hist.shape[0]][:] = True
            
            # errant counts
            counts_perpix = np.sum(hist, axis=-1)
            max          = np.max(counts_perpix)
            sig           = 0.1 * max
            bad_pix       = np.where(np.abs(counts_perpix - max) > sig)[0] 
            
            if len(bad_pix) > 0 :
                print '\n Warning: found', len(bad_pix), 'pixels with counts more than 10% from the max', int(max-sig), int(max+sig)
                print 'masking...'
                pix_map['valid'][start + bad_pix] = False
                        
            
            # counts
            if d['counts']['value'] is not None :
                if d['counts']['value'].shape[0] == 1 :
                    varno = np.where( [id(var) == id(d['vars'][0]) for var in vars] )[0][0]
                    
                    pix_map['n']['v'][start : start + hist.shape[0], varno][:] = 1.
                    pix_map['n']['up'][start : start + hist.shape[0], :]       = False
                else :
                    for v in range(d['counts']['value'].shape[0]):
                        # which var number is this count ?
                        varno = np.where( [id(var) == id(d['vars'][v]) for var in vars] )[0][0]
                        
                        ns = d['counts']['value'][v].astype(np.float128)
                        ms = np.where(pix_map['valid'][start : start + hist.shape[0]])
                        pix_map['n']['v'][start : start + hist.shape[0], varno][ms] = ns[ms] / counts_perpix.astype(np.float128)[ms]
                        
                    if d['counts']['update'] :
                        pix_map['n']['up'][start : start + hist.shape[0], :] = True
            
            start += hist.shape[0]

        pix_map['mu']['up'][np.where(pix_map['valid'] == False)]      = False
        pix_map['g']['up'][np.where(pix_map['valid'] == False)]       = False
        pix_map['n']['up'][np.where(pix_map['valid'] == False)[0], :] = False
        
        for v in range(len(vars)) :
            if vars[v]['function']['value'] is not None :
                print '\n setting', vars[v]['name'], 'var number',v,'to the input value'
                Xs[v]['v'][:]  = vars[v]['function']['value'].astype(np.float128)
            
            if vars[v]['function'].has_key('adus') :
                print '\n applying the adu mask of', vars[v]['name'], 'var number',v,'to the input value'
                Xs[v]['up'][:]  = vars[v]['function']['adus']
            else :
                Xs[v]['up'][:]  = True
        
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
                pix_map['n']['v'][start : start + hist.shape[0], varno].fill(1.)
            
            start += hist.shape[0]

        # the adu distributions
        for v in range(len(vars)) :
            if vars[v]['function']['value'] is None :
                print '\n setting', vars[v]['name'],' to the sum of the histograms'
                Xs[v]['v'] = f 
        return pix_map, Xs

    def unshift_ungain(self, pix_map):
        if rank == 0 : print '\n calcuating the unshifted and ungained histograms...'
        i = np.arange(pix_map['hist'].shape[1])
        M = float(pix_map['hist'].shape[0])
        
        for m, p in enumerate(pix_map):
            if rank == 0 : update_progress(float(m + 1) / M)
             
            if p['valid'] :
                pix_map[m]['hist_cor'][:] = np.interp((i + p['mu']['v'])/p['g']['v'], i, p['hist']) / p['g']['v'] 
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
        i = np.arange(pix_map['hist'].shape[1])
        M = float(pix_map['hist'].shape[0])
        
        for m, p in enumerate(pix_map):
            if rank == 0 : update_progress(float(m + 1) / M)
            
            if p['valid'] :
                # do not evaluate for pixels where hist == 0
                Is = np.where(p['hist'] > prob_tol)
                
                # evaluate the shifted gained probability function
                f = np.sum( Xs['v'] * p['n']['v'][:, np.newaxis], axis=0)
                f = np.interp(i*p['g']['v'] - p['mu']['v'], i, f.astype(np.float64)) * p['g']['v'] 
                
                # sum the log liklihood errors for this pixel
                e  = p['hist'][Is] * np.log(prob_tol + f[Is])
                
                pix_map[m]['e'] = - np.sum(e) - p['m']
                if pix_map[m]['e'] < 0.0 :
                    print 'Error: pixel', p['pix'], 'has a negative log likelihood (prob > 1). sum(f)', np.sum(f)
                    sys.exit()
            else :
                pix_map[m]['e'] = 0.0
        
        return pix_map
        
    def update_counts(self):
        pass
    def update_gain_offsets(self):
        pass
    def update_Xs(self):
        pass
    def show(self):
        pass

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

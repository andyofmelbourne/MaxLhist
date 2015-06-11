import MaxLhist 
import forward_model as fm
import h5py
from scipy.ndimage import gaussian_filter1d
import numpy as np
import utils as ut
import scipy

processes = 4

# test data
M = 100
N = 3000
I = 250

# 3 random variables
#-------------------
"""
hists, mus, gs, ns, Xv = fm.forward_model_nvars(I=250, M=M, N=N, V=3, sigmas = [5., 7., 9.], \
                                                pos = [100, 120, 150], sigma_mu = 0., sigma_g = 0.0, \
                                                mus=None, ns=None, gs=None, processes = processes)
"""

# 2 random variables
#-------------------
hists, mus, gs, ns, Xv = fm.forward_model_nvars(I=I, M=M, N=N, V=2, sigmas = [5., 7.], \
                                                pos = [100, 130], sigma_mu = 2., sigma_g = 0.1, \
                                                mus=None, ns=None, gs=None, processes = processes)


counts = ns * np.sum(hists, axis=1)


hists2, mus2, gs2, ns2, Xv2 = fm.forward_model_nvars(I=I, M=M, N=N, V=1, sigmas = [5.], \
                                                     pos = [100], sigma_mu = 0., sigma_g = 0.0, \
                                                     mus=mus, ns=None, gs=gs, processes = processes)
Xv = np.array(Xv)
Xv_downsample = np.zeros( (Xv.shape[0], I), dtype=Xv.dtype)
for v in range(len(Xv)) :
    i     = np.arange(I)
    Xv_downsample[v] = np.interp(np.linspace(0, Xv.shape[1]-1, I), np.arange(Xv.shape[1]), Xv[v])
    Xv_downsample[v] = Xv_downsample[v] / np.sum(Xv_downsample[v])

Xv = Xv_downsample

# initial guess
I = 250
i      = np.arange(0, I, 1)
f = np.exp( - (i - 100.).astype(np.float64)**2 / (2. * 20.) )
f = f / np.sum(f)
b = f.copy()

f = np.exp( - (i - 120.).astype(np.float64)**2 / (2. * 20.) )
f = f / np.sum(f)
s = f.copy()

f = np.exp( - (i - 150.).astype(np.float64)**2 / (2. * 20.) )
f = f / np.sum(f)
d = f.copy()


# Random variables
#-----------------
background = {
        'name'      : 'electronic noise',
        'type'      : 'random variable',
        'function'  : {'update': True, 'value' : b},
        #'function'  : {'update': False, 'value' : Xv[0]},
        }

sPhoton = {
        'name'      : 'single photon',
        'type'      : 'random variable',
        'function'  : {'update': True, 'value' : s, 'smooth' : 0.},
        #'function'  : {'update': False, 'value' : Xv[1]},
        }

dPhoton = {
        'name'      : 'double photon',
        'type'      : 'random variable',
        'function'  : {'update': True, 'value' : d},
        #'function'  : {'update': False, 'value' : Xv[2]},
        }

# data
#-----
data2 = {
        'name'       : 'dark run',
        'histograms' : hists2,
        'vars'       : [background], 
        'offset'     : {'update': True, 'value' : None},
        'gain'       : {'update': True, 'value' : None},
        'comment'    : 'testing the X update'
        }

data = {
        'name'       : 'run',
        'histograms' : hists,
        'vars'       : [background, sPhoton], 
        'offset'     : data2['offset'],
        'gain'       : data2['gain'],
        'counts'     : {'update': True, 'value' : None},
        'comment'    : 'testing the X update'
        }

# Retrieve
#---------
result = MaxLhist.refine_seq([data2, data], iterations=3, processes = processes)

print 'fidelity counts :' , np.sum((counts[1:] - result.result['run']['counts'][1:])**2)/np.sum(counts[1:]**2)
print 'fidelity gain   :' , np.sum((gs - result.result['run']['gain'])**2)/np.sum(gs**2)
#print 'fidelity mus    :' , np.sum((mus - result.result['run']['offset'])**2)/np.sum(mus**2)
print 'rms      mus    :' , np.sqrt( np.mean( (mus - result.result['run']['offset'])**2 ) )
for v in result.vars :
    for i in range(Xv.shape[0]) :
        print 'fidelity ', v['name'], ' Xv ', np.sum((v['function']['value'] - Xv[i])**2)/np.sum(Xv[i]**2)
    for X in [b,s] :
        print 'init ', v['name'], ' Xv ', np.sum((v['function']['value'] - X)**2)/np.sum(X**2)

result.show_fit('run', hists)

"""
datas = result.datas
vars  = result.vars

M = datas[0]['histograms'].shape[0]
D = len(datas)
V = len(vars)
I = datas[0]['histograms'].shape[1]
counts = np.zeros((V, M * D), dtype=np.float64)
X      = np.zeros((V, I), dtype=np.float64)

for d in range(0, D):
    if d == 0 :
        hist = ut.ungain_unshift_hist(datas[0]['histograms'], datas[0]['offset']['value'], datas[0]['gain']['value'], processes = processes)
    else : 
        hist = np.concatenate((hist, ut.ungain_unshift_hist(datas[d]['histograms'], datas[d]['offset']['value'], datas[d]['gain']['value'], processes = processes)))
    
    # fill the counts for datas vars
    for v in range(V):
        X[v, :] = vars[v]['function']['value']
        #i = np.where([vars[v] is vt for vt in datas[d]['vars']])
        i = [u for u in range(len(datas[d]['vars'])) if vars[v] is datas[d]['vars'][u]]
        for j in i:
            counts[j, d * M : (d+1) * M] = datas[d]['counts']['value'][v]


ns             = counts / np.sum(hist, axis=-1)
    

Xv_out = np.zeros_like(Xv)
for j in range(hist.shape[1]):
    print j
    ms = np.where(hist[:, j] > 0)[0]
    if len(ms) > 0 :
"""
"""
        gi = len(ms) * np.sum(hist[ms, j]) / np.sum(hist[ms])
        N0 = np.sum(ns[0, ms])
        N1 = np.sum(ns[1, ms])

        bounds_x1 = [( np.max( [0, (gi-N0)/N1] ) , np.min( [1, gi/N1] ) )] 

        def error(X1):
            error = - np.sum( hist[ms, j] * np.log( ns[0, ms] * gi / N0 + X1 * (ns[1, ms] - ns[0, ms] * N1 / N0) + 1.0e-10) )
            return error

        def grad_error(X1):
            error = - np.sum( hist[ms, j] * (ns[1, ms] - ns[0, ms] * N1 / N0) / (ns[0, ms] * gi / N0 + X1 * (ns[1, ms] - ns[0, ms] * N1 / N0) + 1.0e-10) )
            return error

        xs = np.arange(bounds_x1[0][0], bounds_x1[0][1], 0.001)
        errors = np.array([error(x) for x in xs])
        Xv_out[1, j] = xs[np.argmin(errors)]
        Xv_out[0, j] = (gi - N1 * Xv_out[1, j]) / N0
"""
"""
        gi = hist.shape[0] * np.sum(hist[:, j]) / np.sum(hist)
        N0 = np.sum(ns[0, :])
        N1 = np.sum(ns[1, :])

        bounds_x1 = [( np.max( [0, (gi-N0)/N1] ) , np.min( [1, gi/N1] ) )] 

        def error(X1):
            error = - np.sum( hist[ms, j] * np.log( ns[0, ms] * gi / N0 + X1 * (ns[1, ms] - ns[0, ms] * N1 / N0) + 1.0e-10) )
            return error

        def grad_error(X1):
            error = - np.sum( hist[ms, j] * (ns[1, ms] - ns[0, ms] * N1 / N0) / (ns[0, ms] * gi / N0 + X1 * (ns[1, ms] - ns[0, ms] * N1 / N0) + 1.0e-10) )
            return error

        xs = np.arange(bounds_x1[0][0], bounds_x1[0][1], 0.001)
        errors = np.array([error(x) for x in xs])
        Xv_out[1, j] = xs[np.argmin(errors)]
        Xv_out[0, j] = (gi - N1 * Xv_out[1, j]) / N0

"""


"""
print 'Xj god, Xj solved      :', Xv[0][j], Xv[1][j], Xj_out

error_god = fun_graderror_test((hj, total_counts, counts_m, update_vs, nupdate_vs, np.array([Xv[0][j], Xv[1][j]]), ns, bounds, j))
error_res = fun_graderror_test((hj, total_counts, counts_m, update_vs, nupdate_vs, Xj_out, ns, bounds, j))
print 'Error Xj god, Xj solved, res:', error_god, error_res, res.fun
"""



"""
def update_counts_brute(h, Xv, Nv, g, mu):
    hgm = ut.roll_real(h, -mu)
    hgm = ut.gain(hgm, 1 / g)
    
    # we don't need the zeros...
    Is = np.where(hgm >= 1)[0]
    hgm = hgm[Is]
    Xv2 = np.array(Xv)[:, Is]
    
    def fun(ns):
        ns2 = np.concatenate( ([ 1 - np.sum(ns)], ns) )
        error = - np.sum( hgm * np.log( np.sum( ns2[:, np.newaxis] * Xv2, axis=0) + 1.0e-10))
        return error
    
    ns0    = Nv / float(np.sum(Nv))
    bounds = ((0, 1.), (0, 1.))
    
    res    = scipy.optimize.minimize(fun, ns0[1:], bounds=bounds)
    print res
    Nv_out = np.concatenate( ([ 1 - np.sum(res.x)], res.x) ) * float(np.sum(Nv))
    return Nv_out

def update_counts_pix_odd_V(h, Xv, Nv, g, mu):
    hgm = ut.roll_real(h, -mu)
    hgm = ut.gain(hgm, 1 / g)

    # we don't need the zeros...
    Is = np.where(hgm >= 1)[0]
    hgm = hgm[Is]
    
    V   = len(Xv)
    Yv  = np.fft.rfftn(np.array(Xv)[:, Is], axes=(0, )).conj() / V
    def fi(Nh):
        Nh_c  = Nh[: (V-1) / 2] + 1J * Nh[(V-1) / 2 :]
        out   = Yv[0].real + 2 * np.sum( Nh_c[:, np.newaxis] * Yv[1 :], axis=0).real
        #out[np.where(out < 0)] = 0
        return out
    
    def fun(Nh):
        error = - np.sum( hgm * np.log( fi(Nh) + 1.0))
        return error
    
    def fprime(Nh):
        out = - 2 * np.sum( hgm * Yv[1 :] / (fi(Nh) + 1.0e-10), axis=1)
        out = np.concatenate( (out.real, out.imag))
        return out

    # initial guess, need to fft then map to real
    N = np.sum(h)
    Nh_r   = np.fft.rfft( Nv / float(N) )[1 :]
    Nh_r   = np.concatenate( (Nh_r.real, Nh_r.imag) )
    
    def gtzero(x, n=0):
        Nh     = x[: (V-1) / 2] + 1J * x[(V-1) / 2 :]
        Nh     = np.concatenate( ( [1], Nh ) )
        Nv_out = np.fft.irfft( Nh, V ) 
        return Nv_out[n]
    
    cons = ({'type': 'ineq', 'fun': lambda x: gtzero(x, 0)}, 
            {'type': 'ineq', 'fun': lambda x: gtzero(x, 1)},
            {'type': 'ineq', 'fun': lambda x: gtzero(x, 2)})
    #res    = scipy.optimize.minimize(fun, Nh_r, jac=fprime, constraints=cons)
    res    = scipy.optimize.minimize(fun, Nh_r, constraints=cons)
    #print res
    
    # solution, need to map to complex then ifft
    Nh     = res.x[: (V-1) / 2] + 1J * res.x[(V-1) / 2 :]
    Nh     = np.concatenate( ( [1], Nh ) )
    Nv_out = np.fft.irfft( Nh, V ) 
    #print res.x.shape, Nh.shape, Nv_out.shape, V, np.array(Xv).shape, np.array(Xv)[:, Is].shape, Yv.shape
    return Nv_out * N


M   = hists.shape[0]

for m in range(5):

    Xv = [var['function']['value'] for var in data['vars']]
    h  = hists[m]
    N  = np.sum(h)
    Nv = data['counts']['value'][:, m] 
    g  = gs[m]
    mu = mus[m]
    #x = update_counts_pix_odd_V(h, Xv, counts[:, m], g, mu)
    Nv0 = np.array([N/3., N/3., N/3.])
    x = update_counts_brute(h, Xv, Nv0, g, mu)
    print x, counts[:, m]

hgm = ut.roll_real(h, -mu)
hgm = ut.gain(hgm, 1 / g)

# we don't need the zeros...
Is = np.where(hgm >= 1)[0]
#Is = np.arange(h.shape[0])
hgm = hgm[Is]

V   = len(Xv)
Yv  = np.fft.rfftn(np.array(Xv)[:, Is], axes=(0, )).conj() / V
print Yv.shape, V
def fi(Nh):
    Nh_c  = Nh[: (V-1) / 2] + 1J * Nh[(V-1) / 2 :]
    out   = Yv[0].real + 2 * np.sum( Nh_c[:, np.newaxis] * Yv[1 :], axis=0).real
    print 'fi:', out.shape, Nh_c
    out[np.where(out < 0)] = 0
    return out

def fun(Nh):
    error = - np.sum( hgm * np.log( fi(Nh) + 1.0e-10))
    return error

def fprime(Nh):
    out = - 2 * np.sum( hgm * Yv[1 :] / (fi(Nh) + 1.0e-10), axis=1)
    out = np.concatenate( (out.real, out.imag))
    print 'fprime:', out.shape
    return out

N = float(np.sum(h))
Nh_r   = np.fft.rfft( Nv / float(N) )[1 :]
Nh_r   = np.concatenate( (Nh_r.real, Nh_r.imag) )
    
errors = []
for nx in np.arange(-1, 1, 0.01):
    for ny in np.arange(-1, 1, 0.01):
        errors.append(fun(np.array([nx, ny])))
errors = np.array(errors).reshape( (np.sqrt(len(errors)), np.sqrt(len(errors))) )


total_counts_v = np.sum(counts, axis=-1)
total_counts   = np.sum(total_counts_v)

print 'ungaining and unshifting the historgam...'
hist_adj = np.zeros_like(hists, dtype=np.float64)
for m in range(hists.shape[0]):
    hist_adj[m] = ut.roll_real(hists[m].astype(np.float64), -mus[m])
    hist_adj[m] = ut.gain(hist_adj[m], 1. / gs[m]) #/ total_counts

Xout = []
ress = []
for j in range(hists.shape[-1]):
#for j in [100]:
    hj     = hist_adj[:, j] 
    ms     = np.where(hj*total_counts>1)
    if np.sum(hj[ms]*total_counts) > 0 :
        def fun_logerror(Xvs):
            fj    = np.sum(ns.T * Xvs, axis=-1)
            error = -np.sum(hj * np.log(fj+1.0e-10))
            return error
        def fun_graderror(Xvs):
            fj    = np.sum(ns.T * Xvs, axis=-1)
            error = 0.0
            for v in range(len(Xvs)):
                #error += (total_counts_v[v]/total_counts - np.sum(ns[v][ms] * hj[ms] / (fj[ms]+1.0e-10)))**2
                error += (total_counts_v[v] - np.sum(ns[v][ms] * hj[ms] / (fj[ms]+1.0e-10)))**2
            return error
        def fun_grad(Xvs):
            fj    = np.sum(ns.T * Xvs, axis=-1)
            error = np.zeros_like(Xvs)
            for v in range(len(Xvs)):
                error[v] = total_counts_v[v]/total_counts - np.sum(ns[v][ms] * hj[ms] / (fj[ms]+1.0e-10))
            return error

        Xvs_0 = np.array([b[j], s[j]])
        res = scipy.optimize.minimize(fun_graderror, Xvs_0, method='L-BFGS-B', bounds=[(0, 1.0),(0, 1.0)]\
                ,options = {'gtol' : 1.0e-10, 'ftol' : 1.0e-10})
        res['pixel'] = j
        print 'optimising adu value:', j, res.fun
        ress.append(res)
        Xout.append(res.x)
    else :
        Xout.append([0.0, 0.0])
Xout = np.array(Xout)

for v in range(Xout.shape[1]):
    Xout[:,v] = gaussian_filter1d(Xout[:,v], 2.)

Xout2 = []
ress = []
for j in range(hists.shape[-1]):
#for j in [100]:
    hj     = hist_adj[:, j] 
    ms     = np.where(hj*total_counts>1)
    if np.sum(hj[ms]*total_counts) > 0 :
        def fun_logerror(Xvs):
            fj    = np.sum(ns.T * Xvs, axis=-1)
            error = -np.sum(hj * np.log(fj+1.0e-10))
            return error
        def fun_graderror(Xvs):
            fj    = np.sum(ns.T * Xvs, axis=-1)
            error = 0.0
            for v in range(len(Xvs)):
                #error += (total_counts_v[v]/total_counts - np.sum(ns[v][ms] * hj[ms] / (fj[ms]+1.0e-10)))**2
                error += (total_counts_v[v] - np.sum(ns[v][ms] * hj[ms] / (fj[ms]+1.0e-10)))**2
            return error
        def fun_grad(Xvs):
            fj    = np.sum(ns.T * Xvs, axis=-1)
            error = np.zeros_like(Xvs)
            for v in range(len(Xvs)):
                error[v] = total_counts_v[v]/total_counts - np.sum(ns[v][ms] * hj[ms] / (fj[ms]+1.0e-10))
            return error

        Xvs_0 = np.array([Xout[j,0], Xout[j, 1]])
        res = scipy.optimize.minimize(fun_graderror, Xvs_0, method='L-BFGS-B', bounds=[(0, 1.0),(0, 1.0)]\
                ,options = {'gtol' : 1.0e-10, 'ftol' : 1.0e-10})
        res['pixel'] = j
        print 'optimising adu value:', j, res.fun
        ress.append(res)
        Xout2.append(res.x)
    else :
        Xout2.append([0.0, 0.0])
"""

#result.dump_to_h5(fnam = 'example/darkcal.h5')



"""
# fidelity
hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 1000, sigma_f = 5., sigma_mu = 20., size = 1000)
hists, M, I, mus, nms, D, S = fm.forward_model_twovars(I = I, M = M, sigma_d = 5., sigma_s = 10., ds = 10., sigma_nm = 0.1, sigma_mu = 20., size = 1000, mus = mus_god, nms = np.ones((M))*0.4 )
i_range = np.arange(I)
d_god  = D.pmf(i_range)
s_god  = S.pmf(i_range)

print 'fidelity noise  :', np.sum(np.abs(d_god - result.result['electronic noise']['function'])**2) / np.sum(np.abs(d_god)**2)
print 'fidelity sPhoton:', np.sum(np.abs(s_god - result.result['single photon']['function'])**2) / np.sum(np.abs(s_god)**2)
"""











import MaxLhist 
import forward_model as fm
import h5py
from scipy.ndimage import gaussian_filter1d
import numpy as np
import utils as ut
import scipy

# test data
M = 1000
N = 2000

hists, mus, gs, ns, Xv = fm.forward_model_nvars(I=250, M=M, N=N, V=3, sigmas = [5., 7., 9.], pos = [100, 120, 150], sigma_mu = 10., sigma_g = 0.1, mus=None, ns=None, gs=None)
counts = ns * np.sum(hists, axis=1)

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

"""
hists, mus, gs, ns, Xv = fm.forward_model_nvars(I=250, M=1000, N=1000, V=2, sigmas = [5., 7.], pos = [100, 120], sigma_mu = 10., sigma_g = 0.1, mus=None, ns=None, gs=None)
counts = ns * np.sum(hists, axis=1)
"""

hists2, mus2, gs2, ns2, Xv2 = fm.forward_model_nvars(I=250, M=M, N=N, V=1, sigmas = [5.], pos = [100], sigma_mu = 0., sigma_g = 0.0, mus=mus, ns=None, gs=gs)

# Random variables
#-----------------
background = {
        'name'      : 'electronic noise',
        'type'      : 'random variable',
        'function'  : {'update': True, 'value' : b},
        }

sPhoton = {
        'name'      : 'single photon',
        'type'      : 'random variable',
        'function'  : {'update': True, 'value' : s},
        }

dPhoton = {
        'name'      : 'double photon',
        'type'      : 'random variable',
        'function'  : {'update': True, 'value' : d},
        }

# data
#-----
data2 = {
        'name'       : 'dark run',
        'histograms' : hists2,
        'vars'       : [background], 
        'offset'     : {'update': True, 'value' : None},
        'gain'       : {'update': True, 'value' : np.ones_like(gs)},
        'comment'    : 'testing the X update'
        }

data = {
        'name'       : 'run',
        'histograms' : hists,
        'vars'       : [background, sPhoton, dPhoton], 
        'offset'     : data2['offset'],
        'gain'       : data2['gain'],
        'counts'     : {'update': True, 'value' : None},
        'comment'    : 'testing the X update'
        }

# Retrieve
#---------
result = MaxLhist.refine([data2, data], iterations=3)

print 'fidelity counts :' , np.sum((counts[1:] - result.result['run']['counts'][1:])**2)/np.sum(counts[1:]**2)
print 'fidelity gain   :' , np.sum((gs - result.result['run']['gain'])**2)/np.sum(gs**2)
print 'fidelity mus    :' , np.sum((mus - result.result['run']['offset'])**2)/np.sum(mus**2)

#result.show_fit('run', hists)
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











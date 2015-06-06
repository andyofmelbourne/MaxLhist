import MaxLhist 
import forward_model as fm
import h5py
from scipy.ndimage import gaussian_filter1d

# test data
hists, mus, gs, ns, Xv = fm.forward_model_nvars(I=250, M=1000, N=1000, V=3, sigmas = [5., 7., 9.], pos = [100, 120, 150], sigma_mu = 10., sigma_g = 0.2, mus=None, ns=None, gs=None)
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

hists2, mus, gs, ns2, Xv2 = fm.forward_model_nvars(I=250, M=1000, N=1000, V=1, sigmas = [5.], pos = [100], sigma_mu = 10., sigma_g = 0.1, mus=mus, ns=None, gs=gs)

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
        'gain'       : {'update': True, 'value' : None},
        'comment'    : 'testing the X update'
        }

data = {
        'name'       : 'run',
        'histograms' : hists,
        'vars'       : [background, sPhoton, dPhoton], 
        'offset'     : data2['offset'],
        'gain'       : data2['gain'],
        'counts'     : {'update': False, 'value' : counts},
        'comment'    : 'testing the X update'
        }

# Retrieve
#---------
result = MaxLhist.refine([data2, data], iterations=5)
result.show_fit('run', hists)

"""
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

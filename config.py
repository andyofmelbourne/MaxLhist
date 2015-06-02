# allowed types:
# offset          -- overall offset of a random variable 
# function        -- the shape of the random variable (adu distribution)
# random variable -- 

import MaxLhist 
import h5py

f    = h5py.File('example/dark_hist_example.h5', 'r')
hist = f['data'].value
f.close()

f     = h5py.File('example/sPhoton_hist_example.h5', 'r')
hist2 = f['data'].value
f.close()

counts = np.sum(hist2, axis=-1)

# Random variables
#-----------------
background = {
        'name'      : 'electronic noise',
        'type'      : 'random variable',
        'function'  : {'update': True, 'value' : None},
        }

sPhoton = {
        'name'      : 'single photon',
        'type'      : 'random variable',
        'function'  : {'update': True, 'value' : None},
        }

# data
#-----
data1 = {
        'name'       : 'dark run',
        'histograms' : hist,
        'vars'       : [background], 
        'offset'     : {'update': True, 'value' : None},
        'comment'    : 'example/dark_hist_example.h5'
        }

data2 = {
        'name'       : 'sPhoton run',
        'histograms' : hist2,
        'vars'       : [background, sPhoton], 
        'counts'     : [counts * 0.6, \
                        counts * 0.4],
        'offset'     : data1['offset'],
        'comment'    : 'example/sPhoton_hist_example.h5'
        }

# Retrieve
#---------
result = MaxLhist.refine([data1, data2], iterations=10)

result.dump_to_h5(fnam = 'example/darkcal.h5')

result.show_fit('sPhoton run', hist2)

"""
# test data
hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 1000, sigma_f = 5., sigma_mu = 20., size = 1000)
f     = h5py.File('example/dark_hist_example.h5', 'w')
f.create_dataset('data', data = hists)
f.close()

hists, M, I, mus, nms, D, S = fm.forward_model_twovars(I = I, M = M, sigma_d = 5., sigma_s = 7., ds = 15., sigma_nm = 0.1, sigma_mu = 20., size = 1000, mus = mus_god, nms = np.ones((M))*0.4 )
f     = h5py.File('example/sPhoton_hist_example.h5', 'w')
f.create_dataset('data', data = hists)
f.close()
"""

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

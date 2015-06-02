# allowed types:
# offset          -- overall offset of a random variable 
# function        -- the shape of the random variable (adu distribution)
# random variable -- 

import MaxLhist 
import h5py

f    = h5py.File('example/dark_hist_example.h5', 'r')
hist = f['data'].value
f.close()

f     = h5py.File('example/dark_hist2_example.h5', 'r')
hist2 = f['data'].value
f.close()

# Random variables
#-----------------
background = {
        'name'      : 'electronic noise',
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
        'name'       : 'dark run2',
        'histograms' : hist2,
        'vars'       : [background], 
        'offset'     : data1['offset'],
        'comment'    : 'example/dark_hist2_example.h5'
        }

# Retrieve
#---------
result = MaxLhist.refine([data1, data2], iterations=10)

result.dump_to_h5(fnam = 'example/darkcal.h5')

result.show_fit('dark run', hist)

"""
# test data
hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 1000, sigma_f = 5., sigma_mu = 20., size = 100)
f     = h5py.File('example/dark_hist_example.h5', 'w')
f.create_dataset('data', data = hists)
f.close()

hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 1000, sigma_f = 5., sigma_mu = 20., size = 100, mus=mus_god)
f     = h5py.File('example/dark_hist2_example.h5', 'w')
f.create_dataset('data', data = hists)
f.close()

# fidelity
hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 1000, sigma_f = 5., sigma_mu = 20., size = 200)
i_range = np.arange(I)
f_god  = F.pmf(i_range)

print 'fidelity:', np.sum(np.abs(f_god - result.result['electronic noise']['function'])**2) / np.sum(np.abs(f_god)**2)
"""

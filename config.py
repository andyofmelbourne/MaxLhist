# allowed types:
# offset          -- overall offset of a random variable 
# function        -- the shape of the random variable (adu distribution)
# random variable -- 

import MaxLhist 
import h5py

f    = h5py.File('example/dark_hist_example.h5', 'r')
hist = f['data'].value
f.close()

f    = h5py.File('example/single_photon_hist_example.h5', 'r')
hist2 = f['data'].value
f.close()

# Random variables
#-----------------
background = {
        'name'      : 'electronic noise',
        'type'      : 'random variable',
        'offset'    : {'update': True, 'init' : None},
        'function'  : {'update': True, 'init' : None},
        }

single_photon = {
        'name'      : 'single photon',
        'type'      : 'random variable',
        'offset'    : {'update': False, \
                       'init' : np.zeros((hist.shape[0], dtype=np.float64))},
        'function'  : {'update': True, 'init' : None},
        }

# data
#-----
data1 = {
        'name'       : 'dark run',
        'histograms' : hist,
        'vars'       : [background], 
        'comment'    : 'example/dark_hist_example.h5'
        }

data2 = {
        'name'       : 'single photon run',
        'histograms' : hist2,
        'vars'       : [background, single_photon], 
        'comment'    : 'example/single_photon_hist_example.h5'
        }

# Retrieve
#---------
result = MaxLhist.refine([data1, data2], iterations=10)

result.dump_to_h5(fnam = 'example/sPhoton.h5')

result.show_fit(data2)

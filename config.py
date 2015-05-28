# allowed types:
# offset          -- overall offset of a random variable 
# function        -- the shape of the random variable (adu distribution)
# random variable -- 

import MaxLhist 
import h5py

f    = h5py.File('example/dark_hist_example.h5', 'r')
hist = f['data'].value
f.close()

shifts = {
        'name'          : 'dark offset',
        'type'          : 'offset',
        'update'        : True,
        'initial guess' : None
        }

dark_dist = {
        'name'          : 'dark shape',
        'type'          : 'function',
        'update'        : True,
        'initial guess' : None
        }

background = {
        'name'      : 'electronic noise',
        'type'      : 'random variable',
        'update'    : True,
        'elements'  : [shifts, dark_dist]
        }

data0 = {
        'name'       : 'dark run',
        'histograms' : hist,
        'vars'       : [background]
        }

result = MaxLhist.refine([data0], iterations=10)

result.dump_to_h5(fnam = 'example/darkcal.h5')

result.show_fit('electronic noise', hist)

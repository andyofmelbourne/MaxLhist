# allowed types:
# offset          -- overall offset of a random variable 
# function        -- the shape of the random variable (adu distribution)
# random variable -- 

import MaxLhist 
import h5py

f    = h5py.File('example/dark_hist_example.h5', 'r')
hist = f['data'].value
f.close()

background = {
        'name'      : 'electronic noise',
        'type'      : 'random variable',
        'offset'    : {'update': True, 'init' : None},
        'function'  : {'update': True, 'init' : None},
        }

data = {
        'name'       : 'dark run',
        'histograms' : hist,
        'vars'       : [background], 
        'comment'    : 'example/dark_hist_example.h5'
        }

result = MaxLhist.refine([data], iterations=10)

result.dump_to_h5(fnam = 'example/darkcal.h5')

result.show_fit(data)

import numpy as np
import scipy.stats
import forward_model as fm
import utils as ut
from scipy.ndimage import gaussian_filter1d
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
rank = comm.rank

def get_uniqe(l):
    lout = []
    for i in l:
        if i not in lout :
            lout.append(l)
    return lout

def get_dims(datas):
    Xs = []
    I  = datas[0]['histograms'].shape[1]
    for d in datas:
        M += d['histograms'].shape[0]
        #
        for X in d['vars']:
            if X not in Xs:
                Xs.append(X)
    V = len(Xs)
    return M, I, V

class Histograms():

    def __init__(self, datas):
        M, I, V = get_dims(datas)
        dt_n  = np.dtype([('v', np.float128, (V,)), ('up', np.bool, (V,))]) 
        dt_g  = np.dtype([('v', np.float128), ('up', np.bool)]) 
        dt_pm = np.dtype([('pix', np.int64), ('hist', np.uint64, (I,)), ('hist_cor', np.float128, (I,)),\
                          ('g', dt_g), ('mu', dt_g), ('n', dt_n), ('valid', np.bool)])
        dt_Xs = np.dtype([('v', np.float128, (I,)), ('up', np.bool, (I,))])
        if rank == 0 :
            pixel_map, Xs = process_input(datas)


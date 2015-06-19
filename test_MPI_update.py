import MaxLhist_MPI
import forward_model as fm
import h5py
from scipy.ndimage import gaussian_filter1d
import numpy as np
import utils as ut
import scipy
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0 :
    # test data
    M = 4000
    N = 1000
    I = 250

    processes = 4

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
                                                    pos = [100, 130], sigma_mu = 20., sigma_g = 0.2, \
                                                    mus=None, ns=None, gs=None, processes = processes)


    counts = ns * np.sum(hists, axis=1)


    hists2, mus2, gs2, ns2, Xv2 = fm.forward_model_nvars(I=I, M=M/2, N=int(N/4.), V=1, sigmas = [5.], \
                                                         pos = [100], sigma_mu = 10., sigma_g = 0.15, \
                                                         mus=None, ns=None, gs=None, processes = processes)
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



    s_mask = np.zeros_like(s, dtype=np.bool)
    s_mask[100 : 150] = True 

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
            'function'  : {'update': True, 'value' : s_mask * s, \
                           'smooth' : 0., 'adus' : np.arange(100, 160, 1)},
            #'function'  : {'update': False, 'value' : Xv[1]},
            }

    dPhoton = {
            'name'      : 'double photon',
            'type'      : 'random variable',
            'function'  : {'update': True, 'value' : None},
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
            'counts'     : {'update': False, 'value' : None},
            'comment'    : 'testing the X update'
            }

    data = {
            'name'       : 'run',
            'histograms' : hists,
            'vars'       : [background, sPhoton], 
            'offset'     : {'update': True, 'value' : None},
            'gain'       : {'update': True, 'value' : None},
            'counts'     : {'update': True, 'value' : None},
            'comment'    : 'testing the X update'
            }
else :
    data2 = data = None

# Retrieve
#---------
H = MaxLhist_MPI.Histograms([data])

for i in range(10):
    H.update_counts()
    H.update_gain_offsets(quadfit=True)
    H.update_Xs()
    H.pixel_errors()
H.gather_pix_map()

if rank == 0 :
    pix = H.datas[0]['histograms']
    print ns.shape, H.pix_map['n']['v'].shape
    print 'fidelity ns     :' , np.sum((ns[1 :] - H.pix_map['n']['v'][pix, 1])**2)/np.sum(ns[1:]**2)
    print 'fidelity gain   :' , np.sum((gs - H.pix_map['g']['v'])**2)/np.sum(gs**2)
    print 'rms      mus    :' , np.sqrt( np.mean( (mus - H.pix_map['mu']['v'])**2 ) )
    
    for v in Xv :
        for i in range(len(H.Xs)) :
            print 'fidelity ', H.Xs[i]['name'], ' Xv ', np.sum((H.Xs[i]['v'] - Xv[i])**2)/np.sum(Xv[i]**2)
H.show()

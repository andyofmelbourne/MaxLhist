import numpy as np

def roll(a, x):
    """
    roll f by x pixels 
    uses np.fft.fft to roll the axis
    
    A_k   =     sum_n=0^N-1 a_n exp{-2 pi i nk / N}
    b_n   = 1/N sum_k=0^N-1 A_k exp{ 2 pi i nk / N} * exp{-2 pi x k_f}
    
    where k_f are the frequency values at the k'th pixel as evaluated by np.fft.fftfreq
    
    Parameters
    ----------
    a : numpy array
        array to be shifted
    x : scalar int or float or complex
        shift amount in pixel units

    Returns
    -------
    b : float numpy array
        Fourier shifted input array.
    """
    A    = np.fft.fft(a)
    ramp = np.exp(-2.0J * np.pi * x * np.fft.fftfreq(a.shape[0]))
    b    = np.fft.ifft(A * ramp)
    return b

def roll_real(a, x):
    """
    roll f by x pixels 
    uses np.fft.rfft to roll the axis
    
    A_k   =     sum_n=0^N-1 a_n exp{-2 pi i nk / N}
    b_n   = 1/N sum_k=0^N-1 A_k exp{ 2 pi i nk / N} * exp{-2 pi x k_f}
    
    where k_f are the frequency values at the k'th pixel as evaluated by np.fft.fftfreq
    
    Parameters
    ----------
    a : float numpy array
        array to be shifted
    x : scalar int or float 
        shift amount in pixel units

    Returns
    -------
    b : float numpy array
        Fourier shifted input array.
    """
    N    = a.shape[0]
    A    = np.fft.rfft(a)
    ramp = np.exp(-2.0J * np.pi * x * np.arange(N//2 + 1) / float(N))
    b    = np.fft.irfft(A * ramp)
    return b

def make_f_real(fhats, f_norm = 0.0):
    """
    Compute the real space representation of f given that 
    f is real and that sum_i f_i = f_norm. The fhats are 
    the complex Fourier space values of f from 1 to 
    N // 2 + 1 where N is the size of the real space array.
    
    Parameters
    ----------
    fhats : 1d complex array
        Fourier space values of f.
    f_norm : float, optional
        Sum of the real space function.
            
    Returns
    -------
    f : 1d float array
        the realspace representation of fhats given f_norm.
    """
    fh = np.concatenate( ([f_norm], fhats) )
    f  = np.fft.irfft(fh)
    return f

def shift_f_real(f, shift):
    """
    Apply the Fourier shift algorithm to f by shift pixels.
    
    Parameters
    ----------
    f : float array
        Real space values of f.
    shift : float
        The value in pixels of the shift amount.
            
    Returns
    -------
    f_shift : 1d float array
        The shifted representation of f.
    """
    fh      = np.fft.rfft(f)
    ramp    = np.exp(-2.0J * np.pi * shift * np.arange(float(fh.shape[0])) \
              / float(f.shape[0]))
    f_shift = np.fft.irfft(fh * ramp)
    return f_shift


def grad_shift_f_real(f, shift):
    """
    Apply the Fourier shift algorithm to f by shift pixels. 
    In addition take the gradient of f.
    
    Parameters
    ----------
    f : float array
        Real space values of f.
    shift : float
        The value in pixels of the shift amount.
            
    Returns
    -------
    f_shift : 1d float array
        The gradient of the shifted representation of f.
    """
    fh      = np.fft.rfft(f)
    lramp   = -2.0J * np.pi * np.arange(float(fh.shape[0])) \
              / float(f.shape[0])
    ramp    = lramp * np.exp(shift * lramp)
    f_shift = np.fft.irfft(fh * ramp)
    return f_shift


def log_likelihood_calc(f, mus, hists, prob_tol = 1.0e-10):
    """
    Calculate the log likelihood error given a probability distribution f,
    a set of shifts mus, the measured histograms for each shift hists.
    
    log likelihood error = - sum_m sum_I hists[m, I] * ln( f[i-mus[m]] )
        
    As the pixel shifts need not be integer f[i-mus[m]] is calculated
    using the Fourier shift theorem and uses the function roll_real.
    
    Parameters
    ----------
    f : float array
        Real space values of f of length I.
    mus : float array
        The value in pixels of the shift amount of length M.
    hists : 2 dimensional integer array
        The measured values sampled from f shifted by the mus.
        hists must have the shape (M, I).
    prob_tol : float, optional
        Amount to add to f to aviod -infinity in the natural logarithm.
            
    Returns
    -------
    log_likelihood_error : float
        The log likelihood error.
    """
    error = 0.0
    for m in range(len(hists)):
        # only look at adu or pixel values that were detected on this pixel
        Is = np.where(hists[m] > 0)
        
        # evaluate the shifted probability function
        fs = roll_real(f, mus[m])[Is] 

        # sum the log liklihood errors for this pixel
        e  = hists[m, Is] * np.log(prob_tol + fs)
        error += np.sum(e)
    return -error


def mu_transform(h):
    """
    Calculate a strange transform. So far only works
    when h.shape[0] is an even number (and h is 1D).
    Note that in the output n = 0 is excluded.
    
    g[n] = 1/M sum_m=0^M-1 h[m]                         for n = 0
    g[n] = 2/M sum_m=0^M-1 e^(-2 pi i m n / M) h[m]     for 0 < n < M/2
    g[n] = 1/M sum_m=0^M-1 (-1)^m h[m]                  for n = M/2
    
    Parameters
    ----------
    h : 1D float array
        Values of h of length M.
            
    Returns
    -------
    g : 1D complex array
        g for n = 1, ..., M/2, so with the shape M/2 
    """
    M = h.shape[0]
    if M % 2 == 1 :
        raise ValueError('input array shape must even for now')
    g = np.zeros((h.shape[0] / 2, ), dtype = np.complex128)
    
    #g[0]     = np.sum(h) / float(M)
    g[0 : -1] = np.fft.fft(h)[1 : M/2] * 2. / float(M)
    g[-1]     = np.sum(h * (-1)**np.arange(h.shape[0]) ) / float(M)
    return g

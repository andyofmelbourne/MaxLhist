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
        the realspace representation of fhats given f_norm.
    """
    fh      = np.fft.rfft(f)
    ramp    = np.exp(-2.0J * np.pi * shift * np.arange(float(fh.shape[0])) \
              / float(f.shape[0]))
    f_shift = np.fft.irfft(fh * ramp)
    return f_shift


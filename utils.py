import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d

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
    ramp    = lramp 
    if shift != 0 :
        ramp *= np.exp(shift * lramp)
    f_shift = np.fft.irfft(fh * ramp)
    return f_shift

def log_likelihood_calc(f, mus, hists, prob_tol = 1.0e-10, pixelwise = False):
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
    if pixelwise :
        error = np.zeros((hists.shape[0]), dtype=np.float64)
    else :
        error = 0.0
    for m in range(len(hists)):
        # only look at adu or pixel values that were detected on this pixel
        Is = np.where(hists[m] > 0)
        
        # evaluate the shifted probability function
        fs = roll_real(f, mus[m])[Is] 
        fs[np.where(fs < 0)] = 0.0

        # sum the log liklihood errors for this pixel
        e  = hists[m, Is] * np.log(prob_tol + fs)
        if pixelwise :
            error[m] = -np.sum(e)
        else :
            error -= np.sum(e)
    return error

def log_likelihood_calc_many(datas, prob_tol = 1.0e-10):
    error = 0.0
    for d in datas:
        hists = d['histograms']
        mus   = d['offset']['value']
        gs    = d['gain']['value'] 
        for m in range(len(hists)):
            f  = forward_fs(d, m)
            Is = np.where(hists[m] > 0)
            
            # evaluate the gained then shifted probability function
            f = gain(f, gs[m]) 
            f = roll_real(f, mus[m])[Is]
            f[np.where(f < 0)] = 0.0

            # sum the log liklihood errors for this pixel
            e  = hists[m, Is] * np.log(prob_tol + f)
            error += np.sum(e)
    return -error

def log_likelihood_calc_pixelwise(f, mus, hists, prob_tol = 1.0e-10):
    """
    """
    from scipy.special import gammaln
    error = np.zeros((hists.shape[0]), dtype=np.float64)
    for m in range(len(hists)):
        # only look at adu or pixel values that were detected on this pixel
        Is = np.where(hists[m] > 0)
        
        # evaluate the shifted probability function
        fs = roll_real(f, mus[m])[Is] 
        fs[np.where(fs < 0)] = 0.0

        # sum the log liklihood errors for this pixel
        e  = hists[m, Is] * np.log(prob_tol + fs)

        # calculate the combinatorial term
        c = gammaln(np.sum(hists[m]) + 1) - np.sum(gammaln(hists[m] + 1))
         
        error[m] = -np.sum(e) - c
    return error

def log_likelihood_calc_pixelwise_many(d, prob_tol = 1.0e-10):
    mus   = d['offset']['value']
    gs    = d['gain']['value']
    hists = d['histograms']
    
    from scipy.special import gammaln
    error = np.zeros((hists.shape[0]), dtype=np.float64)
    for m in range(len(hists)):
        f  = forward_fs(d, m)
        
        # only look at adu or pixel values that were detected on this pixel
        Is = np.where(hists[m] > 0)
        
        # evaluate the shifted probability function
        fs = gain(f, gs[m])
        fs = roll_real(f, mus[m])[Is] 
        fs[np.where(fs < 0)] = 0.0

        # sum the log liklihood errors for this pixel
        e  = hists[m, Is] * np.log(prob_tol + fs)

        # calculate the combinatorial term
        c = gammaln(np.sum(hists[m]) + 1) - np.sum(gammaln(hists[m] + 1))
         
        error[m] = -np.sum(e) - c
    return error

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

def mu_to_muextended(mu):
    """
    Converts the real vector mu into the real independent Fourier
    space variables, when the normalisation of mu is fixed and 
    mu.shape[0] is even. The first half + 1 are the real positive 
    Fourier components of mu while the second half - 1 are the
    imaginary positive Fourier components of mu.

    mu_extended = np.array([np.fft.rfft(mu).real[ 1 : ],  \
                            np.fft.rfft(mu).imag[ 1 : -1])

    Parameters
    ----------
    mu : 1D float array
        Values of mu of length N.
            
    Returns
    -------
    mu_extended : 1D float array
        Values of the "independent" Fourier components of mu when the 
        normalisation of mu is fixed. Of length N - 1.
    """
    muhat       = np.fft.rfft(mu)
    mu_extended = np.concatenate((muhat.real[1 :], muhat.imag[1 : -1]))
    return mu_extended

def muextended_to_mu(mu_extended, norm = 0.0):
    """
    Inverse of mu_to_muextended

    Parameters
    ----------
    mu_extended : 1D float array
        Values of the "independent" Fourier components of mu when the 
        normalisation of mu is fixed. Of length N - 1.

    norm : float
        The normalisation of mu = np.sum(mu).
            
    Returns
    -------
    mu: 1D float array
        Values of mu of length N.
    """
    N                 = mu_extended.shape[0] + 1
    muhat             = np.empty( (N / 2 + 1,) , dtype=np.complex128)
    muhat[0]          = norm
    muhat[1 :]        = mu_extended[: N / 2]
    muhat[1 : -1].imag= mu_extended[N/2 : ]
    mu                = np.fft.irfft(muhat)
    return mu

def mu_bisection(f_alpha, min_step = 1.):
    # find a and b
    a = 0.0
    b = min_step
    fa = f_alpha(a)
    while fa * f_alpha(b) > 0 :
        b = 2 * b
    x0, r = scipy.optimize.bisect(f_alpha, a, b, xtol=1e-2, rtol=4.4408920985006262e-16, maxiter=100, full_output=True, disp=True)
    return x0

def mu_directional_derivative(d, f, muhat, hists, prob_tol = 1.0e-10):
    jacobian = jacobian_mus_calc(f, muhat, hists, prob_tol = 1.0e-10)
    return np.sum((jacobian * np.conj(d)).real)

def f_directional_derivative(d, f, mus, hists, prob_tol = 1.0e-10):
    jacobian = jacobian_fs_calc(f, mus, hists, prob_tol = 1.0e-10)
    return np.sum((jacobian * np.conj(d)).real)

def jacobian_mus_calc(f, mushat, hists, prob_tol = 1.0e-10):
    mus = make_f_real(mushat)
    # this could be more efficient
    f_prime_shift = np.zeros_like(hists, dtype=np.float64)
    for m in range(len(mus)) :
        f_prime_shift[m] = grad_shift_f_real(f, mus[m]) / (prob_tol + roll_real(f, mus[m]))
    
    temp  = hists * f_prime_shift 
    temp  = np.sum(temp, axis=1)
    mus_f = mu_transform(temp)
    return -mus_f

def jacobian_fs_calc(f, mus, hists, prob_tol = 1.0e-10):
    h = np.zeros( f.shape, dtype=f.dtype)
    for m in range(len(mus)) :
        h += roll_real(hists[m].astype(np.float64), - mus[m])
    
    # generate a mask for the gradient vector
    mask = (h >= 1.0)
    
    # calculate the intial gradient (regardless of normalisation / zeros)
    grad = - h / (prob_tol + f)
    
    #print '\n\n\n grad:'
    #print grad * mask

    grad = - grad * mask
    """
    # normalise
    grad = mask * (grad - np.sum(mask * grad) / float(np.sum(mask)))
    #print 'gradient sum', np.sum(grad)
    print '\n\n\n grad:'
    print grad * mask
    
    import sys
    sys.exit()
    """
    return grad

def gain(f, g):
    i     = np.arange(f.shape[0])
    fout  = np.interp(g*i, i, f)
    # preserve normalisation
    f_sum = np.sum(f)
    if np.abs(f_sum) < 1.0e-10 :
        fout = fout - np.sum(fout)
    else :
        fout  = fout * f_sum / np.sum(fout)
    return fout

# update
def update_fs(f0, mus, hists):
    """
    Follow the line of steepest descent.
        mus_i = mus_i-1 - 0.5 * grad_calc(mus_i-1)
    """
    f = f0.copy()

    h = np.zeros( f.shape, dtype=f.dtype)
    for m in range(len(mus)) :
        h += roll_real(hists[m].astype(np.float64), - mus[m])
    # generate a mask for the gradient vector
    mask = (h >= 1.0)
    f = mask * h
    f = f / np.sum(f)
    
    return f

def update_fs_many(var, ds):
    """
    """
    print 'updating ', var['name']
    f = var['function']['value'].copy()
    
    h = np.zeros( f.shape, dtype=f.dtype)
    for d in ds:
        hist   = d['histograms']
        mus    = d['offset']['value']
        vi     = np.where([var is vt for vt in d['vars']])[0][0]
        for m in range(len(mus)) :
            h_m = hist[m].astype(np.float64)
            
            # un-shift and un-gain
            h_m = roll_real(h_m, - mus[m])
            h_m = gain(h_m, 1./d['gain']['value'][m])
            
            ns  = d['counts']['value'][vi][m] / np.sum(h_m)
            h  += ns * h_m
        
    # apply a mask to aviod unconstrained values
    mask = (h >= 1.0)
    f    = mask * h
    f    = f / np.sum(f)
    
    X = float(hist.shape[0] * len(ds)) * f
    # subtract the other stuff
    for d in ds:
        hist   = d['histograms']
        mus    = d['offset']['value']
        for v in d['vars']:
            if v is not var:
                vi = np.where([v is vt for vt in d['vars']])[0][0]
                n  = 0.0
                for m in range(len(mus)) :
                    n  += d['counts']['value'][vi][m] / np.sum(hist[m].astype(np.float64))
                
                X -= n * v['function']['value']
    
    Is = np.where(X < 0)
    if Is[0].shape[0] > 0 :
        print 'warning: ', var['name'] + "'s function had zeros in the update to the sum of ", np.sum(X[Is]) ,". Masking..."
        X[Is] = 0.0
    X = X / np.sum(X)
    return X

def ungain_unshift_hist(hists, mus, gs):
    hist_adj = np.zeros_like(hists, dtype=np.float64)
    for m in range(hists.shape[0]):
        hist_adj[m] = roll_real(hists[m].astype(np.float64), -mus[m])
        hist_adj[m] = gain(hist_adj[m], 1. / gs[m]) #/ total_counts
    return hist_adj

def update_fs_new(vars, datas, normalise = True, smooth = 0):

    
    # join the histograms and counts into a big histogram thing
    M = datas[0]['histograms'].shape[0]
    D = len(datas)
    V = len(vars)
    I = datas[0]['histograms'].shape[1]
    counts = np.zeros((V, M * D), dtype=np.float64)
    X      = np.zeros((V, I), dtype=np.float64)

    update_vs  = np.where([v['function']['update'] for v in vars])[0]
    nupdate_vs = np.where([not v['function']['update'] for v in vars])[0]
    if len(update_vs) == 0 :
        for v in range(V):
            X[v, :] = vars[v]['function']['value']
        return X
    
    for d in range(0, D):
        if d == 0 :
            hist   = ungain_unshift_hist(datas[0]['histograms'], datas[0]['offset']['value'], datas[0]['gain']['value'])
        else : 
            hist = np.concatenate((hist, ungain_unshift_hist(datas[d]['histograms'], datas[d]['offset']['value'], datas[d]['gain']['value'])))
        
        # fill the counts for datas vars
        for v in range(V):
            X[v, :] = vars[v]['function']['value']
            i = np.where([vars[v] is vt for vt in datas[d]['vars']])
            if i[0].shape[0] > 0 :
                i = i[0][0]
                counts[i, d * M : (d+1) * M] = datas[d]['counts']['value'][v]

    total_counts_v = np.sum(counts, axis=-1)
    total_counts   = np.sum(total_counts_v)
    ns             = counts / np.sum(hist, axis=-1)
    
    print hist.shape, ns.shape, X.shape
    for j in range(hist.shape[-1]):
        hj     = hist[:, j] 
        ms     = np.where(hj*total_counts>1)
        Xj     = X[nupdate_vs, j]
        if np.sum(hj[ms]*total_counts) > 0 :
            def fun_graderror(Xvs):
                fj    = np.sum(ns[update_vs, :].T * Xvs, axis=-1)
                fj   += np.sum(ns[nupdate_vs, :].T * Xj, axis=-1)
                error = 0.0
                for v in range(len(Xvs)):
                    error += (total_counts_v[v] - np.sum(ns[v][ms] * hj[ms] / (fj[ms]+1.0e-10)))**2
                return error
            
            Xvs_0 = X[update_vs, j]
            res   = scipy.optimize.minimize(fun_graderror, Xvs_0, method='L-BFGS-B', bounds=[(0, 1.0),(0, 1.0)]\
                    ,options = {'gtol' : 1.0e-10, 'ftol' : 1.0e-10})
            res['pixel'] = j
            print 'optimising adu value:', j, res.fun
            X[update_vs, j] = res.x
        else :
            X[update_vs, j] = res.x

    if smooth != 0 :
        for v in range(V):
            X[v, :] = gaussian_filter1d(X[v], smooth)

    if normalise :
        for v in range(V):
            X[v, :] = X[v, :] / np.sum(X[v, :])
    return X


def update_mus(f, mus0, hists, padd_factor = 1, normalise = True, quadfit = True):
    mus = mus0.copy()

    # calculate the cross-correlation of hists and F
    cor   = np.fft.rfftn(hists.astype(np.float64), axes=(-1, ))
    cor   = cor * np.conj(np.fft.rfft(np.log(f + 1.0e-10)))

    # interpolation factor
    padd_factor = 1
    if padd_factor != 0 and padd_factor != 1 :
        cor   = np.concatenate((cor, np.zeros( (cor.shape[0], cor.shape[1] -1), dtype=cor.dtype)), axis=-1)
        check = np.concatenate((check, np.zeros( (check.shape[0], check.shape[1] -1), dtype=check.dtype)), axis=-1)

    cor  = np.fft.irfftn(cor, axes=(-1, ))

    mus = []
    fftfreq = cor.shape[1] * np.fft.fftfreq(cor.shape[1])
    for m in range(cor.shape[0]):
        mu_0 = np.argmax(cor[m])
        # map to shift coord
        if quadfit :
            mus_0 = np.array([mu_0-1, mu_0, mu_0 + 1]) % cor.shape[1]
            mus_t = fftfreq[mus_0]
            vs    = cor[m][mus_0]
            #print mu_0-1 , mu_0, mu_0 + 2
            #print mus_t
            #print vs
            p     = np.polyfit(mus_t, vs, 2)
            # evaluate the maximum
            mu = - p[1] / (2. * p[0])
            if (mu > mus_t[0]) and (mu < mus_t[-1]) and (p[0] < 0) :
                pass
            else :
                print 'quadratic fit failed', mu_0, mu, mus_t #p
                mu = fftfreq[mu_0]
        else :
            mu = fftfreq[mu_0]
        mus.append(mu / float(padd_factor))
    
    mus = np.array(mus)
    
    if normalise :
        mus = mus - np.sum(mus)/float(len(mus))
     
    return mus

def update_mus_many(offset, ds, normalise = True, quadfit = True):
    """
    """
    mus = offset['value'].copy()

    # calculate the cross-correlation of hists and F
    hshape = ds[0]['histograms'].shape
    if hshape[-1] % 2 == 0 :
        cor_shape = (hshape[1] / 2 + 1, )
    else :
        cor_shape = ((hshape[1]+1) / 2, )
    cor_t = np.zeros(cor_shape, dtype=np.complex128)
    
    fftfreq = hshape[1] * np.fft.fftfreq(hshape[1])
    for m in range(hshape[0]):
        
        # calculate the cross-correlation bw hist and f
        cor_t.fill(0.0)
        for d in ds:
            # generate the forward model of the distributions for this dataset but without the shifts
            f_i = forward_fs(d, m)
            
            temp   = np.fft.rfft(d['histograms'][m].astype(np.float64))
            cor_t += temp * np.conj(np.fft.rfft(np.log(f_i + 1.0e-10)))
        cor  = np.fft.irfft(cor_t)
        
        mu_0 = np.argmax(cor)
        # map to shift coord
        if quadfit :
            mus_0 = np.array([mu_0-1, mu_0, mu_0 + 1]) % cor.shape[0]
            mus_t = fftfreq[mus_0]
            vs    = cor[mus_0]
            p     = np.polyfit(mus_t, vs, 2)
            # evaluate the maximum
            mu = - p[1] / (2. * p[0])
            if (mu > mus_t[0]) and (mu < mus_t[-1]) and (p[0] < 0) :
                pass
            else :
                print 'quadratic fit failed', mu_0, mu, mus_t #p
                mu = fftfreq[mu_0]
        else :
            mu = fftfreq[mu_0]
        mus[m] = mu 
    
    if normalise :
        mus = mus - np.sum(mus)/float(len(mus))
     
    return mus

def forward_fs(data, m):
    f = np.zeros( (data['histograms'].shape[1]), dtype=np.float64)
    for n in range(len(data['vars'])) :
        f += data['counts']['value'][n][m] * data['vars'][n]['function']['value']
    return f / np.sum(f)


def update_mus_gain(ds, normalise = True, quadfit = True):
    M   = ds[0]['histograms'].shape[0]
    mus = np.zeros_like(ds[0]['offset']['value'])
    gs  = np.zeros_like(ds[0]['gain']['value'])
    for m in range(M):
        fms    = [forward_fs(d, m) for d in ds]
        hms    = [d['histograms'][m] for d in ds]
        mu, g  = update_mus_gain_pix(hms, fms, quadfit = quadfit)
        mus[m] = mu
        gs[m]  = g
    
    if normalise :
        mus = mus - np.sum(mus)/float(len(mus))
        gs  = gs / np.mean(gs)
    
    return mus, gs

def update_mus_not_gain(ds, normalise = True, quadfit = True):
    M   = ds[0]['histograms'].shape[0]
    mus = np.zeros_like(ds[0]['offset']['value'])
    gs  = ds[0]['gain']['value']
    for m in range(M):
        fms    = [forward_fs(d, m) for d in ds]
        hms    = [d['histograms'][m] for d in ds]
        mu     = update_mus_not_gain_pix(hms, fms, gs[m], quadfit = quadfit)
        mus[m] = mu
    
    if normalise :
        mus = mus - np.sum(mus)/float(len(mus))
    
    return mus

def update_mus_not_gain_pix(hms, fms, g, quadfit = True):
    hms_hat = np.array([np.fft.rfft(hm.astype(np.float64)) for hm in hms])
    fftfreq = fms[0].shape[0] * np.fft.fftfreq(fms[0].shape[0])

    fms_hat = np.array([np.fft.rfft(np.log(gain(f, g) + 1.0e-10)) for f in fms])
    fms_hat = np.conj(fms_hat)
    
    cor = np.fft.irfft( np.sum( fms_hat * hms_hat, axis=0) )
    
    mu_0     = np.argmax(cor)
    cor_max0 = cor[mu_0]
    
    if quadfit :
        # map to shift coord
        mus_0 = np.array([mu_0-1, mu_0, mu_0 + 1]) % cor.shape[0]
        mus_t = fftfreq[mus_0]
        vs    = cor[mus_0]
        p     = np.polyfit(mus_t, vs, 2)
        # evaluate the new arg maximum 
        mu      = - p[1] / (2. * p[0])
        if (mu > mus_t[0]) and (mu < mus_t[-1]) and (p[0] < 0) :
            pass
        else :
            print 'quadratic fit failed', mu_0, mu, mus_t, p
            mu      = fftfreq[mu_0]
    else :
        mu      = fftfreq[mu_0]
    return mu
    

def update_mus_gain_pix(hms, fms, quadfit = True):
    """    
    calculate the minimum of e(g) = - max [sum_data h .conv. ln(f(g*i))]
    then return g and i
    """    
    hms_hat = np.array([np.fft.rfft(hm.astype(np.float64)) for hm in hms])
    fftfreq = fms[0].shape[0] * np.fft.fftfreq(fms[0].shape[0])
    
    def error_quadfit_fun(g, return_mu = False):
        fms_hat = np.array([np.fft.rfft(np.log(gain(f, g) + 1.0e-10)) for f in fms])
        fms_hat = np.conj(fms_hat)
        
        cor = np.fft.irfft( np.sum( fms_hat * hms_hat, axis=0) )
        
        mu_0     = np.argmax(cor)
        cor_max0 = cor[mu_0]
        
        # map to shift coord
        mus_0 = np.array([mu_0-1, mu_0, mu_0 + 1]) % cor.shape[0]
        mus_t = fftfreq[mus_0]
        vs    = cor[mus_0]
        p     = np.polyfit(mus_t, vs, 2)
        # evaluate the new arg maximum 
        mu      = - p[1] / (2. * p[0])
        if (mu > mus_t[0]) and (mu < mus_t[-1]) and (p[0] < 0) :
            cor_max = p[0] * mu**2 + p[1] * mu + p[2]
        else :
            print 'quadratic fit failed', mu_0, mu, mus_t, cor.shape
            mu      = fftfreq[mu_0]
            cor_max = cor_max0
        
        # keep mu, which we get for free kind of
        if return_mu :
            return -cor_max, mu
        else :
            return -cor_max
    
    def error_fun(g, return_mu = False):
        fms_hat = np.array([np.fft.rfft(np.log(gain(f, g) + 1.0e-10)) for f in fms])
        fms_hat = np.conj(fms_hat)
        
        cor = np.fft.irfft( np.sum( fms_hat * hms_hat, axis=0) )
        
        # keep mu, which we get for free kind of
        if return_mu :
            mu = np.argmax(cor)
            mu = fftfreq[mu]
            return -np.max(cor), mu
        else :
            return -np.max(cor)
        
    if quadfit :
        errorf = error_quadfit_fun
    else :
        errorf = error_fun
    
    res   = scipy.optimize.minimize_scalar(errorf, method='bounded', bounds=(0.5, 1.5))
    g     = res.x
    e, mu = errorf(g, True)
    return mu, g


import numpy as np

from cgls.python_scripts import cgls_nonlinear

class CGLS_background_from_histogram(cgls_nonlinear.Cgls):
    
    def __init__(self, x0, mus0, hists):
        """
        Solve for the dark background probability distribution given an array of histograms.

        x0 = the initial guess for the probablity distribution of the background.
             must be a normalised numpy array of floats of shape (I,) (I is the adu range)
        
        mus0 = the initial guess for the mean of the probability distribution at each pixel
               must be a numpy array of integers of shape (N,) (N is the number of pixels)

        hists = the histograms of the pixels 
                must be a numpy array of integers of shape (N, I)
        """
        # my state variable
        x = np.array([mus0, x0], dtype=np.float64)
        
    def f(self, x):
        """
        error function (must take x as its argument)
        """
        error = log_likelihood_calc(x[1], x[0], self.data)
        return error

    def df(self, x, smooth = 2.):
        """
        gradient of the error function with respect to x [mu, adus]
        """
        fprime = np.gradient(x[1])

        out = np.zeros_like(x)
        out[0] = np.gradient(x[0])
        out[1] = np.gradient(x[1])

        if smooth is not False :
            # smooth the gradient
            out[0] = np.convolve(out[0], np.ones((smooth), dtype=np.float)/ float(smooth), mode='same')
            out[1] = np.convolve(out[1], np.ones((smooth), dtype=np.float)/ float(smooth), mode='same')
        return out 

    def fd(self, x):
        """
        x . Jacobian of the error function with respect to x
        """
        return

    def dfd(self, x):
        """
        xT . Hessian of the error function with respect to x . x
        """
        pass

# Jacobian
def jacobian_mus_calc(f, mus, hists, prob_tol = 1.0e-10, continuity = 2.):
    out = np.zeros_like(mus)
    
    # calculate the derivative of f
    fprime = np.gradient(f)
    if continuity != 0 :
        fprime = np.convolve(fprime, np.ones((continuity), dtype=np.float)/ float(continuity), mode='same')
    
    Ns, Is = np.where(hists > 0)
    for n in Ns :
        fs      = f[Is - np.rint(mus[n])]
        fprimes = fprime[Is - np.rint(mus[n])]
        
        out[n] = np.sum(hists[n] * fprimes / (fs + prob_tol))
    return out
    
def jacobian_f_calc(f, mus, hists, prob_tol = 1.0e-10):
    out = np.zeros_like(f)
    
    arcf = 1 / (f + prob_tol)
    Ns, Is = np.where(hists > 0)
    for n in Ns :
        out += np.sum(hists[n] * arcf[Is - np.rint(mus[n])])
    return out


# error metric
def log_likelihood_calc(f, mus, hists, prob_tol = 1.0e-10):
    error = 0.0
    Ns, Is = np.where(hists > 0)
    for n in Ns :
        fs = f[Is - np.rint(mus[n])]
        e = hists[n] * np.log(prob_tol + fs)
        error += np.sum(e)
    return -error

# generate a random variable
x = np.arange(-100, 150, 1)
f = np.exp( - x.astype(np.float64)**2 / (2. * 5.**2)) 
f = f / np.sum(f)

mu = np.exp( - x.astype(np.float64)**2 / (2. * 20.**2)) 
mu = mu / np.sum(mu)

from scipy import stats
X   = stats.rv_discrete(name='background', values = (x, f))
MU  = stats.rv_discrete(name='mu', values = (x, mu))

# make some histograms
counts = 100
N = 10
hists = []
mus = []
for n in range(N):
    mu = MU.rvs(1)
    f  = X.rvs(size = counts)
    hist, bins = np.histogram(f, bins = x)
    hists.append(np.roll(hist, mu))
    mus.append(mu)

hists = np.array(hists)

f0   = hists[0].astype(np.float64)
f0   = f0 / np.sum(f0)
mus0 = np.ones_like(mus) * float(np.argmax(f0))

cgls = cgls_nonlinear.Cgls(f0, error_calc


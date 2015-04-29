import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

from MaxLhist import utils as ut
from MaxLhist import forward_model as fm
from MaxLhist.retrieve_mus import jacobian_mus_manual_calc

#from cgls.cgls.line_search import line_search_secant

import matplotlib.pyplot as plt
import scipy
import numpy as np


def test_jacobian_mus_calc(f, mushat, hists, prob_tol = 1.0e-10):
    mus = ut.make_f_real(mushat)
    # this could be more efficient
    f_prime_shift = np.zeros_like(hists, dtype=np.float64)
    for m in range(len(mus)) :
        f_prime_shift[m] = ut.grad_shift_f_real(f, mus[m]) / (prob_tol + ut.roll_real(f, mus[m]))
    
    temp  = hists * f_prime_shift 
    return temp

def jacobian_mus_calc(f, mushat, hists, prob_tol = 1.0e-10):
    mus = ut.make_f_real(mushat)
    # this could be more efficient
    f_prime_shift = np.zeros_like(hists, dtype=np.float64)
    for m in range(len(mus)) :
        f_prime_shift[m] = ut.grad_shift_f_real(f, mus[m]) / (prob_tol + ut.roll_real(f, mus[m]))
    
    temp  = hists * f_prime_shift 
    temp  = np.sum(temp, axis=1)
    mus_f = ut.mu_transform(temp)
    return -mus_f

def jacobian_mushat_manual_calc(f, mushat, hists, error_func, step = 0.1):
    """
    Calculate d error / d mus using the central difference algorithm:
    d error / d mus_i = [error(mus_i + step) - error(mus_i - step)] / (2 * step)
    REF:
        http://en.wikipedia.org/wiki/Numerical_differentiation
        and
        http://en.wikipedia.org/wiki/Finite_difference_coefficient
    """
    J_mus = np.zeros_like(mushat, dtype=np.float64)
    for i in range(len(mushat)):
        mushat_temp     = mushat
        mushat_temp[i] += step
        mus_temp        = ut.make_f_real(mushat_temp)
        e1 = error_func(f, mus_temp, hists)
        
        mushat_temp     = mushat
        mushat_temp[i] -= step
        mus_temp        = ut.make_f_real(mushat_temp)
        e2 = error_func(f, mus_temp, hists)
        
        d = (e1 - e2) / (2. * step)
        
        J_mus[i] = d
    return J_mus

#hists, M, I, mus_god, F = fm.forward_model(I = 250, M = 10, sigma_f = 5., sigma_mu = 20.)
M = 2
I = 250
sigma_f = 5.

# the "adu" range
i      = np.arange(0, I, 1)
i_bins = np.arange(0, I+1, 1)

# the probability function or "background"
f = np.exp( - (i - 100).astype(np.float64)**2 / (2. * sigma_f**2)) 
f = f / np.sum(f)
F = scipy.stats.rv_discrete(name='background', values = (i, f))

# make some histograms
hists = []
mus = np.array([-20., 20.])
for n in range(M):
    mu = mus[n]
    
    # create a new random variable with the shifted background
    f_shift = ut.roll_real(f, mu)
    F_shift = scipy.stats.rv_discrete(name='background', values = (i, f_shift))
    ff = F_shift.rvs(size = 1000)
    hist, bins = np.histogram(ff, bins = i_bins)
    hists.append(hist)
    
hists = np.array(hists)

if False :
    # plot the histograms
    ax = plt.subplot(111)
    ax.bar(i, hists[0], alpha = 0.5, color='r', width=1.0, label='pixel 0')
    ax.bar(i, hists[1], alpha = 0.5, color='b', width=1.0, label='pixel 1')
    ax.set_xlabel('adus')
    ax.set_ylabel('frequency')
    ax.legend()
    plt.show()

mus_god = mus.copy()

# look at the error as a function of mhat1
errors = []
mhs = np.arange(100., -200, -0.1)
for mh in mhs:
    mus = ut.make_f_real([mh])
    errors.append(ut.log_likelihood_calc(f, mus, hists, prob_tol = 1.0e-10))
errors = np.array(errors)

# evaluate the Jacobian at mhat1 = 0 and -40:
mh         = np.array([-40.])
error_func = lambda x,y,z: ut.log_likelihood_calc(x, y, z, prob_tol = 1.0e-10)
de_dmu = jacobian_mus_calc(f, mh, hists, prob_tol = 1.0e-10)
# grad_line = mx + c
e         = error_func(f, ut.make_f_real(mh), hists)
grad_line = de_dmu * mhs + (e - de_dmu * mh)

mh         = np.array([0.])
de_dmu = jacobian_mus_calc(f, mh, hists, prob_tol = 1.0e-10)
# grad_line = mx + c
e         = error_func(f, ut.make_f_real(mh), hists)
grad_line_0 = de_dmu * mhs + (e - de_dmu * mh)

# 
thing = test_jacobian_mus_calc(f, mh, hists, prob_tol = 1.0e-10)
if False :
    ax = plt.subplot(111)
    ax.plot(i, thing[0], alpha = 0.8, color='k')
    ax.set_xlabel('adus')
    ax.set_ylabel('thing')
    plt.show()


# plot the line and the error
if False :
    ax = plt.subplot(111)
    ax.plot(mhs, errors, alpha = 0.8, color='k', label='log likelihood error')
    ax.plot(mhs, grad_line, alpha = 0.8, color='b', label='tangent line for mhat = 0')
    ax.plot(mhs, grad_line_0, alpha = 0.8, color='r', label='tangent line for mhat = -40')
    ax.set_xlabel('mhat')
    ax.set_ylabel('error')
    ax.set_ylim([errors.min() * 0.7, errors.max() * 1.1])
    ax.legend()
    plt.show()


mu_0    = np.array([0.0, 0.0], dtype=np.float64)
muhat_0 = np.fft.rfft(mu_0)[1 :]
mue_0   = ut.mu_to_muextended(mu_0)

# find the descent direction
d = jacobian_mus_calc(f, muhat_0, hists, prob_tol = 1.0e-10)
d = d / np.sqrt(np.sum(np.abs(d)**2))

# evaluate the directional derivative

d = np.array([1.0], dtype=np.complex128)

eprime_alpha = lambda alpha : ut.mu_directional_derivative(d, f, muhat_0 + d * alpha, hists, prob_tol = 1.0e-10)

alphas = np.arange(-200.0, 100.0, 1.0)
eprime_alphas = []

for a in alphas :
    eprime_alphas.append(eprime_alpha(a))
eprime_alphas = np.array(eprime_alphas)

# bisection
def mu_bisection(f_alpha, min_step = 1.):
    # find a and b
    a = 0.0
    b = min_step
    fa = f_alpha(a)
    while fa * f_alpha(b) > 0 :
        b = 2 * b
        print b
    x0, r = scipy.optimize.bisect(f_alpha, a, b, xtol=1e-2, rtol=4.4408920985006262e-16, maxiter=100, full_output=True, disp=True)
    return x0

# run the bisection line minimisation algorithm
mu_0    = np.array([-50.0, 50.0], dtype=np.float64)
muhat_0 = np.fft.rfft(mu_0)[1 :]
mue_0   = ut.mu_to_muextended(mu_0)

# find the descent direction
d = -jacobian_mus_calc(f, muhat_0, hists, prob_tol = 1.0e-10)
d = d / np.sqrt(np.sum(np.abs(d)**2))

eprime_alpha = lambda alpha : mu_directional_derivative(d, f, muhat_0 + d * alpha, hists, prob_tol = 1.0e-10)

alpha0 = mu_bisection(eprime_alpha)
muhat_1 = muhat_0 + d * alpha0
mu_1    = ut.make_f_real(muhat_1)
print alpha0, muhat_0 + d * alpha0

# plot the line and the error
if True :
    ax = plt.subplot(211)
    ax.plot(mhs, errors, alpha = 0.8, color='k')
    ax.set_xlim([mhs.min(), mhs.max()])
    ax.set_xlabel('mhat')
    ax.set_ylabel('log likelihood error')

    ax.scatter(mue_0, ut.log_likelihood_calc(f, mu_0, hists, prob_tol = 1.0e-10))
    ax.scatter(muhat_1[0].real, ut.log_likelihood_calc(f, mu_1, hists, prob_tol = 1.0e-10))
    
    ax = plt.subplot(212)
    ax.plot(alphas + 100, eprime_alphas, alpha = 0.8, color='b')
    ax.set_xlim([(alphas+100).min(), (alphas+100).max()])
    ax.set_xlabel('alpha')
    ax.set_ylabel('directional derivative')

    ax.scatter(mue_0 + 100, eprime_alpha(0.0))
    ax.scatter(muhat_1[0].real + 100, eprime_alpha(alpha0))
    plt.show()


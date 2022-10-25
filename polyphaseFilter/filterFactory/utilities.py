import numpy as np
from scipy import signal
import scipy as sp
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
################################################################################
###############################Filter utilities:###############################
def checkOnAnalogAllpassFilter(b, a, verbose=False):
    '''
    This function checks the branch filter on validity.

    b:          Monoms of numerator polynomial (descending order)
    a:          Monoms of denominator polynomial (descending order)
    verbose:    Plot/Print additional information

    return:     -
    '''

    z = np.roots(b)
    assert np.all(z.real >= 0) == True, "The numerator roots must lie within the positive half-plane in the s domain!"

    p = np.roots(a)
    assert np.all(p.real <= 0) == True, "The denominator roots must lie within the negative half-plane in the s domain!"

    #check on allpass behaviour
    assert np.array_equal(-1*p.real,z.real) == True, "Branch filters must be allpass filters!"

    if verbose:
        sPlot(a,b)

def sPlot(a,b,filename=None):
    """
    Plot the complex s-domain of a given transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=1.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def allpass_s2z(a):
    '''
    This function transforms an analog allpass to a digital allpass. The returend values are monomes, which
    follow the following pattern:
    b = b[0] + b[1]*z**(-1)+...+ b[N]*z**(-N)
    a = a[0] + a[1]*z**(-1)+...+ a[N]*z**(-N)

    Parameters:
    ___________
    a:          denominator polynomial of analog allpass (descending order)
    b:          Numerator polynomial of analog allpass (descending order)

    return:
    _______
    b           Numerator polynomial in descending order
    a           Denominator polynomial in descending order
    '''
    #1.Step: Determine denominator polynomial
    #Transform denominator into first and second order factors
    P_n = np.poly1d(a)
    roots = P_n.r

    first_order_roots = roots[(roots.imag==0)]
    second_order_roots = roots[(roots.imag>0)]               #only extract one part of the conjugate pairs

    #Determine denominator polynomials of digital allpass
    N = len(first_order_roots) + len(second_order_roots)
    allpasses_a = np.zeros((N,3))                               #descending order

    for i, root in enumerate(first_order_roots):
        P = np.poly1d([root], r=True)                           #P[1]*s + P[0] = s + a1
        allpasses_a[i,1:] = [1, (P[0]-1)/(P[0]+1)]

    offset = len(first_order_roots)
    for i, root in enumerate(second_order_roots):

        P = np.poly1d([root, np.conj(root)], r=True)            #P[2]*s**2 + P[1]*s + P[0] = s**2 + a1*s + a2
        allpasses_a[offset + i, 0] = 1
        allpasses_a[offset + i, 1] = 2*(P[0] - 1)/(P[2] + P[1] + P[0])
        allpasses_a[offset + i, 2] = (1-P[1]+P[0])/(P[2] + P[1] + P[0])

    #determine overall branch filter (must be a digital allpass)
    P_n = np.poly1d(1)
    #print(P_n)
    for a in allpasses_a:
        P_n *= np.poly1d(a)

    a = P_n.c

    #2. Step: Determine numerator polynomial
    zeros = mirrorAtUnityCircle(P_n.r)
    b = np.poly1d(zeros, r=True).c

    #get unity gain
    k = a[-1]
    b *= k

    return b,a

def checkOnDigitalAllpassFilter(b, a, verbose=False):
    '''
    This function checks the given filter on allpass behaviour

    b:          Numerator polynomial in descending order (monoms)
    a:          denominator polynomial in descending order (monoms)
    verbose:    Prints/Plots extra information

    return:     -
    '''
    z = np.roots(b)
    assert np.all(np.abs(z) > 1) == True, "The numerator roots must lie outside of the unity circle in the z-domain!"

    p = np.roots(a)
    assert np.all(np.abs(p) < 1) == True, "The denominator roots must lie inside of the unity circle in the z-domain!"

    #check on allpass behaviour

    assert np.allclose(np.sort(np.round(p,2)), np.sort(np.round(mirrorAtUnityCircle(z),2)),atol=0.01) == True, '''Poles and zeros location
    do not match poles and zeros of a digital allpass filter!'''

    #check on conjugate pairs
    z_second_order = z[(z.imag!=0)]
    p_second_order = p[(z.imag!=0)]

    for zero in z_second_order:
        assert (zero in z_second_order)==True, "Poles and zeros location do not match poles and zeros of a digital allpass filter!"

    for pole in p_second_order:
        assert (pole in p_second_order)==True, "Poles and zeros location do not match poles and zeros of a digital allpass filter!"

    #plot filter
    if verbose:
        zPlot(b,a)

    return

def zPlot(b,a,filename=None):
    """Plot the complex z-plane of the given transfer function.
       Todo: Add line ticks

    b:      Numerator polynomial
    a:      denominator polynomial
    """

    # Get the poles and zeros
    b, a = negDes2posDes(b, a)
    z, p, k = signal.tf2zpk(b,a)

    # get a figure/plot
    fig = plt.figure(figsize=(10,7))
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)
    yAxis = ax.axvline(0, color='black')
    xAxis = ax.axhline(0, color='black')
    ax.axis('equal')

    # Plot the zeros and set marker properties
    plt.plot(z.real, z.imag, 'go', markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    plt.plot(p.real, p.imag, 'rx', markersize=12.0, markeredgewidth=1.0,
              markeredgecolor='r', markerfacecolor='r')


    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    return

def freqz_expanded(b,a,L, worN):
    '''
    This function returns the frequency response of the filter and its image
    frequency. The maximum frequency is which is shown in L*pi

    Parameters:
    ___________
    b:              Numerator of filter (descendig order)
    a:              Denominator of filter (descending order)
    L:              Factor which indicates how many multiples of pi should be
                    included in response
    worN:           Number of bins
    return:
    _______
    wBins:          Normalized circular frequency bins at which frequency reponse
                    is evaluated
    H:              Frequency response
    '''
    b_up = np.zeros((len(b)-1)*L+1)
    a_up = np.zeros((len(a)-1)*L+1)

    b_up[::L] = b
    a_up[::L] = a

    wBins, H = signal.freqz(b_up, a_up, worN=worN)

    return wBins, H

def negDes2posDes(b,a):
    N = len(b) if len(b) > len(a) else len(a)

    b_ = np.zeros((N,))
    a_ = np.zeros((N,))

    b_[:len(b)] = b
    a_[:len(a)] = a

    b_ = np.trim_zeros(b_, trim='f')
    a_ = np.trim_zeros(a_, trim='f')

    return b_, a_

def computationalComplexity(b,a):
    '''
    This functions returns the number of multiplications and additions per input
    sample. If ap_decompostion is True then the complexity is calculated on the
    basis of a filter which is decomposed into distinct allpass filters. Otherwise
    the complexity is deduced on the basis of the direct-form-1.

    Note: The filter should not contain any pure delay elments as these interfere
    the result.

    Parameter:
    __________
    b:                  Numerator polynomial (descending order)
    a:                  Denominator polynomial (descending order)

    return:
    _______
    Nm:             Number of multiplications per input sample
    Na:             Number of additions per input sample
    '''
    #Nm = number of multiplications, Na = number of additions

    Nm = len(b)
    Na = len(b) - 1

    if len(a) > 1:
        #IIR Filter
        Nm += len(a) - 1
        Na += len(a) - 1

    return Nm, Na

def computationalComplexity_polyphaseFilters(b,a):
    '''
    This function computes the overall computational complexity of a given poly-
    phase filter. Note: b,a are 2 dimensional arrays, where the rows denote the
    distinct branch filters and the columns denote the filter coefficients.

    Parameter:
    __________
    b:                  Numerator polynomials (descending order, negative powers)
    a:                  Denominator polynomials (descending order, negative powers)

    return:
    _______
    Nm:             Number of multiplications per input sample
    Na:             Number of additions per input sample
    '''
    L = np.shape(b)[0]              #number of branches
    Nm = 0
    Na = L

    for b_,a_ in zip(b,a):
        nm, na = computationalComplexity(b_, a_)
        Nm += nm
        Na += na

    return Nm, Na

def FIR_polyphase_filter(b,L):
    '''
    This function converts the given filter into a polyphase filter. The output
    sub filters are all equal in length. Some multipliers might be zeros.

    Parameters:
    __________
    b:                  Filter polynomial (descending order, negative powers)
    L:                  Decimation/Interpolation factor

    return:
    _______
    b:                  Polyphase filter polynomials (number of rows represent,
                        the number of branches, the columns define the distinct
                        branch filters)
    '''
    N = len(b)%L
    b = np.pad(b,(0,L-N),'constant', constant_values=(0, 0))

    Nb = len(b)
    mask = np.arange(0,Nb,1).reshape(Nb//L,L).T
    b = b[:,np.newaxis][mask][:,:,0]

    return b

def plot_FIR_polyphase_filter(b):
    L = np.shape(b)[0]

    worN = 512
    H = np.zeros((worN), dtype=complex)
    wBins = np.zeros((worN))

    for branch in range(L):
        b_ = np.zeros((L*len(b[branch,:]),))
        b_[::L] = b[branch,:]
        b_ = np.pad(b_,(branch,0),'constant',constant_values=(0,0))
        wBins, H_ = signal.freqz(b_,[1],worN)
        H += H_

    fig = plt.figure(figsize=(10,8))
    plt.plot(wBins, 20*np.log10(np.abs(H+0.000001)),label="FIR Halfband Filter")
    plt.legend()
    plt.show()

    return

def FIR_lowpass(wc, wd, fs):
    '''
    This functions return the filter taps for the given lowpass filter.

    Parameters:
    ___________
    wc:                     Cutoff Frequency (Normalized by Nyquist rate)
    wd:                     Transition width from pass to stop
                            band (Normalized by Nyquist rate)
    fs:                     Sampling frequency

    return:
    _______
    b:                          Filter taps
    '''

    fc = wc/2/np.pi*fs
    fd = wd/2/np.pi*fs

    numtaps = 600      # Size of the FIR filter.
    b = sp.signal.remez(numtaps, [0, fc, fc + fd, 0.5*fs], [1, 0], Hz=fs)

    return b

def FIR_filter_groupDelay(b, mode):
    '''
    This fuctions returns the group delay for the given FIR filter.

    Parameters:
    ___________
    b:                  Filter taps
    mode:               Averaging or averaging with weighting (av/avW)

    return:
    _______
    gd:                 GroupDelay of EMQF halfband filter [samples]
    '''

    wBins, H = sp.signal.freqz(b, [1])

    wc = wBins[(20*np.log10(np.abs(H))) < -3][0]

    H = H[(wBins<=wc)]
    wBins = wBins[(wBins<=wc)]

    phi = np.unwrap(np.angle(H))                #phase response
    H = np.abs(H)**2                            #transform response to squared magnitude

    supportedModes = ['av', 'avW']

    if mode in supportedModes == False:
        raise Exception("Mode not supported!")

    #Determine group delay
    gd = 0
    if mode == supportedModes[0]:
        #average in passband
        gd = -np.diff(phi)/np.diff(wBins)
        gd = np.mean(gd)
    elif mode == supportedModes[1]:
        gd = -np.diff(phi)/np.diff(wBins)
        gd = np.sum(gd*H[1:])/np.sum(H[1:])

    return gd
################################################################################
###########Utilities for getting Polynomial from Phase##########################
def alphaCoeffs4ArbitraryPhasePolynomial(phase, w, alpha, n):
    '''
    This function returns the alhpa coefficients which are necessary to de-
    duce a polynomial which specifies the phase response of an filter. The coefficients
    are calculated for given phase samples.

    phase:      Phase samples (used as specification)
    w:          Frequency bins of phase samples
    alpha:      Empty array, used for coefficient storage
    n:          Order - 1

    return:     Alpha coefficients

    Source: The Generation of Arbitrary-Phase Polynomials by Recurrence Formulae, Henk Támas
    '''
    if n >= 1:
        alpha = alphaCoeffs4ArbitraryPhasePolynomial(phase, w, alpha, n-1)
    else:
        alpha[0] = w[1] / np.tan(phase[1])
        return alpha
    #alpha coefficients for n > 0 are based on continued fractions:
    step = 0
    denominator = w[n+1]/np.tan(phase[n+1])        #last element
    while(n > step):
        denominator = (w[n+1]**2 - w[step+1]**2)/(alpha[step] - denominator)
        step +=1
    alpha[n] = denominator
    return alpha

def betaCoeffs4ArbitraryPhasePolynomial(phase, w, beta, n):
    '''
    This function returns the beta coefficients which are necessary for
    a polynomial which specifies the phase response of an filter. The polynomial
    is deduced by the recurrence formula of T. Henk. The coefficients are calculated
    for given phase samples.

    phase:      Phase samples (used as specification)
    w:          Frequency bins of phase samples
    alpha:      Empty array, used for coefficient storage
    n:          Depth of recurrence formula

    return:     Beta-coefficients

    Source: The Generation of Arbitrary-Phase Polynomials by Recurrence Formulae, Henk Támas
    '''
    if n >= 1:
        beta = betaCoeffs4ArbitraryPhasePolynomial(phase, w, beta, n-1)
    else:
        #beta "root" element
        beta[0] = np.tan(phase[1])/w[1]
        return beta
    step = 0
    denominator = 1 - beta[0]*w[n+1]/np.tan(phase[n+1])
    step += 1
    while n > step:
        denominator = 1 - beta[step]*(w[n+1]**2 - w[step]**2)/denominator
        step +=1
    beta[n] = denominator/(w[n+1]**2 - w[n]**2)
    return beta

def alphaCoeffs2betaCoeffs(alpha):
    '''
    This function transforms alpha coefficients to beta coefficients.

    alpha:          Alpha coefficients to be transformed

    return:         Beta coefficients

    Source: The Generation of Arbitrary-Phase Polynomials by Recurrence Formulae, Henk Támas
    '''
    beta = np.zeros_like(alpha, dtype=float)
    beta[0] = 1/alpha[0]
    beta[1:] = 1/(alpha[1:]*alpha[0:-1])
    return beta

def betaCoeffs2alphaCoeffs(beta):
    '''
    This function transforms beta coefficients to alpha coefficients.

    beta:          Beta coefficients to be transformed

    return:        Alpha coefficients

    Source: The Generation of Arbitrary-Phase Polynomials by Recurrence Formulae, Henk Támas
    '''
    alpha = np.zeros_like(beta)
    alpha[0] = 1/beta[0]

    for n in range(1,len(alpha)):
        alpha[n] = np.prod(beta[n-1::-2])/np.prod(beta[n::-2])
    return alpha

def arbitraryPhasePolynomial(w, phase, order, type='alpha'):
    '''
    This function returns the phase polynomial for given phase samples.
    The returned phase polynomial is given by monoms in ascending order:
    x**0 * x**1 * ... * x**p

    phase:      Phase samples (used as specification)
    w:          Frequency bins of phase samples
    order:      Order of polynomial

    return:     Phase polynomial

    Source: The Generation of Arbitrary-Phase Polynomials by Recurrence Formulae, Henk Támas
    '''
    #get coefficients needed for describing phase polynomial, order-1 is used
    #as the functions alphaCoeffs4PhasePolynomial smallest depth is n=0
    if type == 'alpha':
        ## TODO: Multiplication by -1 why?
        if order > 0:
            alpha = np.zeros((order))
            alpha = alphaCoeffs4ArbitraryPhasePolynomial(phase, w, alpha, order-1)

            #get phase polynomial
            monoms_new = np.zeros((order+1))
            monoms_old = np.zeros((order+1))
            p, _ = arbitraryPhasePlyFromAlphaCoeffs(w,alpha, order, monoms_new, monoms_old)
            return p
        else:
            return np.ones((1))
    else:
        if order > 0:
            beta = np.zeros((order))
            alpha = np.zeros((order))
            beta = betaCoeffs4ArbitraryPhasePolynomial(phase, w, beta, order-1)

            #get phase polynomial
            monoms_new = np.zeros((order+1))
            monoms_old = np.zeros((order+1))
            p, _ = arbitraryPhasePlyFromBetaCoeffs(w, beta, order, monoms_new, monoms_old)
            return p
        else:
            return np.ones((1))

def arbitraryPhasePlyFromAlphaCoeffs(w,coeffs, order, monoms_new, monoms_old):
    '''
    This functions returns a polynomial describing a phase response. The polynomial
    is calculated by using alpha coefficients and by applying the recurrence formula:
    P_r = alpha_(r-1)*P_(r-1) + P_(r-2) * (p**2-v_(r-1))

    w:             Frequency bins of phase samples
    coeffs:        Alpha coefficients calculated by the function "alphaCoeffs4PhasePolynomial"
    order:         Order of the polynomial
    monoms_new:    monom of polynomial P_(r)
    monoms_old:    monoms of polynomial P_(r-1)

    return:        Phase polynomial

    Source: The Generation of Arbitrary-Phase Polynomials by Recurrence Formulae, Henk Támas
    '''
    if order > 1:
        monoms_new, monoms_old = arbitraryPhasePlyFromAlphaCoeffs(w,coeffs, order-1, monoms_new, monoms_old)
    else:
        monoms_old[0] = 1
        monoms_new[:2] = [coeffs[0],1]

        return monoms_new, monoms_old

    p = coeffs[order-1]*monoms_new + w[order-1]**2*monoms_old + np.roll(monoms_old, 2)

    monoms_old = monoms_new
    monoms_new = p

    return monoms_new, monoms_old

def arbitraryPhasePlyFromBetaCoeffs(w, coeffs, order, monoms_new, monoms_old):
    '''
    This functions returns a polynomial describing a phase response. The polynomial
    is calculated by using beta coefficients and by applying the recurrence formula:
    P_r = P_(r-1) + P_(r-2) * (p**2-v_(r-1)) * beta_(r-1)

    w:             Frequency bins of phase samples
    coeffs:        Beta coefficients calculated by the function "alphaCoeffs4PhasePolynomial"
    order:         Order of the polynomial
    monoms_new:    monom of polynomial P_(r)
    monoms_old:    monoms of polynomial P_(r-1)

    return:        Phase polynomial

    Source: The Generation of Arbitrary-Phase Polynomials by Recurrence Formulae, Henk Támas
    '''
    if order > 1:
        monoms_new, monoms_old = arbitraryPhasePlyFromBetaCoeffs(w, coeffs, order-1, monoms_new, monoms_old)
    else:
        monoms_old[0] = 1
        monoms_new[:2] = [1, coeffs[0]]

        return monoms_new, monoms_old

    p = monoms_new + np.roll(monoms_old, 2)*coeffs[order-1] + monoms_old*w[order-1]**2*coeffs[order-1]

    monoms_old = monoms_new
    monoms_new = p

    return monoms_new, monoms_old





################################################################################
####################Utilities for Other Stuff###################################
def mirrorAtUnityCircle(a):
    '''
    This function mirrors the given array at the unity circle. Note: Passing roots
    lying in the origin leads to undefined behaviour

    a:          Array of values which are mirrored at the unity circle.

    return:     Mirrored values
    '''

    a_m = np.zeros_like(a)
    a_m = 1/np.conj(a)
    return a_m






#%%

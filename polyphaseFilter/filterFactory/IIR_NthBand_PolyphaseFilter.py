import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

from polyphaseFilter.filterFactory import utilities
################################################################################
def nth_band_IIR_polyphaseFilter(L, wp, As, K=None, eps=0.0000001, nIter=20, logger=None, verbose=True):
    #get number of adjustable attenuation zeros
    if K==None:
        n, R = numberOfAdjustableZeros(As, wp, eps)
    else:
        #zeros: R adjustable zeros + 1 fixed at w = 0
        R = int(np.floor(K/(L-1)))

    if verbose:
        print("Order of Branch Filters: {}".format(n))
        print("Number of Attenuation Zeros: {}".format(R))

    b_digital, a_digital = nth_band_IIR_polyphaseFilter_linear(L, wp, R, eps, nIter, logger)

    return b_digital, a_digital

def iir_polyphase_directForm(b,a, L):
    '''
    This function transforms the given IIR filter into a polyphase filter structure,
    which comprising one polyphase FIR filter and one IIR all-pole filter.
    '''
    #check on parameters
    Nb = len(b)
    assert L>0, "Decimation/Interpolation factor must be greater than zero!"
    assert Nb>L,"The tab number of the non-recursive part (numerator) must be \n" \
                    "greater than the interpolation factor!"
    #compose IRR all-pole filter
    b_iir = 1
    a_iir = a

    #compose FIR polyphase structure
    b_fir = b
    if(Nb%L) != 0:
        b_fir = np.pad(b, (0, L-Nb%L), 'constant', constant_values=(0,0))
    Nb = len(b_fir)
    mask = np.arange(0,Nb,1).reshape(Nb//L,L).T
    b_fir = b_fir[:,np.newaxis][mask][:,:,0]

    return b_fir, b_iir, a_iir

################################################################################
###Functions for creating different types of Nth-band IIR Polyphase Filters:####
def nth_band_IIR_polyphaseFilter_linear(L, wp, R, eps=0.0000001, nIter=20, logger=None):
    '''
    This functions generates the branch filter coefficients for a Nth band recursive
    digital filter with linear phase property. For having linear phase property the
    first branch must be composed by pure delay elements. The other branches are
    composed by all pass filters. The returend coefficients are monomes. One branch
    filter is given by:
    b = b[0] + b[1]*z**(-1)+...+ b[N]*z**(-N)
    a = a[0] + a[1]*z**(-1)+...+ a[N]*z**(-N)

    Parameter:
    __________
    L:              Decimation/Interpolation factor
    wp:             Passband edge
    R:              Adjustable attenuation zeros
    eps:            Specification for termination test (Ripple Specification)
    iter:           Maximum number of iterations
    logger:         Creates log file with debug information

    return:
    _______
    b_digital:      Numerator filter coefficients of the individual branches. The
                    number of rows must correspond to the interpolation/decimation
                    factor. (Descending order of coefficients)

    a_digital:      Denominator filter coefficients of the individual branches. The
                    number of rows must correspond to the interpolation/decimation
                    factor. (Descending order of coefficients)
    '''
    if logger != None:
        logger.msg("Specification:", True)
        logger.msg("L:                  {}".format(L))
        logger.msg("Passband Edge:      {}".format(wp))
        logger.msg("Ripple in Passband: {}".format(eps))
        logger.new_section()

    iter = 0
    b_digital = None
    a_digital = None
    #Each transmission zero binds L-1 koefficients. Thus the number of
    k = np.linspace(0,R,R+1,endpoint=True)
    #From now on, we are in the upsampled space
    wp = wp/L

    #The stopbands is definded by: [2*pi/L*r-wp, 2*pi/L*r+wp]. We want to
    #minimize the amplification within the stopband. For this reason, we minimize
    #the maxima within the passband.

    #Step 1: initial guess for maxima location within the passband, w_r element of [0,wp]
    w_r = 2/L*np.arcsin(np.sin(L*wp/2)*np.sin(k*np.pi/(2*R+1)))

    assert np.any(w_r > wp) == False, "Attenuation zeros must not be greater than passband edge!"
    assert np.any(w_r < 0) == False, "Attenuation zeros must not be smaller than 0!"

    while True:
        #Step 2: design branch filters
        b_digital, a_digital = designBranchFilters(w_r, L)
        wBins, H_n = freqz_branchFilters(b_digital,
                                         a_digital,
                                         w_r,
                                         L)

        # Calculate the magnitude response
        H = np.abs(1/L*np.sum(H_n, axis=0))

        #Step 3: Error function
        if iter == 0:
            #Within the first iteration, get stopband we use for the optimization
            stopband, stopbandId = getStopbandWithGreatestError(L,
                                                                wp,
                                                                wBins,
                                                                H)

        #Get R+1 maxima within the stopband to be optimized, these maxima define
        #the error function
        wBins_max, F_max = errorFunction(L,               #Decimation/Interpolation factor
                                         wBins,           #Frequency bins at which H has been evaluated
                                         H,               #Magnitude response of recursive polyphase filter design
                                         w_r,             #Attenuation zeros
                                         stopbandId,      #Stopband which has to be optimized
                                         stopband)        #Corresponding frequencies

        if iter == 0:
            #Within the first iteration, get maximum error to be optimized
            maxError=max(F_max)
            maxErrorIdx = np.argmax(F_max)

        #step 4: termination test
        if max(F_max)/min(F_max) < 1+eps:
            break

        #step 5: Optimize maximum error on the basis of Newton-Raphson method
        J, derr, maxError = stateLinearEquationSystem(maxErrorIdx,      #Index of initial maximum error
                                                      maxError,         #maximum error to be optimized
                                                      F_max,            #error function
                                                      w_r,              #attenuation zeros
                                                      L,                #Decimation/interpolation factor
                                                      wBins,            #Frequency bins at which F has been evaluated
                                                      stopband,         #Corresponding stopband frequencies
                                                      stopbandId)       #Sth stopband which has to be optimized



        J_pi = np.linalg.inv(J.T@J) @ J.T
        dw_r = J_pi @ derr
        w_r[1:] += dw_r

        assert np.all(w_r <= wp) == True, "Attenuation zeros must be in range [0,wp]!"
        assert np.all(w_r >= 0) == True, "Attenuation zeros must be in range [0,wp]!"

        if logger != None:
            logger.msg("###############Iteration: {}###############".format(iter))
            logger.msg("Maximum error: {}".format(maxError))
            logger.msg("Deviation from maximum error: {}".format(derr))
            logger.msg("Attenuation zeros: {}".format(w_r))
            log_errorFunction(wBins, H, wBins_max, F_max, stopbandId, stopband, w_r, L, logger)
            logger.new_section()
        iter +=1
        if(iter == nIter):
            break

    return b_digital, a_digital

################################################################################
#####################Functions for designing the branch filters#################
def designBranchFilters(w_r,L):
    '''
    This function design L-1 digital branch filters for the given attenuation zeros.
    Note: The first filter must not be designed as it is a pure delay branch.

    Parameters:
    ___________
    w_r:            Attenuation zeros
    L:              Interpolation/Decimation factor

    return:
    _______
    b_digital:      Numerator of branch filters (descending order)
    a_digital:      Denominator of branch filters (descending order)
    '''
    #The recursive polyphase filter requires equal phase responses of each
    #of each branch within the passband:
    #phi_0(w_i) = phi_1(w_i)=...=phi_L(w_i) for i = [1,R]
    #The individual steps for deducing the branch filters:

    #phase response of delay branch
    R = len(w_r) - 1
    phi_0r = -L*R*w_r

    #state phase constraint for each branch
    branches = np.linspace(1,L-1,L-1,endpoint=True, dtype=int)
    phse_samples = (phi_0r + branches[:,None] @ w_r[None,:])/2       #branches x transmission zero frequencies

    #Design remaining branch filters in s-domain via phase constraint
    b_analog, a_analog = designAllpassFilter_s(L, phse_samples,w_r)

    #Transform branch filters from the s-Domain to the z-Domain
    b_digital = np.zeros((L,R+1))
    a_digital = np.zeros_like(b_digital)

    for branch in branches:
        b_digital[branch], a_digital[branch] = utilities.allpass_s2z(a_analog[branch-1])
        utilities.checkOnDigitalAllpassFilter(b_digital[branch], a_digital[branch])

    #Insert delay branch
    b_digital[0,-1] = 1
    a_digital[0, 0] = 1

    return b_digital, a_digital

def freqz_branchFilters(b_digital, a_digital, w_r,L, wN=4096):
    '''
    This functions determines the frequency response of the branch filters.

    Parameters:
    ___________
    b_digital:          Numerator polynomial of branch filters in descneding order (type: darray)
    a_digital:          Denominator polynomial of branch filters in descending order (type: darray)
    w_r:                Attentuation zeros
    wN:                 Number of normalized circular frequency bins

    return:
    _______
    wBins:              Normalized circular frequency at which frequency response
                        branch filters were evaluated
    H_n:                Frequency responses of branch filters
    '''

    R = len(w_r) - 1
    wBins = np.linspace(0,np.pi,wN,endpoint=True)                    #normalized circular frequencies in z-domain
    wBins = np.insert(wBins, wBins.searchsorted(w_r), w_r)           #Insert attenuation zeros
    wBins = np.unique(wBins)
    wN = len(wBins)

    z = np.exp(-1j*wBins)
    #Note: Branch zero is for pure delay branch
    H_n = np.zeros((L,wN), dtype=complex)

    #Determine allpass branch filters
    branches = np.linspace(0,L-1,L,endpoint=True, dtype=int)

    for branch in branches:
        P_b = np.poly1d(b_digital[branch,::-1])
        P_a = np.poly1d(a_digital[branch,::-1])

        H_n[branch,:] = P_b(z**L)/P_a(z**L) * z**(branch)

    return wBins, H_n

def designAllpassFilter_s(L, phse_samples,w_r):
    '''
    This function designs L-1 allpass filters in the s-domain which exhibit the
    same phase at the attenuation zeros as the given phase samples.

    Parameters:
    ___________
    L:              Decimation/Interpolation factor
    phse_samples:   Given phase samples
    w_r:            Attenuation zeros

    return:
    _______

    b:              Numerator of allpass filter, descending order
    a:              Denominator of allpass filter, descending order
    '''
    #Since the first branch is the delay branch we only have to design the remaining ones
    branches = np.linspace(0,L-2,L-1,endpoint=True, dtype=int)

    R = len(w_r) - 1                                                 #Number of adjustable attenuation zeros
    Wr = np.tan(L*w_r/2)                                             #attenuation zeros in s-domain

    b_analog = np.zeros((L-1,R+1))
    a_analog = np.zeros_like(b_analog)

    for branch in branches:
        #H_n(s) = P_n(s)/P_n(-s)
        b_analog[branch] = utilities.arbitraryPhasePolynomial(Wr,phse_samples[branch,:], R, type='alpha')[::-1]
        a_analog[branch,-1::-2] = b_analog[branch,-1::-2]
        a_analog[branch,-2::-2] = -1 * b_analog[branch,-2::-2]

        utilities.checkOnAnalogAllpassFilter(b_analog[branch], a_analog[branch], verbose=False)

    return b_analog, a_analog

def log_BranchFilters(L, wp, wBins, H_n, logger):
    col = 3
    row = int(np.ceil(L/3))
    fig, axs = plt.subplots(row, col, figsize=(15, 8), sharey=True)
    axs = axs.flatten()

    for i in range(L):
            axs[i].plot(wBins, np.abs(H_n[i,:]), label="Branch {}".format(i))
            axs[i].set_xlabel('Normalized Circular Frequency Bins')
            axs[i].set_ylabel('Linear Magnitude Response')
            axs[i].legend()

    plt.legend()
    plt.ylim([-3, 3])


    fig = plt.figure(figsize=(9,6))
    for i in range(L):
        plt.plot(wBins, np.unwrap(np.angle(H_n[i,:])), label="Branch {}".format(i))
    plt.axvline(wp, color='black', linestyle='dashed', label="Passband edge")
    plt.legend()
    plt.xlabel('Normalized Circular Frequency Bins')
    plt.ylabel('Phase Response of Branch Filters')
    logger.plot("Branch Filter Visualization",
                fig,
                '',
                'png')
    return

################################################################################
###################Functions for Determining Error Function#####################
def errorFunction(L, wBins, H, w_r, stopbandId, stopband):
    '''
    This function returns the error function and the maximal error
    for the given magnitude response.

    Parameter:
    __________
    L:          Decimation/Interpolation factor
    wBins:      Normalized circular frequency bins at which H was evaluated
    H:          Magnitude response
    w_r:        Attentuation zeros

    return:
    _______
    wBins_max:          Normalized circular frequency bins at which error function
                        is evaluated
    F_max:              Error function
    '''

    F = H[(wBins >= stopband[0]) & (wBins <= stopband[-1])]
    N = len(w_r)

    rippleEdges = getRippleEdges(stopbandId,
                                 stopband,
                                 w_r,
                                 L)

    #extract R+1 maxima
    F_max = np.zeros((N,))
    wBins_max = np.zeros((N,))

    for n in range(N):
        F_max[n] = max(F[(stopband >= rippleEdges[n]) & (stopband <= rippleEdges[n+1])])
        wBins_max[n] = stopband[(F_max[n] == F)]

    return wBins_max, F_max

def getRippleEdges(stopbandId, stopband, w_r, L):
    '''
    This function determines the ripple edges within the stopband.

    Parameter:
    _________
    stopbandId:         Id of the stopband to be optimized
    stopband:           Stopband to be optimized
    w_r:                Attenuation zeros
    L:                  Descimation/Interpolation factor

    return:
    _______
    rippleEdges:        Ripple edges
    '''
    N = len(w_r)
    rippleEdges = np.zeros((N+1,))
    if stopbandId%2==1:
        zeros = (stopbandId+1)*np.pi/L - w_r                 #descending
        rippleEdges[1:] = zeros[::-1]
        rippleEdges[0] = stopband[0]
    else:
        zeros = stopbandId*np.pi/L + w_r                     #ascending
        rippleEdges[:-1] = zeros
        rippleEdges[-1] = stopband[-1]

    assert np.all(rippleEdges >= stopband[0]) == True, "Ripple edges must lie in stopband!"
    assert np.all(rippleEdges <= stopband[-1]) == True, "Ripple edges must lie in stopband!"

    return rippleEdges

def getStopbandWithGreatestError(L, wp, wBins, H):

    stopbandId = 0
    errMax = 0
    #get the stopband with the greatest value, the greatest value determines the maximum error
    for s in np.linspace(1,L-1,L-1,endpoint=True):
        if s%2==1:
            upper_stopbandEdge = (s+1)*np.pi/L
            lower_stopbandEdge = upper_stopbandEdge - wp

            assert upper_stopbandEdge <= np.pi, "Upper stopband edge must not be greater than Pi!"
            assert lower_stopbandEdge > wp, "Lower stopband edge must be greater than wp!"
        else:
            lower_stopbandEdge = s*np.pi/L
            upper_stopbandEdge = lower_stopbandEdge + wp

            assert upper_stopbandEdge <= np.pi, "Upper stopband edge must not be greater than Pi!"
            assert lower_stopbandEdge > wp, "Lower stopband edge must be greater than wp!"


        if max(H[(wBins >= lower_stopbandEdge) & (wBins <= upper_stopbandEdge)]) > errMax:
            F = H[(wBins >= lower_stopbandEdge) & (wBins <= upper_stopbandEdge)]
            errMax = max(F)
            stopband = wBins[(wBins >= lower_stopbandEdge) & (wBins <= upper_stopbandEdge)]
            stopbandId = s

    return stopband, stopbandId

def log_errorFunction(wBins, H, wBins_max, F_max, stopbandId, stopband, w_r, L, logger):
    '''
    This function logs the information of the error function.

    Parameter:
    __________
    wBins:          Normalized circular frequency of H (Magnitude response)
    H:              Magnitude response
    wBins_max:      Normalized circular freqency of F_max (error function)
    F_max:          Error function
    stopbandId:     Stopband Id
    stopband:       Stopband from which maxima are extracted
    w_r:            Attenuation zeros
    L:              Interpolation/Decimation factor
    logger:         Logger object for generating log information for hmtl file

    return:
    _______
    -
    '''
    rippleEdges = getRippleEdges(stopbandId, stopband, w_r, L)
    fig = plt.figure(figsize=(9,6))
    plt.plot(wBins, 20*np.log10(np.abs(H)), label="Linear Magnitude Response")
    plt.plot(wBins_max, 20*np.log10(np.abs(F_max)), marker="x", linestyle = 'None', label="Maxima")
    plt.axvline(stopband[0], color="black", linestyle="dashed", linewidth=2)
    plt.axvline(stopband[-1], color="black", linestyle="dashed", linewidth=2)
    for edge in rippleEdges:
        plt.axvline(edge, color='green', linestyle="dashed")
    plt.legend()
    plt.xlabel('Normalized Circular Frequency Bins')
    plt.ylabel('Magnitude Response of Polyphase Filter')
    logger.plot("Visualization of Error Function",
                fig,
                '',
                'png')
    plt.close(fig)
    return
################################################################################
##########################Functions for Optimizing Error########################
def stateLinearEquationSystem(maxErrorIdx, maxError, F_max, w_r, L, wBins, stopband, stopbandId, dw=0.00001):
    '''
    This function states the linear equation system to solve the equation:
    F(W_r,w_max) = max{F(W_r,w_max)}

    Parameters:
    ___________
    maxErrorIdx:    Index/Bin of initial maximum error to be optimized
    maxError:       The maximum error to be optimized
    F_max:          The error array which comprises the R+1 greatest error values of the error function
                    F(W_r,w)
    w_r:            Attenuation zeros which were used to determine F(w_r,w)
    L:              Decimation/Interpolation factor
    wBins:          Normalized circular frequency bins at which F has been evaluated

    return:
    ________
    J               Jacobian matrix
    derr            Deviation of to be optimized maximum error
    maxError        To be optimized maximum error
    '''

    N = len(w_r)                #Number of maxima
    R = len(w_r) - 1            #Adjustable attenuation zeros
    J = np.zeros((N,R))         #Jacobian matrix
    derr = maxError - F_max     #Deviation of to be optimized maximum error


    #Linearize Equation system
    for r in np.linspace(1,R,R,endpoint=True, dtype=int):
        #Displace attenuation zero slightly
        w_r_shifted = w_r
        w_r_shifted[r] += dw
        #Redesign filter branches for new attenuation zeros and get error function
        b_digital, a_digital = designBranchFilters(w_r_shifted, L)
        _, H_n = freqz_branchFilters(b_digital,
                                     a_digital,
                                     w_r_shifted,
                                     L)
        _, H_shifted_n = designBranchFilters(w_r_shifted, L)
        H_shifted = np.abs(1/L*np.sum(H_n, axis=0))

        _, F_max_shifted = errorFunction(L,               #Decimation/Interpolation factor
                                         wBins,           #Frequency bins at which H has been evaluated
                                         H_shifted,       #Magnitude response of recursive polyphase filter design
                                         w_r_shifted,     #Attenuation zeros
                                         stopbandId,
                                         stopband)
        #calculate derivative and deposit derivate in Jacobian matrix
        for n in range(N):
            dF = (F_max_shifted[n] - F_max[n])/dw
            J[n,r-1] = dF

            #add error differential to the maximum error (right hand sided equation system)
            if(n == maxErrorIdx):
                derr[:] += (F_max_shifted[n] - maxError)
                maxError += (F_max_shifted[n] - maxError)


    return J, derr, maxError

def peakCancelling_filter(order, wp, fs, type='bessel'):
    '''
    This filter returns a filter which is used for the suppression of the peaks
    in the stopband. This filter is supposed to act on the low sampling rate.

    Parameters:
    ___________
    order:              Filter order
    wp:                 Passband edge
    fs:                 Sampling rate
    type:               Filter type

    return:
    _______
    b:                  Numerator of filter (descending order)
    a:                  Denominator of filter (descending order)
    '''
    nyq = fs/2

    switcher = {
        'butterworth': signal.butter(order, wp/nyq, btype='lowpass', analog=False,
                       output='ba'),
        'bessel':      signal.bessel(order, wp/nyq, btype='lowpass', analog=False,
                       output='ba', norm='delay')
    }

    return switcher.get(type, "Invalid Filter type")

def numberOfAdjustableZeros(As,wp, eps=0.0000001):
    '''
    This function estimates the minimum number of adjustable attenuation zeros for
    the given attenuation within the stopband, the passband edge and the passband
    ripple. The estimation is based on the order estimation of a chebyshev filter.

    Parameter:
    __________
    As:                 Stopband attenuation
    wp:                 Passband edge
    eps:                Passband ripple

    return:
    n:                  Overall filter order
    R:                  Number of adjustable attenuation zeros
    _______
    '''
    wp /= 2
    ws = np.pi - wp
    #convert to half-cycles/sample
    wp /= np.pi
    ws /= np.pi

    Ap = 20*np.log10((1-0.0000001))

    n, _ = signal.cheb2ord(wp, ws, -Ap, -As)

    n += (n+1)%2

    R = (n - 1)/2

    return int(n), int(R)

################################################################################
###########################Functions for Analysis###############################
def plot_IIR_NthBand_filter(b_pp,a_pp, preFilter=False, b_pre=None, a_pre=None):
    '''
    This functions plot the given Nth-band filter. Note: The given filter must
    not comprise any delay elements (which are common for the polyphase structure).
    This function takes care of that. The input format of the given filter is expected
    as follows: number of rows represent the number of branches, columns represent
    the branch filter.

    Parameter:
    b_pp:                  Numerator polynomial (descending order, negative powers)
    a_pp:                  Denominator polynomial (descending order, negative powers)
    b_pre:                 Numerator polynomial of prefilter (descending order, negative powers)
    a_pre:                 Denominator polynomial of prefilter (descending order, negative powers)
    '''
    #get frequency response of polyphase filter
    L = np.shape(b_pp)[0]
    wBins, H_n = freqz_branchFilters(b_pp, a_pp, [], L)
    H_pp = 1/L*np.sum(H_n, axis=0)

    if(preFilter):
        _, H_pre = utilities.freqz_expanded(b_pre, a_pre, L, len(wBins))

    #plot


    if preFilter:
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        axs[0].plot(wBins, 20*np.log10(np.abs(H_pp)), label="Linear IIR Polyphase Filter")
        axs[0].plot(wBins, 20*np.log10(np.abs(H_pre)), label="Pre-Filter")
        axs[1].plot(wBins, 20*np.log10(np.abs(H_pp*H_pre)), label="Linear IIR Polyphase Filter + Pre-Filter")

        axs[1].set_ylim(-100,1)
        axs[0].set_ylim(-100,1)

        axs[1].legend()
        axs[0].legend()
    else:
        fig, axs = plt.subplots(1, 1, figsize=(18, 8))
        plot(wBins, 20*np.log10(np.abs(H_pp)), label="Linear IIR Polyphase Filter")
        axs.set_ylim(-100,1)
        axs.legend()
    plt.show()

    return

def computationalComplexity(b,a):
    '''
    This function returns the computation complexity of the given filter. The compuational
    complexity is defined by the number of additions and multiplications for one input sample.

    Parameter:
    _________
    b:                  Numerator polynomial (descending order, negative powers)
    a:                  Denominator polynomial (descending order, negative powers)

    return:
    _______
    Nm:                 Number of multiplications
    Na:                 Number of additions
    '''

    L = np.shape(b)[0]                                                        #Decimator/Interpolation factor
    Nm = 1                                                                    #Factor 1/L
    Na = L                                                                    #Accumulation after subfilters
    for branch in range(L):
        b_sub, a_sub = utilities.negDes2posDes(b[branch,:], a[branch,:])      #Convert subfilter to positive powers
        z,p,k = signal.tf2zpk(b_sub, a_sub)                                   #Get poles as measure for allpass sections
        N_ap = len(p[(p!=0)])                                                 #Number of allpasses
        Nm += N_ap
        Na += N_ap*2

    return Nm, Na


#%%

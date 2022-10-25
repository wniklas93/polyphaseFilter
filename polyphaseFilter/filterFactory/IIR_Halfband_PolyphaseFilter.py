import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from polyphaseFilter.filterFactory import utilities
################################################################################
def EMQF_Halfband_filter(wp, As, verbose=False, tfPlot=True):
    #Get remaining parameters so that filter exhibits halfband characteristic
    N = minimum_ellipFilterOrder(wp, As, verbose)                   #Order

    #get EMQF filter:
    betas_hb = EMQF_halfband_coeffs(wp,N)

    #Assemble filter, so that resulting filter becomes an EMQF halfband filter.
    #Therefore, we have to split the filter into 2 branches.To get the branches,
    #we substitute z**(-1) by z
    betas_hb0 = betas_hb[0::2]                     #conjugate complex poles for 1. branch
    betas_hb1 = betas_hb[1::2]                     #conjugate complex poles for 2. branch
    order_hb0 = len(betas_hb0)*2
    order_hb1 = len(betas_hb1)*2
    if order_hb0 != order_hb1:
        #order mismatch between branches
        betas_hb1 = np.pad(betas_hb1,(0,1),'constant',constant_values=(0,0))

    b0, a0, b1, a1 = np.poly1d([1]), np.poly1d([1]), np.poly1d([1]), np.poly1d([1])

    for beta_hb0, beta_hb1 in zip(betas_hb0, betas_hb1):
        #1. branch
        if beta_hb0:
            b0 *= np.poly1d([1,0,beta_hb0])
            a0 *= np.poly1d([beta_hb0,0,1])
        #2. branch
        if beta_hb1:
            b1 *= np.poly1d([1,0,beta_hb1])
            a1 *= np.poly1d([beta_hb1,0,1])

    #Note: inverse order because of substitution
    b0 = b0.c[::-1]
    b1 = b1.c[::-1]
    a0 = a0.c[::-1]
    a1 = a1.c[::-1]

    utilities.checkOnDigitalAllpassFilter(b0, a0)
    utilities.checkOnDigitalAllpassFilter(b1, a1)

    if tfPlot:
        #insert delay element for plot
        b1_4plot = np.poly1d(b1[::-1]) * np.poly1d([1,0])
        b1_4plot = b1_4plot.c[::-1]
        plot_IIR_halfband_filter((b0,b1_4plot),(a0,a1))

    if verbose:
        utilities.zPlot(b0,a0)
        utilities.zPlot(b1,a1)

    return (b0,b1), (a0,a1)

def EMQF_tuneable_Halfband_filter(wp, As, wc, verbose=False, tfPlot=True):
    #check on input
    assert wp/np.pi < 0.5, "Passband edge must be smaller than 0.5*pi!"

    s = np.tan(wc/2)**2/np.tan(wp/2)**2                       #Selectivity factor for tunable filter
    wp_hb = 2*np.arctan(1/np.sqrt(s))                         #Passband filter for halfband filter

    #Prototype filter: EMQF Halfband Filter
    assert wp_hb/np.pi < 0.5, "Passband edge for halfband prototype filter must be smaller than 0.5*pi!"
    N = minimum_ellipFilterOrder(wp_hb, As, verbose)
    betas_hb = EMQF_halfband_coeffs(wp_hb,N)


    betas_hb0 = betas_hb[0::2]                     #conjugate complex poles for 1. branch
    betas_hb1 = betas_hb[1::2]                     #conjugate complex poles for 2. branch
    order_hb0 = len(betas_hb0)*2
    order_hb1 = len(betas_hb1)*2

    #adjust cutoff frequency
    alpha2 = -np.cos(wc)                                      #second-order-section (all-pass)
    alpha1 = 1/alpha2*(1-np.sqrt(1-alpha2**2))                #first-order-section

    #Determine adjusted conjugate complex poles
    betas0 = (betas_hb0+alpha1**2)/(betas_hb0*alpha1**2+1)    #new conjugate complex poles for adjusted filter (1. branch)
    betas1 = (betas_hb1+alpha1**2)/(betas_hb1*alpha1**2+1)    #new conjugate complex poles for adjusted filter (2. branch)

    if order_hb0 != order_hb1:
        #order mismatch between branches
        betas1 = np.pad(betas1,(0,1),'constant',constant_values=(0,0))

    #Determine new filter coefficients: (substitute z**(-1) by z)
    b0, a0, b1, a1 = np.poly1d([1]), np.poly1d([1]), np.poly1d([1]), np.poly1d([1])

    for beta0, beta1 in zip(betas0, betas1):
        if beta0:
            b0 *= np.poly1d([1, alpha2*(1+beta0),beta0])
            a0 *= np.poly1d([beta0,alpha2*(1+beta0),1])

        if beta1:
            b1 *= np.poly1d([1, alpha2*(1+beta1),beta1])
            a1 *= np.poly1d([beta1,alpha2*(1+beta1),1])

    b1 *= np.poly1d([1,alpha1])
    a1 *= np.poly1d([alpha1,1])
    #Note: inverse order because of substitution
    b0 = b0.c[::-1]
    b1 = b1.c[::-1]
    a0 = a0.c[::-1]
    a1 = a1.c[::-1]

    if verbose:
        utilities.zPlot(b0,a0)
        utilities.zPlot(b1,a1)

    utilities.checkOnDigitalAllpassFilter(b0, a0)
    utilities.checkOnDigitalAllpassFilter(b1, a1)

    if tfPlot:
        plot_IIR_halfband_filter((b0,b1),(a0,a1))


    return (b0,b1), (a0,a1)



################################################################################
def minimum_ellipFilterOrder(wp, As, verbose=False):
    '''
    This function calculates the minimum order which is needed to meet the design
    constraints. Since the function is used to implement EMQF filters, the returned
    order is odd.

    Parameter:
    _________
    wp:                         Passband edge
    As:                         Stopband attenuation

    return:
    ______
    order:                      Minimum order
    '''

    ws = np.pi - wp             #stopband edge

    assert np.isclose(np.tan(ws/2)*np.tan(wp/2), 1) == True, "Passband edge and stopband edge are erronous!"

    As_lin = 10**(As/10)
    Ap = 10*np.log10(1+1/(As_lin-1))

    try:
        n, _ = sp.signal.ellipord(wp/np.pi, ws/np.pi, Ap, As, False)
        N = n + (n+1)%2                    #filter order must be odd
    except:
        n = 13
        N = 13

    assert N%2==1, "Elliptic halfband filter must be of odd order!"

    if verbose:
        print("\n")
        print("wp: {}, ws: {}".format(wp/np.pi, ws/np.pi))
        print("Order of Elliptic Filter: {}".format(N))
        print("Intermediate Order: {}".format(n))
        print("Ap: {}".format(Ap))
        print("As: {}".format(As))

    return N

def computationalComplexity(N, adjusted=False):
    '''
    This function returns the computational complexity of an elliptic halfband filter
    for the given order. The computational complextiy is represented by the number
    of additions and multiplications for one input sample.

    Parameter:
    __________
    N:                          Order of elliptic filter
    adjusted:                   True if filter is adjusted by bilinear transform

    return:
    _______
    Nm:                         Number of needed multiplications
    Na:                         Number of needed additions
    '''

    if not adjusted:
        #Number of multiplications: per allpass 1 multiplication, 1 additional multi-
        #plication because of factor 1/2
        Nm = (N-1)/2 + 1

        #Number of additions: per allpass 2 additions and one after subfilter
        Na = (N-1)+1

    else:
        #Number of multiplications: per allpass 2 multiplications, 1 additional multi-
        #plication because of factor 1/2
        Nm = 1 + (N-1)/2*2

        #Number of additions: per allpass 4 additions and one after subfilter
        Na = 1 + (N-1)/2*4

    return Nm, Na

def plot_IIR_halfband_filter(b,a):
    '''
    This function plots the given halfband filter. Note: The given branch filter
    must comprise delay elements. These delay elements are not taken into
    account by this function.

    Parameter:
    __________
    b=(b0,b1):                  Numerator polynomial (decending order negative powers)
    a=(a0,a1):                  Denominator polynomial (descending order, negative powers)
    '''



    #First branch
    b0 = b[0]
    a0 = a[0]

    #Second branch
    b1 = b[1]
    a1 = a[1]

    #Get frequency response of filter
    wBin, H0 = sp.signal.freqz(b0,a0)
    _, H1 = sp.signal.freqz(b1,a1)

    H = (H0+H1)/2
    #Plot phase and frequency response of filter
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    #Frequency response
    axs[0].plot(wBin,20*np.log10(np.abs(H)), label="Frequency Response of IIR Halfband Filter")
    axs[0].set_ylabel('Amplitude [dB]')
    axs[0].set_xlabel('Frequency [rad/sample]')
    axs[0].legend()

    #Phase response
    axs[1].plot(wBin, np.unwrap(np.angle(H)), label="Phase Response of IIR Halfband Filter")
    axs[1].set_xlabel('Frequency [rad/sample]')
    axs[1].set_ylabel('Angle (radians)')
    axs[1].legend()
    return

def EMQF_halfband_coeffs(wp, N):
    '''
    This function returns the conjuate complex poles for an EMQF halfband filter.
    The return format is the following: beta - z**(2), which leads to a conjugate
    complex pole. The return array is returned in ascending order:
    beta0 < beta1 < beta2

    Parameters:
    ___________
    wp:                     Passband edge of the EMQF filter
    N:                      Order of filter

    return:
    _______
    betas:                  Conjugate complex poles for the resulting filter
    '''
    s = 1/np.tan(wp/2)**2                                                       #selectivity

    #Design elliptic filter so that IIR Halfband characteristic is preserved
    N_ap = int((N-1)/2)                                                         #todo check on semantic!!!!

    l = np.linspace(0,N_ap-1,N_ap,endpoint=True, dtype=int)
    Km = sp.special.ellipk(1/s**2)                                              #Complete elliptic integral of the first kind
    u = ((2*l+1)/N+1)*Km
    xl, _, _, _ = sp.special.ellipj(u,1/s**2)

    q_ = np.sqrt((1-xl**2)*(s**2-xl**2))
    betas = (s+xl**2-q_)/(s+xl**2+q_)
    betas = np.sort(betas)
    return betas

def adjust_EMQF_halfband_filter(b,a,wc, verbose=False, tfPlot=True):
    '''
    This filter adjusts the cutoff frequency for the given EMQF halfband filter.
    Note: The given EMQF halfband filter must not comprise any pure delay elements.

    Parameter:
    _________
    b=(b0,b1):          Numerator polynomials of branch filters (descending order, negative powers)
    a=(a0,a1):          Denominator polynomials of branch filters (descending order, negative powers)
    wc:                 Circular normalized cutoff frequency

    return:
    _______
    b=(b0,b1):          Numerator polynomials of branch filters (descending order, negative powers)
    a=(a0,a1):          Denominator polynomials of branch filters (descending order, negative powers)
    '''

    #Get beta/all-pass coefficients of EMQF halfband filter
    b0, a0 = utilities.negDes2posDes(b[0], a[0])
    b1, a1 = utilities.negDes2posDes(b[1], a[1])

    _,p0,_ = sp.signal.tf2zpk(b0,a0)
    _,p1,_ = sp.signal.tf2zpk(b1,a1)

    betas_hb0 = np.abs(p0)**2
    betas_hb1 = np.abs(p1)**2

    betas_hb0 = np.unique(betas_hb0)
    betas_hb1 = np.unique(betas_hb1)

    order_hb0 = len(betas_hb0)*2
    order_hb1 = len(betas_hb1)*2
    if order_hb0 != order_hb1:
        #order mismatch between branches
        betas_hb1 = np.pad(betas_hb1,(0,1),'constant',constant_values=(0,0))

    #Determine parameters for adjustment
    alpha2 = -np.cos(wc)                                      #second-order-section (all-pass)
    alpha1 = 1/alpha2*(1-np.sqrt(1-alpha2**2))                #first-order-section

    #Determine adjusted conjugate complex poles
    betas0 = (betas_hb0+alpha1**2)/(betas_hb0*alpha1**2+1)    #new conjugate complex poles for adjusted filter (1. branch)
    betas1 = (betas_hb1+alpha1**2)/(betas_hb1*alpha1**2+1)    #new conjugate complex poles for adjusted filter (2. branch)

    #Determine new filter coefficients: (substitute z**(-1) by z)
    b0, a0, b1, a1 = np.poly1d([1]), np.poly1d([1]), np.poly1d([1]), np.poly1d([1])

    for beta0, beta1 in zip(betas0, betas1):
        if beta0:
            b0 *= np.poly1d([1, alpha2*(1+beta0),beta0])
            a0 *= np.poly1d([beta0,alpha2*(1+beta0),1])

        if beta1:
            b1 *= np.poly1d([1, alpha2*(1+beta1),beta1])
            a1 *= np.poly1d([beta1,alpha2*(1+beta1),1])

    db = np.poly1d([1,alpha1])                             #Transform of pure delay element
    da = np.poly1d([alpha1,1])                             #Transform of pure delay element
    #Note: inverse order because of substitution
    b0 = b0.c[::-1]
    b1 = b1.c[::-1]
    a0 = a0.c[::-1]
    a1 = a1.c[::-1]

    db = db.c[::-1]
    da = da.c[::-1]


    if verbose:
        utilities.zPlot(b0,a0)
        utilities.zPlot(b1,a1)

    utilities.checkOnDigitalAllpassFilter(b0, a0)
    utilities.checkOnDigitalAllpassFilter(b1, a1)

    if tfPlot:
        b1_4plot = np.poly1d(b1[::-1]) * np.poly1d(db[::-1])
        a1_4plot = np.poly1d(a1[::-1]) * np.poly1d(da[::-1])
        plot_IIR_halfband_filter((b0,b1_4plot.c[::-1]),(a0,a1_4plot.c[::-1]))


    return (b0,b1), (a0,a1), (db,da)

def plot_characteristicCurves_Adjustable_EMQF_halfband_filter(b,a,wc):

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    for wc_ in wc:
        b_, a_, delayTF = adjust_EMQF_halfband_filter(b,a,wc_,False, False)

        #Insert transformed delay element into second branch
        b1_4plot = np.poly1d(b_[1][::-1]) * np.poly1d(delayTF[0][::-1])
        a1_4plot = np.poly1d(a_[1][::-1]) * np.poly1d(delayTF[1][::-1])

        #Get frequency response of filter
        wBin, H0 = sp.signal.freqz(b_[0],a_[0])
        _, H1 = sp.signal.freqz(b1_4plot.c[::-1],a1_4plot.c[::-1])

        H = (H0+H1)/2

        axs[0].plot(wBin,20*np.log10(np.abs(H)),label="wc: {}*pi".format(wc_/np.pi))
        axs[1].plot(wBin, np.unwrap(np.angle(H)), label="wc: {}*pi".format(wc_/np.pi))
        axs[0].legend()
        axs[1].legend()
    plt.show()

def groupDelay(b,a, mode):
    '''
    This fuctions returns the group delay for the given EMQF halfband filter. Since the phase does not
    change for changing cutoff frequencies (when filter adjustment is applied), the group delay is
    valid for all "inheritors" of this prototype filter.

    Parameters:
    ___________
    b=(b0,b1):          Numerator polynomials of branch filters (descending order, negative powers)
    a=(a0,a1):          Denominator polynomials of branch filters (descending order, negative powers)
    mode:               Averaging or averaging with weighting (av/avW)

    return:
    _______
    gd:                 GroupDelay of EMQF halfband filter [samples]
    '''

    #get filter response
    b0, a0 = b[0], a[0]
    b1, a1 = b[1], a[1]

    wBins, H0 = sp.signal.freqz(b0, a0)
    wBins, H1 = sp.signal.freqz(b1, a1)

    H_ges = 0.5*(H1 + H0)
    H_ges = H_ges[(wBins<=0.5*np.pi)]
    wBins = wBins[(wBins<=0.5*np.pi)]

    phi = np.unwrap(np.angle(H_ges))                #phase response
    H_ges = np.abs(H_ges)**2                        #transform response to squared magnitude

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
        gd = np.sum(gd*H_ges[1:])/np.sum(H_ges[1:])

    return gd






#%%

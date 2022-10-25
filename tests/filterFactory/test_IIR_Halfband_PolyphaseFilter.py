import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import signal
from polyphaseFilter.filterFactory import IIR_Halfband_PolyphaseFilter
from polyphaseFilter.filterFactory import utilities
################################################################################
def test_minimum_ellipFilterOrder():
    #1. Subcase: Order for halfband filter
    wp = 0.86*np.pi
    As = 45
    N_ref = 7
    N = IIR_Halfband_PolyphaseFilter.minimum_ellipFilterOrder(wp/2, As, verbose=False)
    assert N==N_ref


def test_EMQF_Halfband_Filter():
    #1. Subcase: 5th order elliptic filter --> no order mismatch between branches
    wp = 0.4*np.pi
    As = 25.2385
    p_ref = np.asarray([0.4863*1j, -0.4863*1j, 0.8453*1j, -0.8453*1j])
    b, a = IIR_Halfband_PolyphaseFilter.EMQF_Halfband_filter(wp, As, False)
    b0 = b[0]
    b1 = b[1]
    a0 = a[0]
    a1 = a[1]
    #convert the filter coefficients so that powers are positive and descending
    b0, a0 = utilities.negDes2posDes(b0, a0)
    b1, a1 = utilities.negDes2posDes(b1, a1)
    _,p0,_ = sp.signal.tf2zpk(b0,a0)
    _,p1,_ = sp.signal.tf2zpk(b1,a1)
    p = np.concatenate((p0,p1))
    p = np.sort(p)
    p_ref = np.sort(p_ref)
    assert np.allclose(p,p_ref,atol=0.001)

    #2. Subcase: 7th order elliptic filter --> order mismatch between branches
    wp = 0.450059*np.pi
    As = 40
    p_ref = np.asarray([0.4359*1j, -0.4359*1j, 0.7429*1j, -0.7429*1j, 0.9274*1j, -0.9274*1j])
    b, a = IIR_Halfband_PolyphaseFilter.EMQF_Halfband_filter(wp, As, False)
    b0 = b[0]
    b1 = b[1]
    a0 = a[0]
    a1 = a[1]

    b0, a0 = utilities.negDes2posDes(b0, a0)
    b1, a1 = utilities.negDes2posDes(b1, a1)
    _,p0,_ = sp.signal.tf2zpk(b0,a0)
    _,p1,_ = sp.signal.tf2zpk(b1,a1)
    p = np.concatenate((p0,p1))
    p = np.round(p,4)
    p = np.sort(p)
    p_ref = np.sort(p_ref)

    assert np.allclose(p,p_ref,atol=0.001)

def test_adjust_EMQF_Filter():
    wp = 0.450059*np.pi
    As = 40
    b_ref, a_ref = IIR_Halfband_PolyphaseFilter.EMQF_Halfband_filter(wp, As, False, False)
    #Adjust Prototype Filter
    wc = 0.5*np.pi
    b,a, delayTF = IIR_Halfband_PolyphaseFilter.adjust_EMQF_halfband_filter(b_ref,a_ref,wc,False, False)

    #check
    assert np.allclose(b_ref[0], b[0], atol=0.001) == True
    assert np.allclose(b_ref[1], b[1], atol=0.001) == True
    assert np.allclose(a_ref[0], a[0], atol=0.001) == True
    assert np.allclose(a_ref[1], a[1], atol=0.001) == True

    #delayTF must be a  pure delay element as we consider an EMQF Halfband filter
    assert np.allclose(delayTF[0], [0,1], atol=0.001) == True
    assert np.allclose(delayTF[1], [1], atol=0.001) == True


def test_computationalComplextity():
    wp = 0.8*np.pi
    As = 36

    N = IIR_Halfband_PolyphaseFilter.minimum_ellipFilterOrder(wp/2,As)
    Nm, Na = IIR_Halfband_PolyphaseFilter.computationalComplexity(N)

    assert Nm == (N-1)/2 + 1
    assert Na == N-1+1

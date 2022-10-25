import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pytest

from polyphaseFilter.filterFactory import utilities
from polyphaseFilter.filterFactory import IIR_Halfband_PolyphaseFilter
################################################################################
############Under Test: Utilities for getting Polynomial from Phase############
def test_polynomialFromAlphaCoefficients():
    ply_ref = np.zeros((6,6))
    ply_ref[0,:] = np.asarray([1, 0, 0, 0, 0, 0])
    ply_ref[1,:] = np.asarray([1, 1, 0, 0, 0, 0])
    ply_ref[2,:] = np.asarray([3, 3, 1, 0, 0, 0])
    ply_ref[3,:] = np.asarray([15, 15, 6, 1, 0, 0])
    ply_ref[4,:] = np.asarray([105, 105, 45, 10, 1, 0])
    ply_ref[5,:] = np.asarray([945, 945, 420, 105, 15, 1])

    for order,row in enumerate(ply_ref):
        b, a = sp.signal.bessel(order, 1, 'low', analog=True, norm='delay')
        #print("a: {}".format(a))
        w, h = sp.signal.freqs(b, a,worN=512)
        phase = np.unwrap(np.angle(h))
        ply_alpha = utilities.arbitraryPhasePolynomial(w[[0,100,200,300,400,500]],
                                                       -phase[[0,100,200,300,400,500]],
                                                       order)


        assert np.all(np.isclose(row[:order+1], ply_alpha, atol=0.001))


def test_alpha_beta_coefficients():
    orders = np.linspace(1,5,5,endpoint=True,dtype=int)
    for order in orders:

        b, a = sp.signal.bessel(order, 1, 'low', analog=True, norm='delay')
        w, h = sp.signal.freqs(b, a, worN=512)
        phase = np.unwrap(np.angle(h))

        alpha = np.zeros((order,))

        alpha = utilities.alphaCoeffs4ArbitraryPhasePolynomial(phase[[0,100,200,300,400,500]],
                                                               w[[0,100,200,300,400,500]],
                                                               alpha,
                                                               order-1)
        beta = np.zeros((order,))
        beta = utilities.betaCoeffs4ArbitraryPhasePolynomial(phase[[0,100,200,300,400,500]],
                                                             w[[0,100,200,300,400,500]],
                                                             beta,
                                                             order-1)

        assert np.all(np.isclose(alpha,
                                 utilities.betaCoeffs2alphaCoeffs(beta),
                                 atol=0.001))

################################################################################
#####################Under Test: Filter Utilities###############################
def test_allpass_s2z():
    #Desing analog allpass filter. Note: a is a descending monom vector
    _, a = sp.signal.butter(4, 100, 'low', analog=True)

    b = np.zeros_like(a)
    b[-1::-2] = a[-1::-2]
    b[-2::-2] = -a[-2::-2]

    utilities.checkOnAnalogAllpassFilter(b,a)
    b, a = utilities.allpass_s2z(a)
    utilities.checkOnDigitalAllpassFilter(b,a)

def test_negDes2posDes():
    #1. subcase
    b = [0,1,2,3]
    a = [1,2,3]
    b_ref = [1,2,3]
    a_ref = [1,2,3,0]

    b, a = utilities.negDes2posDes(b,a)
    assert np.array_equal(b_ref,b) == True
    assert np.array_equal(a_ref,a) == True

    #2. subcase
    b = [1,2,3]
    a = [0,1,2,3]
    b_ref = [1,2,3,0]
    a_ref = [1,2,3]

    b, a = utilities.negDes2posDes(b,a)
    assert np.array_equal(b_ref,b) == True
    assert np.array_equal(a_ref,a) == True

    #3. subcase
    b = [1,2,3,4]
    a = [1,2,3,4]
    b_ref = [1,2,3,4]
    a_ref = [1,2,3,4]

    b, a = utilities.negDes2posDes(b,a)
    assert np.array_equal(b_ref,b) == True
    assert np.array_equal(a_ref,a) == True

def test_computationalComplexity():
    #1. Subcase: FIR Filter
    b = [1,2,3,4,5]
    a = [1]

    Nm_ref = len(b)
    Na_ref = len(b) - 1
    Nm, Na = utilities.computationalComplexity(b,a)

    assert Nm_ref == Nm
    assert Na_ref == Na

    #2. Subcase: IIR Filter
    b = [1,2,3,4]
    a = [1,2,3,4]

    Nm_ref = len(b) + len(a) - 1
    Na_ref = len(b) - 1 + len(a) - 1
    Nm, Na = utilities.computationalComplexity(b,a)

    assert Nm_ref == Nm
    assert Na_ref == Na

    #3. Subcase: IIR filter different numerator and denominator order
    b = [1,2]
    a = [1,2,3,4]

    Nm_ref = len(b) + len(a) - 1
    Na_ref = len(b) - 1 + len(a) - 1
    Nm, Na = utilities.computationalComplexity(b,a)

    assert Nm_ref == Nm
    assert Na_ref == Na


def test_computationalComplexity_polyphaseFilters():
    #FIR Polyphase filter
    b = np.ones((4,5))
    a = np.ones((4,1))

    Nm_ref = 5*4
    Na_ref = 4*4 + 4

    Nm, Na = utilities.computationalComplexity_polyphaseFilters(b,a)
    assert Nm == Nm_ref
    assert Na == Na_ref

    #Parallel IIR Filters
    b = np.ones((5,4))
    a = np.ones((5,4))

    Nm_ref = 5*(len(b[0]) + len(a[0])-1)
    Na_ref = 5*(len(b[0]) - 1 + len(a[0]) - 1) + 5
    Nm, Na = utilities.computationalComplexity_polyphaseFilters(b,a)


    assert Nm_ref == Nm
    assert Na_ref == Na



################################################################################
###################Under Test: Utilities for Other Stuff########################
def test_mirrorAtUnityCircle():
    a = [ 1 +  0j,
          1 + 10j,
          1 - 10j,
         -5 + 10j,
         -5 -  5j]

    b = utilities.mirrorAtUnityCircle(a)
    assert np.allclose(a, utilities.mirrorAtUnityCircle(b), atol=0.1)


################################################################################
###################Under Test: FIR Polyphase Filter#############################
def test_FIR_polyphase_filter():
    b = [1,2,3,4,5,6]
    b_ref = np.asarray([[1,3,5],
                        [2,4,6]])

    b = utilities.FIR_polyphase_filter(b,L=2)

    assert np.array_equal(b,b_ref) == True














################################################################################

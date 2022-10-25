import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pytest

from polyphaseFilter.filterFactory import IIR_NthBand_PolyphaseFilter
from debugLogger.debugLogger import debugLogger
################################################################################
log_dirname = "tests/filterFactory/logFiles"
################################################################################
def test_designBranchFilters():
    loggerName = "test_designBranchFilters"
    filename = "test_designBranchFilters.html"

    logger = debugLogger(loggerName,
                         filename,
                         log_dirname)


    logger.msg('''This test checks on the validity of the branch filter generation.
    ''')
    logger.new_section()

    L = 6                                   #Decimation factor
    wp = 0.5*np.pi                          #passband edge (circular normalized frequency)
    K = 20                                  #Degrees of freedom

    R = int(np.floor(K/(L-1)))
    k = np.linspace(0,R,R+1,endpoint=True)

    logger.msg("Test Specification: \n", True)
    logger.msg("L:                              {}".format(L))
    logger.msg("Passband Edge:                  {}".format(wp))
    logger.msg("Degrees of Freedom:             {}".format(K))
    logger.msg("Adjustable Attenuation zeros:   {}".format(R))
    logger.new_section()

    #create branch filters
    wp = wp/L
    w_r = 2/L*np.arcsin(np.sin(L*wp/2)*np.sin(k*np.pi/(2*R+1)))
    b_digital, a_digital = IIR_NthBand_PolyphaseFilter.designBranchFilters(w_r, L)
    wBins, H_n = IIR_NthBand_PolyphaseFilter.freqz_branchFilters(b_digital,
                                                                 a_digital,
                                                                 w_r,
                                                                 L)

    #log branch filters
    IIR_NthBand_PolyphaseFilter.log_BranchFilters(L, wp, wBins, H_n, logger)

    H = 1/L * np.sum(H_n,axis=0)
    H_r = np.zeros((R+1), dtype=complex)
    for i, w in enumerate(w_r):
        H_r[i] = H[(w==wBins)][0]
    #check on attenuation zeros in wBins
    assert np.all(np.in1d(w_r, wBins))

    #check on the phase at the attenuation zeros
    eps = 0.1                               #Error tolerance
    for w in w_r:
        phase = np.unwrap(np.angle(H_n[:,(wBins==w)]))
        phase = np.reshape(phase, (L,))
        assert np.all((phase >= phase[0] - eps) & (phase <= phase[0] + eps))

    return

def test_recursivePolyphaseFilter():
    loggerName = "test_recursivePolyphaseFilter"
    filename = "test_recursivePolyphaseFilter.html"

    logger = debugLogger(loggerName,
                         filename,
                         log_dirname)
    L = 3                                    #Decimation factor
    wp = 0.8*np.pi                           #passband edge (circular normalized frequency)
    K = 6                                   #Degrees of freedom

    R = int(K/(L-1))

    b_digital, a_digital = IIR_NthBand_PolyphaseFilter.nth_band_IIR_polyphaseFilter_linear(L,
                                                                                wp,
                                                                                R,
                                                                                eps=0.001,
                                                                                nIter=20,
                                                                                logger=logger)

    #get poles of branch filters:
    R = int(np.floor(K/(L-1)))

    p = np.zeros((L,R), dtype=complex)
    for branch in range(L):
        p[branch,:] = np.sort_complex(np.roots(a_digital[branch,:]))

    #compare with values from paper
    p_ref = np.zeros((L,R), dtype=complex)
    p_ref[1,:] = np.sort_complex([-0.590592, 0.271216*np.exp(1j*0.332706*np.pi), 0.271216*np.exp(-1j*0.332706*np.pi)])
    p_ref[2,:] = np.sort_complex([-0.806913, 0.194532*np.exp(1j*0.367246*np.pi), 0.194532*np.exp(-1j*0.367246*np.pi)])

    assert np.all((np.isclose(p,p_ref, atol=0.0001))) == True

import numpy as np
import scipy as sp

from polyphaseFilter.filterFactory import rationalSampleRateAlteration
################################################################################
def test_LM_IIR_polyphase():
    #define numerator
    b = np.poly1d(1)
    b *= np.poly1d([1,1])
    b = b.c
    #define denominator
    a = np.poly1d([1])
    a *= np.poly1d([1,4])
    a *= np.poly1d([1,5])
    a = a.c

    rationalSampleRateAlteration.LM_IIR_polyphase(b,a,1)

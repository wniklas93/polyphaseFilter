import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from polyphaseFilter.filterFactory import utilities
################################################################################
def LM_IIR_polyphase(b,a,L,M):
    '''
    This function transforms the given lowpass filter into a polyphase filter
    structure which can be used to alter the sample rate by the given
    rational factor LM.

    Parameters:
    ___________
    b:                      Numerator polynomial (Descending order and negative powers)
    a:                      Denominator polynomial (Descending order and negative powers)
    L:                      Interpolation factor
    M:                      Decimation factor

    return:
    _______
    (a0,b0):                First filter
    (a1,b1):                Second filter
    (a2,b2):                Third filter
    '''

    #get zeros and poles of filter
    b_pp, a_pp = utilities.negDes2posDes(b,a)         #transform into positive powers
    z,p,_ = sp.signal.tf2zpk(b_pp, a_pp)

    #check on conjugate complex pairs:
    

    #assign poles to interpolator and to decimator:
    p0 = p[:L]
    p2 = p[:L]

    #check on conjugate poles if yes--> pole exchange







#%%
